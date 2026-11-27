import os

# Must pop before PyTorch imports
os.environ.pop('NCCL_MIN_NCHANNELS', None)
os.environ.pop('NCCL_MAX_NCHANNELS', None)
os.environ.pop('NCCL_NVLS_ENABLE', None)

import argparse
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, get_nvlink_gbs, get_physical_domain_size, init_dist
from deep_ep.utils.testing import bench, generate_lognormal_topk, parse_num_bytes

# Period of `make_pattern`'s ramp, chosen so adjacent 12 KiB chunks have different contents
NUM_PATTERN_VALUES = (1 << 12) - 1

# Hidden sizes in fp32 elements, including full chunks, partial tails, and buffer reuse
CORRECTNESS_HIDDENS = (1 << 20, 1 << 18, 16392, 6144, 3080, 3072, 3064, 5120, 1536, 384, 24)


def make_pattern(hidden: int, key: int) -> torch.Tensor:
    """A deterministic fp32 blob: a ramp of period `NUM_PATTERN_VALUES` shifted by `key`.

    Every blob in the test carries a distinct key, so a swapped source shows up as a
    mismatch at every element. Adjacent 12 KiB chunks also have different contents, making
    a one-chunk offset error visible.

    The values are small integers on purpose. `lb_reduce_grads` sums redundant expert gradients into
    `expert_grads` with `cp.reduce.async.bulk`, an element-wise atomic add, so the order in
    which several borrowers land is not fixed. Values this size stay exact in fp32, which
    makes the expected value order-independent and lets the check use `torch.equal` rather
    than a tolerance.

    Plain arithmetic rather than an RNG: it must come out bit-identical on every GPU,
    and RNG launch configs depend on device properties.
    """
    idx = torch.arange(hidden, dtype=torch.int64, device='cuda') + key
    return ((idx % NUM_PATTERN_VALUES) - NUM_PATTERN_VALUES // 2).to(torch.float32)


def expert_grad_key(expert_idx: int) -> int:
    """Key of an original expert gradient. Non-negative, while `redundant_expert_grad_key` is negative, so the two
    families of blobs never share a key."""
    return expert_idx


def redundant_expert_grad_key(rank_idx: int, redundant_expert_idx: int, num_redundant_experts_per_rank: int) -> int:
    """Key of a redundant expert gradient, unique over `(rank_idx, redundant_expert_idx)`."""
    return -1 - (rank_idx * num_redundant_experts_per_rank + redundant_expert_idx)


def expected_expert_grads(redundancy_mapping: torch.Tensor, rank_idx: int, rdma_rank_idx: int,
                          nvl_rank_idx: int, num_nvl_ranks: int, num_redundant_experts_per_rank: int,
                          num_main_experts_per_rank: int, hidden: int) -> torch.Tensor:
    """What `expert_grads` must hold afterwards: its seed plus the gradients from every matching redundant expert.

    A rank never pulls from its own row.
    """
    expected = torch.stack([make_pattern(hidden, expert_grad_key(rank_idx * num_main_experts_per_rank + local_expert_idx))
                            for local_expert_idx in range(num_main_experts_per_rank)])
    for remote_nvl_rank_idx in range(num_nvl_ranks):
        if remote_nvl_rank_idx == nvl_rank_idx:
            continue
        remote_rank_idx = rdma_rank_idx * num_nvl_ranks + remote_nvl_rank_idx
        for redundant_expert_idx in range(num_redundant_experts_per_rank):
            expert_idx = int(redundancy_mapping[remote_nvl_rank_idx, redundant_expert_idx].item())
            if expert_idx >= 0 and expert_idx // num_main_experts_per_rank == nvl_rank_idx:
                expected[expert_idx % num_main_experts_per_rank] += \
                    make_pattern(hidden, redundant_expert_grad_key(remote_rank_idx, redundant_expert_idx, num_redundant_experts_per_rank))
    return expected


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_rdma_ranks, num_nvl_ranks = get_physical_domain_size(group)
    rdma_rank_idx, nvl_rank_idx = rank_idx // num_nvl_ranks, rank_idx % num_nvl_ranks

    # Settings
    num_experts = args.num_experts
    assert num_experts % num_ranks == 0
    num_main_experts_per_rank = num_experts // num_ranks
    # `lb_get_redundancy_mapping` hands back one copy slot per local expert, empty slots marked -1
    num_redundant_experts_per_rank, num_sms = num_main_experts_per_rank, args.num_sms

    # Correctness sweeps a table of hidden sizes; the benchmark runs on its own, much larger
    # one, so that its GB/s is not distorted by launch overhead.
    perf_hidden = args.num_hidden_bytes // torch.float32.itemsize
    hiddens = (*CORRECTNESS_HIDDENS, perf_hidden)

    assert all(h * torch.float32.itemsize % deep_ep.get_num_tma_alignment() == 0 for h in hiddens), \
        'Each expert gradient must be TMA-aligned'
    assert num_experts + num_ranks * num_redundant_experts_per_rank < NUM_PATTERN_VALUES, \
        'Two blobs would share a key modulo the pattern period'

    # Plan every symmetric redundant expert gradient pool once, including the benchmark pools.
    plan = deep_ep.BufferAllocator()
    redundant_expert_grad_sets = [plan.allocate((num_redundant_experts_per_rank, h), torch.float32) for h in hiddens]
    buffer = deep_ep.EPBuffer(group, num_bytes=deep_ep.get_num_allocation_alignment(),
                              lb_allocation_plan_or_num_bytes=plan, explicitly_destroy=True)
    num_sms = buffer.lb_get_theoretical_num_sms() if num_sms == 0 else num_sms

    dist_print(f'Config:\n'
               f' > Ranks: {num_rdma_ranks} x {num_nvl_ranks}\n'
               f' > Experts: {num_experts} ({num_main_experts_per_rank} local), '
               f'redundant experts: {num_redundant_experts_per_rank}\n'
               f' > Correctness hidden sweep: {list(CORRECTNESS_HIDDENS)} fp32\n'
               f' > Perf hidden: {perf_hidden} fp32 '
               f'({perf_hidden * torch.float32.itemsize / 1e6:.1f} MB per expert)\n'
               f' > #SM: {num_sms}\n',
               once_in_node=True)

    # This rank's original expert gradients, one set per hidden size. They stay
    # local -- the kernel only reduces into them, so they never have to be symmetric.
    expert_grad_sets = [torch.empty((num_main_experts_per_rank, h), dtype=torch.float32, device='cuda') for h in hiddens]

    *sweep_expert_grad_sets, perf_expert_grads = expert_grad_sets
    *sweep_redundant_expert_grad_sets, perf_redundant_expert_grads = redundant_expert_grad_sets

    def current_mapping(round_idx: int = 0) -> torch.Tensor:
        """The redundancy mapping this round runs with: `lb_get_redundancy_mapping`'s plan for a lognormal load."""
        topk_idx = generate_lognormal_topk(args.num_tokens, num_experts, args.num_topk, rank_idx, args.seed + round_idx)
        _, redundancy_mapping = buffer.lb_get_redundancy_mapping(topk_idx, num_experts, num_redundant_experts_per_rank, args.token_alignment, 0)
        return redundancy_mapping

    def seed_buffers(hidden: int, expert_grads: torch.Tensor, redundant_expert_grads: torch.Tensor) -> None:
        """Refill both sides. `expert_grads` accumulates, so it must be reseeded every round."""
        for local_expert_idx in range(num_main_experts_per_rank):
            expert_grads[local_expert_idx] = make_pattern(
                hidden, expert_grad_key(rank_idx * num_main_experts_per_rank + local_expert_idx))
        for redundant_expert_idx in range(num_redundant_experts_per_rank):
            redundant_expert_grads[redundant_expert_idx] = make_pattern(hidden, redundant_expert_grad_key(rank_idx, redundant_expert_idx, num_redundant_experts_per_rank))

    # Check correctness over every hidden size.
    dist_print(f'Checking correctness over {len(CORRECTNESS_HIDDENS)} hidden sizes '
               f'x {args.num_rounds} rounds ...', once_in_node=True)
    for hidden, expert_grads, redundant_expert_grads in zip(CORRECTNESS_HIDDENS, sweep_expert_grad_sets, sweep_redundant_expert_grad_sets, strict=True):
        for round_idx in range(args.num_rounds):
            redundancy_mapping = current_mapping(round_idx)

            # The seed must be visible to peers before anyone pulls
            seed_buffers(hidden, expert_grads, redundant_expert_grads)
            torch.cuda.synchronize()
            dist.barrier(group)

            buffer.lb_reduce_grads(
                redundant_expert_grads, expert_grads, redundancy_mapping, num_sms=num_sms,
                previous_event=buffer.capture() if round_idx % 2 else None).wait()
            torch.cuda.synchronize()

            expected = expected_expert_grads(
                redundancy_mapping, rank_idx, rdma_rank_idx, nvl_rank_idx, num_nvl_ranks,
                num_redundant_experts_per_rank, num_main_experts_per_rank, hidden)
            for local_expert_idx in range(num_main_experts_per_rank):
                assert torch.equal(expert_grads[local_expert_idx], expected[local_expert_idx]), \
                    f'{hidden=}, {round_idx=}: gradient of local expert ' \
                    f'{local_expert_idx} is wrong ' \
                    f'({int((expert_grads[local_expert_idx] != expected[local_expert_idx]).sum().item())} ' \
                    f'of {hidden} elements)'

            # Never writes the redundant expert gradient pool. A swapped TMA source/destination would land
            # here rather than in the check above, so it is worth testing explicitly.
            dist.barrier(group)
            for redundant_expert_idx in range(num_redundant_experts_per_rank):
                assert torch.equal(redundant_expert_grads[redundant_expert_idx],
                                   make_pattern(hidden, redundant_expert_grad_key(rank_idx, redundant_expert_idx, num_redundant_experts_per_rank))), \
                    f'{hidden=}, {round_idx=}: redundant expert gradient {redundant_expert_idx} was modified'

    # Perf: count the redundant experts contributing gradients to each rank in this NVLink domain.
    # That is the same set `lb_prefetch_weights` pushes, just read from the other end.
    redundancy_mapping = current_mapping()
    owners = torch.div(redundancy_mapping, num_main_experts_per_rank, rounding_mode='floor')
    redundant_experts_per_owner = [int(((redundancy_mapping >= 0) & (owners == r)).sum().item()) for r in range(num_nvl_ranks)]
    num_bytes_per_redundant_expert = perf_hidden * torch.float32.itemsize
    num_pulled_bytes = redundant_experts_per_owner[nvl_rank_idx] * num_bytes_per_redundant_expert
    num_critical_bytes = max(redundant_experts_per_owner) * num_bytes_per_redundant_expert

    # Values are never checked here, so a zero fill is enough -- it only keeps the repeated
    # accumulation away from whatever `torch.empty` left in the pages.
    perf_expert_grads.zero_()
    perf_redundant_expert_grads.zero_()
    torch.cuda.synchronize()
    dist.barrier(group)

    t, _, _ = bench(lambda: buffer.lb_reduce_grads(
        perf_redundant_expert_grads, perf_expert_grads, redundancy_mapping, num_sms=num_sms).wait())
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > Redundant experts pulled per NVLink rank: {redundant_experts_per_owner} (this rank {redundant_experts_per_owner[nvl_rank_idx]})',
               once_in_node=True)
    dist_print(f' > Rank {rank_idx} perf ({num_sms=}): '
               f'{t * 1e6:7.1f} us | '
               f'this rank {num_pulled_bytes / 1e6:.1f} MB @ {num_pulled_bytes / t / 1e9:5.1f} GB/s | '
               f'critical path {num_critical_bytes / 1e6:.1f} MB @ '
               f'{num_critical_bytes / t / 1e9:5.1f} GB/s '
               f'({num_critical_bytes / t / 1e9 / get_nvlink_gbs() * 100:5.1f}% of NVLink)')

    # Destroy. Redundant expert gradient pools are views into the buffer, so release them first.
    del plan, redundant_expert_grad_sets, sweep_redundant_expert_grad_sets, perf_redundant_expert_grads
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test expert grad reduction')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--num-hidden-bytes', type=parse_num_bytes, default=64 << 20,
                        help='Grad bytes per expert in the benchmark, with an optional binary '
                             'suffix: 64M (default), 512K, or plain bytes.')
    parser.add_argument('--num-rounds', type=int, default=2,
                        help='Copy tables per hidden size, on top of the size sweep')
    parser.add_argument('--num-tokens', type=int, default=8192,
                        help='Tokens per rank fed to lb_get_redundancy_mapping')
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--token-alignment', type=int, default=128)
    parser.add_argument('--num-sms', type=int, default=0, help='0 uses the bandwidth-based SM estimate')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

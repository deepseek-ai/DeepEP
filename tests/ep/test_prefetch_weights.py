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

# Written into every redundant expert slot before a push. Slots whose expert index is `-1`
# must still hold it afterwards, which also catches strayed writes.
SENTINEL = 0xa5

# The three payloads an expert pushes, each checked on its own
PAYLOAD_WEIGHTS, PAYLOAD_SF0, PAYLOAD_SF1 = range(3)
PAYLOAD_NAMES = ('weights', 'sf0', 'sf1')
NUM_PAYLOADS = len(PAYLOAD_NAMES)

# Period of `make_blob`'s ramp, chosen so adjacent 12 KiB chunks have different contents.
NUM_BLOB_VALUES = (1 << 8) - 1

# Payload sizes the correctness sweep covers, in bytes: `(weights, sf0, sf1)` per expert.
PAYLOAD_SWEEP = (
    (4 << 20, 1 << 14, 1 << 13),  # Reuse each warp's buffer through multiple barrier phase changes
    (1 << 20, 1 << 14, 1 << 12),  # Large weights with shorter SF payloads
    (49152, 24576, 12288),        # Exact multiples of the 12 KiB chunk capacity
    (12320, 12256, 32),           # Just above and below one chunk, with a short tail
    (24640, 12320, 64),           # Partial tails after different numbers of full chunks
    (49152, 3072, 1024),          # Full weight chunks with shorter SF chunks
    (20480, 20480, 20480),        # Every payload ends in a partial chunk
    (1152, 544, 288),             # Small aligned payloads all use TMA
    (96, 64, 32),                 # Fewer chunks than the grid has warps
    (64, 32, 32),                 # Minimum-size SF payloads
)


def reshape_weights(weights: list[torch.Tensor], trailing_shape: tuple[int, ...]) -> list[torch.Tensor]:
    return [tensor.unflatten(1, (-1, *trailing_shape)) for tensor in weights]


def make_blob(global_expert_idx: int, num_bytes: int, payload_idx: int) -> torch.Tensor:
    """One payload of one expert: a ramp of period `NUM_BLOB_VALUES`, shifted by a key
    unique to `(global_expert_idx, payload_idx)`.

    A blob pushed from the wrong expert, or into the wrong payload, therefore differs at
    every byte from the expected one rather than at a lucky few.

    Plain arithmetic rather than an RNG: it must come out bit-identical on every GPU,
    and RNG launch configs depend on device properties.
    """
    key = global_expert_idx * NUM_PAYLOADS + payload_idx
    idx = torch.arange(num_bytes, dtype=torch.int64, device='cuda') + key
    return (idx % NUM_BLOB_VALUES).to(torch.uint8)


def check(redundant_expert_weights: torch.Tensor, redundancy_mapping: torch.Tensor,
          nvl_rank_idx: int, domain_expert_offset: int, num_bytes: int, payload_idx: int, where: str) -> None:
    """Every redundant expert slot of this rank must hold its expert's blob, or the sentinel if unassigned."""
    name = PAYLOAD_NAMES[payload_idx]
    for redundant_expert_idx in range(redundant_expert_weights.size(0)):
        expert_idx = int(redundancy_mapping[nvl_rank_idx, redundant_expert_idx].item())
        if expert_idx < 0:
            assert (redundant_expert_weights[redundant_expert_idx] == SENTINEL).all(), \
                f'{where}: {name} redundant expert {redundant_expert_idx} is unassigned but was written'
        else:
            assert torch.equal(
                redundant_expert_weights[redundant_expert_idx],
                make_blob(domain_expert_offset + expert_idx, num_bytes, payload_idx),
            ), f'{where}: {name} redundant expert {redundant_expert_idx} does not match expert {expert_idx}'


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

    # Correctness sweeps a table of payload sizes.
    perf_payload = (args.num_hidden_bytes, args.num_sf0_bytes, args.num_sf1_bytes)
    payloads = (*PAYLOAD_SWEEP, perf_payload)

    assert all(n % deep_ep.get_num_tma_alignment() == 0 for payload in payloads for n in payload), \
        'Each expert payload must be TMA-aligned'
    assert num_experts * NUM_PAYLOADS <= NUM_BLOB_VALUES, \
        'Two blobs would share a key modulo the pattern period'

    # Plan every symmetric redundant expert pool once, including the benchmark pools.
    plan = deep_ep.BufferAllocator()
    redundant_expert_weight_sets = [[plan.allocate((num_redundant_experts_per_rank, n), torch.uint8) for n in payload]
                  for payload in payloads]
    buffer = deep_ep.EPBuffer(group, num_bytes=deep_ep.get_num_allocation_alignment(),
                              lb_allocation_plan_or_num_bytes=plan, explicitly_destroy=True)
    num_sms = buffer.lb_get_theoretical_num_sms() if num_sms == 0 else num_sms

    dist_print(f'Config:\n'
               f' > Ranks: {num_rdma_ranks} x {num_nvl_ranks}\n'
               f' > Experts: {num_experts} ({num_main_experts_per_rank} local), '
               f'redundant experts: {num_redundant_experts_per_rank}\n'
               f' > Correctness payload sweep: {[list(payload) for payload in PAYLOAD_SWEEP]} B\n'
               f' > Perf bytes per expert: {perf_payload[PAYLOAD_WEIGHTS]} weights, '
               f'{perf_payload[PAYLOAD_SF0]}/{perf_payload[PAYLOAD_SF1]} SF '
               f'({sum(perf_payload) / 1e6:.1f} MB total)\n'
               f' > #SM: {num_sms}\n',
               once_in_node=True)

    # This rank's original expert weights, one set per payload size.
    expert_weight_sets = [[torch.empty((num_main_experts_per_rank, n), dtype=torch.uint8, device='cuda') for n in payload]
                 for payload in payloads]

    *sweep_expert_weight_sets, perf_expert_weights = expert_weight_sets
    *sweep_redundant_expert_weight_sets, perf_redundant_expert_weights = redundant_expert_weight_sets

    for payload, expert_weights in zip(PAYLOAD_SWEEP, sweep_expert_weight_sets, strict=True):
        for local_expert_idx in range(num_main_experts_per_rank):
            expert_idx = rank_idx * num_main_experts_per_rank + local_expert_idx
            for payload_idx, num_bytes in enumerate(payload):
                expert_weights[payload_idx][local_expert_idx] = make_blob(expert_idx, num_bytes, payload_idx)

    def current_mapping(round_idx: int = 0) -> torch.Tensor:
        """The redundancy mapping this round runs with: `lb_get_redundancy_mapping`'s plan for a lognormal load."""
        topk_idx = generate_lognormal_topk(args.num_tokens, num_experts, args.num_topk, rank_idx, args.seed + round_idx)
        _, redundancy_mapping = buffer.lb_get_redundancy_mapping(topk_idx, num_experts, num_redundant_experts_per_rank, args.token_alignment, 0)
        return redundancy_mapping

    def prefetch(redundant_expert_weights: list, expert_weights: list, redundancy_mapping: torch.Tensor,
                 with_sf: bool, previous_event=None):
        return buffer.lb_prefetch_weights(
            redundant_expert_weights if with_sf else redundant_expert_weights[:1],
            expert_weights if with_sf else expert_weights[:1],
            redundancy_mapping, num_sms=num_sms, previous_event=previous_event)

    # Check correctness over every payload size.
    dist_print(f'Checking correctness over {len(PAYLOAD_SWEEP)} payload sizes '
               f'x {args.num_rounds} rounds ...', once_in_node=True)
    for payload, redundant_expert_weights, expert_weights in zip(PAYLOAD_SWEEP, sweep_redundant_expert_weight_sets, sweep_expert_weight_sets, strict=True):
        for with_sf in (True, False):
            for round_idx in range(args.num_rounds):
                where = f'{payload}, {with_sf=}, {round_idx=}'
                redundancy_mapping = current_mapping(round_idx)

                # The sentinel fill must be visible to peers before anyone pushes
                for tensor in redundant_expert_weights:
                    tensor.fill_(SENTINEL)
                torch.cuda.synchronize()
                dist.barrier(group)

                # Alternate flat tensors with 3D/4D views; the expert byte payloads stay the same.
                prefetch(
                    reshape_weights(redundant_expert_weights, (2, 8)) if round_idx % 2 else redundant_expert_weights,
                    reshape_weights(expert_weights, (16,)) if round_idx % 2 else expert_weights,
                    redundancy_mapping, with_sf, previous_event=buffer.capture() if round_idx % 2 else None).wait()
                torch.cuda.synchronize()

                num_pushed = NUM_PAYLOADS if with_sf else 1
                domain_expert_offset = rdma_rank_idx * num_nvl_ranks * num_main_experts_per_rank
                for payload_idx in range(num_pushed):
                    check(redundant_expert_weights[payload_idx], redundancy_mapping, nvl_rank_idx, domain_expert_offset,
                          payload[payload_idx], payload_idx, where)
                for payload_idx in range(num_pushed, NUM_PAYLOADS):
                    assert (redundant_expert_weights[payload_idx] == SENTINEL).all(), \
                        f'{where}: {PAYLOAD_NAMES[payload_idx]} was pushed even though it was not requested'

    # Perf: count the redundant experts requesting weights from each rank in this NVLink domain.
    redundancy_mapping = current_mapping()
    owners = torch.div(redundancy_mapping, num_main_experts_per_rank, rounding_mode='floor')
    redundant_experts_per_sender = [int(((redundancy_mapping >= 0) & (owners == r)).sum().item()) for r in range(num_nvl_ranks)]
    num_bytes_per_redundant_expert = sum(perf_payload)
    num_pushed_bytes = redundant_experts_per_sender[nvl_rank_idx] * num_bytes_per_redundant_expert
    num_critical_bytes = max(redundant_experts_per_sender) * num_bytes_per_redundant_expert

    t, _, _ = bench(lambda: prefetch(perf_redundant_expert_weights, perf_expert_weights, redundancy_mapping, True).wait())
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > Redundant experts pushed per NVLink rank: {redundant_experts_per_sender} (this rank {redundant_experts_per_sender[nvl_rank_idx]})',
               once_in_node=True)
    dist_print(f' > Rank {rank_idx} perf ({num_sms=}): '
               f'{t * 1e6:7.1f} us | '
               f'this rank {num_pushed_bytes / 1e6:.1f} MB @ {num_pushed_bytes / t / 1e9:5.1f} GB/s | '
               f'critical path {num_critical_bytes / 1e6:.1f} MB @ '
               f'{num_critical_bytes / t / 1e9:5.1f} GB/s '
               f'({num_critical_bytes / t / 1e9 / get_nvlink_gbs() * 100:5.1f}% of NVLink)')

    # Destroy. Redundant expert weight pools are views into the buffer, so release them first.
    del plan, redundant_expert_weight_sets, sweep_redundant_expert_weight_sets, perf_redundant_expert_weights
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test weight prefetch')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--num-hidden-bytes', type=parse_num_bytes, default=64 << 20,
                        help='Weight bytes per expert in the benchmark, with an optional binary '
                             'suffix: 64M (default), 512K, or plain bytes. Correctness sweeps '
                             'its own table of sizes instead')
    parser.add_argument('--num-sf0-bytes', type=parse_num_bytes, default=1 << 14,
                        help='First scale-factor payload in the benchmark')
    parser.add_argument('--num-sf1-bytes', type=parse_num_bytes, default=1 << 12,
                        help='Second scale-factor payload in the benchmark')
    parser.add_argument('--num-rounds', type=int, default=2,
                        help='Copy tables per payload size, on top of the size sweep')
    parser.add_argument('--num-tokens', type=int, default=8192,
                        help='Tokens per rank fed to lb_get_redundancy_mapping')
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--token-alignment', type=int, default=128)
    parser.add_argument('--num-sms', type=int, default=0, help='0 uses the bandwidth-based SM estimate')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

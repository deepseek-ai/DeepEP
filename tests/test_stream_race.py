"""
Test for GitHub issue #589: buffer.combine() produces NaN when
torch.use_deterministic_algorithms(True) due to stream race on output tensor.

The bug is a write-write CUDA stream race between PyTorch's NaN-fill kernel
(on compute_stream) and DeepEP's combine kernel (on comm_stream). When
fill_uninitialized_memory=True, torch::empty() launches a NaN-fill kernel on
the current stream. Because stream_wait happens before the allocation, the
comm_stream never waits for this fill kernel.

Usage:
    python test_stream_race.py [--num-processes N]

All four configs assert zero NaN. Config 1 is the regression gate — before
the fix it probabilistically produces NaN (~18% per iteration on 8xH20),
but after the fix it deterministically produces zero. The race is probabilistic
(GPU kernel scheduling is nondeterministic), but the fix is deterministic
(stream_wait guarantees ordering). So we assert the post-fix invariant.

Test matrix (all assert zero NaN):
    Config 1: deterministic=True,  fill=True,  alloc_on_comm=False  -> regression gate
    Config 2: deterministic=True,  fill=True,  alloc_on_comm=True   -> control (same stream)
    Config 3: deterministic=True,  fill=False, alloc_on_comm=False  -> control (no fill kernel)
    Config 4: deterministic=False, fill=default, alloc_on_comm=False -> control (no fill kernel)
"""

import argparse
import os
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, inplace_unique


NUM_ITERS = 200


def run_dispatch_combine_loop(buffer, x, topk_idx, topk_weights,
                              num_tokens_per_rank, is_token_in_rank,
                              num_tokens_per_expert, config, num_iters,
                              allocate_on_comm_stream):
    """Run cached dispatch+combine in a tight loop, return NaN count."""
    # Initial dispatch to get handle
    recv_x, recv_topk_idx, recv_topk_weights, _, handle, event = buffer.dispatch(
        x=x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config, async_finish=True,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    event.current_stream_wait()

    # Tight loop: cached dispatch + combine
    nan_count = 0
    for _ in range(num_iters):
        recv_x, _, _, _, _, event = buffer.dispatch(
            x=x, handle=handle, config=config, async_finish=True,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        event.current_stream_wait()

        combined_x, _, event = buffer.combine(
            x=recv_x, handle=handle, config=config, async_finish=True,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        event.current_stream_wait()
        torch.cuda.synchronize()

        if torch.isnan(combined_x).any().item():
            nan_count += 1

    return nan_count


def test_main(local_rank, num_ranks, rank, buffer, group):
    num_tokens = 4096
    hidden = 4096
    num_topk = 8
    num_experts = 256
    assert num_experts % num_ranks == 0

    # Random data
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Layout
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)

    config = deep_ep.Config(24, 8, 256)

    # ── Config 1: deterministic=True, fill=True, alloc_on_comm=False ──
    # This is the bug: NaN-fill races with combine kernel
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    group.barrier()
    nan_count = run_dispatch_combine_loop(
        buffer, x, topk_idx, topk_weights,
        num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        config, NUM_ITERS, allocate_on_comm_stream=False)

    # Reduce across ranks
    nan_tensor = torch.tensor([nan_count], dtype=torch.int64, device='cuda')
    dist.all_reduce(nan_tensor, group=group)
    total_nans_config1 = nan_tensor.item()

    if local_rank == 0:
        status = 'PASSED' if total_nans_config1 == 0 else 'FAILED'
        print(f'[config 1] {status}: deterministic=True, fill=True, '
              f'alloc_on_comm=False -> {total_nans_config1} NaN', flush=True)
    assert total_nans_config1 == 0, \
        f'Config 1 produced {total_nans_config1} NaN iterations (stream race bug #589)'

    # ── Config 2: deterministic=True, fill=True, alloc_on_comm=True ──
    # alloc_on_comm_stream=True serializes fill+combine on same stream
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    group.barrier()
    nan_count = run_dispatch_combine_loop(
        buffer, x, topk_idx, topk_weights,
        num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        config, NUM_ITERS, allocate_on_comm_stream=True)

    nan_tensor = torch.tensor([nan_count], dtype=torch.int64, device='cuda')
    dist.all_reduce(nan_tensor, group=group)
    total_nans = nan_tensor.item()

    if local_rank == 0:
        status = 'PASSED' if total_nans == 0 else 'FAILED'
        print(f'[config 2] {status}: deterministic=True, fill=True, '
              f'alloc_on_comm=True -> {total_nans} NaN', flush=True)
    assert total_nans == 0, f'Config 2 should never produce NaN, got {total_nans}'

    # ── Config 3: deterministic=True, fill=False, alloc_on_comm=False ──
    # No fill kernel -> nothing to race with
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = False

    group.barrier()
    nan_count = run_dispatch_combine_loop(
        buffer, x, topk_idx, topk_weights,
        num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        config, NUM_ITERS, allocate_on_comm_stream=False)

    nan_tensor = torch.tensor([nan_count], dtype=torch.int64, device='cuda')
    dist.all_reduce(nan_tensor, group=group)
    total_nans = nan_tensor.item()

    if local_rank == 0:
        status = 'PASSED' if total_nans == 0 else 'FAILED'
        print(f'[config 3] {status}: deterministic=True, fill=False, '
              f'alloc_on_comm=False -> {total_nans} NaN', flush=True)
    assert total_nans == 0, f'Config 3 should never produce NaN, got {total_nans}'

    # ── Config 4: deterministic=False, alloc_on_comm=False ──
    # No deterministic mode -> no fill kernel
    torch.use_deterministic_algorithms(False)

    group.barrier()
    nan_count = run_dispatch_combine_loop(
        buffer, x, topk_idx, topk_weights,
        num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        config, NUM_ITERS, allocate_on_comm_stream=False)

    nan_tensor = torch.tensor([nan_count], dtype=torch.int64, device='cuda')
    dist.all_reduce(nan_tensor, group=group)
    total_nans = nan_tensor.item()

    if local_rank == 0:
        status = 'PASSED' if total_nans == 0 else 'FAILED'
        print(f'[config 4] {status}: deterministic=False, '
              f'alloc_on_comm=False -> {total_nans} NaN', flush=True)
    assert total_nans == 0, f'Config 4 should never produce NaN, got {total_nans}'

    # Restore defaults
    torch.use_deterministic_algorithms(False)


def test_loop(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    buffer = deep_ep.Buffer(group,
                            int(2e9),
                            0,
                            low_latency_mode=False,
                            num_qps_per_rank=1,
                            explicitly_destroy=True,
                            allow_mnnvl=False,
                            use_fabric=False)
    torch.manual_seed(rank)

    if local_rank == 0:
        print(f'Testing stream race (issue #589) with {num_ranks} ranks, '
              f'{NUM_ITERS} iterations per config', flush=True)
        print('', flush=True)

    test_main(local_rank, num_ranks, rank, buffer, group)

    if local_rank == 0:
        print('', flush=True)
        print('All controls passed. Config 1 result above shows whether '
              'the stream race was triggered.', flush=True)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test stream race (issue #589)')
    parser.add_argument('--num-processes', type=int, default=8,
                        help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-iters', type=int, default=200,
                        help='Iterations per config (default: 200)')
    args = parser.parse_args()
    NUM_ITERS = args.num_iters
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args),
                                nprocs=args.num_processes)

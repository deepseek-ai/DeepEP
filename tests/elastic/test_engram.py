import argparse
import os
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist, dist_print
from deep_ep.utils.math import per_token_cast_to_fp8
from deep_ep.utils.testing import bench_kineto


# noinspection PyUnboundLocalVariable,PyShadowingNames
@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    dtype = torch.float8_e4m3fn if args.use_fp8 else torch.bfloat16
    num_gpu_bytes, num_cpu_bytes = deep_ep.ElasticBuffer.get_engram_storage_size_hint(
        args.num_entries, args.hidden, args.num_tokens * args.num_entries_per_token, dtype)

    # 1 QP uses 1 SM
    num_qps = args.num_qps
    if num_qps == 0:
        num_qps = torch.cuda.get_device_properties('cuda').multi_processor_count

    # Allocate buffer
    dist_print(f'Config:\n'
               f' > Ranks: {num_ranks}\n'
               f' > QPs: {num_qps}\n'
               f' > Entries per rank: {args.num_entries}, hidden: {args.hidden}\n'
               f' > Tokens to fetch: {args.num_tokens} x {args.num_entries_per_token} entries\n'
               f' > Storage per rank: {args.num_entries * args.hidden * dtype.itemsize / 1024 / 1024:.1f} MB\n',
               once_in_node=True)
    buffer = deep_ep.ElasticBuffer(
        group,
        num_bytes=num_gpu_bytes + num_cpu_bytes, num_cpu_bytes=num_cpu_bytes,
        explicitly_destroy=True, num_allocated_qps=num_qps,
        allow_hybrid_mode=args.allow_hybrid_mode, allow_multiple_reduction=False)

    # Write buffer: each rank writes its own local storage into the NCCL window
    local_bf16 = torch.randn((args.num_entries, args.hidden), dtype=torch.bfloat16, device='cuda')
    local_sf = None
    if args.use_fp8:
        local_storage, local_sf = per_token_cast_to_fp8(local_bf16)
    else:
        local_storage = local_bf16

    # Replicate the scaling factors so any rank can fetch any entry's factors locally
    sf = None
    if args.use_fp8:
        sf = torch.empty((num_ranks * args.num_entries, local_sf.shape[1]), dtype=local_sf.dtype, device='cuda')
        dist.all_gather_into_tensor(sf, local_sf, group)

    buffer.engram_write(local_storage, sf=sf)

    # Generate random indices to fetch
    indices = torch.randint(0, num_ranks * args.num_entries,
                            (args.num_tokens, args.num_entries_per_token), device='cuda', dtype=torch.int)

    # Correctness check
    if not args.skip_check:
        global_storage = torch.empty((num_ranks * args.num_entries, args.hidden), dtype=dtype, device='cuda')
        dist.all_gather_into_tensor(global_storage, local_storage, group)
        ref_data = global_storage[indices.view(-1)].view(args.num_tokens, -1)
        ref_sf = sf[indices.view(-1)].view(args.num_tokens, -1) if args.use_fp8 else None

        for use_tma_aligned_col_major_sf in (False, True) if args.use_fp8 else (False,):
            data, fetched_sf = buffer.engram_fetch(indices,
                                                   use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf)()
            assert torch.equal(ref_data, data), 'data mismatch'
            if args.use_fp8:
                assert torch.equal(ref_sf, fetched_sf), f'fp8 scaling-factor mismatch ({use_tma_aligned_col_major_sf=})'

    # Performance test
    dist_print('Running performance test ...', once_in_node=True)
    msg_bytes = args.hidden * dtype.itemsize
    num_fetched_bytes = args.num_tokens * args.num_entries_per_token * msg_bytes

    # Measure fetch + wait (end-to-end)
    def fetch_and_wait():
        # noinspection PyShadowingNames
        hook = buffer.engram_fetch(indices, use_tma_aligned_col_major_sf=True)
        hook()

    issue_t, wait_t = bench_kineto(
        fetch_and_wait,
        kernel_names=('engram_fetch_impl', 'engram_fetch_wait_impl'),
        barrier_comm_profiling=True,
        barrier=buffer.barrier,
        trace_path=f'{args.dump_profile_traces}/engram_fetch_rank{buffer.rank_idx}.json' if args.dump_profile_traces else None)
    mpps = args.num_tokens * args.num_entries_per_token / (issue_t + wait_t) / 1e6
    dist_print(f' > Rank {rank:3}/{num_ranks} | '
               f'issue: {issue_t * 1e6:.1f} us, '
               f'wait: {wait_t * 1e6:.1f} us, '
               f'{num_fetched_bytes / (issue_t + wait_t) / 1e9:.1f} GB/s, '
               f'bytes: {num_fetched_bytes / 1024 / 1024:.1f} MB, '
               f'{mpps:.2f} MPPS ({msg_bytes} B/msg)')
    dist_print('', once_in_node=True)

    # Destroy the runtime and communication group
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test engram fetch kernels')
    parser.add_argument('--num-processes', type=int, default=4, help='Number of processes to spawn')
    parser.add_argument('--num-qps', type=int, default=0, help='Number of QPs used (0 for maximum)')
    parser.add_argument('--num-entries', type=int, default=524288, help='Number of entries per rank')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num-tokens', type=int, default=512, help='Number of tokens to fetch')
    parser.add_argument('--num-entries-per-token', type=int, default=24, help='Number of entries concatenated per token')
    parser.add_argument('--skip-check', action='store_true', help='Skip correctness check')
    parser.add_argument('--use-fp8', action='store_true', help='Store entries in FP8 with replicated scaling factors')
    parser.add_argument('--allow-hybrid-mode', action='store_true', help='Enable hybrid mode (multi-plane)')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    # Launch
    num_processes = args.num_processes
    torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)

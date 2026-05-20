import argparse
import os
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist, dist_print
from deep_ep.utils.testing import bench_kineto


# noinspection PyUnboundLocalVariable,PyShadowingNames
@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_gpu_bytes, num_cpu_bytes = deep_ep.ElasticBuffer.get_engram_storage_size_hint(
        args.num_entries, args.hidden, args.num_tokens, torch.bfloat16)

    # 1 QP uses 1 SM
    num_qps = args.num_qps
    if num_qps == 0:
        num_qps = torch.cuda.get_device_properties('cuda').multi_processor_count

    # Allocate buffer
    dist_print(f'Config:\n'
               f' > Ranks: {num_ranks}\n'
               f' > QPs: {num_qps}\n'
               f' > Entries per rank: {args.num_entries}, hidden: {args.hidden}\n'
               f' > Tokens to fetch: {args.num_tokens}\n'
               f' > Storage per rank: {args.num_entries * args.hidden * 2 / 1024 / 1024:.1f} MB\n',
               once_in_node=True)
    buffer = deep_ep.ElasticBuffer(
        group,
        num_bytes=num_gpu_bytes + num_cpu_bytes, num_cpu_bytes=num_cpu_bytes,
        explicitly_destroy=True, num_allocated_qps=num_qps,
        allow_hybrid_mode=args.allow_hybrid_mode, allow_multiple_reduction=False)

    # Write buffer: each rank writes its own local storage into the NCCL window
    local_storage = torch.randn((args.num_entries, args.hidden), dtype=torch.bfloat16, device='cuda')
    global_storage = torch.empty((num_ranks * args.num_entries, args.hidden), dtype=torch.bfloat16, device='cuda')
    dist.all_gather_into_tensor(global_storage, local_storage, group)
    buffer.engram_write(local_storage)

    # Generate random indices to fetch
    indices = torch.randint(0, num_ranks * args.num_entries, (args.num_tokens, ), device='cuda', dtype=torch.int)

    # Correctness check
    ref_fetched = global_storage[indices]
    hook = buffer.engram_fetch(indices)
    fetched = hook()
    if not args.skip_check:
        assert torch.equal(ref_fetched, fetched), f'{(ref_fetched - fetched).abs().max().item()}'

    # Performance test
    dist_print('Running performance test ...', once_in_node=True)
    msg_bytes = args.hidden * 2  # bfloat16
    num_fetched_bytes = args.num_tokens * msg_bytes

    # Measure fetch + wait (end-to-end)
    def fetch_and_wait():
        # noinspection PyShadowingNames
        hook = buffer.engram_fetch(indices)
        hook()

    issue_t, wait_t = bench_kineto(
        fetch_and_wait,
        kernel_names=('engram_fetch_impl', 'engram_fetch_wait_impl'),
        barrier_comm_profiling=True,
        barrier=buffer.barrier,
        trace_path=f'{args.dump_profile_traces}/engram_fetch_rank{buffer.rank_idx}.json' if args.dump_profile_traces else None)
    mpps = args.num_tokens / (issue_t + wait_t) / 1e6
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
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens to fetch')
    parser.add_argument('--skip-check', action='store_true', help='Skip correctness check')
    parser.add_argument('--allow-hybrid-mode', action='store_true', help='Enable hybrid mode (multi-plane)')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    # Launch
    num_processes = args.num_processes
    torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)

import argparse
import os
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist, dist_print
from deep_ep.utils.math import ceil_div, per_token_cast_to_fp8
from deep_ep.utils.testing import bench_kineto


# noinspection PyUnboundLocalVariable,PyShadowingNames
@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    dtype = torch.float8_e4m3fn if args.use_fp8 else torch.bfloat16
    num_entries_per_layer = [int(x) for x in args.num_entries_per_layer.split(',')]
    num_layers = len(num_entries_per_layer)
    num_sf_packs = ceil_div(args.hidden, 128) if args.use_fp8 else 0
    num_gpu_bytes, num_rdma_storage_bytes = deep_ep.EngramBuffer.get_storage_size_hint(
        group, num_entries_per_layer, args.hidden, args.num_tokens, args.num_entries_per_token,
        dtype, num_sf_packs)

    num_allocated_qps, qp_depth, restrict_rd_atomic = deep_ep.EngramBuffer.get_theoretical_config(
        group, num_layers, args.num_tokens, args.num_entries_per_token)
    num_qps = args.num_qps if args.num_qps else num_allocated_qps

    # Allocate buffer
    dist_print(f'Config:\n'
               f' > Ranks: {num_ranks}\n'
               f' > QPs: {num_qps}\n'
               f' > Entries per rank: {num_entries_per_layer}, hidden: {args.hidden}\n'
               f' > Tokens to fetch: {args.num_tokens} x {args.num_entries_per_token} entries x {num_layers} layers\n'
               f' > Storage per rank: {sum(num_entries_per_layer) * args.hidden * dtype.itemsize / 1024 / 1024:.1f} MB\n',
               once_in_node=True)
    buffer = deep_ep.EngramBuffer(
        group, num_gpu_bytes, num_rdma_storage_bytes, args.use_cpu_rdma_storage, num_allocated_qps, qp_depth, restrict_rd_atomic,
        allow_hybrid_mode=args.allow_hybrid_mode, explicitly_destroy=True)
    buffer.set_config(num_entries_per_layer, args.hidden,
                      args.num_tokens, args.num_entries_per_token, dtype, num_sf_packs)

    # Write buffer: each rank writes its own local storages into the NCCL window
    local_storages, sfs = [], [] if args.use_fp8 else None
    for num_entries in num_entries_per_layer:
        local_bf16 = torch.randn((num_entries, args.hidden), dtype=torch.bfloat16, device='cuda')
        if args.use_fp8:
            local_storage, local_sf = per_token_cast_to_fp8(local_bf16)
            # Replicate the scaling factors so any rank can fetch any entry's factors locally
            sf = torch.empty((num_ranks * num_entries, local_sf.shape[1]), dtype=local_sf.dtype, device='cuda')
            dist.all_gather_single(sf, local_sf, group=group)
            sfs.append(sf)
        else:
            local_storage = local_bf16
        local_storages.append(local_storage)

    buffer.write(local_storages, sfs=sfs)

    def gen_indices():
        indices_per_layer = []
        for num_entries in num_entries_per_layer:
            table_size = num_ranks * num_entries // args.num_entries_per_token
            table_offsets = torch.arange(
                args.num_entries_per_token, device='cuda', dtype=torch.int) * table_size
            indices_per_layer.append(table_offsets + torch.randint(
                0, table_size, (args.num_tokens, args.num_entries_per_token),
                device='cuda', dtype=torch.int))
        return torch.stack(indices_per_layer)

    # Correctness check
    if not args.skip_check:
        global_storages = []
        for local_storage in local_storages:
            global_storage = torch.empty((num_ranks * local_storage.shape[0], args.hidden), dtype=dtype, device='cuda')
            dist.all_gather_single(global_storage, local_storage, group=group)
            global_storages.append(global_storage)

        # Verify that layers can be waited in both forward and reverse order
        for layer_order in (range(num_layers), reversed(range(num_layers))):
            indices = gen_indices()
            ref_data_per_layer = [
                global_storages[layer_idx][indices[layer_idx].view(-1)].view(args.num_tokens, -1)
                for layer_idx in range(num_layers)
            ]
            ref_sf_per_layer = [
                sfs[layer_idx][indices[layer_idx].view(-1)].view(args.num_tokens, -1)
                for layer_idx in range(num_layers)
            ] if args.use_fp8 else None

            hooks = buffer.fetch(indices, num_qps=num_qps)
            for layer_idx in layer_order:
                fetched = hooks[layer_idx]()
                data, fetched_sf = fetched if args.use_fp8 else (fetched, None)
                assert torch.equal(ref_data_per_layer[layer_idx], data), f'data mismatch ({layer_idx=})'
                if args.use_fp8:
                    assert torch.equal(ref_sf_per_layer[layer_idx], fetched_sf), \
                        f'fp8 scaling-factor mismatch ({layer_idx=})'

    # Performance test
    dist_print('Running performance test ...', once_in_node=True)
    indices = gen_indices()
    msg_bytes = args.hidden * dtype.itemsize
    num_fetched_bytes = num_layers * args.num_tokens * args.num_entries_per_token * msg_bytes

    # Measure fetch + wait (end-to-end)
    def fetch_and_wait():
        # noinspection PyShadowingNames
        hooks = buffer.fetch(indices, num_qps=num_qps)
        for hook in hooks:
            hook()

    issue_t, wait_t = bench_kineto(
        fetch_and_wait,
        kernel_names=('engram_fetch_impl', 'engram_fetch_wait_impl'),
        barrier_comm_profiling=True,
        barrier=buffer.barrier,
        trace_path=f'{args.dump_profile_traces}/engram_fetch_rank{buffer.rank_idx}.json' if args.dump_profile_traces else None)
    # NOTES: `wait_t` is a per-launch average, while each iteration waits once per layer
    total_t = issue_t + wait_t * num_layers
    mpps = num_layers * args.num_tokens * args.num_entries_per_token / total_t / 1e6
    dist_print(f' > Rank {rank:3}/{num_ranks} | '
               f'issue: {issue_t * 1e6:.1f} us, '
               f'wait: {wait_t * num_layers * 1e6:.1f} us, '
               f'{num_fetched_bytes / total_t / 1e9:.1f} GB/s, '
               f'bytes: {num_fetched_bytes / 1024 / 1024:.1f} MB, '
               f'{mpps:.2f} MPPS ({msg_bytes} B/msg)')
    dist_print('', once_in_node=True)

    # Destroy the runtime and communication group
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test engram fetch kernels')
    parser.add_argument('--num-processes', type=int, default=4, help='Number of processes to spawn')
    parser.add_argument('--num-qps', type=int, default=0, help='Number of QPs used (0 for automatic)')
    parser.add_argument('--num-entries-per-layer', type=str, default='524288,524309',
                        help='Comma-separated number of entries per rank for each layer')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--num-tokens', type=int, default=512, help='Number of tokens to fetch')
    parser.add_argument('--num-entries-per-token', type=int, default=24, help='Number of entries concatenated per token')
    parser.add_argument('--skip-check', action='store_true', help='Skip correctness check')
    parser.add_argument('--use-fp8', action='store_true', help='Store entries in FP8 with replicated scaling factors')
    parser.add_argument('--use-cpu-rdma-storage', action='store_true', help='Store Engram shards in CPU memory')
    parser.add_argument('--allow-hybrid-mode', action='store_true', help='Enable hybrid mode (multi-plane)')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    # Launch
    num_processes = args.num_processes
    torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)

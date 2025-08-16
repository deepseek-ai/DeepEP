import sys
import argparse
import os
import random
import torch
import time
from functools import partial

import rank_server
import deep_ep
from plan import Plan

# Add tests directory to path to import test utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back


def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, buffer: deep_ep.Buffer,
              use_logfmt: bool = False, seed: int = 0, kineto: bool = False):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    x_list = [x]
    for i in range(4 if use_logfmt else 0):
        # NOTES: make more LogFMT casts and also with some BF16
        x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.5 * random.random())
    # NOTES: the last one is for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.1)

    torch.manual_seed(seed + rank + 1000)
    torch.cuda.manual_seed(seed + rank + 1000)
    random.seed(seed + rank)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # Randomly mask some positions
    for i in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='cuda')
    for r in range(num_ranks):
        # Use same deterministic reset as above (seed + r + 1000)
        torch.manual_seed(seed + r + 1000)
        torch.cuda.manual_seed(seed + r + 1000)
        r_random = random.Random(seed + r)
        r_scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
        r_topk_idx = torch.topk(r_scores, num_topk, dim=-1, largest=True, sorted=True)[1]
        # Apply same random masking
        for i in range(10):
            r_topk_idx[r_random.randint(0, num_tokens - 1), r_random.randint(0, num_topk - 1)] = -1
        all_topk_idx[r] = r_topk_idx

    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0
    for current_x in x_list:
        for return_recv_hook in (False, True):
            for dispatch_use_fp8 in (False, True):
                for round_scale in (False, True) if dispatch_use_fp8 else (False, ):
                    for use_ue8m0 in (False, True) if round_scale else (False, ):
                        num_times += 1
                        for i in range((num_times % 2) + 1):
                            cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
                            packed_recv_x, packed_recv_count, handle, event, hook = \
                                buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                                            use_fp8=dispatch_use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,
                                                            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                                            async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
                            hook() if return_recv_hook else event.current_stream_wait()
                        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x
                        simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape) \
                            if dispatch_use_fp8 else packed_recv_x.clone()

                        # Make sure dispatch finished successfully
                        torch.cuda.synchronize()

                        for i in range(num_local_experts if do_check else 0):
                            expert_id = rank * num_local_experts + i
                            recv_x = per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]
                            recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]

                            # Check expert indices
                            int_mask = (2 ** 32) - 1
                            num_valid_tokens = recv_count.item()
                            assert cumulative_local_expert_recv_stats[i].item() == num_valid_tokens, f'{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}'
                            assert num_valid_tokens == (recv_layout_range & int_mask).sum().item(), f'{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()'
                            assert num_valid_tokens == (all_topk_idx == expert_id).sum().item(), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum().item()}'

                            if num_valid_tokens == 0:
                                continue
                            # Check received data
                            if current_x is x:
                                recv_x = recv_x[:num_valid_tokens]
                                recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                                recv_src_info = recv_src_info[:num_valid_tokens]
                                assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))
                                if round_scale:
                                    assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007
                                else:
                                    assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0
                                for j in range(num_ranks):
                                    begin_idx, count = (recv_layout_range[j] >> 32).item(), (recv_layout_range[j] & int_mask).item()
                                    if not round_scale:
                                        assert (recv_x_amin == j - rank_offset).sum().item() == (all_topk_idx[j] == expert_id).sum().item()
                                        assert (recv_x[begin_idx:begin_idx + count, :-128] - j + rank_offset).sum().item() == 0
                            if dispatch_use_fp8:
                                hash_value ^= hash_tensor(packed_recv_x[0][i, :num_valid_tokens])
                                hash_value ^= hash_tensor(packed_recv_x[1][i, :num_valid_tokens])
                            else:
                                hash_value ^= hash_tensor(packed_recv_x[i, :num_valid_tokens])

                        # Check combine correctness
                        for zero_copy in (False, ) if use_logfmt else (False, True):
                            if zero_copy:
                                buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
                            out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                            combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                                                use_logfmt=use_logfmt,
                                                                                async_finish=not return_recv_hook, zero_copy=zero_copy,
                                                                                return_recv_hook=return_recv_hook, out=out)
                            hook() if return_recv_hook else event.current_stream_wait()
                            if do_check:
                                diff = calc_diff(current_x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
                                assert torch.isnan(combined_x).sum().item() == 0
                                assert diff < (9e-4 if dispatch_use_fp8 else 1e-5), f'Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}'
                                hash_value ^= hash_tensor(combined_x)

    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((8192, 8192), dtype=torch.float)
        mat_1 = torch.randn((8192, 8192), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_func(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=True, async_finish=False, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                             use_logfmt=use_logfmt, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None

    def test_barrier():
        buffer.low_latency_sync()

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += (num_logfmt10_bytes if use_logfmt else num_bf16_bytes) * num_selections

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
    print(f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
          f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)

    # Separate profiling
    if not kineto:
        return

    for return_recv_hook in (False, True):
        buffer.low_latency_sync()
        dispatch_t, combine_t = bench_kineto(partial(test_func, return_recv_hook=return_recv_hook),
                                             kernel_names=('dispatch', 'combine'), barrier_comm_profiling=True,
                                             suppress_kineto_output=False, num_kernels_per_period=2 if return_recv_hook else 1,
                                             barrier_fn=test_barrier)
        if not return_recv_hook:
            print(f'[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
                  f'Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us', flush=True)
        else:
            print(f'[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | '
                  f'Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us', flush=True)


def worker(torch_rank: int, args: argparse.Namespace):
    local_rank, global_rank = rank_server.get_rank(args.rank_server if args.rank_server else "127.0.0.1")
    plan = Plan(args.plan, global_rank)
    max_num_ranks = plan.get_max_rank() + 1
    print(f"Process {torch_rank} -> global_rank={global_rank}, local_rank={local_rank}", flush=True)

    # Initialize torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank % 8)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(0)

    # Initialize UCX
    pxb_nics = ["mlx5_0", "mlx5_3", "mlx5_4", "mlx5_5", "mlx5_6", "mlx5_9", "mlx5_10", "mlx5_11"]
    tcp_nics = ',ibp154s0,ibp192s0,ibp206s0,ibp220s0,ibp94s0'
    os.environ['UCX_NET_DEVICES'] = f'cuda0-{pxb_nics[local_rank]}:1' + tcp_nics

    # Initialize NIXL
    os.environ['NIXL_ETCD_ENDPOINTS'] = args.etcd_server

    # Initialize deep_ep buffer
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(args.num_tokens, args.hidden_dim, max_num_ranks, args.num_experts_per_rank * max_num_ranks)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)

    buffer = deep_ep.Buffer.nixl_buffer(rank=global_rank, low_latency_mode=True, low_latency_nvlink_backend=args.nvlink_backend, explicitly_destroy=True)

    buffer.update_memory_buffers(num_ranks=max_num_ranks, num_experts_per_rank=args.num_experts_per_rank, num_nvl_bytes=0, num_rdma_bytes=num_rdma_bytes)

    released_ranks = set() # Track other ranks that have been released

    while True:
        print(f"global_rank={global_rank}, local_rank={local_rank} -> start phase {plan.get_phase()}", flush=True)

        added_ranks = plan.get_new_ranks()
        removed_ranks = plan.get_removed_ranks()

        # If this rank is being removed in this phase, exit gracefully
        if global_rank in removed_ranks:
            print(f"global_rank={global_rank}, local_rank={local_rank} -> this rank is being removed in this phase, exiting", flush=True)
            break

        if len(added_ranks) > 0:
            # TODO: remove once we support adding ranks that have been released before
            if released_ranks.intersection(added_ranks):
                raise RuntimeError(f"[ERROR] global_rank={global_rank}, local_rank={local_rank} -> Cannot add previously released ranks {list(released_ranks.intersection(added_ranks))}.")

            print(f"global_rank={global_rank}, local_rank={local_rank} -> adding connections to {added_ranks}", flush=True)
            buffer.connect_ranks(added_ranks)

        if len(removed_ranks) > 0:
            print(f"global_rank={global_rank}, local_rank={local_rank} -> removing connections to {removed_ranks}", flush=True)
            released_ranks.update(removed_ranks)
            buffer.remove_ranks(removed_ranks)

        test_main(args.num_tokens, args.hidden_dim, args.num_experts_per_rank * len(plan.get_active_ranks()), args.num_topk,
                global_rank, len(plan.get_active_ranks()), buffer, kineto=args.kineto)

        print(f"global_rank={global_rank}, local_rank={local_rank} -> end phase {plan.get_phase()}", flush=True)

        if not plan.next():
            break

    buffer.destroy()

    print(f"global_rank={global_rank}, local_rank={local_rank} -> done", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Elastic EP Test")
    parser.add_argument("--plan", type=str, default="plan.json", help="Path to plan file")
    parser.add_argument("--num-processes", type=int, default=8, help="Number of worker processes to launch")
    parser.add_argument("--num-tokens", type=int, default=128, help="Number of tokens")
    parser.add_argument("--num-experts-per-rank", type=int, default=2, help="Number of experts per rank")
    parser.add_argument("--hidden-dim", type=int, default=7168, help="Hidden dimension")
    parser.add_argument("--num-topk", type=int, default=8, help="Number of topk")
    parser.add_argument("--etcd-server", type=str, default="http://127.0.0.1:2379", help="ETCD server address for NIXL (default: http://127.0.0.1:2379)")
    parser.add_argument("--rank-server", type=str, help="Rank server address. If not set, a rank server will be started locally.")
    parser.add_argument("--kineto", action="store_true", help="Enable kineto profiling")
    parser.add_argument('--nvlink-backend', choices=['nixl', 'ipc', 'none'], default='nixl', help='NVLink backend to use')

    args = parser.parse_args()

    if not args.rank_server:
        rank_server.start_server()

    try:
        if args.num_processes > 1:
            torch.multiprocessing.spawn(
                worker,
                args=(args,),
                nprocs=args.num_processes,
                join=True,
                daemon=False,
                start_method="spawn",
            )
        else:
            worker(0, args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

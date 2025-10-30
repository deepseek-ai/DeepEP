import argparse
import random
import torch
import torch.distributed as dist
from functools import partial
from typing import Literal, Set
import pandas as pd
import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back, get_global_token_indices


def simulate_failure_and_skip(rank: int, api: Literal["dispatch", "combine", "clean"], expected_masked_ranks: Set[int]):
    # Simulates rank failure when the rank first calls the corresponding communication API
    failed_api_ranks = {
        # API -> rank to fail (rank fails when it first calls the corresponding communication API)
        'dispatch': 1,
        'combine': 3,
        'clean': 5
    }
    if rank in expected_masked_ranks:
        # Rank already failed
        return True
    if api in failed_api_ranks.keys():
        expected_masked_ranks.add(failed_api_ranks[api])
        if failed_api_ranks[api] == rank:
            print(f"Rank {rank} failed when first calling {api} communication API, exit...", flush=True)
            return True
    return False


def query_mask_buffer_and_check(api: Literal["dispatch", "combine", "clean"], buffer: deep_ep.Buffer, mask_status: torch.Tensor,
                                expected_masked_ranks: Set[int]):
    buffer.low_latency_query_mask_buffer(mask_status)
    assert set(mask_status.nonzero().squeeze(-1).tolist()) == expected_masked_ranks


def test_main(num_tokens: int,
              hidden: int,
              num_experts: int,
              num_topk: int,
              rank: int,
              num_ranks: int,
              group: dist.ProcessGroup,
              buffer: deep_ep.Buffer,
              imbalance_factor: float = 1.0,
              distribution: str = 'lognormal',
              print_res: bool = True,
              use_logfmt: bool = False,
              shrink_test: bool = False,
              seed: int = 0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    x_list = [x]
    for _ in range(4 if use_logfmt else 0):
        # NOTES: make more LogFMT casts and also with some BF16
        x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.5 * random.random())
    # NOTES: the last one is for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.1)

    scatter_list = None
    if rank == 0:
        global_topk_idx = get_global_token_indices(distribution, num_experts, num_tokens, num_ranks, num_topk, imbalance_factor, seed)
        scatter_list = [chunk.contiguous() for chunk in torch.chunk(global_topk_idx, num_ranks, dim=0)]
    topk_idx = torch.empty(num_tokens, num_topk, dtype=torch.long, device='cuda')
    dist.scatter(tensor=topk_idx, scatter_list=scatter_list, src=0, group=group)

    results = {}
    results['topk_idx'] = topk_idx.cpu()

    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # Randomly mask some positions
    for _ in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='cuda')
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

    # For failure simulation and shrink testing
    mask_status = torch.zeros((num_ranks, ), dtype=torch.int, device='cuda')
    expected_masked_ranks = set()

    # Check dispatch correctness
    do_check = True
    hash_value, num_times = 0, 0
    for current_x in x_list:
        for return_recv_hook in (False, True):
            for dispatch_use_fp8 in (False, True):
                for round_scale in (False, True) if dispatch_use_fp8 else (False, ):
                    for use_ue8m0 in (False, True) if round_scale else (False, ):
                        if shrink_test and simulate_failure_and_skip(rank, "dispatch", expected_masked_ranks):
                            break
                        num_times += 1
                        for _ in range((num_times % 2) + 1):
                            cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
                            packed_recv_x, packed_recv_count, handle, event, hook = \
                                buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                                            use_fp8=dispatch_use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,
                                                            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                                            async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
                            hook() if return_recv_hook else event.current_stream_wait()
                        if shrink_test:
                            query_mask_buffer_and_check("dispatch", buffer, mask_status, expected_masked_ranks)
                        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x
                        simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape) \
                            if dispatch_use_fp8 else packed_recv_x.clone()
                        for i in range(num_local_experts if do_check else 0):
                            expert_id = rank * num_local_experts + i
                            recv_x = per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]
                            recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]

                            # Check expert indices
                            int_mask = (2**32) - 1
                            num_valid_tokens = recv_count.item()
                            assert cumulative_local_expert_recv_stats[i].item(
                            ) == num_valid_tokens, f'{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}'
                            assert num_valid_tokens == (
                                recv_layout_range
                                & int_mask).sum().item(), f'{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()'
                            assert num_valid_tokens == (all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status == 0].sum().item(
                            ), f'{num_valid_tokens} != {(all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status==0].sum().item()}'

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
                                    if shrink_test and mask_status[j]:
                                        continue
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
                        if shrink_test and simulate_failure_and_skip(rank, "combine", expected_masked_ranks):
                            break
                        for zero_copy in (False, ) if use_logfmt else (False, True):
                            if zero_copy:
                                buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
                            out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                            combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,
                                                                                 topk_idx,
                                                                                 topk_weights,
                                                                                 handle,
                                                                                 use_logfmt=use_logfmt,
                                                                                 async_finish=not return_recv_hook,
                                                                                 zero_copy=zero_copy,
                                                                                 return_recv_hook=return_recv_hook,
                                                                                 out=out)
                            hook() if return_recv_hook else event.current_stream_wait()
                            if shrink_test:
                                query_mask_buffer_and_check("combine", buffer, mask_status, expected_masked_ranks)
                            if do_check:
                                if shrink_test:
                                    owner_by_expert = (torch.arange(num_experts, device='cuda') // num_local_experts)
                                    fail_owner_mask = (mask_status == 1).index_select(0, owner_by_expert)
                                    valid_topk_idx = topk_idx >= 0
                                    failed_topk_idx = torch.zeros_like(topk_idx, device='cuda', dtype=torch.bool)
                                    failed_topk_idx[valid_topk_idx] = fail_owner_mask.index_select(0, topk_idx[valid_topk_idx])
                                    topk_idx[failed_topk_idx] = -1
                                diff = calc_diff(current_x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
                                assert torch.isnan(combined_x).sum().item() == 0
                                if not round_scale:
                                    assert diff < (9e-4 if dispatch_use_fp8 else 1e-5), f'Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}'
                                hash_value ^= hash_tensor(combined_x)

                        # Clean buffer API
                        if shrink_test:
                            if simulate_failure_and_skip(rank, "clean", expected_masked_ranks):
                                break

                            buffer.clean_low_latency_buffer(num_tokens, hidden, num_experts)
                            query_mask_buffer_and_check("clean", buffer, mask_status, expected_masked_ranks)

    if shrink_test:
        return

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
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x,
                                                             topk_idx,
                                                             topk_weights,
                                                             handle,
                                                             use_logfmt=use_logfmt,
                                                             return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None

    # Calculate bandwidth
    num_fp8_bytes, num_bf16_bytes = (hidden + hidden / 128 * 4 + 16), hidden * 2
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_fp8_bytes * num_selections
        num_combine_comm_bytes += (num_logfmt10_bytes if use_logfmt else num_bf16_bytes) * num_selections
    results['dispatch_comm_bytes'] = num_dispatch_comm_bytes
    results['combine_comm_bytes'] = num_combine_comm_bytes

    # Dispatch + combine testing
    avg_t, min_t, max_t = bench(partial(test_func, return_recv_hook=False))
    if print_res:
        print(
            f'[rank {rank}] Dispatch + combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
            f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us',
            flush=True)
    results['total_time_avg'] = avg_t * 1e6
    results['total_time_min'] = min_t * 1e6
    results['total_time_max'] = max_t * 1e6

    # Separate profiling
    for return_recv_hook in (False, True):
        group.barrier()
        dispatch_t, combine_t = bench_kineto(partial(test_func, return_recv_hook=return_recv_hook),
                                             kernel_names=('dispatch', 'combine'),
                                             barrier_comm_profiling=True,
                                             suppress_kineto_output=True,
                                             num_kernels_per_period=2 if return_recv_hook else 1)
        if not return_recv_hook:
            if print_res:
                print(
                    f'[rank {rank}] Dispatch bandwidth: {num_dispatch_comm_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
                    f'Combine bandwidth: {num_combine_comm_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us',
                    flush=True)
            results['dispatch_time'] = dispatch_t * 1e6
            results['combine_time'] = combine_t * 1e6
        else:
            if print_res:
                print(
                    f'[rank {rank}] Dispatch send/recv time: {dispatch_t[0] * 1e6:.2f} + {dispatch_t[1] * 1e6:.2f} us | '
                    f'Combine send/recv time: {combine_t[0] * 1e6:.2f} + {combine_t[1] * 1e6:.2f} us',
                    flush=True)
            results['dispatch_send_time'] = dispatch_t[0] * 1e6
            results['dispatch_recv_time'] = dispatch_t[1] * 1e6
            results['combine_send_time'] = combine_t[0] * 1e6
            results['combine_recv_time'] = combine_t[1] * 1e6
    return results, hash_value


def process_and_display_results(all_results, num_experts, num_ranks, imbalance_factor):
    all_topk_idx_list = [result['topk_idx'] for result in all_results]
    global_topk_idx = torch.cat(all_topk_idx_list, dim=0)

    num_local_experts = num_experts // num_ranks
    rank_counts = torch.zeros(num_ranks, dtype=torch.int64)

    valid_indices = global_topk_idx[global_topk_idx >= 0]

    for rank in range(num_ranks):
        start_expert = rank * num_local_experts
        end_expert = (rank + 1) * num_local_experts

        mask = (valid_indices >= start_expert) & (valid_indices < end_expert)
        rank_counts[rank] = mask.sum().item()

    max_count = rank_counts.max().item()
    min_count = rank_counts.min().item()
    avg_count = rank_counts.float().mean().item()
    median_count = rank_counts.float().median().item()
    std_count = rank_counts.float().std().item()
    actual_max_avg = max_count / avg_count if avg_count > 0 else 0

    avg_dispatch_bytes = sum(result['dispatch_comm_bytes'] for result in all_results) / len(all_results)
    avg_combine_bytes = sum(result['combine_comm_bytes'] for result in all_results) / len(all_results)

    avg_total_time = sum(result['total_time_avg'] for result in all_results) / len(all_results)
    avg_dispatch_time = sum(result['dispatch_time'] for result in all_results) / len(all_results)
    avg_combine_time = sum(result['combine_time'] for result in all_results) / len(all_results)

    total_bw = (avg_dispatch_bytes + avg_combine_bytes) / 1e9 / (avg_total_time / 1e6)
    dispatch_bw = avg_dispatch_bytes / 1e9 / (avg_dispatch_time / 1e6)
    combine_bw = avg_combine_bytes / 1e9 / (avg_combine_time / 1e6)

    for result in all_results:
        if 'topk_idx' in result:
            del result['topk_idx']
        for key in ['dispatch_comm_bytes', 'combine_comm_bytes']:
            if key in result:
                del result[key]

    df = pd.DataFrame(all_results)
    mean_series = df.mean()

    mean_series['total_bw'] = total_bw
    mean_series['dispatch_bw'] = dispatch_bw
    mean_series['combine_bw'] = combine_bw
    mean_series['imbalance_factor'] = imbalance_factor
    mean_series['max_count'] = float(max_count)
    mean_series['min_count'] = float(min_count)
    mean_series['avg_count'] = float(avg_count)
    mean_series['median_count'] = float(median_count)
    mean_series['std_count'] = float(std_count)
    mean_series['actual_max_avg'] = float(actual_max_avg)

    return mean_series


def print_summary_tables(final_df):
    print("\n" + "=" * 120)
    print("  PERFORMANCE SUMMARY (Statistics across all ranks)")
    print("=" * 120)

    # Table 1: Token Distribution Statistics
    print("\n--- Token Distribution per Rank ---")
    imbalance_df = final_df[['max_count', 'min_count', 'avg_count', 'median_count', 'std_count']].copy()
    imbalance_df.columns = ['Max', 'Min', 'Avg', 'Median', 'Std Dev']

    formatters = {
        'Max': lambda x: f"{x:4.0f}",
        'Min': lambda x: f"{x:3.0f}",
        'Avg': lambda x: f"{x:5.1f}",
        'Median': lambda x: f"{x:6.1f}",
        'Std Dev': lambda x: f"{x:6.1f}"
    }

    for col, formatter in formatters.items():
        imbalance_df[col] = imbalance_df[col].apply(formatter)

    print(imbalance_df.to_string())

    # Table 2: Total Performance
    print("\n--- Total Performance (Dispatch + Combine) ---")
    total_perf_df = final_df[['total_bw', 'total_time_avg', 'total_time_min', 'total_time_max']].copy()
    total_perf_df.columns = ['Total BW', 'Avg Time', 'Min Time', 'Max Time']

    formatters = {
        'Total BW': lambda x: f"{x:.2f} GB/s",
        'Avg Time': lambda x: f"{x:.2f} us",
        'Min Time': lambda x: f"{x:.2f} us",
        'Max Time': lambda x: f"{x:.2f} us"
    }

    for col, formatter in formatters.items():
        total_perf_df[col] = total_perf_df[col].apply(formatter)

    print(total_perf_df.to_string())

    # Table 3: Separate Performance
    print("\n--- Separate Dispatch & Combine Performance ---")
    separate_perf_df = final_df[['dispatch_bw', 'dispatch_time', 'combine_bw', 'combine_time']].copy()
    separate_perf_df.columns = ['Dispatch BW', 'Dispatch Time', 'Combine BW', 'Combine Time']

    formatters = {
        'Dispatch BW': lambda x: f"{x:.2f} GB/s",
        'Dispatch Time': lambda x: f"{x:.2f} us",
        'Combine BW': lambda x: f"{x:.2f} GB/s",
        'Combine Time': lambda x: f"{x:.2f} us"
    }

    for col, formatter in formatters.items():
        separate_perf_df[col] = separate_perf_df[col].apply(formatter)

    print(separate_perf_df.to_string())

    # Table 4: Hook Performance
    print("\n--- Send/Recv Timings (Hook=True) ---")

    hook_data = []
    for idx in final_df.index:
        row = final_df.loc[idx]
        hook_data.append([
            f"{row['dispatch_send_time']:>6.2f} us", f"{row['dispatch_recv_time']:>6.2f} us", f"{row['combine_send_time']:>6.2f} us",
            f"{row['combine_recv_time']:>6.2f} us"
        ])

    columns = pd.MultiIndex.from_tuples([('        ', 'Send'), ('Dispatch', 'Recv'), ('       ', 'Send'), ('Combine', 'Recv')])

    hook_df = pd.DataFrame(hook_data, index=final_df.index, columns=columns)

    print(hook_df.to_string())

    print("\n" + "=" * 120)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group,
                            num_rdma_bytes=num_rdma_bytes,
                            low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink,
                            explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl,
                            enable_shrink=args.shrink_test)
    test_main(num_tokens,
              hidden,
              num_experts,
              num_topk,
              rank,
              num_ranks,
              group,
              buffer,
              use_logfmt=args.use_logfmt,
              shrink_test=args.shrink_test,
              seed=1)
    dist.barrier()
    if rank == 0:
        all_imbalance_summaries = []
    for imbalance_factor in args.imbalance_factors:
        if rank == 0:
            print(f"\n--> Running test for target imbalance factor: {imbalance_factor}", flush=True)
        results_dict, _ = test_main(num_tokens,
                                    hidden,
                                    num_experts,
                                    num_topk,
                                    rank,
                                    num_ranks,
                                    group,
                                    buffer,
                                    imbalance_factor=imbalance_factor,
                                    distribution=args.distribution,
                                    print_res=False,
                                    seed=1)
        if rank == 0:
            all_results = [None] * num_ranks
            dist.gather_object(results_dict, all_results, dst=0, group=group)

            mean_series = process_and_display_results(all_results, num_experts, num_ranks, imbalance_factor)
            all_imbalance_summaries.append(mean_series)
        else:
            dist.gather_object(results_dict, None, dst=0, group=group)

    if rank == 0 and all_imbalance_summaries:
        df = pd.DataFrame(all_imbalance_summaries)

        df['display_index'] = df.apply(lambda row: f"{row['imbalance_factor']:.1f} (Actual: {row['actual_max_avg']:.2f})", axis=1)
        df.set_index('display_index', inplace=True)
        df.index.name = "Max/Avg Ratio"
        print_summary_tables(df)
    do_pressure_test = args.pressure_test
    for seed in range(int(1e9) if do_pressure_test else 0):
        if local_rank == 0:
            print(f'Testing with seed {seed} ...', flush=True)
        __, ref_hash = test_main(num_tokens,
                                 hidden,
                                 num_experts,
                                 num_topk,
                                 rank,
                                 num_ranks,
                                 group,
                                 buffer,
                                 use_logfmt=args.use_logfmt,
                                 seed=seed)
        for _ in range(20):
            __, current_hash = test_main(num_tokens,
                                         hidden,
                                         num_experts,
                                         num_topk,
                                         rank,
                                         num_ranks,
                                         group,
                                         buffer,
                                         use_logfmt=args.use_logfmt,
                                         seed=seed)
            assert current_hash == ref_hash, f'Error: seed={seed}'

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`
    parser = argparse.ArgumentParser(description='Test low-latency EP kernels')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128, help='Number of tokens (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=288, help='Number of experts (default: 288)')
    parser.add_argument('--allow-mnnvl', action="store_true", help='Allow MNNVL for communication')
    parser.add_argument('--disable-nvlink', action='store_true', help='Whether to disable NVLink for testing')
    parser.add_argument('--use-logfmt', action='store_true', help='Whether to test LogFMT combine')
    parser.add_argument("--pressure-test", action='store_true', help='Whether to do pressure test')
    parser.add_argument("--shrink-test", action='store_true', help='Whether to simulate failure and test shrink mode')
    parser.add_argument('--imbalance-factors',
                        type=float,
                        nargs='+',
                        default=[1.0, 2.0, 3.0],
                        help='A list of target max/avg ratios for per-rank expert load (tokens per expert). '
                        'Higher values create more load imbalance (e.g., 1.0, 2.0, 3.0).'
                        'Note: actual ratios may be lower than targets due to token count constraints.')
    parser.add_argument('--distribution', type=str, default='lognormal', choices=['lognormal', 'powerlaw', 'gamma'])
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)

import argparse
import os
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, create_grouped_scores, inplace_unique, per_token_cast_to_fp8, per_token_cast_back, hash_tensor

# Test compatibility with low latency functions
import test_low_latency


# noinspection PyShadowingNames
def test_main(args: argparse.Namespace, num_sms: int,
              local_rank: int, num_local_ranks: int, num_ranks: int, num_nodes: int, rank: int,
              buffer: deep_ep.Buffer, group: dist.ProcessGroup, skip_benchmark: bool = False):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk_groups, num_topk, num_experts = args.num_topk_groups, args.num_topk, args.num_experts
    zero_copy = args.use_zero_copy
    disable_stride = args.disable_stride
    disable_bias = args.disable_bias
    if zero_copy:
        if not disable_bias:
            disable_bias = True
            if local_rank == 0:
                print("[config] Disabling bias because zero-copy combine doesn't support it yet")
        if not disable_stride:
            disable_stride = True
            if local_rank == 0:
                print("[config] Disabling stride because zero-copy dispatch doesn't support it yet")

    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}', flush=True)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_e4m3 = per_token_cast_to_fp8(x)
    x_pure_rand_e4m3 = per_token_cast_to_fp8(x_pure_rand)
    if not disable_stride:
        x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)
    hash_value = 0

    # RDMA dispatch counts
    rdma_idx = topk_idx // (num_experts // num_nodes)
    rdma_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rdma_idx, num_nodes)
    num_rdma_token_sent = rdma_idx.ne(-1).sum().item()

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='cuda')
    num_tokens_per_rdma_rank = torch.empty((num_nodes, ), dtype=torch.int, device='cuda')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')
    for i in range(num_nodes):
        num_tokens_per_rdma_rank[i] = (rdma_rank_idx == i).sum()
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    ref_num_tokens_per_rank, ref_num_tokens_per_rdma_rank, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_rdma_rank, num_tokens_per_rdma_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
        print('', flush=True)
    group.barrier()
    time.sleep(1)

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (24, 48, 96, 144, 160) else 512)
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size, 10 if zero_copy else 16, rdma_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    zcopy_buffer_id = -1

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in (x_pure_rand, x, x_pure_rand_e4m3, x_e4m3):
                for with_topk in (False, True):
                    is_rand = current_x is x_pure_rand or current_x is x_pure_rand_e4m3
                    zcopy_buffer_id = (zcopy_buffer_id + 1) % 2
                    if local_rank == 0:
                        print(f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...', flush=True, end='')
                    dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,  'is_token_in_rank': is_token_in_rank,
                                     'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': async_mode,
                                     'zero_copy': zero_copy, 'zcopy_buffer_id': zcopy_buffer_id}
                    if with_topk:
                        dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights_pure_rand if is_rand else topk_weights})
                    if zero_copy:
                        use_fp8 = True if isinstance(current_x, tuple) else False
                        current_x_hidden, current_x_scales = current_x if isinstance(current_x, tuple) else (current_x, None)
                        dispatch_buffer_layout = buffer.get_internode_dispatch_buffer(num_tokens, hidden, num_topk, with_topk, use_fp8, zcopy_buffer_id)
                        if current_x_scales is not None:
                            dispatch_buffer_layout[0][:, :] = current_x_hidden
                            dispatch_buffer_layout[1][:, :] = current_x_scales
                        else:
                            dispatch_buffer_layout[0][:, :] = current_x
                        if with_topk:
                            dispatch_buffer_layout[2][:, :] = topk_idx
                            dispatch_buffer_layout[3][:, :] = topk_weights_pure_rand if is_rand else topk_weights
                    if previous_mode:
                        dispatch_args.update({'previous_event': buffer.capture()})
                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)
                    event.current_stream_wait() if async_mode else ()

                    if current_x is x_pure_rand or current_x is x:
                        hash_value += hash_tensor(recv_x)
                    else:
                        hash_value += hash_tensor(recv_x[0])
                        hash_value += hash_tensor(recv_x[1])

                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    # Checks
                    recv_gbl_rank_prefix_sum = handle[-5]
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                    if not is_rand:
                        check_data(recv_x, recv_gbl_rank_prefix_sum)
                    if with_topk:
                        # Check `topk_idx`
                        assert (recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        if not is_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode, 'zero_copy': zero_copy, 'zcopy_buffer_id': zcopy_buffer_id}
                        if zero_copy:
                            use_fp8 = True if isinstance(current_x, tuple) else False
                            current_x_hidden, current_x_scales = current_x if isinstance(current_x, tuple) else (current_x, None)
                            dispatch_buffer_layout = buffer.get_internode_dispatch_buffer(num_tokens, hidden, num_topk, with_topk, use_fp8, zcopy_buffer_id)
                            if current_x_scales is not None:
                                dispatch_buffer_layout[0][:, :] = current_x_hidden
                                dispatch_buffer_layout[1][:, :] = current_x_scales
                            else:
                                dispatch_buffer_layout[0][:, :] = current_x
                        if previous_mode:
                            dispatch_args.update({'previous_event': buffer.capture()})
                        recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                        if not is_rand:
                            check_data(recv_x, recv_gbl_rank_prefix_sum)

                    # Test combine
                    if disable_bias:
                        bias_0 = bias_1 = None
                    else:
                        bias_0 = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                        bias_1 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    combine_args = {'x': recv_x, 'bias': (bias_0, bias_1), 'handle': handle, 'config': config, 'async_finish': async_mode, 'zero_copy': zero_copy, 'zcopy_buffer_id': zcopy_buffer_id}
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if zero_copy:
                        num_tokens_to_combine = recv_x.size(0)
                        combine_buffer_layout = buffer.get_internode_combine_buffer(num_tokens_to_combine, hidden, num_topk, with_topk, zcopy_buffer_id)
                        if with_topk:
                            combine_buffer_layout[1][:, :] = recv_topk_weights
                        combine_buffer_layout[0][:, :] = recv_x
                    if previous_mode:
                        combine_args.update({'previous_event': buffer.capture()})
                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                    event.current_stream_wait() if async_mode else ()
                    check_x = (combined_x.float() if disable_bias else combined_x.float() - bias_0.float() - bias_1.float()) / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    ref_x = x_pure_rand if is_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-4 if current_x is x_pure_rand_e4m3 else 5e-6 
                    if with_topk:
                        check_topk_weights = combined_topk_weights if is_rand else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))
                        ref_topk_weights = topk_weights_pure_rand if is_rand else topk_weights
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    hash_value += hash_tensor(recv_x)

                    # For later tuning
                    dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
                    combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print('', flush=True)

    if skip_benchmark:
        return hash_value

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (dispatch_bf16_rdma_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_send_bytes
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        for nvl_chunk_size in range(4, 45, 4):
            for rdma_chunk_size in range(4, 13, 2) if zero_copy else range(4, 33, 4):
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
                tune_args = {'x': current_x, 'handle': handle, 'config': config, 'zero_copy': zero_copy}
                if zero_copy:
                    use_fp8 = True if isinstance(current_x, tuple) else False
                    with_topk = False
                    current_x_hidden, current_x_scales = current_x if isinstance(current_x, tuple) else (current_x, None)
                    dispatch_buffer_layout = buffer.get_internode_dispatch_buffer(num_tokens, hidden, num_topk, with_topk, use_fp8)
                    dispatch_buffer_layout[0][:, :] = current_x_hidden
                    if use_fp8:
                        dispatch_buffer_layout[1][:, :] = current_x_scales
                # Tuning use cached dispatch with does not allow supplying TopK
                t, notify_t = bench_kineto(lambda: buffer.dispatch(**tune_args), ('dispatch', 'notify'), suppress_kineto_output=True)
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size, notify_t)
                if local_rank == 0:
                    print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: '
                          f'{notify_t * 1e6:.0f} + {t * 1e6:.0f} us, '
                          f'{rdma_send_bytes / 1e9 / t:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL) ', flush=True)
        if local_rank == 0:
            print(f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: '
                  f'{best_results[3] * 1e6:.0f} + {best_time * 1e6:.0f} us, '
                  f'{rdma_send_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL)', flush=True)
            print('', flush=True)

        if isinstance(current_x, tuple):
            # Gather FP8 the best config from rank 0
            best_dispatch_results = torch.tensor([best_results[0], best_results[1], best_results[2]], dtype=torch.int32, device='cuda')
            all_best_fp8_results_list = [torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())]
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size, best_dispatch_results[2], rdma_buffer_size)

    dispatch_args = {'x': x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,
                     'is_token_in_rank': is_token_in_rank, 'num_tokens_per_expert': num_tokens_per_expert,
                     'config': dispatch_config if dispatch_config is not None else config, 'zero_copy': zero_copy}
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in range(1, 8, 1):
        for rdma_chunk_size in range(12 if num_nodes == 2 else 8, 33, 4):
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
            tune_args = {'x': recv_x, 'handle': handle, 'config': config, 'zero_copy': zero_copy}
            if zero_copy:
                num_tokens_to_combine = recv_x.size(0)
                combine_buffer_layout = buffer.get_internode_combine_buffer(num_tokens_to_combine, hidden, 0, False)
                combine_buffer_layout[0][:, :] = recv_x
            t, notify_t = bench_kineto(lambda: buffer.combine(**tune_args), ('combine', 'notify'), suppress_kineto_output=True)
            if local_rank == 0:
                print(f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: '
                      f'{notify_t * 1e6:.0f} + {t * 1e6:.0f} us, '
                      f'{combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (RDMA), '
                      f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL) ', flush=True)
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size, notify_t)

    if local_rank == 0:
        print(f'[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, '
              f'{best_results[3] * 1e6:.2f} + {best_time * 1e6:.2f} us, '
              f'{combine_bf16_rdma_recv_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL)', flush=True)
        print('', flush=True)
    return hash_value


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if args.test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9

    if args.use_zero_copy:
        assert num_nodes >= 2, "Zero copy is only implemented for >=2 nodes"

    if args.num_sms is None:
        if args.use_zero_copy:
            num_sms = 8 if num_nodes > 2 else 12
        else:
            num_sms = 24
    else:
        num_sms = args.num_sms
    num_qps_per_rank = max(num_sms, ll_num_experts // num_ranks if args.test_ll_compatibility else 0)

    buffer = deep_ep.Buffer(group, int(2e9), int(1e9), low_latency_mode=args.test_ll_compatibility,
                            num_qps_per_rank=num_qps_per_rank, explicitly_destroy=True, enable_zcopy=args.use_zero_copy, num_zcopy_buffers=2)
    assert num_local_ranks == 8 and num_ranks > 8

    for seed in range(int(1e9)):
        if local_rank == 0:
            print(f'Testing with seed {seed} ...', flush=True)
        torch.manual_seed(rank + seed)
        ref_hash = 0
        for i in (num_sms, ):
            ref_hash += test_main(args, i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group, args.pressure_test_mode == 1)
            if local_rank == 0:
                print('', flush=True)
        if args.pressure_test_mode == 0:
            break
        
        if local_rank == 0:
            print(f'{ref_hash=}')
            print('', flush=True)

        for j in range(20):
            torch.manual_seed(rank + seed)
            current_hash = 0
            for i in (num_sms, ):
                current_hash += test_main(args, i, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group, args.pressure_test_mode == 1)
                if local_rank == 0:
                    print('', flush=True)
            assert current_hash == ref_hash

    # Test compatibility with low latency functions
    if args.test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
        test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test internode EP kernels')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=4096,
                       help='Number of tokens (default: 4096)')
    parser.add_argument('--hidden', type=int, default=7168,
                       help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk-groups', type=int, default=None,
                       help='Number of top-k groups (default: `min(num_nodes, 4)`)')
    parser.add_argument('--num-topk', type=int, default=8,
                       help='Number of top-k experts (default: 8)')
    parser.add_argument('--pressure-test-mode', type=int, default=0,
                       help='Pressure test mode. 0: don\'t do pressure test, 1: do pressure test without benchmarks, 2: do pressure test with benchmarks')
    parser.add_argument('--num-experts', type=int, default=256,
                       help='Number of experts (default: 256')
    parser.add_argument('--test-ll-compatibility', action='store_true',
                        help='whether to test compatibility with low-latency kernels')
    parser.add_argument('--use-zero-copy', action='store_true',
                        help='whether to test the zero-copy internode kernels')
    parser.add_argument('--disable-bias', action='store_true',
                        help='whether to disable combine with bias')
    parser.add_argument('--disable-stride', action='store_true',
                        help='whether to disable dispatch with stride')
    parser.add_argument('--num-sms', type=int, default=None,
                        help='number of SMs to use')
    args = parser.parse_args()

    # Set default `num_topk_groups` if not provided
    if args.num_topk_groups is None:
        num_nodes = int(os.getenv('WORLD_SIZE', 1))
        args.num_topk_groups = min(num_nodes, 4)

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)

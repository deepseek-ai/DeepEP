import os
import time
import torch
import torch.distributed as dist
from functools import partial

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, create_grouped_scores, inplace_unique, per_token_cast_to_fp8, per_token_cast_back

# Test compatibility with low latency functions
import test_low_latency


def test_main_decoupled(num_sms: int, num_tokens: int, num_max_dispatch_tokens_per_rank: int, hidden: int, num_topk_groups: int, num_topk: int, num_experts: int, 
                        local_rank: int, num_local_ranks: int, num_ranks: int, num_nodes: int, rank: int, buffer: deep_ep.Buffer, group: dist.ProcessGroup):
    # Settings
    assert num_experts % num_ranks == 0 and num_local_ranks == 8
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk_groups={num_topk_groups}, num_topk={num_topk}', flush=True)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_e4m3 = per_token_cast_to_fp8(x)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=num_topk_groups, dim=-1, sorted=False).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)
    rdma_rank_idx = rank_idx // num_local_ranks
    rdma_rank_idx.masked_fill_(rank_idx == -1, -1)
    inplace_unique(rdma_rank_idx, num_nodes)

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
        print(f'[layout] get_dispatch_layout() Kernel performance: {t * 1000:.3f} ms', flush=True)
        print('', flush=True)
    group.barrier()
    time.sleep(1)

    # Config
    rdma_buffer_size, nvl_buffer_size = 128, (720 if num_ranks in (144, 160) else 512)
    nvl_buffer_size = 64 * num_nodes  ## for my testing environment 2/4 nodes, I use this config
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size, 16, rdma_buffer_size)

    # noinspection PyShadowingNames
    def check_data(check_x, recv_gbl_rank_prefix_sum):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = recv_gbl_rank_prefix_sum[i].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    # Test dispatch: all modes
    for previous_mode in (False, True):
        for async_mode, return_recv_hook in [(True, False), (False, False), (False, True)]:  ## all modes
            for current_x in (x_pure_rand, x, x_e4m3):
                for with_topk in (False, True):
                    if local_rank == 0:
                        print(f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, return_recv_hook={return_recv_hook}, previous={previous_mode}) ...', flush=True, end='')
                    dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,  'is_token_in_rank': is_token_in_rank,
                                     'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': async_mode, 'return_recv_hook': return_recv_hook}
                    if with_topk:
                        dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights_pure_rand if current_x is x_pure_rand else topk_weights})
                    if previous_mode:
                        dispatch_args.update({'previous_event': buffer.capture()})
                    if return_recv_hook:
                        dispatch_args.update({'num_max_dispatch_tokens_per_rank': num_max_dispatch_tokens_per_rank})

                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event, hook = buffer.dispatch(**dispatch_args)
                    event.current_stream_wait() if async_mode else ()
                    if return_recv_hook:
                        hook()

                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    # Checks
                    recv_gbl_rank_prefix_sum = handle[-5]
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                    if current_x is not x_pure_rand:
                        check_data(recv_x, recv_gbl_rank_prefix_sum)
                    if with_topk:
                        # Check `topk_idx`
                        assert (recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        if current_x is not x_pure_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                            check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode, 'return_recv_hook': return_recv_hook}
                        if previous_mode:
                            dispatch_args.update({'previous_event': buffer.capture()})
                        recv_x, _, _, _, _, event, hook = buffer.dispatch(**dispatch_args)

                        event.current_stream_wait() if async_mode else ()
                        if return_recv_hook:
                            hook()
                            # torch.cuda.synchronize()

                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                        if current_x is not x_pure_rand:
                            check_data(recv_x, recv_gbl_rank_prefix_sum)

                    # Test combine
                    bias_0 = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    bias_1 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    combine_args = {'x': recv_x, 'bias': (bias_0, bias_1), 'handle': handle, 'config': config, 'async_finish': async_mode, 'return_recv_hook': return_recv_hook}
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if previous_mode:
                        combine_args.update({'previous_event': buffer.capture()})

                    combined_x, combined_topk_weights, event, hook = buffer.combine(**combine_args)

                    event.current_stream_wait() if async_mode else ()
                    if return_recv_hook:
                        hook()

                    check_x = (combined_x.float() - bias_0.float() - bias_1.float()) / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6
                    if with_topk:
                        check_topk_weights = combined_topk_weights if (current_x is x_pure_rand) else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))
                        ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning
                    dispatch_bf16_rdma_send_bytes = num_rdma_token_sent * hidden * 2
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
                    combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

                    torch.cuda.synchronize()

                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print('General Tests for All Normal Modes Complete', flush=True)
        print('', flush=True)


    # Border Test for dispatch: hook mode
    def calc_pass():
        pass

    def calc_sleep_1ms():
        time.sleep(0.001)

    def calc_sleep_10ms():
        time.sleep(0.01)

    def calc_sleep_100ms():
        time.sleep(0.1)

    def calc_sync():
        torch.cuda.synchronize()

    torch.cuda.synchronize()

    calc_func_list = [calc_pass, calc_sleep_1ms, calc_sleep_10ms, calc_sleep_100ms, calc_sync]
    calc_func_name_list = ["calc_pass", "calc_sleep_1ms", "calc_sleep_10ms", "calc_sleep_100ms", "calc_sync"]

    for i in range(len(calc_func_list)):
        calc_func = calc_func_list[i]
        calc_func_name = calc_func_name_list[i]
        for previous_mode in (False, True):
            for async_mode, return_recv_hook in [(False, True),]:  ## hook mode
                for current_x in (x_pure_rand, x, x_e4m3):
                    for with_topk in (False, True):
                        if local_rank == 0:
                            print(f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (calc_func={calc_func_name}, async={async_mode}, return_recv_hook={return_recv_hook}, previous={previous_mode}) ...', flush=True, end='')
                        dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank,  'is_token_in_rank': is_token_in_rank,
                                        'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': async_mode, 'return_recv_hook': return_recv_hook}
                        if with_topk:
                            dispatch_args.update({'topk_idx': topk_idx, 'topk_weights': topk_weights_pure_rand if current_x is x_pure_rand else topk_weights})
                        if previous_mode:
                            dispatch_args.update({'previous_event': buffer.capture()})
                        if return_recv_hook:
                            dispatch_args.update({'num_max_dispatch_tokens_per_rank': num_max_dispatch_tokens_per_rank})

                        recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event, hook = buffer.dispatch(**dispatch_args)
                        calc_func()
                        hook()

                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                        # Checks
                        recv_gbl_rank_prefix_sum = handle[-5]
                        assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                        assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                        if current_x is not x_pure_rand:
                            check_data(recv_x, recv_gbl_rank_prefix_sum)
                        if with_topk:
                            # Check `topk_idx`
                            assert (recv_topk_idx.eq(-1) | ((recv_topk_idx >= 0) & (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                            for i, count in enumerate(recv_num_tokens_per_expert_list):
                                assert recv_topk_idx.eq(i).sum().item() == count

                            # Check `topk_weights`
                            if current_x is not x_pure_rand:
                                recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                                check_data(recv_topk_weights, recv_gbl_rank_prefix_sum)

                        # Test cached dispatch (must without top-k staffs)
                        if not with_topk:
                            dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode, 'return_recv_hook': return_recv_hook}
                            if previous_mode:
                                dispatch_args.update({'previous_event': buffer.capture()})
                            recv_x, _, _, _, _, event, hook = buffer.dispatch(**dispatch_args)
                            event.current_stream_wait() if async_mode else ()

                            calc_func()
                            hook()

                            recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                            if current_x is not x_pure_rand:
                                check_data(recv_x, recv_gbl_rank_prefix_sum)

                        # Test combine
                        combine_args = {'x': recv_x, 'handle': handle, 'config': config, 'async_finish': async_mode, 'return_recv_hook': return_recv_hook}
                        if with_topk:
                            combine_args.update({'topk_weights': recv_topk_weights})
                        if previous_mode:
                            combine_args.update({'previous_event': buffer.capture()})
                        combined_x, combined_topk_weights, event, hook = buffer.combine(**combine_args)
                        event.current_stream_wait() if async_mode else ()

                        calc_func()
                        hook()

                        check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)
                        ref_x = x_pure_rand if current_x is x_pure_rand else x
                        assert calc_diff(check_x, ref_x) < 5e-6
                        if with_topk:
                            check_topk_weights = combined_topk_weights if (current_x is x_pure_rand) else (combined_topk_weights / is_token_in_rank.sum(dim=1).unsqueeze(1))
                            ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                            assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                        torch.cuda.synchronize()
                        if local_rank == 0:
                            print(' passed', flush=True)
    if local_rank == 0:
        print('Border Tests for Hook Mode Complete', flush=True)
        print('', flush=True)


    ### Tune decoupled mode dispatch performance
    # noinspection PyShadowingNames
    def large_gemm_with_hook(hook):
        mat_0 = torch.randn((4096, 4096), dtype=torch.float)
        mat_1 = torch.randn((4096, 4096), dtype=torch.float)
        mat_0 @ mat_1
        hook()

    # noinspection PyShadowingNames
    def test_dispatch_hook(x, config, handle, return_recv_hook):
        _, _, _, _, _, _, hook = \
            buffer.dispatch(x=x, config=config, handle=handle, async_finish=False, return_recv_hook=return_recv_hook, num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank)
        large_gemm_with_hook(hook) if return_recv_hook else None
        torch.cuda.synchronize()

    def test_combine_hook(x, config, handle, return_recv_hook):
        _, _, _, hook = \
            buffer.combine(x=x, config=config, handle=handle, async_finish=False, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None
        torch.cuda.synchronize()

    def test_dispatch_combine_hook(x, config, handle, return_recv_hook):
        recv_x, _, _, _, _, _, hook = \
            buffer.dispatch(x=x, config=config, handle=handle, async_finish=False, return_recv_hook=return_recv_hook, num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank)
        large_gemm_with_hook(hook) if return_recv_hook else None

        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

        _, _, _, hook = \
            buffer.combine(x=recv_x, config=config, handle=handle, async_finish=False, return_recv_hook=return_recv_hook)
        large_gemm_with_hook(hook) if return_recv_hook else None
        torch.cuda.synchronize()


    ## Hook mode Dispatch
    torch.cuda.synchronize()
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (dispatch_bf16_rdma_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_send_bytes
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        for nvl_chunk_size in range(2, 45, 2):
            for rdma_chunk_size in range(4, 33, 4):
                for return_recv_hook in (True,):
                    config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
                    dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank, 'is_token_in_rank': is_token_in_rank,
                                    'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': False, 'return_recv_hook': True, 
                                    'num_max_dispatch_tokens_per_rank': num_max_dispatch_tokens_per_rank}
                    _, _, _, _, handle_hook_tuning, _, hook = buffer.dispatch(**dispatch_args)
                    hook()
                    torch.cuda.synchronize()

                    # trace_path = f'/home/nas/zhiyihu/traces/trace_rank_{rank}_nvl_chunk_size_{nvl_chunk_size}_rdma_chunk_size_{rdma_chunk_size}.json'
                    # dispatch_t, gemm_t = bench_kineto(partial(test_dispatch_hook, x=current_x, config=config, handle=handle_hook_tuning, return_recv_hook=return_recv_hook),
                    #                     kernel_names=('dispatch', 'gemm'), trace_path=trace_path, barrier_comm_profiling=True, suppress_kineto_output=False)
                    dispatch_t, gemm_t = bench_kineto(partial(test_dispatch_hook, x=current_x, config=config, handle=handle_hook_tuning, return_recv_hook=return_recv_hook),
                                        kernel_names=('dispatch', 'gemm'))

                    if dispatch_t < best_time:
                        best_time, best_results = dispatch_t, (num_sms, nvl_chunk_size, rdma_chunk_size)
                    if local_rank == 0:
                        print(f'[Tuning Decoupled Mode Dispatch, With Recv Hook] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: Dispatch send kernel time plus recv kernel time: {dispatch_t * 2 * 1e6:.2f} us, GEMM kernel time: {gemm_t * 1e6:.2f} us ', flush=True)

        if local_rank == 0:
            print(f'[Tuning Decoupled Mode Dispatch, With Recv Hook] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}, rdma_send_bytes {rdma_send_bytes}, nvl_recv_bytes {nvl_recv_bytes}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: {best_time * 2 * 1e6:.2f} us, {rdma_send_bytes / 1e9 / (best_time * 2):.2f} GB/s (RDMA) ', flush=True)
            print('', flush=True)

    ## Hook mode Combine
    torch.cuda.synchronize()
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (dispatch_bf16_rdma_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_send_bytes
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        for nvl_chunk_size in range(1, 13, 1):
            for rdma_chunk_size in range(8, 33, 4):
                for return_recv_hook in (True,):
                    config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
                    dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank, 'is_token_in_rank': is_token_in_rank,
                                    'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': False, 'return_recv_hook': True, 
                                    'num_max_dispatch_tokens_per_rank': num_max_dispatch_tokens_per_rank}
                    recv_x, _, _, _, handle_hook_tuning, _, hook = buffer.dispatch(**dispatch_args)
                    hook()
                    torch.cuda.synchronize()

                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    combine_t, gemm_t = bench_kineto(partial(test_combine_hook, x=recv_x, config=config, handle=handle_hook_tuning, return_recv_hook=return_recv_hook),
                                        kernel_names=('combine', 'gemm'))

                    if combine_t < best_time:
                        best_time, best_results = combine_t, (num_sms, nvl_chunk_size, rdma_chunk_size)
                    if local_rank == 0:
                        print(f'[Tuning Decoupled Mode Combine, With Recv Hook] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: Combine send kernel time plus recv kernel time: {combine_t * 2 * 1e6:.2f} us, GEMM kernel time: {gemm_t * 1e6:.2f} us ', flush=True)

        if local_rank == 0:
            print(f'[Tuning Decoupled Mode Combine, With Recv Hook] Best combine (nvl_send_bytes {combine_bf16_nvl_send_bytes}, rdma_recv_bytes {combine_bf16_rdma_recv_bytes}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: {best_time * 2 * 1e6:.2f} us, {combine_bf16_rdma_recv_bytes / 1e9 / (best_time * 2):.2f} GB/s (RDMA) ', flush=True)
            print('', flush=True)

    ## Hook mode Dispatch + Combine
    torch.cuda.synchronize()
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (dispatch_bf16_rdma_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_send_bytes
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        for nvl_chunk_size in range(1, 13, 1):
            for rdma_chunk_size in range(8, 33, 4):
                for return_recv_hook in (True,):
                    config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
                    dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank, 'is_token_in_rank': is_token_in_rank,
                                    'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': False, 'return_recv_hook': True, 
                                    'num_max_dispatch_tokens_per_rank': num_max_dispatch_tokens_per_rank}
                    _, _, _, _, handle_hook_tuning, _, hook = buffer.dispatch(**dispatch_args)
                    hook()
                    torch.cuda.synchronize()

                    dispatch_t, combine_t, gemm_t = bench_kineto(partial(test_dispatch_combine_hook, x=current_x, config=config, handle=handle_hook_tuning, return_recv_hook=return_recv_hook),
                                        kernel_names=('dispatch', 'combine', 'gemm'))

                    if dispatch_t + combine_t < best_time:
                        best_time, best_results = dispatch_t + combine_t, (num_sms, nvl_chunk_size, rdma_chunk_size)
                    if local_rank == 0:
                        print(f'[Tuning Decoupled Mode Dispatch + Combine, With Recv Hook] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: Dispatch + Combine send kernel time plus recv kernel time: {(dispatch_t + combine_t) * 2 * 1e6:.2f} us, Dispatch send kernel time plus recv kernel time: {dispatch_t * 2 * 1e6:.2f} us, Combine send kernel time plus recv kernel time: {combine_t * 2 * 1e6:.2f} us, GEMM kernel time: {gemm_t * 1e6:.2f} us ', flush=True)

        if local_rank == 0:
            print(f'[Tuning Decoupled Mode Dispatch + Combine, With Recv Hook] Best dispatch {"FP8" if isinstance(current_x, tuple) else "BF16"} + combine BF16 (rdma_send_bytes {rdma_send_bytes}, nvl_recv_bytes {nvl_recv_bytes}, nvl_send_bytes {combine_bf16_nvl_send_bytes}, rdma_recv_bytes {combine_bf16_rdma_recv_bytes}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: {best_time * 2 * 1e6:.2f} us ', flush=True)
            print('', flush=True)


    ### Tune native (non-decoupled) mode dispatch performance
    # noinspection PyShadowingNames
    def test_func_native(x, config, handle):
        _, _, _, _, _, event, _ = \
            buffer.dispatch(x=x, config=config, handle=handle, async_finish=False, return_recv_hook=False)

    torch.cuda.synchronize()
    num_sms = 24
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (dispatch_bf16_rdma_send_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_rdma_send_bytes
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        for nvl_chunk_size in range(4, 45, 4):
            for rdma_chunk_size in range(4, 33, 4):
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
                dispatch_args = {'x': current_x, 'num_tokens_per_rank': num_tokens_per_rank, 'num_tokens_per_rdma_rank': num_tokens_per_rdma_rank, 'is_token_in_rank': is_token_in_rank,
                                'num_tokens_per_expert': num_tokens_per_expert, 'config': config, 'async_finish': async_mode, 
                                'num_max_dispatch_tokens_per_rank': num_max_dispatch_tokens_per_rank}
                _, _, _, _, handle_native, _, _ = buffer.dispatch(**dispatch_args)
                avg_t = bench(partial(test_func_native, x=current_x, config=config, handle=handle_native))[0]
                if avg_t < best_time:
                    best_time, best_results = avg_t, (num_sms, nvl_chunk_size, rdma_chunk_size)
                if local_rank == 0:
                    print(f'[Tuning Native Mode Dispatch] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: Dispatch kernel time: {avg_t * 1e6:.2f} us, Dispatch bandwidth: {rdma_send_bytes / 1e9 / avg_t:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / avg_t:.2f} GB/s (NVL) ', flush=True)
        if local_rank == 0:
            print(f'[Tuning Native Mode Dispatch] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}, rdma_send_bytes {rdma_send_bytes}, nvl_recv_bytes {nvl_recv_bytes}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: {best_time * 1e6:.2f} us, {rdma_send_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL)', flush=True)
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
                     'config': dispatch_config if dispatch_config is not None else config}
    recv_x, _, _, _, handle_native, _, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in range(1, 13, 1):
        for rdma_chunk_size in range(8, 33, 4):
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
            tune_args = {'x': recv_x, 'handle': handle_native, 'config': config}
            avg_t = bench(lambda: buffer.combine(**tune_args))[0]
            if local_rank == 0:
                print(f'[Tuning Native Mode Combine] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}: Combine kernel time: {avg_t * 1e6:.2f} us, Combine bandwidth: {combine_bf16_rdma_recv_bytes / 1e9 / avg_t:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / avg_t:.2f} GB/s (NVL) ', flush=True)
                if avg_t < best_time:
                    best_time, best_results = avg_t, (num_sms, nvl_chunk_size, rdma_chunk_size)

    if local_rank == 0:
        print(f'[Tuning Native Mode Combine] Best combine BF16: SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}: {best_time * 1e6:.2f} us, {combine_bf16_rdma_recv_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL)', flush=True)
        print('', flush=True)



# noinspection PyUnboundLocalVariable
def test_loop_decoupled(local_rank: int, num_local_ranks: int):
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    num_tokens, hidden, num_topk_groups, num_topk, num_experts = 4096, 7168, min(num_nodes, 4), 8, (256 // num_ranks) * num_ranks

    # num_max_dispatch_tokens_per_rank = num_tokens + 100
    num_max_dispatch_tokens_per_rank = num_tokens

    test_ll_compatibility = True
    if test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9

    num_sms= 64
    num_qps_per_rank = max(num_sms, ll_num_experts // num_ranks if test_ll_compatibility else 0)

    return_recv_hook = True
    num_rdma_bytes = deep_ep.Buffer.get_normal_hook_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_nodes, num_sms, return_recv_hook)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group, num_nvl_bytes=int(8e9), num_rdma_bytes=num_rdma_bytes, low_latency_mode=test_ll_compatibility,
                            num_qps_per_rank=num_qps_per_rank)
    assert num_local_ranks == 8 and num_ranks > 8
    torch.manual_seed(rank)

    for i in (num_sms, ):
        test_main_decoupled(i, num_tokens, num_max_dispatch_tokens_per_rank, hidden, num_topk_groups, num_topk, num_experts, 
                            local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group)
        if local_rank == 0:
            print('', flush=True)

    # Test compatibility with low latency functions
    if test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
        test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the communication group
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    num_processes = 8
    torch.multiprocessing.spawn(test_loop_decoupled, args=(num_processes, ), nprocs=num_processes)

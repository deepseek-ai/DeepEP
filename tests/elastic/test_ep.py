import argparse
import os
import torch
import torch.distributed as dist
from typing import Union, Tuple, Optional

import deep_ep
from deep_ep.utils.math import (
    align, count_bytes, calc_diff,
    per_token_cast_back, per_token_cast_to_fp8,
    safe_div
)
from deep_ep.utils.gate import get_unbalanced_scores
from deep_ep.utils.envs import init_dist, init_seed, dist_print
from deep_ep.utils.refs import dispatch as ref_dispatch
from deep_ep.utils.refs import combine as ref_combine
from deep_ep.utils.refs import generate_pre_combine_data, ordered_accumulate
from deep_ep.utils.testing import bench_kineto


# noinspection PyUnusedLocal,PyShadowingNames
def enumerate_ep_modes(expert_alignment=None, use_fp8_dispatch=None, num_bias=None):
    for do_handle_copy in (1, 0):
        for ea in (128, 1):
            if expert_alignment is not None and ea != expert_alignment:
                continue
            for fp8 in (1, 0):
                if use_fp8_dispatch is not None and fp8 != use_fp8_dispatch:
                    continue
                for nb in (0, 1, 2):
                    if num_bias is not None and nb != num_bias:
                        continue
                    for with_previous_event in (0, 1):
                        for async_with_compute_stream in (0, 1):
                            for allocate_on_comm_stream in ((1, ) if with_previous_event else (0, 1)):
                                yield (do_handle_copy, ea, fp8, nb,
                                       with_previous_event, async_with_compute_stream, allocate_on_comm_stream)


def launch(buffer: deep_ep.ElasticBuffer, name: str,
           with_previous_event: int, async_with_compute_stream: int,
           params: dict):
    if with_previous_event:
        params.update(previous_event=buffer.capture())
    values = getattr(buffer, name)(**params)
    values[-1].current_stream_wait() if async_with_compute_stream else ()
    return values


def fold_expanded(expanded: Union[Tuple[torch.Tensor], torch.Tensor],
                  indices: torch.Tensor, valid_mask: torch.Tensor):
    if not isinstance(expanded, torch.Tensor):
        return tuple(fold_expanded(t, indices, valid_mask) for t in expanded)

    gathered = expanded[indices]
    first_valid_idx = valid_mask.to(torch.int).argmax(dim=1)
    folded = gathered[torch.arange(gathered.shape[0], device='cuda'), first_valid_idx]
    result = (gathered == folded.unsqueeze(1)).all(dim=-1)
    result = result | (~valid_mask)
    assert result.all()
    return folded


def clone_recv_x(recv_x: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
    if isinstance(recv_x, tuple):
        return tuple(t.clone() for t in recv_x)
    return recv_x.clone()


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_dispatch_combine(buffer: deep_ep.ElasticBuffer, args: argparse.Namespace):
    # Settings
    num_scaleout_ranks, num_scaleup_ranks = buffer.get_logical_domain_size()
    num_max_tokens_per_rank, num_tokens, hidden = args.num_tokens, max(1, args.num_tokens - dist.get_rank()), args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_local_experts = num_experts // buffer.num_ranks
    num_sms = buffer.get_theoretical_num_sms(num_experts, num_topk) if args.num_sms == 0 else args.num_sms
    num_qps = buffer.get_theoretical_num_qps(num_sms) if args.num_qps == 0 else args.num_qps
    dist_print(f'Config:\n'
               f' > Ranks: {num_scaleout_ranks} x {num_scaleup_ranks}\n'
               f' > Experts: {num_topk}/{num_experts}\n'
               f' > Tokens: {num_tokens} (max: {num_max_tokens_per_rank}), hidden: {hidden}\n'
               f' > #SM: {num_sms}, #QPs: {num_qps}/{buffer.num_allocated_qps}\n'
               f' > Mode: {"DRV_X" if os.environ.get("EP_V2_EXPERIMENTAL_DIRECT_RECV_X", "0") == "1" else "baseline"}',
               once_in_node=True)

    # Construct expert selections first (may have an unbalanced ratio here)
    scores = get_unbalanced_scores(num_tokens, num_experts, buffer.num_ranks, num_topk, args.unbalanced_ratio, args.precise_unbalanced_ratio)
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    if args.masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
        topk_weights.masked_fill_(topk_idx < 0, 0)

    # Run all tests
    dist_print('Running all test cases:', once_in_node=True)
    for (do_handle_copy, expert_alignment, use_fp8_dispatch, num_bias,
         with_previous_event, async_with_compute_stream, allocate_on_comm_stream) in enumerate_ep_modes(
             expert_alignment=args.expert_alignment, use_fp8_dispatch=args.use_fp8_dispatch, num_bias=args.num_bias):
        use_direct_recv_x = os.environ.get('EP_V2_EXPERIMENTAL_DIRECT_RECV_X', '0') == '1'
        run_expanded_dispatch = not use_direct_recv_x
        # DRV_X restrictions: no cached handle, no previous event, no deterministic dispatch
        if use_direct_recv_x and (do_handle_copy or with_previous_event or args.deterministic):
            continue

        dist_print(f' > Testing with '
                   f'{do_handle_copy=}, {expert_alignment=}, {use_fp8_dispatch=}, {num_bias=}, '
                   f'{with_previous_event=}, {async_with_compute_stream=}, {allocate_on_comm_stream=} ...',
                   once_in_node=True)

        # Random data
        # TODO: support top-k groups
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        x = per_token_cast_to_fp8(x) if use_fp8_dispatch else x
        bias = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') if num_bias == 1 else None
        if num_bias == 2:
            bias = tuple(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') for _ in range(num_bias))
            assert len(bias) == 2   # To prevent linter warning

        def get_recv_x_bf16(recv_x) -> torch.Tensor:
            if use_fp8_dispatch:
                return per_token_cast_back(recv_x[0], recv_x[1])
            else:
                return recv_x

        # Test correctness with NCCL reference
        if not args.skip_check:
            ref_recv_x, ref_recv_topk_idx, ref_recv_topk_weights, \
                ref_recv_src_token_idx, ref_num_recv_tokens_per_rank = \
                ref_dispatch(x, topk_idx, topk_weights, num_max_tokens_per_rank, num_experts)
            ref_recv_x_bf16 = get_recv_x_bf16(ref_recv_x)

            if args.allow_multiple_reduction:
                # Should be the same as the trigger condition of DeepEP's hybrid combine, which performs intra-scaleup reduction first
                if args.allow_hybrid_mode and num_scaleout_ranks > 1:
                    reduced_combine_recipe = (True, True)
                    combine_recipe = (True, True)
                else:
                    reduced_combine_recipe = (True, False)
                    combine_recipe = (True, False)
            else:
                reduced_combine_recipe = (False, False)
                combine_recipe = (True, False)
            ref_y = generate_pre_combine_data(
                dist.get_rank() * num_max_tokens_per_rank + torch.arange(num_tokens, device='cuda'),
                num_max_tokens_per_rank, num_topk, hidden)
            ref_y[topk_idx == -1] = 0
            ref_reduced_combined_y = ref_combine(
                ref_y, topk_idx,
                num_scaleout_ranks, num_scaleup_ranks, num_experts,
                bias,
                *reduced_combine_recipe
            )
            ref_combined_y = ref_combine(
                ref_y, topk_idx,
                num_scaleout_ranks, num_scaleup_ranks,
                num_experts, bias,
                *combine_recipe
            )  # Reduce within rank, then globally, for non-expand combine mode
            torch.cuda.synchronize()

        # Do dispatch
        dispatch_args = dict(
            x=x, topk_idx=topk_idx, topk_weights=topk_weights,
            num_sms=num_sms, num_qps=num_qps,
            num_max_tokens_per_rank=num_max_tokens_per_rank, num_experts=num_experts,
            expert_alignment=expert_alignment,
            async_with_compute_stream=async_with_compute_stream,
            allocate_on_comm_stream=allocate_on_comm_stream,
            do_handle_copy=do_handle_copy, do_cpu_sync=args.do_cpu_sync)
        recv_x, recv_topk_idx, recv_topk_weights, handle, dispatch_event = \
            launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, dispatch_args)
        # Direct recv modes return a view into the symmetric window tail. Later dispatches and
        # perf benches reuse that window, so preserve the compact result for correctness checks.
        if use_direct_recv_x:
            recv_x = clone_recv_x(recv_x)
        recv_x_bf16 = get_recv_x_bf16(recv_x)

        # Expanding mode
        if run_expanded_dispatch:
            expanded_dispatch_args = dispatch_args | dict(do_expand=True, use_tma_aligned_col_major_sf=True)
            expanded_recv_x, expanded_recv_topk_idx, expanded_recv_topk_weights, expanded_handle, expanded_dispatch_event = \
                launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, expanded_dispatch_args)
            if use_direct_recv_x:
                expanded_recv_x = clone_recv_x(expanded_recv_x)
            expanded_recv_x_bf16 = get_recv_x_bf16(expanded_recv_x)

        # Cached mode
        if not use_direct_recv_x:
            cached_dispatch_args = dict(
                x=x,
                num_sms=num_sms, num_qps=num_qps,
                async_with_compute_stream=async_with_compute_stream,
                allocate_on_comm_stream=allocate_on_comm_stream,
                handle=handle)
            cached_recv_x, cached_recv_topk_idx, cached_recv_topk_weights, cached_handle, cached_dispatch_event = \
                launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, cached_dispatch_args)
            if use_direct_recv_x:
                cached_recv_x = clone_recv_x(cached_recv_x)

        # Cached expanding mode with zero padding
        if run_expanded_dispatch:
            cached_expanded_dispatch_args = cached_dispatch_args | dict(
                topk_weights=topk_weights, do_expand=True, use_tma_aligned_col_major_sf=True,
                do_zero_padding=True, handle=expanded_handle)
            cached_expanded_recv_x, _, cached_expanded_recv_topk_weights, _, _ = \
                launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, cached_expanded_dispatch_args)
            if use_direct_recv_x:
                cached_expanded_recv_x = clone_recv_x(cached_expanded_recv_x)
        # Count the number of received tokens
        num_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
        if run_expanded_dispatch:
            assert num_recv_tokens == expanded_handle.psum_num_recv_tokens_per_scaleup_rank[-1].item(), \
                   'Expand should not affect the number of received tokens.'
            num_expanded_tokens = expanded_handle.psum_num_recv_tokens_per_expert[-1].item()

        # Construction the input data for DeepEP combine
        src_token_global_idx = handle.recv_src_metadata[:num_recv_tokens, 0]
        if not args.skip_check:
            sorted_src_token_global_idx = torch.sort(src_token_global_idx).values
            assert torch.equal(ref_recv_src_token_idx, sorted_src_token_global_idx), \
                f'{ref_recv_src_token_idx=}, {sorted_src_token_global_idx=}'
        local_y = generate_pre_combine_data(src_token_global_idx, num_max_tokens_per_rank, num_topk, hidden)  # [num_recv_tokens, topk, hidden]
        local_y[recv_topk_idx[:num_recv_tokens] == -1] = 0
        local_reduced_y = ordered_accumulate(local_y)
        input_for_combine = torch.empty_like(recv_x_bf16, dtype=torch.bfloat16, device='cuda')
        input_for_combine[:num_recv_tokens] = local_reduced_y

        # Expanded combine data
        if run_expanded_dispatch:
            expanded_src_token_global_idx = expanded_handle.recv_src_metadata[:num_recv_tokens, 0]
            if not args.skip_check:
                sorted_expanded_src_token_global_idx = torch.sort(expanded_src_token_global_idx).values
                assert torch.equal(ref_recv_src_token_idx, sorted_expanded_src_token_global_idx), \
                    f'{ref_recv_src_token_idx=}, {sorted_expanded_src_token_global_idx=}'
            local_y_expand = generate_pre_combine_data(expanded_src_token_global_idx, num_max_tokens_per_rank, num_topk, hidden)  # [num_recv_tokens, topk, hidden]
            # We put an extra row to conveniently handle the -1 index
            input_for_expand_combine = torch.empty((expanded_recv_x_bf16.shape[0] + 1, hidden), dtype=torch.bfloat16, device='cuda')
            input_for_expand_combine[expanded_handle.recv_src_metadata[:num_recv_tokens, 2:].flatten()] = local_y_expand.view(-1, hidden)
            input_for_expand_combine = input_for_expand_combine[:-1, ...]

        # Do combine
        combine_args = dict(
            x=input_for_combine, topk_weights=recv_topk_weights, bias=bias,
            handle=handle,
            num_sms=num_sms, num_qps=num_qps,
            async_with_compute_stream=async_with_compute_stream,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        combined_x, combined_topk_weights, combine_event = \
            launch(buffer, 'combine', with_previous_event, async_with_compute_stream, combine_args)

        # Reduced combine
        if run_expanded_dispatch:
            reduced_combine_args = dict(
                x=input_for_expand_combine, bias=bias,
                handle=expanded_handle,
                num_sms=num_sms, num_qps=num_qps,
                async_with_compute_stream=async_with_compute_stream,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
            # NOTES: expand mode requires `allow_multiple_reduction = True` to support topk_weights
            if args.allow_multiple_reduction:
                reduced_combine_args['topk_weights'] = expanded_recv_topk_weights
            reduced_combined_x, reduced_combined_topk_weights, reduced_combine_event = \
                launch(buffer, 'combine', with_previous_event, async_with_compute_stream, reduced_combine_args)

        assert not (args.dump_profile_traces and args.skip_perf_test), '`--skip-perf-test` should not be specified when `--dump-profile-traces` is provided'
        if not args.skip_perf_test:
            # Profiling
            def get_trace_path(prefix: str):
                return None if not args.dump_profile_traces else f'{args.dump_profile_traces}/{prefix}_rank{buffer.rank_idx}.json'

            # Calculate the number of tokens that are sent to the other scaleout peers
            dst_scaleout_rank_idx = topk_idx // (num_experts // num_scaleout_ranks)
            num_scaleout_send_tokens = 0
            for i in range(num_scaleout_ranks if num_scaleout_ranks > 1 else 0):
                if args.ignore_local_traffic and i == dist.get_rank() // num_scaleup_ranks:
                    continue
                num_scaleout_send_tokens += (dst_scaleout_rank_idx == i).any(dim=1).sum().item()

            # Calculate the number of tokens that are received via the other scaleup peers
            num_scaleup_recv_tokens = num_recv_tokens
            if args.ignore_local_traffic:
                num_scaleup_recv_tokens -= (src_token_global_idx // num_max_tokens_per_rank % num_scaleup_ranks == dist.get_rank() % num_scaleup_ranks).sum().item()

            # Test dispatch performance
            num_bytes_per_dispatch_token = safe_div(count_bytes(recv_x, recv_topk_idx, recv_topk_weights), recv_topk_idx.size(0))
            num_scaleup_bytes = num_bytes_per_dispatch_token * num_scaleup_recv_tokens  # Received via scaleup
            num_scaleout_bytes = num_bytes_per_dispatch_token * num_scaleout_send_tokens    # Send via scaleout
            t, copy_t = bench_kineto(lambda: buffer.dispatch(**dispatch_args),
                                    kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),
                                    barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('dispatch'))
            dist_print(f'   * EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                    f'dispatch: '
                    f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                    f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                    f'copy: {2 * num_recv_tokens * num_bytes_per_dispatch_token / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')

            # Test expanded dispatch performance
            if run_expanded_dispatch:
                num_bytes_per_dispatch_token_meta = safe_div(count_bytes(expanded_handle.recv_src_metadata), expanded_handle.recv_src_metadata.size(0))
                t, copy_t = bench_kineto(lambda: buffer.dispatch(**expanded_dispatch_args),
                                        kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),
                                        barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('expanded_dispatch'))
                dist_print(f'   - EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                        f'expanded dispatch: '
                        f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                        f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                        f'copy: {(num_recv_tokens * (num_bytes_per_dispatch_token_meta + num_bytes_per_dispatch_token) + num_expanded_tokens * num_bytes_per_dispatch_token) / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')

            # Test cached dispatch performance
            if not use_direct_recv_x:
                t, copy_t = bench_kineto(lambda: buffer.dispatch(**cached_dispatch_args),
                                        kernel_names=('dispatch_impl', 'dispatch_copy_epilogue_impl'),
                                        barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('cached_dispatch'))
                dist_print(f'   # EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                        f'cached dispatch: '
                        f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                        f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                        f'copy: {2 * num_scaleup_bytes / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')

            # Test combine performance
            num_bytes_per_combine_token = safe_div(count_bytes(recv_x_bf16, recv_topk_weights), recv_x_bf16.size(0))
            num_bias_bytes = count_bytes(bias)
            num_reduction_write_bytes = count_bytes(combined_x, combined_topk_weights)

            def get_combine_bytes(is_expand_mode: bool) -> Tuple[float, float, float]:
                num_experts_per_rank = num_experts // (num_scaleup_ranks * num_scaleout_ranks)
                num_experts_per_scaleout_rank = num_experts_per_rank * num_scaleup_ranks

                def get_unique_and_valid_dst_count(dst_idx: torch.Tensor,
                                                   ignored_nums_l: Optional[int] = None, ignored_nums_r: Optional[int] = None,
                                                   max_num_in_dst_idx: int = num_experts - 1) -> int:
                    """
                    Get the number of valid destinations, with deduplication within each token and numbers within `[ignored_nums_l, ignored_nums_r)` being ignored
                    """
                    dst_idx = dst_idx.clone()
                    ignore_mask = dst_idx == -1
                    if args.ignore_local_traffic and ignored_nums_l is not None:
                        assert ignored_nums_r is not None
                        ignore_mask |= ((dst_idx >= ignored_nums_l) & (dst_idx < ignored_nums_r))
                    dst_idx = dst_idx + torch.arange(0, dst_idx.shape[0], dtype=dst_idx.dtype, device=dst_idx.device).unsqueeze(-1) * (max_num_in_dst_idx + 1)  # So that different rows will have different values
                    dst_idx[ignore_mask] = dst_idx[0][0].item()  # So that these `-1`s won't affect the count of unique numbers
                    return torch.unique(dst_idx, sorted=False).numel()

                if not args.allow_multiple_reduction:
                    # No multiple reduction
                    if not is_expand_mode:
                        num_scaleup_tokens = num_scaleup_recv_tokens
                        num_scaleout_tokens = get_unique_and_valid_dst_count(
                            topk_idx // num_experts_per_rank, buffer.scaleout_rank_idx * num_scaleup_ranks, (buffer.scaleout_rank_idx + 1) * num_scaleup_ranks)
                        num_reduction_read_tokens = get_unique_and_valid_dst_count(topk_idx // num_experts_per_rank)
                    else:
                        tokens_src_rank_idx = src_token_global_idx//num_max_tokens_per_rank
                        if args.ignore_local_traffic:
                            num_scaleup_tokens = (recv_topk_idx[:num_recv_tokens] != -1)[tokens_src_rank_idx % num_scaleup_ranks != buffer.scaleup_rank_idx].sum().item()
                        else:
                            num_scaleup_tokens = (recv_topk_idx[:num_recv_tokens] != -1).sum().item()
                        num_scaleout_tokens = get_unique_and_valid_dst_count(
                            topk_idx, buffer.scaleout_rank_idx * num_experts_per_scaleout_rank, (buffer.scaleout_rank_idx + 1) * num_experts_per_scaleout_rank)
                        num_reduction_read_tokens = get_unique_and_valid_dst_count(topk_idx)
                else:
                    # With `allow_multiple_reduction`, "combine" has exactly the same number of tokens as "dispatch"
                    num_scaleup_tokens = num_scaleup_recv_tokens
                    num_scaleout_tokens = num_scaleout_send_tokens
                    if args.allow_hybrid_mode:
                        num_reduction_read_tokens = get_unique_and_valid_dst_count(topk_idx // num_experts_per_scaleout_rank)
                    else:
                        num_reduction_read_tokens = get_unique_and_valid_dst_count(topk_idx // num_experts_per_rank)
                if not args.ignore_local_traffic and num_scaleout_ranks == 1:
                    num_scaleout_tokens = 0
                return num_scaleout_tokens * num_bytes_per_combine_token, num_scaleup_tokens * num_bytes_per_combine_token, num_reduction_read_tokens * num_bytes_per_combine_token

            num_scaleout_bytes, num_scaleup_bytes, num_reduction_read_bytes = get_combine_bytes(False)
            t, copy_t = bench_kineto(lambda: buffer.combine(**combine_args),
                                    kernel_names=('combine_impl', 'combine_reduce_epilogue_impl'),
                                    barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('combine'))
            dist_print(f'   @ EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                    f'combine: '
                    f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                    f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                    f'reduce: {(num_bias_bytes + num_reduction_read_bytes + num_reduction_write_bytes) / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')

            # Test reduced combine performance
            if run_expanded_dispatch:
                num_scaleout_bytes, num_scaleup_bytes, num_reduction_read_bytes = get_combine_bytes(True)
                t, copy_t = bench_kineto(lambda: buffer.combine(**reduced_combine_args),
                                        kernel_names=('combine_impl', 'combine_reduce_epilogue_impl'),
                                        barrier_comm_profiling=True, barrier=buffer.barrier, trace_path=get_trace_path('reduced_combine'))
                dist_print(f'   + EP: {buffer.rank_idx:3}/{buffer.num_ranks} | '
                        f'reduced combine: '
                        f'{num_scaleout_bytes / t / 1e9:.0f} GB/s (SO), '
                        f'{num_scaleup_bytes / t / 1e9:.0f} GB/s (SU), {t * 1e6:.3f} us, {num_scaleup_bytes:.0f} bytes | '
                        f'reduce: {(num_bias_bytes + num_reduction_read_bytes + num_reduction_write_bytes) / copy_t / 1e9:.0f} GB/s, {copy_t * 1e6:.3f} us')
            dist_print(once_in_node=True)

        # Checks
        # NOTES: we do checks after the performance tests, as we may modify some tensors
        if not args.skip_check:
            # Handle copy checks
            assert (topk_idx.data_ptr() != handle.topk_idx.data_ptr()) == do_handle_copy
            if not use_direct_recv_x:
                assert (topk_idx.data_ptr() != cached_handle.topk_idx.data_ptr()) == do_handle_copy
                assert handle.topk_idx.data_ptr() == cached_handle.topk_idx.data_ptr()

            # Make the valid part of the whole tensor for no CPU sync mode
            if not args.do_cpu_sync:
                if use_fp8_dispatch:
                    recv_x = (recv_x[0][:num_recv_tokens], recv_x[1][:num_recv_tokens])
                    if not use_direct_recv_x:
                        cached_recv_x = (cached_recv_x[0][:num_recv_tokens], cached_recv_x[1][:num_recv_tokens])
                else:
                    recv_x = recv_x[:num_recv_tokens]
                    if not use_direct_recv_x:
                        cached_recv_x = cached_recv_x[:num_recv_tokens]
                recv_x_bf16 = recv_x_bf16[:num_recv_tokens]
                recv_topk_idx = recv_topk_idx[:num_recv_tokens]
                recv_topk_weights = recv_topk_weights[:num_recv_tokens]
                if not use_direct_recv_x:
                    cached_recv_topk_idx = cached_recv_topk_idx[:num_recv_tokens]
                handle.recv_src_metadata = handle.recv_src_metadata[:num_recv_tokens]
                if run_expanded_dispatch:
                    expanded_handle.recv_src_metadata = expanded_handle.recv_src_metadata[:num_recv_tokens]

            if run_expanded_dispatch:
                expanded_indices = expanded_handle.recv_src_metadata[:, 2:]
                expanded_mask = expanded_indices >= 0
                valid_expanded_indices = expanded_indices[expanded_mask]

            # Make sure deterministic mode works by doing the dispatch twice
            if args.deterministic:
                recv_x_twice, recv_topk_idx_twice, recv_topk_weights_twice, handle_twice, dispatch_event_twice = \
                    launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, dispatch_args)
                if not args.do_cpu_sync:
                    assert num_recv_tokens == handle_twice.psum_num_recv_tokens_per_scaleup_rank[-1].item()
                    handle_twice.recv_src_metadata = handle_twice.recv_src_metadata[:num_recv_tokens]
                recv_x_twice_bf16 = get_recv_x_bf16(recv_x_twice)
                assert torch.equal(recv_x_bf16, recv_x_twice_bf16[:num_recv_tokens])
                assert torch.equal(recv_topk_idx, recv_topk_idx_twice[:num_recv_tokens])
                assert torch.equal(recv_topk_weights, recv_topk_weights_twice[:num_recv_tokens])
                assert torch.equal(handle.recv_src_metadata[:, :2], handle_twice.recv_src_metadata[:, :2])

                if run_expanded_dispatch:
                    expanded_recv_x_twice, expanded_recv_topk_idx_twice, expanded_recv_topk_weights_twice, expanded_handle_twice, expanded_dispatch_event_twice = \
                        launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, expanded_dispatch_args)
                    expanded_recv_x_twice_bf16 = get_recv_x_bf16(expanded_recv_x_twice)
                    assert torch.equal(expanded_recv_x_bf16[valid_expanded_indices], expanded_recv_x_twice_bf16[valid_expanded_indices])
                    if expert_alignment == 1:
                        assert torch.equal(expanded_recv_topk_weights, expanded_recv_topk_weights_twice)  # Only check for `topk_weight` if `expert_alignment == 1`, since its padding isn't initialized

            # Test cumulative stats counter
            cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
            dispatch_args['cumulative_local_expert_recv_stats'] = cumulative_local_expert_recv_stats
            launch(buffer, 'dispatch', with_previous_event, async_with_compute_stream, dispatch_args)

            # Expanded checks
            if run_expanded_dispatch:
                assert expanded_recv_topk_idx is None
                assert expanded_handle.recv_src_metadata.size(0) == num_recv_tokens
                expanded_indices = expanded_handle.recv_src_metadata[:, 2:]
                expanded_mask = expanded_indices >= 0
                valid_expanded_indices = expanded_indices[expanded_mask]
                expanded_safe_indices = expanded_indices.clone()
                expanded_safe_indices[~expanded_mask] = 0

                if not use_direct_recv_x:
                    # Cached expand checks: compare valid token slots and verify padding is zeroed
                    cached_expanded_recv_x_bf16 = get_recv_x_bf16(cached_expanded_recv_x)
                    assert torch.equal(expanded_recv_x_bf16[valid_expanded_indices], cached_expanded_recv_x_bf16[valid_expanded_indices])
                    assert torch.equal(expanded_recv_topk_weights[valid_expanded_indices], cached_expanded_recv_topk_weights[valid_expanded_indices])

                    # Zero padding checks
                    for expert_idx in range(num_local_experts):
                        start = expanded_handle.psum_num_recv_tokens_per_expert[expert_idx].item()
                        end = align(start, expert_alignment)
                        assert (cached_expanded_recv_x_bf16[start:end] == 0).all()
                        assert (cached_expanded_recv_topk_weights[start:end] == 0).all()

                # Fold expanded
                expanded_recv_x = fold_expanded(expanded_recv_x, expanded_safe_indices, expanded_mask)
                expanded_recv_topk_weights = expanded_recv_topk_weights[expanded_safe_indices]

            # Cached checks
            if not use_direct_recv_x:
                if use_fp8_dispatch:
                    assert torch.equal(recv_x[0], cached_recv_x[0])
                    assert torch.equal(recv_x[1], cached_recv_x[1])
                else:
                    assert torch.equal(recv_x, cached_recv_x)
                assert torch.equal(recv_topk_idx, cached_recv_topk_idx)
                assert torch.equal(handle.dst_buffer_slot_idx, cached_handle.dst_buffer_slot_idx)
                assert torch.equal(handle.psum_num_recv_tokens_per_scaleup_rank, cached_handle.psum_num_recv_tokens_per_scaleup_rank)
                assert handle.num_recv_tokens_per_expert_list == cached_handle.num_recv_tokens_per_expert_list

            # Check dispatch expert count
            assert recv_x_bf16.size() == ref_recv_x_bf16.size(), f'{recv_x_bf16.size()=}, {ref_recv_x_bf16.size()=}'
            assert recv_x_bf16.size(0) == num_recv_tokens
            for i in range(num_local_experts if args.do_cpu_sync else 0):
                ref_count = (ref_recv_topk_idx == i).sum().item()
                aligned_ref_count = align(ref_count, expert_alignment)
                assert ref_count == cumulative_local_expert_recv_stats[i].item(),\
                    f'{i}, {ref_count}, {cumulative_local_expert_recv_stats[i].item()}'
                assert aligned_ref_count == handle.num_recv_tokens_per_expert_list[i]
            psum_num_recv_tokens_per_expert_list = [0] + handle.psum_num_recv_tokens_per_expert.tolist()
            if run_expanded_dispatch:
                expanded_psum_num_recv_tokens_per_expert_list = [0] + expanded_handle.psum_num_recv_tokens_per_expert.tolist()
            for i in range(num_local_experts):
                ref_count = (ref_recv_topk_idx == i).sum().item()
                count = psum_num_recv_tokens_per_expert_list[i + 1] - psum_num_recv_tokens_per_expert_list[i]
                assert align(ref_count, expert_alignment) == count, f'{buffer.rank_idx=}, {i=}, {ref_count=}, {count=}'
                if run_expanded_dispatch:
                    expanded_count = (expanded_psum_num_recv_tokens_per_expert_list[i + 1] -
                                      align(expanded_psum_num_recv_tokens_per_expert_list[i], expert_alignment))
                    assert ref_count == expanded_count, f'{ref_count=}, {expanded_count=}'

            # Check dispatch scale-up received token psum
            psum_num_recv_tokens_per_scaleup_rank_list = [0] + handle.psum_num_recv_tokens_per_scaleup_rank.tolist()
            for i in range(num_scaleup_ranks):
                count = psum_num_recv_tokens_per_scaleup_rank_list[i + 1] - psum_num_recv_tokens_per_scaleup_rank_list[i]
                ref_count = sum(ref_num_recv_tokens_per_rank[i::num_scaleup_ranks])
                assert count == ref_count, f'{ref_count=}, {count=}'

            # Check dispatch data
            check_tuples = [(recv_x, recv_topk_idx, recv_topk_weights, handle)]
            if run_expanded_dispatch:
                check_tuples.append((expanded_recv_x, None, expanded_recv_topk_weights, expanded_handle))
            for check_recv_x, check_recv_topk_idx, check_recv_topk_weights, check_handle in check_tuples:
                for i in range(buffer.num_ranks):
                    rank_start_idx = sum(ref_num_recv_tokens_per_rank[:i])
                    rank_end_idx = rank_start_idx + ref_num_recv_tokens_per_rank[i]
                    sorted_metadata = torch.sort(check_handle.recv_src_metadata[:, 0])
                    sorted_indices = sorted_metadata.indices[rank_start_idx:rank_end_idx]
                    sorted_values = sorted_metadata.values[rank_start_idx:rank_end_idx]
                    assert torch.equal(ref_recv_src_token_idx[rank_start_idx:rank_end_idx], sorted_values)

                    # Data should be bitwise identical
                    check_list = [(ref_recv_topk_weights, check_recv_topk_weights, True)]
                    if check_recv_topk_idx is not None:
                        check_list.append((ref_recv_topk_idx, check_recv_topk_idx, False))
                    if use_fp8_dispatch:
                        check_list.append((ref_recv_x[0], check_recv_x[0], False))
                        check_list.append((ref_recv_x[1], check_recv_x[1], False))
                    else:
                        check_list.append((ref_recv_x, check_recv_x, False))
                    ref_mask = ref_recv_topk_idx[rank_start_idx:rank_end_idx] < 0
                    for ref_t, t, do_mask in check_list:
                        ref_t = ref_t[rank_start_idx:rank_end_idx]
                        t = t[sorted_indices]
                        if do_mask:
                            ref_t = ref_t.masked_fill(ref_mask, 0)
                            t = t.masked_fill(ref_mask, 0)
                        assert torch.equal(ref_t, t), f'{ref_t=}, {t=}'

            # Combined data should also be bitwise-identical
            assert torch.equal(combined_x, ref_combined_y), \
                f'Diff: {calc_diff(combined_x, ref_combined_y)}'
            if run_expanded_dispatch:
                assert torch.equal(reduced_combined_x, ref_reduced_combined_y), \
                    f'Diff: {calc_diff(reduced_combined_x, ref_reduced_combined_y)}'
            assert torch.equal(combined_topk_weights, topk_weights), \
                f'{calc_diff(combined_topk_weights, topk_weights)}'
            if run_expanded_dispatch and args.allow_multiple_reduction:
                assert torch.equal(reduced_combined_topk_weights, topk_weights), \
                    f'{calc_diff(reduced_combined_topk_weights, topk_weights)}'

        # Break on the first test case
        if args.test_first_only:
            break
    dist_print('', once_in_node=True)


# noinspection PyUnboundLocalVariable,PyShadowingNames
@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    # Set experimental direct-recv env vars before any DeepEP imports/construction
    if args.direct_recv_x:
        os.environ['EP_V2_EXPERIMENTAL_DIRECT_RECV_X'] = '1'

    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    def construct_elastic_buffer():
        return deep_ep.ElasticBuffer(group,
                                     num_max_tokens_per_rank=args.num_tokens, hidden=args.hidden,
                                     num_topk=args.num_topk,
                                     deterministic=args.deterministic,
                                     allow_hybrid_mode=args.allow_hybrid_mode,
                                     allow_multiple_reduction=args.allow_multiple_reduction,
                                     prefer_overlap_with_compute=bool(args.prefer_overlap_with_compute),
                                     sl_idx=args.sl_idx,
                                     num_allocated_qps=max(args.num_allocated_qps, args.num_qps),
                                     explicitly_destroy=True,
                                     num_gpu_timeout_secs=args.num_gpu_timeout_secs,
                                     num_cpu_timeout_secs=args.num_cpu_timeout_secs)

    buffer = construct_elastic_buffer()

    # Warning in case of precise unbalanced ratio
    if args.precise_unbalanced_ratio:
        dist_print('\033[33mWarning: Using precise unbalanced ratio mode. '
                   'Test data is manually constructed and may differ from real world distribution.\033[0m',
                   once_in_node=True)

    # Test MoE kernels
    test_dispatch_combine(buffer, args)

    # Pressure tests
    for seed in range(int(1e9) if args.do_pressure_test else 0):
        if not args.reuse_elastic_buffer:
            # Recreate elastic buffer
            buffer.destroy()
            buffer = construct_elastic_buffer()

        assert not args.skip_check
        dist_print(f'Testing with {seed=} ...', once_in_node=True)
        init_seed(seed)
        test_dispatch_combine(buffer, args)

    # Destroy the runtime and communication group
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test elastic EP kernels')

    # Resource settings
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-sms', type=int, default=0, help='Number of SMs to use (0 means auto)')
    parser.add_argument('--num-qps', type=int, default=0, help='Number of QPs to use (0 means auto)')
    parser.add_argument('--num-allocated-qps', type=int, default=0, help='Number of QPs to allocate (0 means auto)')
    parser.add_argument('--num-gpu-timeout-secs', type=int, default=100, help='Timeout in seconds (GPU side)')
    parser.add_argument('--num-cpu-timeout-secs', type=int, default=100, help='Timeout in seconds (CPU side)')
    parser.add_argument('--sl-idx', type=int, default=0, help='SL index')

    # Model settings
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of top-k experts')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts')

    # Scenario settings
    parser.add_argument('--do-cpu-sync', type=int, default=1, help='Whether to do CPU sync')
    parser.add_argument('--allow-hybrid-mode', type=int, default=1, help='Whether to allow hybrid mode')
    parser.add_argument('--allow-multiple-reduction', type=int, default=1, help='Whether to allow multiple reductions')
    parser.add_argument('--prefer-overlap-with-compute', type=int, default=0, help='Whether to prefer overlap with compute')
    parser.add_argument('--direct-recv-x', action='store_true', help='Enable EP_V2_EXPERIMENTAL_DIRECT_RECV_X (non-cached, non-expanded)')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic algorithm')

    # Mode filter settings (None = test all values for that dimension)
    parser.add_argument('--expert-alignment', type=int, default=None, choices=[1, 128], help='Filter: only test this expert alignment')
    parser.add_argument('--use-fp8-dispatch', type=int, default=None, choices=[0, 1], help='Filter: only test FP8 (1) or BF16 (0) dispatch')
    parser.add_argument('--num-bias', type=int, default=None, choices=[0, 1, 2], help='Filter: only test this number of bias tensors')

    # Test settings
    parser.add_argument('--seed', type=int, default=0, help='Default seed for pressure tests')
    parser.add_argument('--skip-check', action='store_true', help='Whether to skip correctness checks')
    parser.add_argument('--skip-perf-test', action='store_true', help='Whether to skip performance tests')
    parser.add_argument('--do-pressure-test', action='store_true', help='Whether to do pressure test')
    parser.add_argument('--reuse-elastic-buffer', action='store_true', help='Whether to reuse elastic buffer for each test')
    parser.add_argument('--test-first-only', action='store_true', help='Only test the first case')
    parser.add_argument('--unbalanced-ratio', type=float, default=1.0, help='The MoE unbalanced ratio')
    parser.add_argument('--precise-unbalanced-ratio', action='store_true', help='Generate topk index with precise unbalanced ratio')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--ignore-local-traffic', action='store_true', help='Whether to ignore local traffic during bandwidth calculation')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    # Launch test processes
    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)

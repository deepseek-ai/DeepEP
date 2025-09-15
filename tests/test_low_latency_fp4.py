import argparse
import random
import time
import os
import torch
import torch.distributed as dist
import numpy as np
from functools import partial
from typing import Optional

from sgl_kernel import scaled_fp4_grouped_quant

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back, get_global_token_idxs, recover_experts_swizzled_scales, get_pair_token_idx

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer,
              use_logfmt: bool = False, seed: int = 0, args: argparse.Namespace = None):
    #############################################################
    # set configurations
    #############################################################
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    # torch.set_printoptions(threshold=2000, edgeitems=8) 
    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks
    padded_m = (((num_ranks * num_tokens) + 128 - 1) // 128) * 128
    padded_k = ((hidden + 64 - 1) // 64) * 64
    if rank == 0:
        print(f'Start testing nvfp4 dispatch')

    #############################################################
    # prepare data to dispatch.
    # Because we can not ensure the order of tokens after dispatch,
    # to make the correctness check easier, we supposed each token has the same value.
    # Otherwise, we have to get the order of received tokens before the correctness check
    #############################################################
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x[:, 0:16] = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], device='cuda').to(torch.bfloat16) + 1
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1) + rank * num_tokens
    
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()
    x_max = torch.max(torch.abs(x))
    x_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x_max.to(torch.float32)
    dist.all_reduce(x_global_scale, op=dist.ReduceOp.MIN, group=group)

    #############################################################
    # dispatch with bf16 data format
    #############################################################
    return_recv_hook = False
    cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
    packed_recv_x, packed_recv_count, handle, event, hook = \
        buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                    use_fp8=False, use_nvfp4=False, x_global_scale=x_global_scale,
                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                    async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
    hook() if return_recv_hook else event.current_stream_wait()
    recv_x = packed_recv_x
    recv_count, recv_src_info, recv_layout_range = packed_recv_count, handle[0], handle[1]

    #############################################################
    # dispatch with nvfp4 data format
    #############################################################
    packed_recv_x_pre_quant, packed_recv_count_pre_quant, handle_pre_quant, event_pre_quant, hook_pre_quant = \
        buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                    use_fp8=False, use_nvfp4=True, x_global_scale=x_global_scale,
                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                    async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
    hook_pre_quant() if return_recv_hook else event_pre_quant.current_stream_wait()
    recv_x_pre_quant = packed_recv_x_pre_quant[0]
    recv_x_pre_quant_scales = packed_recv_x_pre_quant[1]
    recv_count_pre_quant, recv_src_info_pre_quant, recv_layout_range_pre_quant = packed_recv_count_pre_quant, handle_pre_quant[0], handle_pre_quant[1]
    #############################################################
    # prepare data to for checking correctness
    #############################################################
    global_token_idxs_ret = get_global_token_idxs(recv_count, recv_src_info, recv_layout_range, num_local_experts, num_ranks, num_tokens)
    global_token_idxs_test = get_global_token_idxs(recv_count_pre_quant, recv_src_info_pre_quant, recv_layout_range_pre_quant, num_local_experts, num_ranks, num_tokens)

    #############################################################
    # correctness checking.
    # the reference is got by dispatching with bf16 data format
    # and then quantizing the output of dispatch with nvfp4 data format
    #############################################################
    if args.CUDA_ARCH >= 100:
        if rank == 0:
            print(f'Compare nvfp4 dispatch output with grouped quantize output')
        mask = recv_count
        x_sf_global = torch.ones((num_local_experts, ), dtype=torch.float32, device='cuda') * x_global_scale
        recv_x_post_quant, recv_x_scales_post_quant = scaled_fp4_grouped_quant(
            recv_x,
            x_sf_global,
            mask,
        )
        # refer to  https://github.com/sgl-project/sglang/blob/19d64f2b725889cfbdb000937a2d57c07db5cfa8/sgl-kernel/tests/test_fp4_quantize.py#L194
        # # output in logical (m, k, l), but its physical layout is (l, m, k).
        # # So permute first to (l, m, k).
        # output = output.permute(2, 0, 1)
        # # output_scale in logical (32, 4, rm, 4, rk, l), but its physical layout is (l, rm, rk, 32, 4, 4).
        # # So permute first to (l, rm, rk, 32, 4, 4).
        # padded_m = ((m + 128 - 1) // 128) * 128
        # output_scales = output_scales.permute(5, 2, 4, 0, 1, 3).view(l, padded_m, -1)
        recv_x_ref = recv_x_post_quant.permute(2, 0, 1)
        recv_x_scales_ref = recv_x_scales_post_quant.permute(5, 2, 4, 0, 1, 3).view(num_local_experts, -1)
        recv_x_scales_ref = recover_experts_swizzled_scales(recv_x_scales_ref, num_local_experts, padded_m, padded_k)
        
        recv_x_test = recv_x_pre_quant.permute(2, 0, 1)
        recv_x_scales_test = recv_x_pre_quant_scales.permute(5, 2, 4, 0, 1, 3).view(num_local_experts, -1)
        recv_x_scales_test = recover_experts_swizzled_scales(recv_x_scales_test, num_local_experts, padded_m, padded_k)

        for local_expert in range(num_local_experts):
            num_valid_tokens = recv_count[local_expert].item()
            for test_token_idx in range(num_valid_tokens):
                # get the pair token index
                ref_token_idx, global_token_idxs = get_pair_token_idx(global_token_idxs_test, global_token_idxs_ret, local_expert, test_token_idx)
                # check recv_x
                recv_x_bf16_ref_per_token = recv_x[local_expert, ref_token_idx]
                recv_x_ref_per_token = recv_x_ref[local_expert, ref_token_idx]
                recv_x_test_per_token = recv_x_test[local_expert, test_token_idx]
                assert torch.equal(recv_x_ref_per_token, recv_x_test_per_token), f'rank {rank}, recv_x_ref_per_token: {recv_x_ref_per_token}, recv_x_test_per_token: {recv_x_test_per_token}'
                # check recv_x_scales
                recv_x_scales_ref_per_token = recv_x_scales_ref[local_expert, ref_token_idx]
                recv_x_scales_test_per_token = recv_x_scales_test[local_expert, test_token_idx]
                assert torch.equal(recv_x_scales_ref_per_token, recv_x_scales_test_per_token), f'rank {rank}, recv_x_scales_ref_per_token: {recv_x_scales_ref_per_token}, recv_x_scales_test_per_token: {recv_x_scales_test_per_token}'
    
    
    #############################################################
    # correctness checking.
    # the reference is got by dispatching with bf16 data format,
    # and then the reference is compared with dequantized output of nvfp4 dispatch
    #############################################################
    if rank == 0:
        print(f'Compare dequantized nvfp4 dispatch output with bf16 dispatch output')
    recv_x_test = per_token_cast_back(recv_x_pre_quant, recv_x_pre_quant_scales, x_global_scale, src_data_format='nvfp4')
    for local_expert in range(num_local_experts):
        num_valid_tokens = recv_count[local_expert].item()
        assert recv_count_pre_quant[local_expert].item() == num_valid_tokens, f'num_valid_tokens_pre_quant: {num_valid_tokens_pre_quant}, num_valid_tokens: {num_valid_tokens}'
        for test_token_idx in range(num_valid_tokens):
            # get the pair token index
            ref_token_idx, global_token_idxs = get_pair_token_idx(global_token_idxs_test, global_token_idxs_ret, local_expert, test_token_idx)
            # check recv_x
            recv_x_ref_per_token = recv_x[local_expert, ref_token_idx]
            recv_x_test_per_token = recv_x_test[local_expert, test_token_idx]
            diff = calc_diff(recv_x_ref_per_token, recv_x_test_per_token)
            assert diff < 1e-1, f'diff: {diff}'
    if rank == 0:
        print(f'Test nvfp4 dispatch passed')
    return


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink, explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl)
    test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
              use_logfmt=args.use_logfmt, seed=1, args=args)

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`
    parser = argparse.ArgumentParser(description='Test low-latency EP kernels')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128,
                       help='Number of tokens (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168,
                       help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8,
                       help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=288,
                       help='Number of experts (default: 288)')
    parser.add_argument('--allow-mnnvl', action="store_true",
                        help='Allow MNNVL for communication')
    parser.add_argument('--disable-nvlink', action='store_true',
                        help='Whether to disable NVLink for testing')
    parser.add_argument('--use-logfmt', action='store_true',
                        help='Whether to test LogFMT combine')
    parser.add_argument("--pressure-test", action='store_true',
                        help='Whether to do pressure test')
    parser.add_argument("--no-kineto-profile", action='store_true',
                        help='Whether to do torch profile')
    parser.add_argument("--CUDA_ARCH", type=int, default=90,
                        help='Whether to do torch profile')

    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)

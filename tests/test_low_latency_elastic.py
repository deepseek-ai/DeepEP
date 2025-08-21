import argparse
import random
import time
import os
import torch
import torch.distributed as dist
import numpy as np
from functools import partial
from typing import Optional
from enum import Enum

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back

class Api(Enum):
    dispatch = 0
    combine = 1
    clean = 2
    allgather = 3

failure_cases = {
    # round -> [rank, Api]
    1: [1, Api.dispatch],
    3: [3, Api.combine],
    5: [5, Api.clean],
    7: [7, Api.allgather],
}

allgather_intput_ints = 10

def api_loop(num_tokens: int, hidden: int, num_experts: int, rank: int, num_ranks: int, rank_offset: int,
             x: torch.Tensor, topk_idx: torch.Tensor, all_topk_idx: torch.Tensor, topk_weights: torch.Tensor,
             return_recv_hook: bool, dispatch_use_fp8: bool, round_scale: bool, use_ue8m0: bool, 
             zero_copy: bool, use_logfmt: bool,
             buffer_params: dict,
             test_rounds: int = 10,
             dispatch_check_recv_data: bool = True):

    buffer = deep_ep.Buffer(**buffer_params)

    num_local_experts = num_experts // num_ranks
    hash_value = 0

    mask_status = torch.zeros((num_ranks, ), dtype=torch.int, device='cuda')

    gather_tensor = torch.zeros(num_ranks, allgather_intput_ints, dtype=torch.int32, device='cuda')

    expected_failed_ranks = set()

    def simulate_failure_and_exit(r, api):
        if r in failure_cases.keys() and failure_cases[r][1] == api:
            if rank == failure_cases[r][0]:
                logging.info(f"Rank {rank} failed at round {r} before {api} communication, exit...")
                buffer.destroy()
                return True
            expected_failed_ranks.add(failure_cases[r][0])
        return False

    def check_mask_buffer_query(r, api):
        torch.cuda.synchronize()
        buffer.low_latency_query_mask_buffer(mask_status)
        torch.cuda.synchronize()
        detected_failed_ranks = set(mask_status.nonzero().squeeze(-1).tolist())
        assert expected_failed_ranks == detected_failed_ranks, f"check_mask_buffer_query failed: expected {expected_failed_ranks}, got {detected_failed_ranks}"
        if rank % 8 == 0:
            logging.info(f"Rank {rank} pass {api} test at round {r}")

    def check_mask_buffer_clean():
        torch.cuda.synchronize()
        buffer.low_latency_clean_mask_buffer()
        torch.cuda.synchronize()
        buffer.low_latency_query_mask_buffer(mask_status)
        detected_failed_ranks = set(mask_status.nonzero().squeeze(-1).tolist())
        assert len(detected_failed_ranks) == 0, f"clean_mask_buffer failed: got {detected_failed_ranks}"
        
    for r in range(test_rounds):
        # Dispatch
        if simulate_failure_and_exit(r, Api.dispatch):
            return

        cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
        packed_recv_x, packed_recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts,
                                        use_fp8=dispatch_use_fp8, round_scale=round_scale, use_ue8m0=use_ue8m0,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
        hook() if return_recv_hook else event.current_stream_wait()

        check_mask_buffer_query(r, Api.dispatch)

        packed_recv_x = (packed_recv_x[0], packed_recv_x[1].contiguous()) if dispatch_use_fp8 else packed_recv_x
        simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape) \
            if dispatch_use_fp8 else packed_recv_x.clone()

        ## Check dispatch result correctness
        for i in range(num_local_experts):
            expert_id = rank * num_local_experts + i
            recv_x = per_token_cast_back(packed_recv_x[0][i], packed_recv_x[1][i]) if dispatch_use_fp8 else packed_recv_x[i]
            recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]

            # Check expert indices
            int_mask = (2 ** 32) - 1
            num_valid_tokens = recv_count.item()
            assert cumulative_local_expert_recv_stats[i].item() == num_valid_tokens, f'{cumulative_local_expert_recv_stats[i].item()} != {num_valid_tokens}'
            assert num_valid_tokens == (recv_layout_range & int_mask).sum().item(), f'{num_valid_tokens} != {recv_layout_range & int_mask}.sum().item()'
            assert num_valid_tokens == (all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status==0].sum().item(), \
                f'{rank=} round {r} {num_valid_tokens} != {(all_topk_idx == expert_id).sum(dim=[1, 2])[mask_status==0].sum().item()}'

            if num_valid_tokens == 0:
                continue
            # Check received data
            if dispatch_check_recv_data:
                recv_x = recv_x[:num_valid_tokens]
                recv_x_amin = recv_x[:, :-128].amin(dim=-1)
                recv_src_info = recv_src_info[:num_valid_tokens]
                assert torch.equal(recv_x_amin, recv_x[:, :-128].amax(dim=-1))
                if round_scale:
                    assert calc_diff(recv_x[:, -1], recv_src_info.view(-1)) < 0.007
                else:
                    assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0
                for j in range(num_ranks):
                    if mask_status[j]: continue
                    begin_idx, count = (recv_layout_range[j] >> 32).item(), (recv_layout_range[j] & int_mask).item()
                    if not round_scale:
                        assert (recv_x_amin == j - rank_offset).sum().item() == (all_topk_idx[j] == expert_id).sum().item()
                        assert (recv_x[begin_idx:begin_idx + count, :-128] - j + rank_offset).sum().item() == 0
            if dispatch_use_fp8:
                hash_value ^= hash_tensor(packed_recv_x[0][i, :num_valid_tokens])
                hash_value ^= hash_tensor(packed_recv_x[1][i, :num_valid_tokens])
            else:
                hash_value ^= hash_tensor(packed_recv_x[i, :num_valid_tokens])


        # Combine
        if simulate_failure_and_exit(r, Api.combine):
            return

        if zero_copy:
            buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x
        out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                            use_logfmt=use_logfmt,
                                                            async_finish=not return_recv_hook, zero_copy=zero_copy,
                                                            return_recv_hook=return_recv_hook, out=out)
        hook() if return_recv_hook else event.current_stream_wait()
        check_mask_buffer_query(r, Api.combine)

        ## Check combine result correctness
        owner_by_expert = (torch.arange(num_experts, device='cuda') // num_local_experts)
        fail_owner_mask = (mask_status==1).index_select(0, owner_by_expert)
        valid_topk_idx = topk_idx >= 0
        failed_topk_idx = torch.zeros_like(topk_idx, device='cuda', dtype=torch.bool)
        failed_topk_idx[valid_topk_idx] = fail_owner_mask.index_select(0, topk_idx[valid_topk_idx])
        new_topk_idx = topk_idx.clone()
        new_topk_idx[failed_topk_idx] = -1
        diff = calc_diff(x * topk_weights.masked_fill(new_topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
        assert torch.isnan(combined_x).sum().item() == 0
        assert diff < (9e-4 if dispatch_use_fp8 else 1e-5), f'Error: {diff=}, {dispatch_use_fp8=}, {zero_copy=}'
        hash_value ^= hash_tensor(combined_x)

        # Clean low latency buffer, without correctness checking
        if simulate_failure_and_exit(r, Api.clean):
            return
        buffer.clean_low_latency_buffer(num_tokens, hidden, num_experts)
        check_mask_buffer_query(r, Api.clean)

        # Allgather
        if simulate_failure_and_exit(r, Api.allgather):
            return

        gather_tensor.fill_(0)
        gather_tensor[rank] = rank_offset - rank
        buffer.low_latency_allgather(gather_tensor)
        check_mask_buffer_query(r, Api.allgather)

        ## Check allgather result correctness
        expected = torch.zeros_like(gather_tensor)
        for src_rank in range(num_ranks):
            if src_rank not in expected_failed_ranks:
                expected[src_rank] = rank_offset - src_rank
        assert torch.allclose(gather_tensor, expected), f"Rank {rank} allgather result is not correct"

    check_mask_buffer_clean()
    
    buffer.destroy()

def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer_params: dict,
              use_logfmt: bool = False, seed: int = 0):
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
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

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # Randomly mask some positions
    for _ in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='cuda')
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

    for current_x in x_list:
        for return_recv_hook in (False, True):
            for dispatch_use_fp8 in (False, True):
                for round_scale in (False, True) if dispatch_use_fp8 else (False, ):
                    for use_ue8m0 in (False, True) if round_scale else (False, ):
                        for zero_copy in (False, True):
                            if rank % 8 == 0:
                                logging.info(f"Testing with params {return_recv_hook=} {dispatch_use_fp8=}, {round_scale=}, "
                                      f"{use_ue8m0=}, {zero_copy=}, {failure_cases=} ...")
                            api_loop(
                                    num_tokens, hidden, num_experts, rank, num_ranks, rank_offset,
                                    current_x, topk_idx, all_topk_idx, topk_weights,
                                    return_recv_hook, dispatch_use_fp8, round_scale,
                                    use_ue8m0, zero_copy, use_logfmt, buffer_params,
                                    dispatch_check_recv_data = current_x is x,
                            )
                            if rank % 8 == 0:
                                logging.info(" passed")
                            dist.barrier(group=group)
                            time.sleep(2)

# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        logging.info(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...')
    buffer_params = {
        'group': group, 
        'num_rdma_bytes': num_rdma_bytes, 
        'low_latency_mode': True,
        'num_qps_per_rank': num_experts // num_ranks,
        'allow_nvlink_for_low_latency_mode': not args.disable_nvlink, 
        'explicitly_destroy': True,
        'allow_mnnvl': args.allow_mnnvl, 
        'enable_elastic': True,
        'num_coll_buffer_bytes': num_ranks * allgather_intput_ints * 4
    }
    test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer_params,
              use_logfmt=args.use_logfmt, seed=1)

    # Destroy the buffer runtime and communication group
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
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
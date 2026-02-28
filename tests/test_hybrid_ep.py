# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
import paddle
paddle.enable_compat()

import argparse
import time
import torch
import torch.distributed as dist
import os
import deep_ep
import logging

from utils import TorchRef, bench, bench_kineto, init_dist, count_rdma_send_from_routing_map

import contextlib
from paddle.distributed import fleet
from paddle.distributed.communication.group import Group

HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
MAX_NUM_OF_TOKENS_PER_RANK = int(os.environ.get("MAX_NUM_OF_TOKENS_PER_RANK", 4096))
# NUM_TOKENS_PER_RANK should equal or less than MAX_NUM_OF_TOKENS_PER_RANK
NUM_TOKENS_PER_RANK = int(os.environ.get("NUM_TOKENS_PER_RANK", 4096))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 32))
NUM_OF_RANKS_PER_NODE = int(os.environ.get("NUM_OF_RANKS_PER_NODE", 8))
NUM_OF_NODES = int(os.environ.get("NUM_OF_NODES", 1))
TOPK = int(os.environ.get("TOPK", 8))
PAD_MULTIPLE = int(os.environ.get("PAD_MULTIPLE", 128))
NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES
ITERATIONS = int(os.environ.get("ITERATIONS", 100))
SEED = int(os.environ.get("SEED", 42))
USE_MNNVL = os.environ.get("USE_MNNVL", "0").strip().lower() in {"1", "true", "t", "yes", "y", "on"}
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# Will be set after the process group is initialized
NUM_OF_RANKS_PER_NODE = None
NUM_OF_NODES = None
NUM_OF_EXPERTS = None

def print_in_order(msg: str):
    """Print message in order by rank to avoid interleaved output"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for i in range(world_size):
        if i == rank:
            print(msg, flush=True)
        dist.barrier()

def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape or a.device != b.device:
        return False
    a_bytes = a.contiguous().view(torch.uint8)
    b_bytes = b.contiguous().view(torch.uint8)
    return torch.equal(a_bytes, b_bytes)

def init_tensor(
    hidden_dim: int,
    seq_len: int,
    topk: int,
    num_of_experts: int,
    use_fp8: bool = False,
):
    if use_fp8:
        hidden = torch.randint(
            low=0,
            high=256,
            size=(seq_len, hidden_dim),
            dtype=torch.int32,
        ).cuda().cast(torch.uint8)
    else:
        hidden = torch.randn(seq_len, hidden_dim, dtype=torch.bfloat16).cuda()
    probs = torch.zeros(seq_len, num_of_experts, dtype=torch.float32).cuda()
    topk_idx = torch.zeros(seq_len, topk, dtype=torch.int64).cuda()
    topk_weights = torch.zeros(seq_len, topk, dtype=torch.float32).cuda()
    scaling_factor = torch.randn(
        seq_len, hidden_dim // 128, dtype=torch.float32
    ).cuda()

    routing_map = torch.zeros(seq_len, num_of_experts, dtype=torch.bool).cuda()

    for i in range(seq_len):
        # Force balanced routing for testing
        # selected_experts = torch.tensor([
        #     ((i * topk) % num_of_experts + val) % num_of_experts for val in range(topk)
        # ])
        selected_experts = torch.randperm(num_of_experts).cuda()[:topk]
        topk_idx[i, :] = selected_experts.to(torch.int64)
        topk_weights[i, :] = torch.ones(topk, dtype=torch.float32).cuda()
        routing_map[i, selected_experts] = True
        probs[i, selected_experts] = topk_weights[i, :]

    return hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights


def test_hybrid_ep_correctness(buffer: deep_ep.HybridEPBuffer, ref: TorchRef, use_fp8: bool):
    hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights  = init_tensor(
        hidden_dim=HIDDEN_DIM,
        seq_len=NUM_TOKENS_PER_RANK,
        topk=TOPK,
        num_of_experts=NUM_OF_EXPERTS,
        use_fp8=use_fp8,
    )
    print(f"hidden: {hidden}, shape: {hidden.shape}")
    print(f"probs: {probs}, shape: {probs.shape}")
    print(f"scaling_factor: {scaling_factor}, shape: {scaling_factor.shape}")
    print(f"routing_map: {routing_map}, shape: {routing_map.shape}")
    print(f"topk_idx: {topk_idx}, shape: {topk_idx.shape}")
    print(f"topk_weights: {topk_weights}, shape: {topk_weights.shape}")

    # Dispatch correctness check
    for with_probs in [True]:
        # The check for the dispatch
        dispatched_hidden_ref, dispatched_probs_ref, dispatched_scaling_factor_ref = (
            ref.dispatch(
                hidden, routing_map, probs if with_probs else None, scaling_factor
            )
        )
        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
            handle,
        ) = buffer.dispatch(
            hidden=hidden, 
            scaling_factor=scaling_factor, 
            topk_idx=topk_idx, 
            topk_weights=topk_weights if with_probs else None, 
            num_of_experts=NUM_OF_EXPERTS,
        )
        print(f"dispatched_hidden: {dispatched_hidden}, shape: {dispatched_hidden.shape}")
        print(f"dispatched_probs: {dispatched_probs}, shape: {dispatched_probs.shape}")
        print(f"dispatched_scaling_factor: {dispatched_scaling_factor}, shape: {dispatched_scaling_factor.shape}")

        assert bitwise_equal(dispatched_hidden_ref, dispatched_hidden)
        if dispatched_probs is not None and dispatched_probs_ref is not None:
            start, end = ref._local_expert_range_per_node()
            masked_probs = torch.zeros_like(dispatched_probs)
            masked_probs[:, start:end] = dispatched_probs[:, start:end]
            assert bitwise_equal(dispatched_probs_ref, dispatched_probs[:, start:end])
            dispatched_probs = masked_probs
        if (
            dispatched_scaling_factor is not None
            and dispatched_scaling_factor_ref is not None
        ):
            assert bitwise_equal(
                dispatched_scaling_factor_ref, dispatched_scaling_factor
            )

        _, _, _, num_dispatched_tokens, local_expert_routing_map, _, _ = handle

        print(f"num_dispatched_tokens: {num_dispatched_tokens}, shape: {num_dispatched_tokens.shape}")
        print(f"local_expert_routing_map: {local_expert_routing_map}, shape: {local_expert_routing_map.shape}")
        
        num_dispatched_tokens = num_dispatched_tokens.cpu()
        local_expert_routing_map = local_expert_routing_map[
            : num_dispatched_tokens.item()
        ]
        # Simulate the permute and expert and unpermute. The expert is identity op
        copy_times = local_expert_routing_map.sum(dim=1)
        dispatched_hidden = dispatched_hidden.to(torch.bfloat16)  
        print(f"copy_times: {copy_times}, shape: {copy_times.shape}")
        print(f"dispatched_hidden: {dispatched_hidden}, shape: {dispatched_hidden.shape}")
        # The combine only support bf16
        hidden_to_combine = dispatched_hidden * copy_times.unsqueeze(1)
        probs_to_combine = dispatched_probs
        print(f"hidden_to_combine: {hidden_to_combine}, shape: {hidden_to_combine.shape}")
        print(f"dispatched_hidden: {probs_to_combine}, shape: {probs_to_combine.shape}")

        # The check for the combine
        combined_hidden, combined_probs = buffer.combine(
            hidden_to_combine, probs_to_combine, handle
        )
        print(f"combined_hidden: {combined_hidden}, shape: {combined_hidden.shape}")
        print(f"combined_probs: {combined_probs}, shape: {combined_probs.shape}")

        # The reconstucted value should be TOPK times larger than the input hidden
        combined_hidden = combined_hidden / TOPK
        print(f"combined_hidden new: {combined_hidden}, shape new: {combined_hidden.shape}")

        assert torch.allclose(combined_hidden, hidden.to(torch.bfloat16), atol=2e-5, rtol=1e-2)
        if combined_probs is not None and probs is not None:
            assert bitwise_equal(combined_probs, probs)

    # # Dispatch with permute correctness check
    # for with_probs in [True]:
    #     # The check for the dispatch
    #     (
    #         dispatched_hidden,
    #         dispatched_probs,
    #         dispatched_scaling_factor,
    #         tokens_per_expert,
    #         handle,
    #     ) = buffer.dispatch_with_permute(
    #         hidden=hidden,
    #         routing_map=routing_map,
    #         probs=probs if with_probs else None,
    #         scaling_factor=scaling_factor,
    #         pad_multiple=PAD_MULTIPLE,
    #     )
    #     if dist.get_rank() == 0:
    #         print("dispatched_hidden: ", dispatched_hidden, dispatched_hidden.shape)
    #         print("dispatched_probs: ", dispatched_probs, dispatched_probs.shape)
    #         print("dispatched_scaling_factor: ", dispatched_scaling_factor, dispatched_scaling_factor.shape)
    #         print("tokens_per_expert: ", tokens_per_expert, tokens_per_expert.shape)
    #     _, _, _, num_dispatched_tokens_tensor, local_expert_routing_map, _, _, _ = (
    #         handle
    #     )
    #     if dist.get_rank() == 0:
    #         print("num_dispatched_tokens_tensor: ", num_dispatched_tokens_tensor, num_dispatched_tokens_tensor.shape)
    #         print("local_expert_routing_map: ", local_expert_routing_map, local_expert_routing_map.shape)
    #     num_dispatched_tokens_tensor = num_dispatched_tokens_tensor.cpu()
    #     local_expert_routing_map = local_expert_routing_map[
    #         : num_dispatched_tokens_tensor.item()
    #     ]
    #     # The out_token_num of permutation is the sum of the tokens_per_expert
    #     out_token_num = tokens_per_expert.sum().item()
    #     (
    #         dispatched_hidden_ref,
    #         dispatched_probs_ref,
    #         dispatched_scaling_factor_ref,
    #     ) = ref.dispatch(
    #         hidden,
    #         routing_map,
    #         probs if with_probs else None,
    #         scaling_factor,
    #         local_expert_routing_map=local_expert_routing_map,
    #         out_token_num=out_token_num,
    #         pad_multiple=PAD_MULTIPLE,
    #         enable_permute=True,
    #     )

    #     assert bitwise_equal(dispatched_hidden_ref, dispatched_hidden)
    #     if dispatched_probs is not None and dispatched_probs_ref is not None:
    #         assert bitwise_equal(dispatched_probs_ref, dispatched_probs)
    #     if (
    #         dispatched_scaling_factor is not None
    #         and dispatched_scaling_factor_ref is not None
    #     ):
    #         assert bitwise_equal(
    #             dispatched_scaling_factor_ref, dispatched_scaling_factor
    #         )

    #     # The combine only support bf16
    #     dispatched_hidden = dispatched_hidden.to(torch.bfloat16)  
    #     hidden_to_combine = dispatched_hidden
    #     probs_to_combine = dispatched_probs
 
    #     # The check for the combine
    #     combined_hidden, combined_probs = buffer.combine_with_unpermute(
    #         hidden=hidden_to_combine,
    #         probs=probs_to_combine,
    #         handle=handle,
    #         pad_multiple=PAD_MULTIPLE,
    #     )

    #     # The reconstucted value should be TOPK times larger than the input hidden
    #     combined_hidden = combined_hidden / TOPK

    #     assert torch.allclose(
    #         combined_hidden, hidden.to(torch.bfloat16), atol=2e-5, rtol=1e-2
    #     )
    #     if combined_probs is not None and probs is not None:
    #         assert bitwise_equal(combined_probs, probs)

    # print_in_order(f'[rank {dist.get_rank()}] Correctness check passed ({"FP8" if hidden.dtype == torch.uint8 else "BF16"})')


def test_hybrid_ep_benchmark(buffer: deep_ep.HybridEPBuffer, group: Group, use_fp8: bool, nsys_profile: bool):
    hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights = init_tensor(
        hidden_dim=HIDDEN_DIM,
        seq_len=NUM_TOKENS_PER_RANK,
        topk=TOPK,
        num_of_experts=NUM_OF_EXPERTS,
        use_fp8=use_fp8,
    )

    # warmup
    for _ in range(10):
        dispatched_hidden, dispatched_probs, _, handle = (
            buffer.dispatch(hidden=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights, num_of_experts=NUM_OF_EXPERTS)
        )
        # The combine only support bf16
        dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)
        dispatched_probs = None
        _, _ = buffer.combine(dispatched_hidden_bf16, dispatched_probs, handle)

    rank = dist.get_rank()
    fp8_factor = (1 + 4 / 128) / 2
    dispatch_bf16_nvl_recv_bytes = dispatched_hidden.numel() * 2
    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes
    if NUM_OF_NODES > 1:
        local_node_id = rank // NUM_OF_RANKS_PER_NODE
        num_rdma_send = count_rdma_send_from_routing_map(routing_map, local_node_id, NUM_OF_NODES)
        dispatch_bf16_rdma_send_bytes = num_rdma_send * HIDDEN_DIM * 2
        combine_bf16_rdma_recv_bytes = dispatch_bf16_rdma_send_bytes

    '''
    Benchmark of the dispatch and combine torch API without permute
    '''

    dispatched_hidden, dispatched_probs, _, handle= (
        buffer.dispatch(hidden=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights, num_of_experts=NUM_OF_EXPERTS)
    )
    dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)

    dispatch_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'topk_idx': topk_idx, 'topk_weights': topk_weights, 'num_of_experts': NUM_OF_EXPERTS, 'handle': handle}
    t = bench(lambda: buffer.dispatch(**dispatch_args))[0]
    nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if hidden.dtype == torch.uint8 else dispatch_bf16_nvl_recv_bytes
    if NUM_OF_NODES > 1:
        rdma_send_bytes = dispatch_bf16_rdma_send_bytes * fp8_factor if hidden.dtype == torch.uint8 else dispatch_bf16_rdma_send_bytes
    print_in_order(f'[rank {rank}] HybridEP dispatch torch API ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): '
            f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, nvl_recv_bytes: {nvl_recv_bytes / 1e6:.2f} MB')
    if NUM_OF_NODES > 1:
        print_in_order(f'[rank {rank}] HybridEP dispatch torch API ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): '
                f'{rdma_send_bytes / 1e9 / t:.2f} GB/s (IB), t: {t * 1e6:.2f} us, rdma_send_bytes: {rdma_send_bytes / 1e6:.2f} MB')

    combine_args = {'hidden': dispatched_hidden_bf16, 'probs': dispatched_probs, 'handle': handle}
    t = bench(lambda: buffer.combine(**combine_args))[0]
    print_in_order(f'[rank {rank}] HybridEP combine torch API: '
            f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, combine_send_bytes: {combine_bf16_nvl_send_bytes / 1e6:.2f} MB')
    if NUM_OF_NODES > 1:
        print_in_order(f'[rank {rank}] HybridEP combine torch API: '
                    f'{combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (IB), t: {t * 1e6:.2f} us, rdma_recv_bytes: {combine_bf16_rdma_recv_bytes / 1e6:.2f} MB')

    '''
    Benchmark of the dispatch and combine with permute extension
    '''
    dispatched_hidden_with_permute, dispatched_probs_with_permute, _, tokens_per_expert, handle_with_permute= (
        buffer.dispatch_with_permute(hidden=hidden, scaling_factor=scaling_factor, routing_map=routing_map, probs=probs, pad_multiple=PAD_MULTIPLE)
    )
    num_permuted_tokens = tokens_per_expert.sum().item()
    dispatched_hidden_bf16_with_permute = dispatched_hidden_with_permute.to(torch.bfloat16)

    dispatch_with_permute_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'routing_map': routing_map, 'probs': probs, 'pad_multiple': PAD_MULTIPLE, 'handle': handle_with_permute, 'num_permuted_tokens': num_permuted_tokens}
    t = bench(lambda: buffer.dispatch_with_permute(**dispatch_with_permute_args))[0]
    nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if hidden.dtype == torch.uint8 else dispatch_bf16_nvl_recv_bytes
    print_in_order(f'[rank {rank}] HybridEP dispatch+permute torch API ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): '
            f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, nvl_recv_bytes: {nvl_recv_bytes / 1e6:.2f} MB')
    if NUM_OF_NODES > 1:
        print_in_order(f'[rank {rank}] HybridEP dispatch+permute torch API ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): '
                f'{rdma_send_bytes / 1e9 / t:.2f} GB/s (IB), t: {t * 1e6:.2f} us, rdma_send_bytes: {rdma_send_bytes / 1e6:.2f} MB')

    combine_with_unpermute_args = {'hidden': dispatched_hidden_bf16_with_permute, 'probs': dispatched_probs_with_permute, 'handle': handle_with_permute, 'pad_multiple': PAD_MULTIPLE}
    t = bench(lambda: buffer.combine_with_unpermute(**combine_with_unpermute_args))[0]
    print_in_order(f'[rank {rank}] HybridEP combine+unpermute torch API: '
            f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, combine_send_bytes: {combine_bf16_nvl_send_bytes / 1e6:.2f} MB')
    if NUM_OF_NODES > 1:
        print_in_order(f'[rank {rank}] HybridEP combine+unpermute torch API: '
                f'{combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (IB), t: {t * 1e6:.2f} us, rdma_recv_bytes: {combine_bf16_rdma_recv_bytes / 1e6:.2f} MB')

    if not nsys_profile:
        # noinspection PyShadowingNames
        def test_func():
            dispatched_hidden, dispatched_probs, _, handle = (
                buffer.dispatch(hidden=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights, num_of_experts=NUM_OF_EXPERTS)
            )
            # The combine only support bf16
            dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)
            dispatched_probs = None
            _, _ = buffer.combine(dispatched_hidden_bf16, dispatched_probs, handle)

        group.barrier()
        dispatch_t, combine_t = bench_kineto(test_func,
                                             kernel_names=('dispatch_kernel', 'combine_kernel'), barrier_comm_profiling=True,
                                             suppress_kineto_output=True)
        print_in_order(f'[rank {rank}] HybridEP dispatch kernel(NVL) ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): {nvl_recv_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
              f'HybridEP combine kernel(NVL): {combine_bf16_nvl_send_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us')
        if NUM_OF_NODES > 1:
            print_in_order(f'[rank {rank}] HybridEP dispatch kernel(IB) ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): {rdma_send_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
                  f'HybridEP combine kernel(IB): {combine_bf16_rdma_recv_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us')
    else:
        torch.cuda.profiler.start()
        with torch.cuda.nvtx.range(f"hybrid-ep dispatch ({'FP8' if hidden.dtype == torch.uint8 else 'BF16'})"):
            if rank == 0:
                print(f"profile hybrid-ep dispatch ({'FP8' if hidden.dtype == torch.uint8 else 'BF16'})", flush=True)
            dispatch_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'topk_idx': topk_idx, 'topk_weights': topk_weights, 'num_of_experts': NUM_OF_EXPERTS}
            bench(lambda: buffer.dispatch(**dispatch_args))
        with torch.cuda.nvtx.range("hybrid-ep combine"):
            if rank == 0:
                print(f"profile hybrid-ep combine", flush=True)
            combine_args = {'hidden': dispatched_hidden_bf16, 'probs': dispatched_probs, 'handle': handle}
            bench(lambda: buffer.combine(**combine_args))
        with torch.cuda.nvtx.range(f"hybrid-ep dispatch+permute ({'FP8' if hidden.dtype == torch.uint8 else 'BF16'})"):
            if rank == 0:
                print(f"profile hybrid-ep dispatch+permute ({'FP8' if hidden.dtype == torch.uint8 else 'BF16'})", flush=True)
            dispatch_with_permute_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'routing_map': routing_map, 'probs': probs, 'pad_multiple': PAD_MULTIPLE}
            bench(lambda: buffer.dispatch_with_permute(**dispatch_with_permute_args))
        with torch.cuda.nvtx.range("hybrid-ep combine+unpermute"):
            if rank == 0:
                print(f"profile hybrid-ep combine+unpermute", flush=True)
            combine_with_unpermute_args = {'hidden': dispatched_hidden_bf16_with_permute, 'probs': dispatched_probs_with_permute, 'handle': handle_with_permute, 'pad_multiple': PAD_MULTIPLE}
            bench(lambda: buffer.combine_with_unpermute(**combine_with_unpermute_args))
        time.sleep(1)
        torch.cuda.profiler.stop()


def test_main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    _, _, group = init_dist(local_rank, num_local_ranks)
    print("group: ", group.id)

    # Set missing global vars
    global NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_EXPERTS
    if USE_MNNVL:
        NUM_OF_RANKS_PER_NODE = group.world_size
        NUM_OF_NODES = 1
        NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES
    else:
        NUM_OF_RANKS_PER_NODE = args.num_processes
        NUM_OF_NODES = group.world_size // NUM_OF_RANKS_PER_NODE
        NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES
        
    for use_fp8 in [True]:
        print("use_fp8: ", use_fp8)
        buffer = deep_ep.HybridEPBuffer(
            group=group,
            hidden_dim=HIDDEN_DIM,
            max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
            num_local_experts=NUM_LOCAL_EXPERTS,
            num_of_hybrid_ep_ranks_per_nvlink_domain=NUM_OF_RANKS_PER_NODE,
            use_mnnvl=USE_MNNVL,
            use_fp8=use_fp8
        )
        print("buffer: ", buffer)
        
        ref = TorchRef(
            ep_group=group,
            num_of_experts=NUM_OF_EXPERTS,
            num_of_ranks_per_node=NUM_OF_RANKS_PER_NODE,
        )

        test_hybrid_ep_correctness(buffer, ref, use_fp8)
        # test_hybrid_ep_benchmark(buffer, group, use_fp8, args.nsys_profile)
    dist.barrier()
    dist.destroy_process_group()

def init_dist_env(world_size, seed=20):
    context = contextlib.nullcontext()
    with context:
        # start to init distributed env
        strategy = fleet.DistributedStrategy()

        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": world_size,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": seed}

        fleet.init(is_collective=True, strategy=strategy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 4)')
    parser.add_argument('--nsys-profile', action='store_true', default=False,
                       help='benchmark with nsys profile or not (default: False)')
    args = parser.parse_args()

    if dist.get_world_size() > 1:
        init_dist_env(dist.get_world_size())

    rank = dist.get_rank()
    num_ranks = dist.get_world_size()

    test_main(rank, num_ranks, args)

    # torch.multiprocessing.spawn(test_main, args=(args.num_processes, args), nprocs=args.num_processes)

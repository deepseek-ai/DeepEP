# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
import argparse
import time
import torch
import torch.distributed as dist
import os
import deep_ep

from utils import TorchRef, bench, bench_kineto, init_dist, count_rdma_send_from_routing_map

HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
MAX_NUM_OF_TOKENS_PER_RANK = int(os.environ.get("MAX_NUM_OF_TOKENS_PER_RANK", 4096))
# NUM_TOKENS_PER_RANK should equal or less than MAX_NUM_OF_TOKENS_PER_RANK
NUM_TOKENS_PER_RANK = int(os.environ.get("NUM_TOKENS_PER_RANK", 4096))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 8))
NUM_OF_RANKS_PER_NODE = int(os.environ.get("NUM_OF_RANKS_PER_NODE", 4))
NUM_OF_NODES = int(os.environ.get("NUM_OF_NODES", 1))
TOPK = int(os.environ.get("TOPK", 8))
PAD_MULTIPLE = int(os.environ.get("PAD_MULTIPLE", 32))
NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES
ITERATIONS = int(os.environ.get("ITERATIONS", 100))
SEED = int(os.environ.get("SEED", 42))
USE_MNNVL = os.environ.get("USE_MNNVL", "0").strip().lower() in {"1", "true", "t", "yes", "y", "on"}
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
            device="cuda",
            dtype=torch.uint8,
        )
    else:
        hidden = torch.randn(seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
    probs = torch.zeros(seq_len, num_of_experts, device="cuda", dtype=torch.float32)
    scaling_factor = torch.randn(
        seq_len, hidden_dim // 128, device="cuda", dtype=torch.float32
    )

    routing_map = torch.zeros(seq_len, num_of_experts, device="cuda", dtype=torch.bool)
    for i in range(seq_len):
        # Force balanced routing for testing
        selected_experts = torch.tensor([
            ((i * topk) % num_of_experts + val) % num_of_experts for val in range(topk)
        ], device="cuda")
        routing_map[i, selected_experts] = True

    return hidden, probs, scaling_factor, routing_map


def test_hybrid_ep_correctness(buffer: deep_ep.HybridEPBuffer, ref: TorchRef, use_fp8: bool, with_probs: bool):
    # Construct the input
    hidden, probs, scaling_factor, routing_map = init_tensor(
        hidden_dim=HIDDEN_DIM,
        seq_len=NUM_TOKENS_PER_RANK,
        topk=TOPK,
        num_of_experts=NUM_OF_EXPERTS,
        use_fp8=use_fp8,
    )
    num_permuted_tokens = NUM_TOKENS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES * TOPK
    graph = torch.cuda.CUDAGraph()

    # Warm up, and get the preprocessing results
    (
        dispatched_hidden,
        dispatched_probs,
        dispatched_scaling_factor,
        tokens_per_expert,
        handle,
    ) = buffer.dispatch_with_permute(
        hidden=hidden,
        routing_map=routing_map,
        probs=probs if with_probs else None,
        scaling_factor=scaling_factor,
        pad_multiple=PAD_MULTIPLE,
        num_permuted_tokens=num_permuted_tokens,
        non_blocking=True,
    )
    _, _, _, num_dispatched_tokens_tensor, local_expert_routing_map, _, _, _, _ = (
        handle
    )
    num_dispatched_tokens_tensor = num_dispatched_tokens_tensor.cpu()
    local_expert_routing_map = local_expert_routing_map[
        : num_dispatched_tokens_tensor.item()
    ]
    num_permuted_tokens_runtime = tokens_per_expert.sum().item()
    _, _ = buffer.combine_with_unpermute(
        hidden=dispatched_hidden.to(torch.bfloat16),
        probs=dispatched_probs,
        handle=handle,
        pad_multiple=PAD_MULTIPLE
    )

    # Get the reference
    dispatched_hidden_ref, dispatched_probs_ref, dispatched_scaling_factor_ref = ref.dispatch(
        hidden,
        routing_map,
        probs if with_probs else None,
        scaling_factor,
        local_expert_routing_map=local_expert_routing_map,
        out_token_num=tokens_per_expert.sum().item(),
        pad_multiple=PAD_MULTIPLE,
        enable_permute=True,
    )

    graph_dispatched_hidden = None
    graph_dispatched_probs = None
    graph_dispatched_scaling_factor = None
    graph_tokens_per_expert = None
    graph_handle = None
    graph_combined_hidden = None
    graph_combined_probs = None

    with torch.cuda.graph(graph):
        (
            graph_dispatched_hidden,
            graph_dispatched_probs,
            graph_dispatched_scaling_factor,
            graph_tokens_per_expert,
            graph_handle,
        ) = buffer.dispatch_with_permute(
            hidden=hidden,
            routing_map=routing_map,
            probs=probs if with_probs else None,
            scaling_factor=scaling_factor,
            pad_multiple=PAD_MULTIPLE,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=True,
        )
        dispatched_hidden_bf16 = graph_dispatched_hidden.to(torch.bfloat16)
        (
            graph_combined_hidden,
            graph_combined_probs,
        ) = buffer.combine_with_unpermute(
            hidden=dispatched_hidden_bf16,
            probs=graph_dispatched_probs,
            handle=graph_handle,
            pad_multiple=PAD_MULTIPLE
        )
    graph.replay()
    torch.cuda.synchronize()

    # Check the correctness
    assert bitwise_equal(dispatched_hidden_ref, graph_dispatched_hidden[:num_permuted_tokens_runtime,:])
    if with_probs:
        assert bitwise_equal(dispatched_probs_ref, graph_dispatched_probs[:num_permuted_tokens_runtime])
    if use_fp8:
        assert bitwise_equal(
            dispatched_scaling_factor_ref, graph_dispatched_scaling_factor[:num_permuted_tokens_runtime,:]
        )
    reconstructed_hidden = graph_combined_hidden / TOPK
    assert torch.allclose(
        reconstructed_hidden, hidden.to(torch.bfloat16), atol=2e-5, rtol=1e-2
    )
    if with_probs:
        assert bitwise_equal(graph_combined_probs, probs)

    print_in_order(f'[rank {dist.get_rank()}] Correctness check passed ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}, with_probs={with_probs})')

    # Benchmark the graphed hybrid ep in nsys profile
    torch.cuda.profiler.start()
    for _ in range(ITERATIONS):
        graph.replay()
    torch.cuda.profiler.stop()


def test_main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    _, _, group = init_dist(local_rank, num_local_ranks)
    for use_fp8 in [False, True]:
        for with_probs in [True, False]:
            buffer = deep_ep.HybridEPBuffer(
                group=group,
                hidden_dim=HIDDEN_DIM,
                max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
                num_local_experts=NUM_LOCAL_EXPERTS,
                num_of_hybrid_ep_ranks_per_nvlink_domain=NUM_OF_RANKS_PER_NODE,
                use_mnnvl=USE_MNNVL,
                use_fp8=use_fp8
            )
            ref = TorchRef(
                ep_group=group,
                num_of_experts=NUM_OF_EXPERTS,
                num_of_ranks_per_node=NUM_OF_RANKS_PER_NODE,
            )

            test_hybrid_ep_correctness(buffer, ref, use_fp8, with_probs)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')
    parser.add_argument('--num-processes', type=int, default=4,
                       help='Number of processes to spawn (default: 4)')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_main, args=(args.num_processes, args), nprocs=args.num_processes)

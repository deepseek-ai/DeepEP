# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
import argparse
from operator import truediv
import time
import torch
import torch.distributed as dist
import os
import deep_ep
from torch.profiler import ExecutionTraceObserver

from utils import TorchRef, bench, bench_kineto, init_dist, count_rdma_send_from_routing_map

HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
MAX_NUM_OF_TOKENS_PER_RANK = int(os.environ.get("MAX_NUM_OF_TOKENS_PER_RANK", 4096))
# NUM_TOKENS_PER_RANK should equal or less than MAX_NUM_OF_TOKENS_PER_RANK
NUM_TOKENS_PER_RANK = int(os.environ.get("NUM_TOKENS_PER_RANK", 4096))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 8))
TOPK = int(os.environ.get("TOPK", 8))
PAD_MULTIPLE = int(os.environ.get("PAD_MULTIPLE", 32))
ITERATIONS = int(os.environ.get("ITERATIONS", 100))
SEED = int(os.environ.get("SEED", 42))
USE_MNNVL = os.environ.get("USE_MNNVL", "0").strip().lower() in {"1", "true", "t", "yes", "y", "on"}
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Will be set after the process group is initialized
NUM_OF_RANKS_PER_NODE = None
NUM_OF_NODES = None
NUM_OF_EXPERTS = None

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
    topk_idx = torch.zeros(seq_len, topk, device="cuda", dtype=torch.int64)
    topk_weights = torch.zeros(seq_len, topk, device="cuda", dtype=torch.float32)
    scaling_factor = torch.randn(
        seq_len, hidden_dim // 128, device="cuda", dtype=torch.float32
    )

    routing_map = torch.zeros(seq_len, num_of_experts, device="cuda", dtype=torch.bool)

    for i in range(seq_len):
        # Force balanced routing for testing
        # selected_experts = torch.tensor([
        #     ((i * topk) % num_of_experts + val) % num_of_experts for val in range(topk)
        # ], device="cuda")
        selected_experts = torch.randperm(num_of_experts, device="cuda")[:topk]
        topk_idx[i, :] = selected_experts.to(torch.int64)
        topk_weights[i, :] = torch.ones(topk, device="cuda", dtype=torch.float32)
        routing_map[i, selected_experts] = True
        probs[i, selected_experts] = topk_weights[i, :]

    return hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights


def test_main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    _, _, group = init_dist(local_rank, num_local_ranks)

    # Set missing global vars
    global NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_EXPERTS
    if USE_MNNVL:
        NUM_OF_RANKS_PER_NODE = group.size()
        NUM_OF_NODES = 1
        NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES
    else:
        NUM_OF_RANKS_PER_NODE = args.num_processes
        NUM_OF_NODES = group.size() // NUM_OF_RANKS_PER_NODE
        NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        et = ExecutionTraceObserver().register_callback(f"rank-{dist.get_rank()}.json")
        et.start()

        buffer = deep_ep.HybridEPBuffer(
            group=group,
            hidden_dim=HIDDEN_DIM,
            max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
            num_local_experts=NUM_LOCAL_EXPERTS,
            use_fp8=True
        )

        hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights = init_tensor(
            hidden_dim=HIDDEN_DIM,
            seq_len=NUM_TOKENS_PER_RANK,
            topk=TOPK,
            num_of_experts=NUM_OF_EXPERTS,
            use_fp8=truediv,
        )

        dispatched_hidden, dispatched_probs, _, handle = (
            buffer.dispatch(hidden=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights, num_of_experts=NUM_OF_EXPERTS)
        )
        # The combine only support bf16
        dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)
        dispatched_probs = None
        _, _ = buffer.combine(dispatched_hidden_bf16, dispatched_probs, handle)

        et.stop()
        et.unregister_callback()
    
    time.sleep(10)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')
    parser.add_argument('--num-processes', type=int, default=4,
                       help='Number of processes to spawn (default: 4)')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_main, args=(args.num_processes, args), nprocs=args.num_processes)

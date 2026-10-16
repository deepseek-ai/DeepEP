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
TOPK = int(os.environ.get("TOPK", 8))
PAD_MULTIPLE = int(os.environ.get("PAD_MULTIPLE", 32))
ITERATIONS = int(os.environ.get("ITERATIONS", 100))
SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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


def test_hybrid_ep_correctness(buffer: deep_ep.HybridEPBuffer, ref: TorchRef, use_fp8: bool, with_probs: bool, fused_permute_dispatch: bool):
    # Construct the input
    hidden, probs, scaling_factor, routing_map = init_tensor(
        hidden_dim=HIDDEN_DIM,
        seq_len=NUM_TOKENS_PER_RANK,
        topk=TOPK,
        num_of_experts=NUM_OF_EXPERTS,
        use_fp8=use_fp8,
    )
    graph = torch.cuda.CUDAGraph()
    # Pre-allocate buffer size; non_blocking mode requires this upfront
    num_permuted_tokens = NUM_TOKENS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES * TOPK

    # Warm up to get runtime token count
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
        fuse_permute_dispatch=fused_permute_dispatch,
    )
    num_permuted_tokens_runtime = tokens_per_expert.sum().item()
    buffer.combine_with_unpermute(
        hidden=dispatched_hidden.to(torch.bfloat16),
        probs=dispatched_probs,
        handle=handle,
        pad_multiple=PAD_MULTIPLE,
        fuse_unpermute_combine=fused_permute_dispatch,
    )

    # Get the reference (no token_per_expert needed)
    dispatched_hidden_ref, dispatched_probs_ref, dispatched_scaling_factor_ref = ref.dispatch(
        hidden,
        routing_map,
        probs if with_probs else None,
        scaling_factor,
        pad_multiple=PAD_MULTIPLE,
        enable_permute=True,
    )

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
            fuse_permute_dispatch=fused_permute_dispatch,
        )
        dispatched_hidden_bf16 = graph_dispatched_hidden.to(torch.bfloat16)
        (
            graph_combined_hidden,
            graph_combined_probs,
        ) = buffer.combine_with_unpermute(
            hidden=dispatched_hidden_bf16,
            probs=graph_dispatched_probs,
            handle=graph_handle,
            pad_multiple=PAD_MULTIPLE,
            fuse_unpermute_combine=fused_permute_dispatch,
        )
    graph.replay()
    torch.cuda.synchronize()

    # Check correctness
    assert bitwise_equal(dispatched_hidden_ref, graph_dispatched_hidden[:num_permuted_tokens_runtime, :]), \
        f"Dispatch hidden mismatch (with_probs={with_probs}, fused={fused_permute_dispatch})"
    if with_probs:
        assert bitwise_equal(dispatched_probs_ref, graph_dispatched_probs[:num_permuted_tokens_runtime]), \
            f"Dispatch probs mismatch (with_probs={with_probs}, fused={fused_permute_dispatch})"
    if use_fp8:
        assert bitwise_equal(
            dispatched_scaling_factor_ref, graph_dispatched_scaling_factor[:num_permuted_tokens_runtime, :]
        ), f"Dispatch scaling_factor mismatch (with_probs={with_probs}, fused={fused_permute_dispatch})"
    reconstructed_hidden = graph_combined_hidden / TOPK
    assert torch.allclose(
        reconstructed_hidden, hidden.to(torch.bfloat16), atol=2e-5, rtol=1e-2
    ), f"Combine hidden mismatch (with_probs={with_probs}, fused={fused_permute_dispatch})"
    if with_probs:
        assert bitwise_equal(graph_combined_probs, probs), \
            f"Combine probs mismatch (with_probs={with_probs}, fused={fused_permute_dispatch})"

    dtype_str = "FP8" if hidden.dtype == torch.uint8 else "BF16"
    fused_str = "fused" if fused_permute_dispatch else "non-fused"
    print_in_order(f'[rank {dist.get_rank()}] Correctness check passed ({dtype_str}, with_probs={with_probs}, {fused_str})')

    # Benchmark the graphed hybrid ep in nsys profile
    torch.cuda.profiler.start()
    for _ in range(ITERATIONS):
        graph.replay()
    torch.cuda.profiler.stop()


def test_plain_dispatch_capture(buffer: deep_ep.HybridEPBuffer):
    """Cover capturing dispatch() itself.

    The cases above always hand dispatch_with_permute an explicit
    num_permuted_tokens, so they never exercise the no-permute path whose
    output shape has to come from the caller. Without a count that shape is
    unknowable at capture time, and honouring a supplied count is what keeps
    the recorded allocation and D2D copies correct.
    """
    hidden, probs, scaling_factor, routing_map = init_tensor(
        hidden_dim=HIDDEN_DIM,
        seq_len=NUM_TOKENS_PER_RANK,
        topk=TOPK,
        num_of_experts=NUM_OF_EXPERTS,
    )

    # Warm up outside capture. This also leaves the warmup's own count in the
    # pinned buffer, which is what the explicit count below is checked against.
    warm_token, _, _, _ = buffer.dispatch(
        hidden=hidden, scaling_factor=scaling_factor,
        routing_map=routing_map, probs=probs)
    warm_count = warm_token.shape[0]
    torch.cuda.synchronize()

    # Without a count the shape cannot be known, so this has to be refused
    # rather than capturing whatever the pinned buffer happens to hold.
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            buffer.dispatch(hidden=hidden, scaling_factor=scaling_factor,
                            routing_map=routing_map, probs=probs)
        raise AssertionError("capturing dispatch() without a count should be refused")
    except RuntimeError as exc:
        assert "num_dispatched_tokens" in str(exc), f"unclear error: {exc}"

    torch.cuda.synchronize()

    # Stay below the warmup count so the sized copies are in bounds, while
    # still differing from the value left in the pinned buffer.
    target = max(1, warm_count - 128)
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        out_token, _, _, _ = buffer.dispatch(
            hidden=hidden, scaling_factor=scaling_factor,
            routing_map=routing_map, probs=probs,
            num_dispatched_tokens=target)
    assert out_token.shape[0] == target, (
        f"output has {out_token.shape[0]} rows, expected {target}; the count was "
        f"read back instead of honoured (warmup count was {warm_count})")

    torch.cuda.synchronize()
    print_in_order("test_plain_dispatch_capture passed")


def test_main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    _, _, group = init_dist(local_rank, num_local_ranks)

    for use_fp8 in [False, True]:
        for with_probs in [True, False]:
            for fused_permute_dispatch in [False, True]:
                buffer = deep_ep.HybridEPBuffer(
                    group=group,
                    hidden_dim=HIDDEN_DIM,
                    max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
                    num_local_experts=NUM_LOCAL_EXPERTS,
                    use_fp8=use_fp8
                )

                # Set missing global vars - use buffer's detected values
                global NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_EXPERTS
                # The NVLink domain the buffer detects may span several nodes
                # (MNNVL), so it is not derivable from the launcher's per-node
                # process count. The dispatched prob vector is indexed by
                # rank-within-domain, and the reference must use the same width
                # or it slices the wrong experts.
                NUM_OF_RANKS_PER_NODE = buffer.num_of_hybrid_ep_ranks_per_nvlink_domain
                NUM_OF_NODES = buffer.num_of_nodes
                NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES
                if group.rank() == 0:
                    # Surface the topology: if it is ever wrong, the symptom is
                    # an expert-slicing mismatch, a confusing way to find out.
                    print(f"[topology] ranks_per_nvlink_domain={NUM_OF_RANKS_PER_NODE} "
                          f"num_of_nodes={NUM_OF_NODES} num_of_experts={NUM_OF_EXPERTS} "
                          f"(from the buffer; override with "
                          f"NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN)", flush=True)

                ref = TorchRef(
                    ep_group=group,
                    num_of_experts=NUM_OF_EXPERTS,
                    num_of_ranks_per_node=NUM_OF_RANKS_PER_NODE,
                )
                test_hybrid_ep_correctness(buffer, ref, use_fp8, with_probs, fused_permute_dispatch)

                if not use_fp8 and with_probs and not fused_permute_dispatch:
                    test_plain_dispatch_capture(buffer)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')
    parser.add_argument('--num-processes', type=int, default=4,
                       help='Number of processes to spawn (default: 4)')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_main, args=(args.num_processes, args), nprocs=args.num_processes)

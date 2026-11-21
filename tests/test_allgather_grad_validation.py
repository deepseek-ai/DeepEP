# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression coverage for HybridEP custom-allgather and NCCL fallback paths.

This extends the dedicated allgather output comparison through combine and the
documented manual backward communication sequence:

    forward:  dispatch -> expert -> combine
    backward: dispatch(combined grads) -> expert backward -> combine
"""

import argparse
import os

import torch
import torch.distributed as dist

from utils import init_dist


NUM_TOKENS = int(os.environ.get("NUM_TOKENS_PER_RANK", 1024))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 8))
TOPK = int(os.environ.get("TOPK", 8))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 1024))
REPEATS = int(os.environ.get("REPEATS", 3))
SEED = int(os.environ.get("SEED", 1025))
EXPECTED_NUM_NODES = int(os.environ.get("EXPECTED_NUM_NODES", 0))


def assert_equal(label, lhs, rhs):
    assert (lhs is None) == (rhs is None), f"{label}: None mismatch"
    if lhs is None:
        return
    assert lhs.shape == rhs.shape, f"{label}: shape {lhs.shape} != {rhs.shape}"
    assert lhs.dtype == rhs.dtype, f"{label}: dtype {lhs.dtype} != {rhs.dtype}"
    if not torch.equal(
        lhs.contiguous().view(torch.uint8), rhs.contiguous().view(torch.uint8)
    ):
        mismatches = (lhs != rhs).sum().item()
        if lhs.is_floating_point():
            max_diff = (lhs.float() - rhs.float()).abs().max().item()
            raise AssertionError(
                f"{label}: {mismatches} values differ, max_abs_diff={max_diff}"
            )
        raise AssertionError(f"{label}: {mismatches} values differ")


def assert_dispatch_equal(label, custom, nccl):
    for name, custom_tensor, nccl_tensor in zip(
        ("hidden", "probs", "scaling_factor"), custom[:3], nccl[:3]
    ):
        assert_equal(f"{label}.dispatch.{name}", custom_tensor, nccl_tensor)

    custom_handle, nccl_handle = custom[3], nccl[3]
    assert len(custom_handle) == len(nccl_handle)
    assert_equal(
        f"{label}.handle.num_dispatched_tokens",
        custom_handle[3],
        nccl_handle[3],
    )
    custom_count = int(custom_handle[3].item())
    nccl_count = int(nccl_handle[3].item())
    assert custom_count == nccl_count
    assert_equal(
        f"{label}.handle.local_expert_routing_map",
        custom_handle[4][:custom_count],
        nccl_handle[4][:nccl_count],
    )
    assert custom_handle[5] == nccl_handle[5]


def make_inputs(rank, num_experts):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED + rank)
    hidden = torch.randn(
        NUM_TOKENS,
        HIDDEN_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    scores = torch.randn(
        NUM_TOKENS, num_experts, device="cuda", dtype=torch.float32, generator=generator
    )
    topk_idx = scores.topk(TOPK, dim=-1, sorted=False).indices
    topk_weights = torch.rand(
        NUM_TOKENS, TOPK, device="cuda", dtype=torch.float32, generator=generator
    )
    routing_map = torch.zeros(
        NUM_TOKENS, num_experts, device="cuda", dtype=torch.bool
    ).scatter_(1, topk_idx, True)
    probs = torch.zeros(
        NUM_TOKENS, num_experts, device="cuda", dtype=torch.float32
    ).scatter_(1, topk_idx, topk_weights)
    scaling_factor = torch.randn(
        NUM_TOKENS,
        HIDDEN_DIM // 128,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    return generator, hidden, topk_idx, topk_weights, routing_map, probs, scaling_factor


def run_case(
    label,
    buffer_custom,
    buffer_nccl,
    generator,
    hidden,
    dispatch_kwargs,
):
    custom = buffer_custom.dispatch(hidden=hidden, **dispatch_kwargs)
    nccl = buffer_nccl.dispatch(hidden=hidden, **dispatch_kwargs)
    assert_dispatch_equal(label, custom, nccl)

    # Forward expert output and combine outputs.
    expert_custom = custom[0] * torch.tensor(1.25, device="cuda", dtype=torch.bfloat16)
    expert_nccl = nccl[0] * torch.tensor(1.25, device="cuda", dtype=torch.bfloat16)
    combined_custom = buffer_custom.combine(expert_custom, custom[1], custom[3])
    combined_nccl = buffer_nccl.combine(expert_nccl, nccl[1], nccl[3])
    assert_equal(f"{label}.combine.hidden", combined_custom[0], combined_nccl[0])
    assert_equal(f"{label}.combine.probs", combined_custom[1], combined_nccl[1])

    # Backward of combine: dispatch gradients from tokens back to expert outputs.
    grad_combined_hidden = torch.randn(
        hidden.shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    num_experts = dispatch_kwargs.get("num_of_experts")
    if num_experts is None:
        num_experts = dispatch_kwargs["routing_map"].shape[1]
    grad_combined_probs = None
    if custom[1] is not None:
        grad_combined_probs = torch.randn(
            hidden.shape[0],
            num_experts,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    grad_expert_custom = buffer_custom.dispatch(
        hidden=grad_combined_hidden,
        probs=grad_combined_probs,
        num_of_experts=num_experts,
        handle=custom[3],
    )
    grad_expert_nccl = buffer_nccl.dispatch(
        hidden=grad_combined_hidden,
        probs=grad_combined_probs,
        num_of_experts=num_experts,
        handle=nccl[3],
    )
    assert_dispatch_equal(
        f"{label}.backward_combine", grad_expert_custom, grad_expert_nccl
    )

    # Backward of dispatch: combine expert-input gradients back to token inputs.
    grad_dispatched_hidden = torch.randn(
        custom[0].shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    grad_dispatched_probs = None
    if custom[1] is not None:
        grad_dispatched_probs = torch.randn(
            custom[1].shape, device="cuda", dtype=torch.float32, generator=generator
        )
    grad_input_custom = buffer_custom.combine(
        grad_dispatched_hidden, grad_dispatched_probs, custom[3]
    )
    grad_input_nccl = buffer_nccl.combine(
        grad_dispatched_hidden, grad_dispatched_probs, nccl[3]
    )
    assert_equal(
        f"{label}.backward_dispatch.hidden_grad",
        grad_input_custom[0],
        grad_input_nccl[0],
    )
    assert_equal(
        f"{label}.backward_dispatch.probs_grad",
        grad_input_custom[1],
        grad_input_nccl[1],
    )


def worker(local_rank, num_local_ranks, _args):
    import deep_ep

    _, _, group = init_dist(local_rank, num_local_ranks)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_experts = NUM_LOCAL_EXPERTS * world_size
    assert TOPK <= num_experts
    assert HIDDEN_DIM % 128 == 0

    generator, hidden, topk_idx, topk_weights, routing_map, probs, scaling_factor = (
        make_inputs(rank, num_experts)
    )
    buffer_custom = deep_ep.HybridEPBuffer(
        group=group,
        hidden_dim=HIDDEN_DIM,
        max_num_of_tokens_per_rank=NUM_TOKENS,
        num_local_experts=NUM_LOCAL_EXPERTS,
        enable_custom_allgather=True,
    )
    buffer_nccl = deep_ep.HybridEPBuffer(
        group=group,
        hidden_dim=HIDDEN_DIM,
        max_num_of_tokens_per_rank=NUM_TOKENS,
        num_local_experts=NUM_LOCAL_EXPERTS,
        enable_custom_allgather=False,
    )
    if EXPECTED_NUM_NODES:
        assert buffer_custom.num_of_nodes == EXPECTED_NUM_NODES
        assert buffer_nccl.num_of_nodes == EXPECTED_NUM_NODES
    if rank == 0:
        print(
            f"TOPOLOGY ranks={world_size}, nodes={buffer_nccl.num_of_nodes}, "
            f"ranks_per_node={buffer_nccl.num_of_hybrid_ep_ranks_per_nvlink_domain}",
            flush=True,
        )

    for repeat in range(REPEATS):
        for mode in ("indices", "sparse"):
            for with_probs in (False, True):
                if mode == "indices":
                    kwargs = {
                        "topk_idx": topk_idx,
                        "topk_weights": topk_weights if with_probs else None,
                        "num_of_experts": num_experts,
                        "scaling_factor": scaling_factor,
                    }
                else:
                    kwargs = {
                        "routing_map": routing_map,
                        "probs": probs if with_probs else None,
                        "scaling_factor": scaling_factor,
                    }
                label = f"repeat={repeat},mode={mode},with_probs={with_probs}"
                run_case(
                    label,
                    buffer_custom,
                    buffer_nccl,
                    generator,
                    hidden,
                    kwargs,
                )
                dist.barrier()
                if rank == 0:
                    print(f"PASS {label}", flush=True)

    if rank == 0:
        print(
            f"ALL OUTPUT/GRAD CHECKS PASSED: ranks={world_size}, repeats={REPEATS}, "
            f"tokens={NUM_TOKENS}, hidden={HIDDEN_DIM}, experts={num_experts}, topk={TOPK}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=8)
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        worker, args=(args.num_processes, args), nprocs=args.num_processes
    )

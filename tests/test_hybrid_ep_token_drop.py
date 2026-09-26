# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
import argparse
import os

import torch
import torch.distributed as dist

import deep_ep

from utils import init_dist

HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
MAX_NUM_OF_TOKENS_PER_RANK = int(os.environ.get("MAX_NUM_OF_TOKENS_PER_RANK", 4096))
# NUM_TOKENS_PER_RANK should equal or less than MAX_NUM_OF_TOKENS_PER_RANK
NUM_TOKENS_PER_RANK = int(os.environ.get("NUM_TOKENS_PER_RANK", 4096))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 8))
PAD_MULTIPLE = int(os.environ.get("PAD_MULTIPLE", 32))
SEED = int(os.environ.get("SEED", 1025))
# Route every token to a single expert, so a dst token is either combined from
# exactly 1 src token or fully dropped. That makes the expected output of every
# token unambiguous: the token itself, or zero.
TOPK = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def init_tensor(hidden_dim: int, seq_len: int, num_of_experts: int):
    hidden = torch.randn(seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)
    token_ids = torch.arange(seq_len, device="cuda")
    # Balanced round-robin routing, so every local expert of every rank receives
    # the same number of tokens and the capacity below is exceeded everywhere.
    expert_ids = token_ids % num_of_experts

    routing_map = torch.zeros(seq_len, num_of_experts, device="cuda", dtype=torch.bool)
    routing_map[token_ids, expert_ids] = True
    probs = torch.zeros(seq_len, num_of_experts, device="cuda", dtype=torch.float32)
    # Keep the probabilities away from zero: a zero row marks a dropped token below.
    probs[token_ids, expert_ids] = torch.rand(seq_len, device="cuda", dtype=torch.float32) * 0.5 + 0.5

    return hidden, probs, routing_map


def test_token_drop(
    buffer: deep_ep.HybridEPBuffer,
    num_of_experts: int,
    with_probs: bool,
    fuse_permute_dispatch: bool,
):
    hidden, probs, routing_map = init_tensor(HIDDEN_DIM, NUM_TOKENS_PER_RANK, num_of_experts)

    # Balanced top-1 routing sends NUM_TOKENS_PER_RANK tokens to every rank. Give
    # the permuted token buffer half of that, so the tail local experts of every
    # rank overflow and their tokens are dropped.
    num_permuted_tokens = ((NUM_TOKENS_PER_RANK // 2) // PAD_MULTIPLE) * PAD_MULTIPLE
    assert num_permuted_tokens > 0, "NUM_TOKENS_PER_RANK is too small to trigger token drop."

    context = f"(with_probs={with_probs}, fused={fuse_permute_dispatch})"

    (
        dispatched_hidden,
        dispatched_probs,
        _dispatched_scaling_factor,
        _tokens_per_expert,
        handle,
    ) = buffer.dispatch_with_permute(
        hidden=hidden,
        routing_map=routing_map,
        probs=probs if with_probs else None,
        pad_multiple=PAD_MULTIPLE,
        num_permuted_tokens=num_permuted_tokens,
        non_blocking=True,
        fuse_permute_dispatch=fuse_permute_dispatch,
    )
    # The fused combine-unpermute kernel hangs here instead of returning if the
    # combine side is built without token drop support.
    combined_hidden, combined_probs = buffer.combine_with_unpermute(
        hidden=dispatched_hidden.to(torch.bfloat16),
        probs=dispatched_probs,
        handle=handle,
        pad_multiple=PAD_MULTIPLE,
        fuse_unpermute_combine=fuse_permute_dispatch,
        non_blocking=True,
    )
    torch.cuda.synchronize()

    overflow_flag = handle[-1]
    assert overflow_flag.item() != 0, f"Token drop was not triggered {context}"

    assert torch.isfinite(combined_hidden.float()).all(), \
        f"Combined hidden has non-finite values {context}"

    # A dst token whose only src token was dropped must come back as zero, every
    # other one as the token itself(the expert is the identity here).
    dropped = (combined_hidden == 0).all(dim=1)
    kept = ~dropped
    assert dropped.any(), f"No token was dropped, the test did not exercise token drop {context}"
    assert kept.any(), f"All tokens were dropped {context}"
    assert torch.equal(combined_hidden[kept], hidden[kept]), \
        f"Surviving tokens must be combined unchanged {context}"

    if with_probs:
        assert torch.equal(combined_probs[kept], probs[kept]), \
            f"Surviving token probs must be combined unchanged {context}"
        assert (combined_probs[dropped] == 0).all(), \
            f"Dropped token probs must be combined to zero {context}"

    num_dropped = int(dropped.sum().item())
    fused_str = "fused" if fuse_permute_dispatch else "non-fused"
    if dist.get_rank() == 0:
        print(
            f"  token drop (with_probs={with_probs}, {fused_str}): PASS "
            f"({num_dropped}/{NUM_TOKENS_PER_RANK} tokens dropped)",
            flush=True,
        )


def test_main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    _, _, group = init_dist(local_rank, num_local_ranks)

    num_of_ranks_per_node = args.num_processes
    num_of_nodes = group.size() // num_of_ranks_per_node
    num_of_experts = NUM_LOCAL_EXPERTS * num_of_ranks_per_node * num_of_nodes

    for with_probs in [False, True]:
        for fuse_permute_dispatch in [False, True]:
            buffer = deep_ep.HybridEPBuffer(
                group=group,
                hidden_dim=HIDDEN_DIM,
                max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
                num_local_experts=NUM_LOCAL_EXPERTS,
                use_fp8=False,
            )
            test_token_drop(buffer, num_of_experts, with_probs, fuse_permute_dispatch)
            dist.barrier()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test hybrid EP token drop in non-blocking mode')
    parser.add_argument('--num-processes', type=int, default=4,
                        help='Number of processes to spawn (default: 4)')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_main, args=(args.num_processes, args), nprocs=args.num_processes)

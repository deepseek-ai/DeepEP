# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
"""
Correctness tests for ragged (per-rank unequal) token counts in HybridEP.

Each rank dispatches its own N_i tokens (optionally zero on some ranks) while
all ranks pass the same group-uniform `num_of_tokens_per_rank` slot count. The
reference is the uniform-count TorchRef fed with locally zero-padded inputs:
pad rows route nowhere, so the reference outputs are exactly the expected
ragged outputs.
"""
import argparse
import os

import torch
import torch.distributed as dist

import deep_ep

from utils import TorchRef, init_dist

HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 2048))
MAX_NUM_OF_TOKENS_PER_RANK = int(os.environ.get("MAX_NUM_OF_TOKENS_PER_RANK", 2048))
# Per-rank valid token counts, cycled by rank. Deliberately includes a
# non-multiple-of-16 count and a very small count.
RAGGED_NUM_TOKENS = [
    int(v) for v in os.environ.get("RAGGED_NUM_TOKENS", "1000,16,512,2048").split(",")
]
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 8))
TOPK = int(os.environ.get("TOPK", 8))
PAD_MULTIPLE = int(os.environ.get("PAD_MULTIPLE", 32))
SEED = int(os.environ.get("SEED", 1025))

NUM_OF_RANKS_PER_NODE = None
NUM_OF_NODES = None
NUM_OF_EXPERTS = None


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape or a.device != b.device:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def assert_bitwise_equal(name: str, ref: torch.Tensor, test: torch.Tensor, context: str = ""):
    if ref is None or test is None:
        return
    assert ref.shape == test.shape, (
        f"{name} shape mismatch{context}: ref={list(ref.shape)} got={list(test.shape)}"
    )
    if bitwise_equal(ref, test):
        return
    elem_mismatch = (ref != test).sum().item()
    pct = 100.0 * elem_mismatch / max(ref.numel(), 1)
    assert False, (
        f"{name} mismatch{context}: {elem_mismatch}/{ref.numel()} elements "
        f"({pct:.2f}%), shape={list(ref.shape)}"
    )


def align16(x: int) -> int:
    return (x + 15) // 16 * 16


def init_ragged_tensor(
    hidden_dim: int,
    num_valid: int,
    topk: int,
    num_of_experts: int,
    use_fp8: bool,
    seed: int,
):
    """Build per-rank inputs with num_valid rows, each routed to topk experts."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    if use_fp8:
        hidden = torch.randint(
            low=0, high=256, size=(num_valid, hidden_dim),
            device="cuda", dtype=torch.uint8, generator=gen,
        )
    else:
        hidden = torch.randn(num_valid, hidden_dim, device="cuda", dtype=torch.bfloat16, generator=gen)
    scaling_factor = torch.randn(num_valid, hidden_dim // 128, device="cuda", dtype=torch.float32, generator=gen)

    probs = torch.zeros(num_valid, num_of_experts, device="cuda", dtype=torch.float32)
    routing_map = torch.zeros(num_valid, num_of_experts, device="cuda", dtype=torch.bool)
    topk_idx = torch.full((num_valid, topk), -1, device="cuda", dtype=torch.int64)
    topk_weights = torch.zeros(num_valid, topk, device="cuda", dtype=torch.float32)

    for i in range(num_valid):
        selected = torch.randperm(num_of_experts, device="cuda", generator=gen)[:topk]
        topk_idx[i, :] = selected.to(torch.int64)
        topk_weights[i, :] = 1.0
        routing_map[i, selected] = True
        probs[i, selected] = 1.0

    return hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights


def pad_rows(t: torch.Tensor, target_rows: int):
    """Zero-pad a tensor along dim 0 to target_rows (reference-side padding)."""
    pad = target_rows - t.size(0)
    if pad == 0:
        return t
    filler = torch.zeros((pad,) + tuple(t.shape[1:]), device=t.device, dtype=t.dtype)
    return torch.cat([t, filler])


def check_combined(name, combined, hidden, routing_map, context):
    """combine sums one expert contribution per selected expert, each equal to
    the original token row (weights are 1), so combined[i] == hidden[i] * n_sel[i]."""
    assert combined.shape[0] == hidden.shape[0], (
        f"{name} row count{context}: expected {hidden.shape[0]}, got {combined.shape[0]}"
    )
    if hidden.shape[0] == 0:
        return
    n_sel = routing_map.sum(dim=1).to(torch.float32).unsqueeze(1)
    expected = (hidden.to(torch.float32) * n_sel).to(torch.bfloat16)
    assert torch.allclose(
        combined.to(torch.float32), expected.to(torch.float32), atol=2e-4, rtol=1e-2
    ), f"{name} value mismatch{context}"


def make_ragged_inputs(rank: int, use_fp8: bool, num_valid: int = None):
    if num_valid is None:
        num_valid = RAGGED_NUM_TOKENS[rank % len(RAGGED_NUM_TOKENS)]
    return num_valid, init_ragged_tensor(
        hidden_dim=HIDDEN_DIM,
        num_valid=num_valid,
        topk=TOPK,
        num_of_experts=NUM_OF_EXPERTS,
        use_fp8=use_fp8,
        seed=SEED + rank,
    )


def group_token_slots(num_valid: int) -> int:
    t = torch.tensor([num_valid], device="cuda", dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return align16(int(t.item()))


def test_ragged_dispatch_combine(buffer: deep_ep.HybridEPBuffer, ref: TorchRef, use_fp8: bool):
    rank = dist.get_rank()
    num_valid, (hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights) = (
        make_ragged_inputs(rank, use_fp8)
    )
    num_slots = group_token_slots(num_valid)

    dtype_str = "FP8" if use_fp8 else "BF16"
    if rank == 0:
        print(f"\n=== Ragged dispatch+combine ({dtype_str}, {dist.get_world_size()} ranks, "
              f"slots={num_slots}) ===", flush=True)

    for routing_label, use_dense_topk in [("sparse routing", False), ("dense topk_idx", True)]:
        for with_probs in [True, False]:
            context = f" ({routing_label}, with_probs={with_probs}, {dtype_str})"
            dispatch_kwargs = {
                "hidden": hidden,
                "scaling_factor": scaling_factor,
                "num_of_tokens_per_rank": num_slots,
            }
            if use_dense_topk:
                dispatch_kwargs.update({
                    "topk_idx": topk_idx,
                    "topk_weights": topk_weights if with_probs else None,
                    "num_of_experts": NUM_OF_EXPERTS,
                })
            else:
                dispatch_kwargs.update({
                    "routing_map": routing_map,
                    "probs": probs if with_probs else None,
                })

            # Reference: uniform TorchRef over locally zero-padded inputs.
            ref_hidden, ref_probs, ref_sf = ref.dispatch(
                pad_rows(hidden, num_slots),
                pad_rows(routing_map, num_slots),
                pad_rows(probs, num_slots) if with_probs else None,
                pad_rows(scaling_factor, num_slots),
            )

            dispatched_hidden, dispatched_probs, dispatched_sf, handle = buffer.dispatch(
                **dispatch_kwargs
            )

            assert_bitwise_equal("Ragged dispatch hidden", ref_hidden, dispatched_hidden, context)
            assert_bitwise_equal("Ragged dispatch scaling_factor", ref_sf, dispatched_sf, context)
            if dispatched_probs is not None and ref_probs is not None:
                start, end = ref._local_expert_range_per_node()
                assert_bitwise_equal(
                    "Ragged dispatch probs", ref_probs, dispatched_probs[:, start:end], context
                )
                masked = torch.zeros_like(dispatched_probs)
                masked[:, start:end] = dispatched_probs[:, start:end]
                dispatched_probs = masked

            num_dispatched, local_expert_routing_map = handle[3], handle[4]
            num_dispatched = int(num_dispatched.cpu().item())
            assert num_dispatched == ref_hidden.shape[0], (
                f"num_dispatched_tokens{context}: expected {ref_hidden.shape[0]}, got {num_dispatched}"
            )

            copy_times = local_expert_routing_map[:num_dispatched].sum(dim=1)
            hidden_to_combine = dispatched_hidden.to(torch.bfloat16) * copy_times.unsqueeze(1)
            combined, combined_probs = buffer.combine(hidden_to_combine, dispatched_probs, handle)
            check_combined("Ragged combine hidden", combined, hidden, routing_map, context)
            if combined_probs is not None and with_probs:
                assert combined_probs.shape[0] == num_valid, context
                assert_bitwise_equal("Ragged combine probs", probs, combined_probs, context)

        dist.barrier()
        if rank == 0:
            print(f"  dispatch+combine ({routing_label}): PASS", flush=True)


def test_ragged_dispatch_with_permute(buffer: deep_ep.HybridEPBuffer, ref: TorchRef, use_fp8: bool):
    rank = dist.get_rank()
    num_valid, (hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights) = (
        make_ragged_inputs(rank, use_fp8)
    )
    num_slots = group_token_slots(num_valid)

    dtype_str = "FP8" if use_fp8 else "BF16"
    if rank == 0:
        print(f"\n=== Ragged dispatch_with_permute ({dtype_str}, {dist.get_world_size()} ranks, "
              f"slots={num_slots}) ===", flush=True)

    for routing_label, use_dense_topk in [("sparse routing", False), ("dense topk_idx", True)]:
        for fuse_permute_dispatch in [False, True]:
            for with_probs in [True, False]:
                context = (f" ({routing_label}, with_probs={with_probs}, "
                           f"fuse={fuse_permute_dispatch}, {dtype_str})")
                dispatch_kwargs = {
                    "hidden": hidden,
                    "scaling_factor": scaling_factor,
                    "pad_multiple": PAD_MULTIPLE,
                    "fuse_permute_dispatch": fuse_permute_dispatch,
                    "num_of_tokens_per_rank": num_slots,
                }
                if use_dense_topk:
                    dispatch_kwargs.update({
                        "topk_idx": topk_idx,
                        "topk_weights": topk_weights if with_probs else None,
                        "num_of_experts": NUM_OF_EXPERTS,
                    })
                else:
                    dispatch_kwargs.update({
                        "routing_map": routing_map,
                        "probs": probs if with_probs else None,
                    })

                ref_hidden, ref_probs, ref_sf = ref.dispatch(
                    pad_rows(hidden, num_slots),
                    pad_rows(routing_map, num_slots),
                    pad_rows(probs, num_slots) if with_probs else None,
                    pad_rows(scaling_factor, num_slots),
                    pad_multiple=PAD_MULTIPLE,
                    enable_permute=True,
                )

                (
                    dispatched_hidden,
                    dispatched_probs,
                    dispatched_sf,
                    tokens_per_expert,
                    handle,
                ) = buffer.dispatch_with_permute(**dispatch_kwargs)

                assert_bitwise_equal("Ragged dispatch+permute hidden", ref_hidden, dispatched_hidden, context)
                assert_bitwise_equal("Ragged dispatch+permute probs", ref_probs, dispatched_probs, context)
                assert_bitwise_equal("Ragged dispatch+permute scaling_factor", ref_sf, dispatched_sf, context)

                # Cached-handle replay (the combine-backward pattern): must
                # reproduce the dispatch outputs bitwise.
                (
                    replay_hidden,
                    replay_probs,
                    replay_sf,
                    _tpe,
                    _replay_handle,
                ) = buffer.dispatch_with_permute(
                    hidden=hidden,
                    scaling_factor=scaling_factor,
                    probs=(probs if (with_probs and not use_dense_topk) else None),
                    topk_idx=(topk_idx if use_dense_topk else None),
                    topk_weights=(topk_weights if (with_probs and use_dense_topk) else None),
                    num_of_experts=(NUM_OF_EXPERTS if use_dense_topk else None),
                    handle=handle,
                    num_permuted_tokens=int(tokens_per_expert.sum().item()),
                    pad_multiple=PAD_MULTIPLE,
                    fuse_permute_dispatch=fuse_permute_dispatch,
                )
                assert_bitwise_equal("Ragged replay hidden", dispatched_hidden, replay_hidden, context)
                assert_bitwise_equal("Ragged replay scaling_factor", dispatched_sf, replay_sf, context)
                if with_probs:
                    assert_bitwise_equal("Ragged replay probs", dispatched_probs, replay_probs, context)

                combined, combined_probs = buffer.combine_with_unpermute(
                    hidden=dispatched_hidden.to(torch.bfloat16),
                    probs=dispatched_probs,
                    handle=handle,
                    pad_multiple=PAD_MULTIPLE,
                    fuse_unpermute_combine=fuse_permute_dispatch,
                )
                check_combined("Ragged combine+unpermute hidden", combined, hidden, routing_map, context)
                if combined_probs is not None and with_probs:
                    assert combined_probs.shape[0] == num_valid, context
                    assert_bitwise_equal("Ragged combine+unpermute probs", probs, combined_probs, context)

            dist.barrier()
            if rank == 0:
                fuse_str = "fused" if fuse_permute_dispatch else "non-fused"
                print(f"  dispatch_with_permute ({routing_label}, {fuse_str}): PASS", flush=True)


def test_zero_token_rank(buffer: deep_ep.HybridEPBuffer, ref: TorchRef, use_fp8: bool):
    """One rank dispatches zero tokens; the group must stay in lockstep."""
    rank = dist.get_rank()
    num_valid = 0 if rank == 0 else RAGGED_NUM_TOKENS[rank % len(RAGGED_NUM_TOKENS)]
    num_valid, (hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights) = (
        make_ragged_inputs(rank, use_fp8=use_fp8, num_valid=num_valid)
    )
    num_slots = group_token_slots(num_valid)
    dtype_str = "FP8" if use_fp8 else "BF16"
    if rank == 0:
        print(f"\n=== Ragged zero-token rank ({dtype_str}, slots={num_slots}) ===", flush=True)

    ref_hidden, ref_probs, ref_sf = ref.dispatch(
        pad_rows(hidden, num_slots),
        pad_rows(routing_map, num_slots),
        pad_rows(probs, num_slots),
        pad_rows(scaling_factor, num_slots),
    )
    dispatched_hidden, dispatched_probs, dispatched_sf, handle = buffer.dispatch(
        hidden=hidden,
        scaling_factor=scaling_factor,
        routing_map=routing_map,
        probs=probs,
        num_of_tokens_per_rank=num_slots,
    )
    assert_bitwise_equal("Zero-rank dispatch hidden", ref_hidden, dispatched_hidden, "")
    assert_bitwise_equal("Zero-rank dispatch scaling_factor", ref_sf, dispatched_sf, "")
    start, end = ref._local_expert_range_per_node()
    assert_bitwise_equal("Zero-rank dispatch probs", ref_probs, dispatched_probs[:, start:end], "")
    masked = torch.zeros_like(dispatched_probs)
    masked[:, start:end] = dispatched_probs[:, start:end]

    # Dense int16 top-k mode: a zero-token rank contributes a 0-row topk tensor
    # that pads to num_slots rows of -1 sentinels.
    disp_dense, probs_dense, _sf_dense, _handle_dense = buffer.dispatch(
        hidden=hidden,
        scaling_factor=scaling_factor,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_of_experts=NUM_OF_EXPERTS,
        num_of_tokens_per_rank=num_slots,
    )
    assert_bitwise_equal("Zero-rank dense dispatch hidden", ref_hidden, disp_dense, "")
    assert_bitwise_equal("Zero-rank dense dispatch probs", ref_probs, probs_dense[:, start:end], "")

    num_dispatched, local_expert_routing_map = handle[3], handle[4]
    num_dispatched = int(num_dispatched.cpu().item())
    copy_times = local_expert_routing_map[:num_dispatched].sum(dim=1)
    combined, combined_probs = buffer.combine(
        dispatched_hidden.to(torch.bfloat16) * copy_times.unsqueeze(1), masked, handle
    )
    check_combined("Zero-rank combine hidden", combined, hidden, routing_map, "")
    assert combined_probs.shape[0] == num_valid
    assert_bitwise_equal("Zero-rank combine probs", probs, combined_probs, "")

    # Permute path with a zero-token rank (the production API), both modes.
    for fuse in [False, True]:
        ref_hidden_p, ref_probs_p, ref_sf_p = ref.dispatch(
            pad_rows(hidden, num_slots),
            pad_rows(routing_map, num_slots),
            pad_rows(probs, num_slots),
            pad_rows(scaling_factor, num_slots),
            pad_multiple=PAD_MULTIPLE,
            enable_permute=True,
        )
        disp_p, probs_p, sf_p, _tpe_p, handle_p = buffer.dispatch_with_permute(
            hidden=hidden,
            scaling_factor=scaling_factor,
            routing_map=routing_map,
            probs=probs,
            pad_multiple=PAD_MULTIPLE,
            fuse_permute_dispatch=fuse,
            num_of_tokens_per_rank=num_slots,
        )
        assert_bitwise_equal("Zero-rank permute hidden", ref_hidden_p, disp_p, f" (fuse={fuse})")
        assert_bitwise_equal("Zero-rank permute probs", ref_probs_p, probs_p, f" (fuse={fuse})")
        combined_p, combined_probs_p = buffer.combine_with_unpermute(
            hidden=disp_p.to(torch.bfloat16),
            probs=probs_p,
            handle=handle_p,
            pad_multiple=PAD_MULTIPLE,
            fuse_unpermute_combine=fuse,
        )
        check_combined("Zero-rank combine+unpermute hidden", combined_p, hidden, routing_map, f" (fuse={fuse})")
        assert combined_probs_p.shape[0] == num_valid

    dist.barrier()
    if rank == 0:
        print("  zero-token rank: PASS", flush=True)


def test_uniform_unaligned_default(buffer: deep_ep.HybridEPBuffer, ref: TorchRef):
    """Legacy calling convention (no num_of_tokens_per_rank kwarg) with a
    uniform token count that is NOT a multiple of 16: the slot count is
    rounded up internally and combine must still return exactly N rows."""
    rank = dist.get_rank()
    num_valid = 1000  # not a multiple of 16, same on every rank
    num_valid, (hidden, probs, scaling_factor, routing_map, _ti, _tw) = (
        make_ragged_inputs(rank, use_fp8=False, num_valid=num_valid)
    )
    num_slots = align16(num_valid)
    if rank == 0:
        print(f"\n=== Uniform unaligned default path (BF16, N={num_valid}) ===", flush=True)

    ref_hidden, ref_probs, ref_sf = ref.dispatch(
        pad_rows(hidden, num_slots),
        pad_rows(routing_map, num_slots),
        pad_rows(probs, num_slots),
        pad_rows(scaling_factor, num_slots),
    )
    dispatched_hidden, dispatched_probs, dispatched_sf, handle = buffer.dispatch(
        hidden=hidden,
        scaling_factor=scaling_factor,
        routing_map=routing_map,
        probs=probs,
    )
    assert_bitwise_equal("Unaligned dispatch hidden", ref_hidden, dispatched_hidden, "")
    start, end = ref._local_expert_range_per_node()
    assert_bitwise_equal("Unaligned dispatch probs", ref_probs, dispatched_probs[:, start:end], "")
    masked = torch.zeros_like(dispatched_probs)
    masked[:, start:end] = dispatched_probs[:, start:end]

    num_dispatched, local_expert_routing_map = handle[3], handle[4]
    num_dispatched = int(num_dispatched.cpu().item())
    copy_times = local_expert_routing_map[:num_dispatched].sum(dim=1)
    combined, combined_probs = buffer.combine(
        dispatched_hidden.to(torch.bfloat16) * copy_times.unsqueeze(1), masked, handle
    )
    check_combined("Unaligned combine hidden", combined, hidden, routing_map, "")
    assert combined.shape[0] == num_valid and combined_probs.shape[0] == num_valid
    dist.barrier()
    if rank == 0:
        print("  uniform unaligned default: PASS", flush=True)


def test_stale_handle_rejected(buffer: deep_ep.HybridEPBuffer, ref: TorchRef):
    """Buffer growth reallocates the communication buffers and resets the flag
    protocol; replaying a handle created before the growth must fail loudly.
    Must run last: it grows the shared buffer (collective realloc + JIT)."""
    rank = dist.get_rank()
    num_valid, (hidden, probs, scaling_factor, routing_map, _ti, _tw) = (
        make_ragged_inputs(rank, use_fp8=False, num_valid=256)
    )
    if rank == 0:
        print("\n=== Stale handle rejection after buffer growth ===", flush=True)

    disp, _disp_probs, _sf, handle = buffer.dispatch(
        hidden=hidden,
        scaling_factor=scaling_factor,
        routing_map=routing_map,
        probs=probs,
        num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
    )

    # Grow the buffer: a slot count above the current capacity triggers a
    # collective reallocation (and JIT recompile) on every rank.
    disp2, disp_probs2, _sf2, handle2 = buffer.dispatch(
        hidden=hidden,
        scaling_factor=scaling_factor,
        routing_map=routing_map,
        probs=probs,
        num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK + 512,
    )

    # The pre-growth handle must be rejected on the cached-handle paths.
    for op in (
        lambda: buffer.combine(disp.to(torch.bfloat16), None, handle),
        lambda: buffer.dispatch(
            hidden=hidden, scaling_factor=scaling_factor,
            routing_map=routing_map, probs=probs, handle=handle,
        ),
    ):
        try:
            op()
        except RuntimeError as e:
            assert "generation" in str(e), f"unexpected error: {e}"
        else:
            raise AssertionError("stale pre-growth handle was not rejected")

    # The post-growth handle still works end to end.
    num_dispatched, lerm = handle2[3], handle2[4]
    num_dispatched = int(num_dispatched.cpu().item())
    copy_times = lerm[:num_dispatched].sum(dim=1)
    start, end = ref._local_expert_range_per_node()
    masked = torch.zeros_like(disp_probs2)
    masked[:, start:end] = disp_probs2[:, start:end]
    combined, _combined_probs = buffer.combine(
        disp2.to(torch.bfloat16) * copy_times.unsqueeze(1), masked, handle2
    )
    check_combined("Post-growth combine hidden", combined, hidden, routing_map, "")
    dist.barrier()
    if rank == 0:
        print("  stale handle rejected: PASS", flush=True)


def test_main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    _, _, group = init_dist(local_rank, num_local_ranks)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fp8_modes = [False] if args.only_bf16 else [False, True]
        for use_fp8 in fp8_modes:
            buffer = deep_ep.HybridEPBuffer(
                group=group,
                hidden_dim=HIDDEN_DIM,
                max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
                num_local_experts=NUM_LOCAL_EXPERTS,
                use_fp8=use_fp8,
            )

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
                # Surface the topology: if it is ever wrong, the symptom is an
                # expert-slicing mismatch, which is a confusing way to find out.
                print(f"[topology] ranks_per_nvlink_domain={NUM_OF_RANKS_PER_NODE} "
                      f"num_of_nodes={NUM_OF_NODES} num_of_experts={NUM_OF_EXPERTS} "
                      f"(from the buffer; override with "
                      f"NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN)", flush=True)

            ref = TorchRef(
                ep_group=group,
                num_of_experts=NUM_OF_EXPERTS,
                num_of_ranks_per_node=NUM_OF_RANKS_PER_NODE,
            )

            test_ragged_dispatch_combine(buffer, ref, use_fp8)
            test_ragged_dispatch_with_permute(buffer, ref, use_fp8)
            test_zero_token_rank(buffer, ref, use_fp8)
            if not use_fp8:
                test_uniform_unaligned_default(buffer, ref)
                test_stale_handle_rejected(buffer, ref)
    dist.barrier()
    if dist.get_rank() == 0:
        print("\nAll ragged HybridEP tests PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ragged (unequal per-rank token count) HybridEP")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--only-bf16", action="store_true", default=False)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_main, args=(args.num_processes, args), nprocs=args.num_processes)

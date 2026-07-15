# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
import argparse
import time
import torch
import torch.distributed as dist
import os
import deep_ep
import hybrid_ep_cpp

from utils import TorchRef, bench, bench_kineto, init_dist, count_rdma_send_from_routing_map

HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
MAX_NUM_OF_TOKENS_PER_RANK = int(os.environ.get("MAX_NUM_OF_TOKENS_PER_RANK", 4096))
# NUM_TOKENS_PER_RANK should equal or less than MAX_NUM_OF_TOKENS_PER_RANK
NUM_TOKENS_PER_RANK = int(os.environ.get("NUM_TOKENS_PER_RANK", 4096))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 8))
TOPK = int(os.environ.get("TOPK", 8))
PAD_MULTIPLE = int(os.environ.get("PAD_MULTIPLE", 32))
SEED = int(os.environ.get("SEED", 1025))

def _optional_int(env_key):
    v = os.environ.get(env_key, "").strip()
    return int(v) if v else None

NUM_SMS_DISPATCH     = _optional_int("NUM_SMS_DISPATCH")
NUM_SMS_COMBINE      = _optional_int("NUM_SMS_COMBINE")
NUM_BLOCKS_PERMUTE   = _optional_int("NUM_BLOCKS_PERMUTE")
NUM_BLOCKS_UNPERMUTE = _optional_int("NUM_BLOCKS_UNPERMUTE")

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
LOG_LABEL_WIDTH = 48

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


def assert_bitwise_equal(name: str, ref: torch.Tensor, test: torch.Tensor, context: str = ""):
    if ref is None or test is None:
        return
    if bitwise_equal(ref, test):
        return
    mismatch = (ref.contiguous().view(torch.uint8) != test.contiguous().view(torch.uint8)).sum().item()
    elem_mismatch = (ref != test).sum().item()
    pct = 100.0 * elem_mismatch / max(ref.numel(), 1)
    msg = (f"{name} mismatch{context}: {elem_mismatch}/{ref.numel()} elements "
           f"({pct:.2f}%), {mismatch} bytes differ, shape={list(ref.shape)}")
    flat_ref = ref.contiguous().view(-1)
    flat_test = test.contiguous().view(-1)
    diff_idx = (flat_ref != flat_test).nonzero(as_tuple=True)[0][:5]
    for idx in diff_idx:
        i = idx.item()
        msg += f"\n    [{i}]: ref={flat_ref[i].item()}, got={flat_test[i].item()}"
    assert False, msg


def supports_dense_topk_routing() -> bool:
    """Return whether the installed HybridEP supports dense topk_idx metadata."""
    try:
        # Test-only compatibility check: production code still falls back to sparse routing
        # when a dense-capable build receives a layout outside dense kernel limits.
        return hasattr(hybrid_ep_cpp.HybridEpConfigInstance(), "topk")
    except (TypeError, ValueError, AttributeError):
        return False

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


def test_hybrid_ep_correctness(buffer: deep_ep.HybridEPBuffer, ref: TorchRef, use_fp8: bool):
    hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights  = init_tensor(
        hidden_dim=HIDDEN_DIM,
        seq_len=NUM_TOKENS_PER_RANK,
        topk=TOPK,
        num_of_experts=NUM_OF_EXPERTS,
        use_fp8=use_fp8,
    )
    dtype_str = "FP8" if hidden.dtype == torch.uint8 else "BF16"
    supports_dense_topk = supports_dense_topk_routing()
    dist.barrier()
    if dist.get_rank() == 0:
        print(f'\n=== Correctness Check ({dtype_str}, {dist.get_world_size()} ranks) ===', flush=True)

    # Dispatch correctness check. Older hybrid-ep branches do not expose dense
    # topk_idx metadata yet, so skip the dense routing mode when unsupported.
    routing_modes = [("sparse routing", False)]
    if supports_dense_topk:
        routing_modes.append(("dense topk_idx", True))

    for routing_label, use_dense_topk in routing_modes:
        for with_probs in [True, False]:
            context = f" ({routing_label}, with_probs={with_probs})"
            dispatch_kwargs = {"hidden": hidden, "scaling_factor": scaling_factor}
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

            dispatched_hidden_ref, dispatched_probs_ref, dispatched_scaling_factor_ref = (
                ref.dispatch(
                    hidden, routing_map, probs if with_probs else None, scaling_factor
                )
            )
            (
                dispatched_hidden_dense,
                dispatched_probs_dense,
                dispatched_scaling_factor_dense,
                handle_dense,
            ) = buffer.dispatch(**dispatch_kwargs)

            assert_bitwise_equal("Dispatch hidden", dispatched_hidden_ref, dispatched_hidden_dense, context)
            assert_bitwise_equal("Dispatch scaling_factor", dispatched_scaling_factor_ref, dispatched_scaling_factor_dense, context)
            if dispatched_probs_dense is not None and dispatched_probs_ref is not None:
                start, end = ref._local_expert_range_per_node()
                assert_bitwise_equal("Dispatch probs", dispatched_probs_ref, dispatched_probs_dense[:, start:end], context)
                masked_probs = torch.zeros_like(dispatched_probs_dense)
                masked_probs[:, start:end] = dispatched_probs_dense[:, start:end]
                dispatched_probs_dense = masked_probs

            _, _, _, num_dispatched_tokens, local_expert_routing_map, _, _ = handle_dense
            num_dispatched_tokens = num_dispatched_tokens.cpu()
            local_expert_routing_map = local_expert_routing_map[
                : num_dispatched_tokens.item()
            ]
            copy_times = local_expert_routing_map.sum(dim=1)
            hidden_to_combine = dispatched_hidden_dense.to(torch.bfloat16) * copy_times.unsqueeze(1)
            combined_hidden, combined_probs = buffer.combine(
                hidden_to_combine, dispatched_probs_dense, handle_dense
            )
            combined_hidden = combined_hidden / TOPK

            assert torch.allclose(combined_hidden, hidden.to(torch.bfloat16), atol=2e-5, rtol=1e-2), \
                f"Combine hidden mismatch{context}"
            if combined_probs is not None and probs is not None:
                assert_bitwise_equal("Combine probs", probs, combined_probs, context)

        dist.barrier()
        if dist.get_rank() == 0:
            dense_suffix = " (dense routing)" if use_dense_topk else ""
            print(f'  dispatch+combine API{dense_suffix}: PASS', flush=True)

    if not supports_dense_topk:
        dist.barrier()
        if dist.get_rank() == 0:
            print('  dispatch+combine API (dense routing): SKIP (unsupported)', flush=True)

    # Dispatch with permute correctness check. Dense topk_idx mode exercises
    # scan with enable_permute=True, i.e. the path that produces
    # dense_chunk_layout and dense_to_expert_map.
    for routing_label, use_dense_topk in routing_modes:
        for fuse_permute_dispatch in [False, True]:
            for with_probs in [True, False]:
                context = (f" ({routing_label}, with_probs={with_probs}, "
                           f"fuse_permute_dispatch={fuse_permute_dispatch})")
                dispatch_kwargs = {
                    "hidden": hidden,
                    "scaling_factor": scaling_factor,
                    "pad_multiple": PAD_MULTIPLE,
                    "fuse_permute_dispatch": fuse_permute_dispatch,
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

                (
                    dispatched_hidden,
                    dispatched_probs,
                    dispatched_scaling_factor,
                    _tokens_per_expert,
                    handle,
                ) = buffer.dispatch_with_permute(**dispatch_kwargs)

                (
                    dispatched_hidden_ref,
                    dispatched_probs_ref,
                    dispatched_scaling_factor_ref,
                ) = ref.dispatch(
                    hidden,
                    routing_map,
                    probs if with_probs else None,
                    scaling_factor,
                    pad_multiple=PAD_MULTIPLE,
                    enable_permute=True,
                )

                assert_bitwise_equal("Dispatch+permute hidden", dispatched_hidden_ref, dispatched_hidden, context)
                assert_bitwise_equal("Dispatch+permute probs", dispatched_probs_ref, dispatched_probs, context)
                assert_bitwise_equal("Dispatch+permute scaling_factor", dispatched_scaling_factor_ref, dispatched_scaling_factor, context)

                combined_hidden, combined_probs = buffer.combine_with_unpermute(
                    hidden=dispatched_hidden.to(torch.bfloat16),
                    probs=dispatched_probs,
                    handle=handle,
                    pad_multiple=PAD_MULTIPLE,
                    fuse_unpermute_combine=fuse_permute_dispatch,
                )
                combined_hidden = combined_hidden / TOPK

                assert torch.allclose(
                    combined_hidden, hidden.to(torch.bfloat16), atol=2e-5, rtol=1e-2
                ), f"Combine+unpermute hidden mismatch{context}"
                if combined_probs is not None and probs is not None:
                    assert_bitwise_equal("Combine+unpermute probs", probs, combined_probs, context)

            dist.barrier()
            if dist.get_rank() == 0:
                dense_suffix = "dense routing, " if use_dense_topk else ""
                api_name = (
                    f'dispatch_with_permute + combine_with_unpermute API ({dense_suffix}non-fused)'
                    if not fuse_permute_dispatch
                    else f'dispatch_with_permute + combine_with_unpermute API ({dense_suffix}fused)'
                )
                print(f'  {api_name}: PASS', flush=True)

    if not supports_dense_topk:
        dist.barrier()
        if dist.get_rank() == 0:
            print('  dispatch_with_permute + combine_with_unpermute API (dense routing): SKIP (unsupported)', flush=True)


def _gather_times(t):
    """Gather a scalar time from all ranks, return list on rank 0."""
    t_tensor = torch.tensor([t], device='cuda', dtype=torch.float64)
    gathered = [torch.zeros(1, device='cuda', dtype=torch.float64) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t_tensor)
    return [x.item() for x in gathered]

def _report_bw(label, t, nvl_bytes, nvl_metric, rdma_bytes=None, rdma_metric=None):
    """Print min/avg/max bandwidth summary on rank 0 (NVL + optional RDMA)."""
    times = _gather_times(t)
    if dist.get_rank() == 0:
        t_min, t_avg, t_max = min(times), sum(times) / len(times), max(times)
        bw_avg = nvl_bytes / 1e9 / t_avg
        label_col = f'{label}:'.ljust(LOG_LABEL_WIDTH)
        print(f'{label_col} {bw_avg:.2f} GB/s (NVL), '
              f't: {t_avg * 1e6:.1f} us [min={t_min * 1e6:.1f}, max={t_max * 1e6:.1f}], '
              f'{nvl_metric}: {nvl_bytes / 1e6:.2f} MB', flush=True)
        if rdma_bytes is not None:
            bw_rdma = rdma_bytes / 1e9 / t_avg
            print(f'{label_col} {bw_rdma:.2f} GB/s (RDMA), '
                  f't: {t_avg * 1e6:.1f} us [min={t_min * 1e6:.1f}, max={t_max * 1e6:.1f}], '
                  f'{rdma_metric}: {rdma_bytes / 1e6:.2f} MB', flush=True)

def _report_kineto(dispatch_label, combine_label, dispatch_t, dispatch_bytes, combine_t, combine_bytes,
                   rdma_dispatch=None, rdma_combine=None):
    """Print min/avg/max kineto dispatch|combine summary on rank 0 (NVL + optional RDMA)."""
    d_times = _gather_times(dispatch_t)
    c_times = _gather_times(combine_t)
    if dist.get_rank() == 0:
        d_min, d_max = min(d_times), max(d_times)
        c_min, c_max = min(c_times), max(c_times)
        d_avg = sum(d_times) / len(d_times)
        c_avg = sum(c_times) / len(c_times)
        dispatch_col_nvl = f'{dispatch_label}(NVL):'.ljust(LOG_LABEL_WIDTH)
        combine_col_nvl = f'{combine_label}(NVL):'.ljust(LOG_LABEL_WIDTH)
        print(f'{dispatch_col_nvl} {dispatch_bytes / 1e9 / d_avg:.2f} GB/s, '
              f'avg_t={d_avg * 1e6:.1f} us [min={d_min * 1e6:.1f}, max={d_max * 1e6:.1f}]', flush=True)
        print(f'{combine_col_nvl} {combine_bytes / 1e9 / c_avg:.2f} GB/s, '
              f'avg_t={c_avg * 1e6:.1f} us [min={c_min * 1e6:.1f}, max={c_max * 1e6:.1f}]', flush=True)
        if rdma_dispatch is not None:
            dispatch_col_rdma = f'{dispatch_label}(RDMA):'.ljust(LOG_LABEL_WIDTH)
            combine_col_rdma = f'{combine_label}(RDMA):'.ljust(LOG_LABEL_WIDTH)
            print(f'{dispatch_col_rdma} {rdma_dispatch / 1e9 / d_avg:.2f} GB/s, '
                  f'avg_t={d_avg * 1e6:.1f} us [min={d_min * 1e6:.1f}, max={d_max * 1e6:.1f}]', flush=True)
            print(f'{combine_col_rdma} {rdma_combine / 1e9 / c_avg:.2f} GB/s, '
                  f'avg_t={c_avg * 1e6:.1f} us [min={c_min * 1e6:.1f}, max={c_max * 1e6:.1f}]', flush=True)


def test_hybrid_ep_benchmark(buffer: deep_ep.HybridEPBuffer, group: dist.ProcessGroup, use_fp8: bool, nsys_profile: bool):
    hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights = init_tensor(
        hidden_dim=HIDDEN_DIM,
        seq_len=NUM_TOKENS_PER_RANK,
        topk=TOPK,
        num_of_experts=NUM_OF_EXPERTS,
        use_fp8=use_fp8,
    )

    rank = dist.get_rank()
    dtype_str = "FP8" if use_fp8 else "BF16"
    multinode = (NUM_OF_NODES > 1)

    # ---- Setup: collect handles, build args dicts (also serves as warmup) ----
    # Non-permute (forward dispatch with probs)
    dispatched_hidden, dispatched_probs, _, handle = (
        buffer.dispatch(hidden=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx,
                        topk_weights=topk_weights, num_of_experts=NUM_OF_EXPERTS))
    dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)
    dispatch_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'topk_idx': topk_idx,
                     'topk_weights': topk_weights, 'num_of_experts': NUM_OF_EXPERTS, 'handle': handle}
    combine_args = {'hidden': dispatched_hidden_bf16, 'probs': dispatched_probs, 'handle': handle}

    # Dispatch/combine with probs=False variants
    dispatch_noprob_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'topk_idx': topk_idx,
                            'topk_weights': None, 'num_of_experts': NUM_OF_EXPERTS, 'handle': handle}
    combine_noprob_args = {'hidden': dispatched_hidden_bf16, 'probs': None, 'handle': handle}

    # Permute (non-fused)
    dispatched_hidden_wp, dispatched_probs_wp, _, tpe_wp, handle_wp = (
        buffer.dispatch_with_permute(hidden=hidden, scaling_factor=scaling_factor,
            routing_map=routing_map, probs=probs, pad_multiple=PAD_MULTIPLE))
    dispatched_hidden_bf16_wp = dispatched_hidden_wp.to(torch.bfloat16)
    dispatch_wp_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'routing_map': routing_map,
                        'probs': probs, 'pad_multiple': PAD_MULTIPLE, 'handle': handle_wp,
                        'num_permuted_tokens': tpe_wp.sum().item()}
    combine_wp_args = {'hidden': dispatched_hidden_bf16_wp, 'probs': dispatched_probs_wp,
                       'handle': handle_wp, 'pad_multiple': PAD_MULTIPLE}

    # Fused permute-dispatch
    dispatched_hidden_fused, dispatched_probs_fused, _, tpe_fused, handle_fused = (
        buffer.dispatch_with_permute(hidden=hidden, scaling_factor=scaling_factor,
            routing_map=routing_map, probs=probs, pad_multiple=PAD_MULTIPLE, fuse_permute_dispatch=True))
    dispatched_hidden_bf16_fused = dispatched_hidden_fused.to(torch.bfloat16)
    dispatch_fused_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'routing_map': routing_map,
                           'probs': probs, 'pad_multiple': PAD_MULTIPLE, 'handle': handle_fused,
                           'num_permuted_tokens': tpe_fused.sum().item(), 'fuse_permute_dispatch': True}
    combine_fused_args = {'hidden': dispatched_hidden_bf16_fused, 'probs': dispatched_probs_fused,
                          'handle': handle_fused, 'pad_multiple': PAD_MULTIPLE, 'fuse_unpermute_combine': True}

    # Bandwidth constants
    fp8_factor = (1 + 4 / 128) / 2
    nvl_dispatch = dispatched_hidden.numel() * 2
    nvl_dispatch_actual = nvl_dispatch * fp8_factor if use_fp8 else nvl_dispatch
    nvl_combine = nvl_dispatch
    rdma_dispatch = rdma_combine = None
    if multinode:
        local_node_id = rank // NUM_OF_RANKS_PER_NODE
        num_rdma = count_rdma_send_from_routing_map(routing_map, local_node_id, NUM_OF_NODES)
        rdma_dispatch = num_rdma * HIDDEN_DIM * 2
        if use_fp8:
            rdma_dispatch = rdma_dispatch * fp8_factor
        rdma_combine = rdma_dispatch

    # ---- Bench: torch API ----
    if rank == 0:
        print(f'\n=== Torch API Benchmark ({dtype_str}, {dist.get_world_size()} ranks) ===', flush=True)
        print(f'  Non-permute:  dispatch = dispatch_kernel + d2d + misc', flush=True)
        print(f'                combine  = d2d + combine_kernel + misc', flush=True)
        print(f'  Permute:      dispatch = dispatch_kernel + permute_kernel + misc', flush=True)
        print(f'                combine  = unpermute_kernel + combine_kernel + misc', flush=True)
        print(f'  Fused:        dispatch = fused_permute_dispatch_kernel + misc', flush=True)
        print(f'                combine  = fused_combine_unpermute_kernel + misc', flush=True)
        print(f'  (misc = device_sync, update_flag, etc.)', flush=True)

    # Non-permute (probs=True)
    t = bench(lambda: buffer.dispatch(**dispatch_args))[0]
    _report_bw(f'dispatch ({dtype_str}, probs=True)', t, nvl_dispatch_actual, 'nvl_recv_bytes', rdma_dispatch, 'rdma_send_bytes')
    t = bench(lambda: buffer.combine(**combine_args))[0]
    _report_bw(f'combine ({dtype_str}, probs=True)', t, nvl_combine, 'combine_send_bytes', rdma_combine, 'rdma_recv_bytes')

    # Non-permute (probs=False)
    t = bench(lambda: buffer.dispatch(**dispatch_noprob_args))[0]
    _report_bw(f'dispatch ({dtype_str}, probs=False)', t, nvl_dispatch_actual, 'nvl_recv_bytes', rdma_dispatch, 'rdma_send_bytes')
    t = bench(lambda: buffer.combine(**combine_noprob_args))[0]
    _report_bw(f'combine ({dtype_str}, probs=False)', t, nvl_combine, 'combine_send_bytes', rdma_combine, 'rdma_recv_bytes')

    # Permute (non-fused)
    t = bench(lambda: buffer.dispatch_with_permute(**dispatch_wp_args))[0]
    _report_bw(f'dispatch+permute ({dtype_str}, probs=True)', t, nvl_dispatch_actual, 'nvl_recv_bytes', rdma_dispatch, 'rdma_send_bytes')
    t = bench(lambda: buffer.combine_with_unpermute(**combine_wp_args))[0]
    _report_bw(f'combine+unpermute ({dtype_str}, probs=True)', t, nvl_combine, 'combine_send_bytes', rdma_combine, 'rdma_recv_bytes')

    # Fused
    t = bench(lambda: buffer.dispatch_with_permute(**dispatch_fused_args))[0]
    _report_bw(f'fused dispatch+permute ({dtype_str}, probs=True)', t, nvl_dispatch_actual, 'nvl_recv_bytes', rdma_dispatch, 'rdma_send_bytes')
    t = bench(lambda: buffer.combine_with_unpermute(**combine_fused_args))[0]
    _report_bw(f'fused combine+unpermute ({dtype_str}, probs=True)', t, nvl_combine, 'combine_send_bytes', rdma_combine, 'rdma_recv_bytes')

    # ---- Kineto / nsys profiling ----
    # Kineto measures pure GPU kernel time only (no CPU overhead, no d2d, no device_sync)
    if not nsys_profile:
        if rank == 0:
            print(f'\n=== Kernel Benchmark ({dtype_str}, {dist.get_world_size()} ranks) ===', flush=True)
            print(f'  Non-fused:  dispatch_kernel only  |  combine_kernel only', flush=True)
            print(f'  Fused:      fused_permute_dispatch_kernel only  |  fused_combine_unpermute_kernel only', flush=True)

        # Non-fused kernel profiling (probs=True)
        group.barrier()
        dispatch_t, combine_t = bench_kineto(
            lambda: (buffer.dispatch(**dispatch_args), buffer.combine(**combine_args)),
            kernel_names=('dispatch_kernel', 'combine_kernel'), barrier_comm_profiling=True, suppress_kineto_output=True)
        _report_kineto(f'dispatch kernel ({dtype_str}, probs=True)', f'combine kernel ({dtype_str}, probs=True)',
                       dispatch_t, nvl_dispatch_actual, combine_t, nvl_combine, rdma_dispatch, rdma_combine)

        # Non-fused kernel profiling (probs=False)
        group.barrier()
        dispatch_t, combine_t = bench_kineto(
            lambda: (buffer.dispatch(**dispatch_noprob_args), buffer.combine(**combine_noprob_args)),
            kernel_names=('dispatch_kernel', 'combine_kernel'), barrier_comm_profiling=True, suppress_kineto_output=True)
        _report_kineto(f'dispatch kernel ({dtype_str}, probs=False)', f'combine kernel ({dtype_str}, probs=False)',
                       dispatch_t, nvl_dispatch_actual, combine_t, nvl_combine, rdma_dispatch, rdma_combine)

        # Fused kernel profiling
        group.barrier()
        dispatch_t, combine_t = bench_kineto(
            lambda: (buffer.dispatch_with_permute(**dispatch_fused_args), buffer.combine_with_unpermute(**combine_fused_args)),
            kernel_names=('dispatch_kernel', 'combine_kernel'), barrier_comm_profiling=True, suppress_kineto_output=True)
        _report_kineto(f'fused dispatch+permute kernel ({dtype_str}, probs=True)', f'fused combine+unpermute kernel ({dtype_str}, probs=True)',
                       dispatch_t, nvl_dispatch_actual, combine_t, nvl_combine, rdma_dispatch, rdma_combine)

        # Non-fused permute/unpermute kernel profiling (isolate permute_kernel and unpermute_kernel times)
        group.barrier()
        dispatch_t, permute_t = bench_kineto(
            lambda: buffer.dispatch_with_permute(**dispatch_wp_args),
            kernel_names=('dispatch_kernel', 'permute_kernel'), barrier_comm_profiling=True, suppress_kineto_output=True)
        combine_t, unpermute_t = bench_kineto(
            lambda: buffer.combine_with_unpermute(**combine_wp_args),
            kernel_names=('combine_kernel', 'unpermute_kernel'), barrier_comm_profiling=True, suppress_kineto_output=True)
        d_times = _gather_times(dispatch_t)
        p_times = _gather_times(permute_t)
        c_times = _gather_times(combine_t)
        u_times = _gather_times(unpermute_t)
        if rank == 0:
            fmt = lambda ts: f'avg={sum(ts)/len(ts)*1e6:.1f} us [min={min(ts)*1e6:.1f}, max={max(ts)*1e6:.1f}]'
            print(f'  {"dispatch_kernel (in dispatch+permute):":<{LOG_LABEL_WIDTH}} {fmt(d_times)}', flush=True)
            print(f'  {"permute_kernel:":<{LOG_LABEL_WIDTH}} {fmt(p_times)}', flush=True)
            print(f'  {"unpermute_kernel:":<{LOG_LABEL_WIDTH}} {fmt(u_times)}', flush=True)
            print(f'  {"combine_kernel (in combine+unpermute):":<{LOG_LABEL_WIDTH}} {fmt(c_times)}', flush=True)
    else:
        if torch.distributed.get_rank() == 0:
            torch.cuda.profiler.start()
        with torch.cuda.nvtx.range(f"hybrid-ep dispatch ({dtype_str}, probs=True)"):
            if rank == 0:
                print(f"profile hybrid-ep dispatch ({dtype_str}, probs=True)", flush=True)
            nsys_dispatch_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'topk_idx': topk_idx, 'topk_weights': topk_weights, 'num_of_experts': NUM_OF_EXPERTS}
            bench(lambda: buffer.dispatch(**nsys_dispatch_args))
        with torch.cuda.nvtx.range(f"hybrid-ep combine ({dtype_str}, probs=True)"):
            if rank == 0:
                print(f"profile hybrid-ep combine ({dtype_str}, probs=True)", flush=True)
            bench(lambda: buffer.combine(**combine_args))
        with torch.cuda.nvtx.range(f"hybrid-ep dispatch+permute ({dtype_str}, probs=True)"):
            if rank == 0:
                print(f"profile hybrid-ep dispatch+permute ({dtype_str}, probs=True)", flush=True)
            nsys_dispatch_wp_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'routing_map': routing_map, 'probs': probs, 'pad_multiple': PAD_MULTIPLE}
            bench(lambda: buffer.dispatch_with_permute(**nsys_dispatch_wp_args))
        with torch.cuda.nvtx.range(f"hybrid-ep combine+unpermute ({dtype_str}, probs=True)"):
            if rank == 0:
                print(f"profile hybrid-ep combine+unpermute ({dtype_str}, probs=True)", flush=True)
            bench(lambda: buffer.combine_with_unpermute(**combine_wp_args))
        with torch.cuda.nvtx.range(f"hybrid-ep dispatch+permute fused ({dtype_str}, probs=True)"):
            if rank == 0:
                print(f"profile hybrid-ep dispatch+permute fused ({dtype_str}, probs=True)", flush=True)
            nsys_dispatch_fused_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'routing_map': routing_map, 'probs': probs, 'pad_multiple': PAD_MULTIPLE, 'fuse_permute_dispatch': True}
            bench(lambda: buffer.dispatch_with_permute(**nsys_dispatch_fused_args))
        with torch.cuda.nvtx.range(f"hybrid-ep combine+unpermute fused ({dtype_str}, probs=True)"):
            if rank == 0:
                print(f"profile hybrid-ep combine+unpermute fused ({dtype_str}, probs=True)", flush=True)
            bench(lambda: buffer.combine_with_unpermute(**combine_fused_args))
        time.sleep(1)
        if torch.distributed.get_rank() == 0:
            torch.cuda.profiler.stop()


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
                num_sms_dispatch_api=NUM_SMS_DISPATCH,
                num_sms_combine_api=NUM_SMS_COMBINE,
                num_blocks_permute=NUM_BLOCKS_PERMUTE,
                num_blocks_unpermute=NUM_BLOCKS_UNPERMUTE,
            )

            # Set missing global vars - use buffer's detected values
            global NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_EXPERTS
            if USE_MNNVL:
                NUM_OF_RANKS_PER_NODE = buffer.num_of_hybrid_ep_ranks_per_nvlink_domain
                NUM_OF_NODES = buffer.num_of_nodes
                NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES
            else:
                NUM_OF_RANKS_PER_NODE = args.num_processes
                NUM_OF_NODES = group.size() // NUM_OF_RANKS_PER_NODE
                NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES

            ref = TorchRef(
                ep_group=group,
                num_of_experts=NUM_OF_EXPERTS,
                num_of_ranks_per_node=NUM_OF_RANKS_PER_NODE,
            )

            test_hybrid_ep_correctness(buffer, ref, use_fp8)
            test_hybrid_ep_benchmark(buffer, group, use_fp8, args.nsys_profile)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')
    parser.add_argument('--num-processes', type=int, default=4,
                       help='Number of processes to spawn (default: 4)')
    parser.add_argument('--nsys-profile', action='store_true', default=False,
                       help='benchmark with nsys profile or not (default: False)')
    parser.add_argument('--only-bf16', action='store_true', default=False,
                       help='Skip FP8 tests, only run BF16 (default: False)')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_main, args=(args.num_processes, args), nprocs=args.num_processes)

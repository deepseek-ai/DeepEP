import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from functools import partial

import deep_gemm
from deep_gemm.utils.math import align, ceil_div, per_block_cast_to_fp8
import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back
from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_post_quant_fwd
from sglang.srt.layers.quantization.fp8_utils import _requant_weight_ue8m0
from deep_gemm.utils.layout import transform_sf_into_required_layout

# --------------------------------------------- main -----------------------------------------------------


def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer, seed: int = 0):
    print(f"[{rank}] test_main start", flush=True)
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceeds the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    print(f"[{rank}] test_main prepare data", flush=True)
    # ref: DeepGEMM - generate_grouped_masked
    x = torch.randn((num_tokens, hidden), device='cuda', dtype=torch.bfloat16)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # Randomly mask some positions
    for i in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    w13_weight_fp8 = create_weight_fp8(num_groups=num_local_experts, n=4096, k=hidden)
    w2_weight_fp8 = create_weight_fp8(num_groups=num_local_experts, n=hidden, k=2048)

    # noinspection PyShadowingNames
    def test_func(fn_mode: str):
        if fn_mode == 'naive':
            f = forward_layer_naive
        elif fn_mode == 'overlap':
            f = forward_layer_overlap
        else:
            raise NotImplementedError

        f(
            hidden_states=x,
            w13_weight_fp8=w13_weight_fp8,
            w2_weight_fp8=w2_weight_fp8,
            buffer=buffer,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens=num_tokens,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
        )

    for fn_mode in [
        # 'naive', # TODO temp rm
        'overlap',
    ]:
        if rank == 0:
            trace_path = str(Path("/data/numa0/tom/temp_sglang_server2local/") / f"{time.time()}-TP-{rank}.trace.json.gz")
        else:
            trace_path = None
        print(f"Execute bench {fn_mode=} {rank=} {trace_path=}", flush=True)
        bench_kineto(partial(test_func, fn_mode=fn_mode),
                     kernel_names=('dispatch', 'combine'), barrier_comm_profiling=True,
                     suppress_kineto_output=False, # NOTE MODIFIED
                     trace_path=trace_path)


def create_weight_fp8(num_groups, n, k):
    # ref: DeepGEMM - generate_grouped_masked
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    b_fp8 = (torch.empty_like(b, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), ceil_div(k, 128)), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i])

    # ref:
    b_fp8 = _requant_weight_ue8m0(*b_fp8, weight_block_size=[128, 128])

    return b_fp8


# noinspection PyShadowingNames
def large_gemm():
    mat_0 = torch.randn((8192, 8192), dtype=torch.float)
    mat_1 = torch.randn((8192, 8192), dtype=torch.float)
    mat_0 @ mat_1


# noinspection PyUnboundLocalVariable
def test_loop(local_rank: int, num_local_ranks: int):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    # num_tokens, hidden, num_topk, num_experts = 4096, 7168, 8, (256 // num_ranks) * num_ranks
    num_tokens = int(os.environ.get("DEEPEP_TEST_NUM_TOKENS", "4096"))
    hidden = int(os.environ.get("DEEPEP_TEST_HIDDEN", "7168"))
    num_topk = int(os.environ.get("DEEPEP_TEST_NUM_TOPK", "8"))
    num_experts = int(os.environ.get("DEEPEP_TEST_NUM_EXPERTS", str((256 // num_ranks) * num_ranks)))

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_mnnvl=bool(int(os.environ.get("DEEPEP_TEST_ALLOW_MNNVL", "0"))))
    test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the communication group
    dist.barrier()
    dist.destroy_process_group()

# --------------------------------------------- layer -----------------------------------------------------


def forward_layer_naive(
    *,
    hidden_states,
    w13_weight_fp8,
    w2_weight_fp8,
    buffer,
    topk_idx,
    topk_weights,
    num_tokens,
    num_experts,
    num_local_experts,
):
    down_input, down_input_scale, comm_handle, expected_m, masked_m, num_groups, m = (
        forward_layer_naive_first_half(
            hidden_states=hidden_states, w13_weight_fp8=w13_weight_fp8,
            buffer=buffer, topk_idx=topk_idx, num_tokens=num_tokens, num_experts=num_experts
        )
    )

    # GroupGemm-1
    n = w2_weight_fp8[0].size(1)
    down_input_fp8 = (down_input, down_input_scale)
    down_output = torch.empty(
        (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
    )
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        down_input_fp8,
        w2_weight_fp8,
        down_output,
        masked_m,
        expected_m,
        recipe=(1, 128, 128),
    )

    combined_x, combine_event, combine_hook = buffer.low_latency_combine(
        down_output, topk_idx, topk_weights, comm_handle,
        return_recv_hook=True,
        async_finish=True, # NOTE
    )
    combine_event.current_stream_wait()
    large_gemm()
    combine_hook()

    return combined_x

def forward_layer_naive_first_half(
        *,
        hidden_states,
        w13_weight_fp8,
        buffer,
        topk_idx,
        num_tokens,
        num_experts,
):
    # src: EPMoE
    fp8_dtype = torch.float8_e4m3fn

    # src: dispatch_a
    expected_m = (hidden_states.shape[0] * buffer.group_size * topk_idx.shape[1] + num_experts) // num_experts

    hidden_states_fp8, recv_count, comm_handle, dispatch_event, dispatch_hook = buffer.low_latency_dispatch(
        hidden_states, topk_idx, num_tokens, num_experts,
        use_fp8=True, async_finish=False, return_recv_hook=True,
        round_scale=True, use_ue8m0=True,
    )
    assert dispatch_event.event is None
    large_gemm()
    dispatch_hook()

    masked_m = recv_count

    # GroupGemm-0
    num_groups, m, k = hidden_states_fp8[0].size()
    n = w13_weight_fp8[0].size(1)
    expected_m = min(expected_m, m)
    gateup_output = torch.empty(
        (num_groups, m, n), device=hidden_states_fp8[0].device, dtype=torch.bfloat16
    )
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        hidden_states_fp8,
        w13_weight_fp8,
        gateup_output,
        masked_m,
        expected_m,
        recipe=(1, 128, 128),
    )

    # Act
    down_input = torch.empty(
        (
            gateup_output.shape[0],
            gateup_output.shape[1],
            gateup_output.shape[2] // 2,
        ),
        device=gateup_output.device,
        dtype=fp8_dtype,
    )
    scale_block_size = 128
    down_input_scale = torch.empty(
        (
            gateup_output.shape[0],
            gateup_output.shape[1],
            gateup_output.shape[2] // 2 // scale_block_size,
        ),
        device=gateup_output.device,
        dtype=torch.float32,
    )
    silu_and_mul_masked_post_quant_fwd(
        gateup_output,
        down_input,
        down_input_scale,
        scale_block_size,
        masked_m,
        scale_ue8m0=True,
    )
    del gateup_output

    # ref: DeepGEMM dispatch.py
    down_input_scale = transform_sf_into_required_layout(
        down_input_scale,
        mn=down_input.shape[1], k=down_input.shape[2],
        recipe=(1, 128, 128),
        num_groups=num_groups, is_sfa=True,
    )

    return down_input, down_input_scale, comm_handle, expected_m, masked_m, num_groups, m

def forward_layer_overlap(
        *,
        hidden_states,
        w13_weight_fp8,
        w2_weight_fp8,
        buffer,
        topk_idx,
        topk_weights,
        num_tokens,
        num_experts,
        num_local_experts,
):
    down_input, down_input_scale, comm_handle, expected_m, masked_m, num_groups, m = (
        forward_layer_naive_first_half(
            hidden_states=hidden_states, w13_weight_fp8=w13_weight_fp8,
            buffer=buffer, topk_idx=topk_idx, num_tokens=num_tokens, num_experts=num_experts
        )
    )

    n = w2_weight_fp8[0].size(1)
    down_input_fp8 = (down_input, down_input_scale)
    down_output = torch.empty((num_groups, m, n), device=down_input.device, dtype=torch.bfloat16)

    src_signals = torch.zeros(num_local_experts, dtype=torch.uint32, device=down_input.device)

    print('hi call low_latency_combine', flush=True)
    combined_x, combine_event, combine_hook = buffer.low_latency_combine(
        down_output, topk_idx, topk_weights, comm_handle,
        return_recv_hook=True,
        async_finish=True, # NOTE
        src_signals=src_signals,
    )

    # NOTE need to change according to DeepEP src code
    # deepep_num_sms = 32
    # deepgemm_num_sms = torch.cuda.get_device_properties(device='cuda').multi_processor_count - deepep_num_sms
    deepgemm_num_sms = 64 # TODO temp

    for local_expert_idx in range(num_local_experts):
        print(f'hi call gemm {local_expert_idx=}', flush=True)
        with configure_deep_gemm_num_sms(deepgemm_num_sms):
            deep_gemm.fp8_m_grouped_gemm_nt_masked(
                _pick_expert_fp8(down_input_fp8, local_expert_idx=local_expert_idx),
                _pick_expert_fp8(w2_weight_fp8, local_expert_idx=local_expert_idx),
                _pick_expert(down_output, local_expert_idx=local_expert_idx),
                masked_m[local_expert_idx:local_expert_idx+1],
                expected_m,
                recipe=(1, 128, 128),
            )

        print(f'hi call notify_src_signals {local_expert_idx=}', flush=True)
        buffer.runtime.notify_src_signals(src_signals, local_expert_idx)

    print(f'hi call current_stream_wait', flush=True)
    combine_event.current_stream_wait()

    # hack
    print(f'hi call sync', flush=True)
    torch.cuda.synchronize()

    print(f'hi call large_gemm', flush=True)
    large_gemm()
    print(f'hi call combine_hook', flush=True)
    combine_hook()
    print(f'hi END', flush=True)

    return combined_x


def _pick_expert_fp8(a, local_expert_idx):
    return (
        _pick_expert(a[0], local_expert_idx),
        _pick_expert(a[1], local_expert_idx),
    )

def _pick_expert(a, local_expert_idx):
    return a[local_expert_idx:local_expert_idx + 1, :, :]


@contextmanager
def configure_deep_gemm_num_sms(num_sms):
    if num_sms is None:
        yield
    else:
        original_num_sms = deep_gemm.config.get_num_sms()
        deep_gemm.config.set_num_sms(num_sms)
        try:
            yield
        finally:
            deep_gemm.config.set_num_sms(original_num_sms)

# --------------------------------------------- SGLANG -----------------------------------------------------


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    num_processes = int(os.getenv("DEEPEP_TEST_NUM_PROCESSES", "8"))
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)

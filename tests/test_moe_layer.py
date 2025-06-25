import json
import os
import random
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
from functools import partial

import sglang.srt.distributed
sglang.srt.distributed.get_tensor_model_parallel_rank = lambda: torch.distributed.get_rank()
sglang.srt.distributed.get_tensor_model_parallel_world_size = lambda: torch.distributed.get_world_size()

import deep_gemm
from deep_gemm.utils.math import align, ceil_div, per_block_cast_to_fp8
import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back
from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_post_quant_fwd
from sglang.srt.layers.quantization.fp8_utils import _requant_weight_ue8m0
from deep_gemm.utils.layout import transform_sf_into_required_layout
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import requant_weight_ue8m0_inplace

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

    if bool(int(os.environ.get("DEEPEP_HACK_BACKGROUND_COPY_ENGINE", "0"))):
        copy_engine_tester = CopyEngineTester()
    else:
        copy_engine_tester = None

    if bool(int(os.environ.get("DEEPEP_HACK_INIT_DEEPGEMM_REDUCE_SM", "0"))):
        deepep_num_sms = 32
        deepgemm_num_sms = torch.cuda.get_device_properties(device='cuda').multi_processor_count - deepep_num_sms
        deep_gemm.config.set_num_sms(deepgemm_num_sms)
        print("HACK: change deepgemm num sms, BUT this may be overriden and useless!")

    layer = MyLayer(
        num_local_experts=num_local_experts, hidden=hidden,
    )

    # noinspection PyShadowingNames
    def execute_forward_layer(fn_mode: str):
        if copy_engine_tester is not None:
            copy_engine_tester()

        return layer.forward_layer(
            fn_mode=fn_mode,
            hidden_states=x,
            buffer=buffer,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens=num_tokens,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            num_ranks=num_ranks,
        )

    # correctness
    if bool(int(os.environ.get("DEEPEP_HACK_DO_CHECK", "1"))):
        out_overlap = execute_forward_layer("overlap").clone()
        out_naive = execute_forward_layer("naive").clone()
        diff = calc_diff(out_naive, out_overlap)
        print(f"Correctness test {diff=}", flush=True)
        print(f"{out_naive=} {out_overlap=}")
        assert diff < 1e-4, f"{diff=} {out_naive=} {out_overlap=}"
        raise Exception("deliberately stop")

    for fn_mode in [
        'naive',
        # 'overlap',
    ]:
        fn_with_mode = partial(execute_forward_layer, fn_mode=fn_mode)
        graph = capture_cuda_graph(fn_with_mode)

        if rank == 0:
            trace_path = str(Path("/data/numa0/tom/temp_sglang_server2local/") / f"{time.time()}-TP-{rank}.trace.json.gz")
        else:
            trace_path = None
        print(f"Execute bench {fn_mode=} {rank=} {trace_path=}", flush=True)
        bench_kineto(fn_with_mode,
                     kernel_names=('dispatch', 'combine'), barrier_comm_profiling=True,
                     suppress_kineto_output=False,  # NOTE MODIFIED
                     trace_path=trace_path)
        print("Execute bench end", flush=True)


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

class MyLayer(torch.nn.Module):
    def __init__(self, *, num_local_experts, hidden):
        super().__init__()
        self._hack_dispatch_fake_overlap_momento = None
        self.w13_weight_fp8 = create_weight_fp8(num_groups=num_local_experts, n=4096, k=hidden)
        self.w2_weight_fp8 = create_weight_fp8(num_groups=num_local_experts, n=hidden, k=2048)
        self.hack_stream = torch.cuda.Stream()

        quant_config = Fp8Config(**{
            'is_checkpoint_fp8_serialized': True,
            'activation_scheme': 'dynamic',
            'ignored_layers': [],
            'weight_block_size': [128, 128],
        })


        self.shared_experts = DeepseekV2MLP(
            hidden_size=hidden,
            # intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            intermediate_size=2048 * 1,
            # hidden_act=config.hidden_act,
            hidden_act="silu",
            quant_config=quant_config,
            reduce_results=False,
            # prefix=add_prefix("shared_experts", prefix),
            tp_rank=0, tp_size=1,
        )

        mlp = self.shared_experts
        weight_block_size = quant_config.weight_block_size
        for module in [
            mlp.gate_up_proj,
            mlp.down_proj,
        ]:
            requant_weight_ue8m0_inplace(
                module.weight, module.weight_scale_inv, weight_block_size
            )

    def forward_layer(self, fn_mode: str, **kwargs):
        if fn_mode == 'naive':
            f = self.forward_layer_naive
        elif fn_mode == 'overlap':
            f = self.forward_layer_overlap
        else:
            raise NotImplementedError
        return f(**kwargs)

    def forward_layer_naive(
        self,
        *,
        hidden_states,
        buffer,
        topk_idx,
        topk_weights,
        num_tokens,
        num_experts,
        num_local_experts,
        num_ranks,
    ):
        down_input, down_input_scale, comm_handle, expected_m, masked_m, num_groups, m = (
            self.forward_layer_naive_first_half(
                hidden_states=hidden_states,
                buffer=buffer, topk_idx=topk_idx, num_tokens=num_tokens, num_experts=num_experts,
            )
        )

        # GroupGemm-1
        n = self.w2_weight_fp8[0].size(1)
        down_input_fp8 = (down_input, down_input_scale)
        down_output = torch.empty(
            (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
        )

        # # NOTE need to change according to DeepEP src code
        # deepep_num_sms = 32
        # deepgemm_num_sms = torch.cuda.get_device_properties(device='cuda').multi_processor_count - deepep_num_sms
        #
        # print("HACK: put deepgemm in another stream (logically wrong)")
        # hack_stream.wait_stream(torch.cuda.current_stream())
        # with torch.cuda.stream(hack_stream):
        #     with configure_deep_gemm_num_sms(deepgemm_num_sms):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            down_input_fp8,
            self.w2_weight_fp8,
            down_output,
            masked_m,
            expected_m,
            recipe=(1, 128, 128),
        )

        combined_x, combine_event, combine_hook = buffer.low_latency_combine(
            down_output, topk_idx, topk_weights, comm_handle,
            return_recv_hook=True,
            # async_finish=True, # NOTE
        )

        shared_experts_output = self.shared_experts(hidden_states)

        assert combine_event.event is None
        # combine_event.current_stream_wait()

        large_gemm()
        combine_hook()

        # print(f"hi forward_layer_naive {combined_x=}")
        return combined_x, shared_experts_output


    def forward_layer_naive_first_half(
            self,
            *,
            hidden_states,
            buffer,
            topk_idx,
            num_tokens,
            num_experts,
    ):
        hack_dispatch_fake_overlap = bool(int(os.environ.get("DEEPEP_HACK_DISPATCH_FAKE_OVERLAP", "0")))

        # src: EPMoE
        fp8_dtype = torch.float8_e4m3fn

        # src: dispatch_a
        expected_m = (hidden_states.shape[0] * buffer.group_size * topk_idx.shape[1] + num_experts) // num_experts

        enable_hack_disptach_fake_overlap_curr_iter = hack_dispatch_fake_overlap and (self._hack_dispatch_fake_overlap_momento is not None)
        if enable_hack_disptach_fake_overlap_curr_iter:
            deepep_num_sms = 32
            deepgemm_num_sms = torch.cuda.get_device_properties(device='cuda').multi_processor_count - deepep_num_sms

            print("HACK: put deepgemm in another stream (logically wrong)")
            self.hack_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.hack_stream):
                with configure_deep_gemm_num_sms(deepgemm_num_sms):
                    deep_gemm.fp8_m_grouped_gemm_nt_masked(**self._hack_dispatch_fake_overlap_momento["gemm_kwargs"])

        hidden_states_fp8, recv_count, comm_handle, dispatch_event, dispatch_hook = buffer.low_latency_dispatch(
            hidden_states, topk_idx, num_tokens, num_experts,
            use_fp8=True, async_finish=False, return_recv_hook=True,
            round_scale=True, use_ue8m0=True,
        )
        assert dispatch_event.event is None

        if enable_hack_disptach_fake_overlap_curr_iter:
            torch.cuda.current_stream().wait_stream(self.hack_stream)

        large_gemm()
        dispatch_hook()

        masked_m = recv_count

        # GroupGemm-0
        num_groups, m, k = hidden_states_fp8[0].size()
        n = self.w13_weight_fp8[0].size(1)
        expected_m = min(expected_m, m)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_fp8[0].device, dtype=torch.bfloat16
        )

        if not enable_hack_disptach_fake_overlap_curr_iter:
            deep_gemm.fp8_m_grouped_gemm_nt_masked(
                hidden_states_fp8,
                self.w13_weight_fp8,
                gateup_output,
                masked_m,
                expected_m,
                recipe=(1, 128, 128),
            )

            if hack_dispatch_fake_overlap:
                self._hack_dispatch_fake_overlap_momento = {
                    "gemm_kwargs": {
                        "a": (hidden_states_fp8[0].clone(), hidden_states_fp8[1].clone()),
                        "b": self.w13_weight_fp8,
                        "d": gateup_output.clone(),
                        "masked_m": masked_m.clone(),
                        "expected_m": expected_m,
                        "recipe": (1, 128, 128),
                    },
                }

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
            self,
            *,
            hidden_states,
            buffer,
            topk_idx,
            topk_weights,
            num_tokens,
            num_experts,
            num_local_experts,
            num_ranks,
    ):
        # # ------------------------------------
        # print("hi prepare deepgemm_kwargs")
        # num_groups, m, k, n = 6, 1024, 2048, 7168
        # a, b, d, ref_d = deepgemm_generate_grouped_masked(num_groups, m, k, n)
        # masked_m = torch.ones((num_groups, ), device='cuda', dtype=torch.int) * m
        # expected_m = min(int(masked_m.float().mean()) + 1, m)
        # deepgemm_kwargs = dict(a=a, b=b, d=d, masked_m=masked_m, expected_m=expected_m)
        # deepgemm_stream = torch.cuda.Stream()
        # # ------------------------------------

        down_input, down_input_scale, comm_handle, expected_m, masked_m, num_groups, m = (
            self.forward_layer_naive_first_half(
                hidden_states=hidden_states,
                buffer=buffer, topk_idx=topk_idx, num_tokens=num_tokens, num_experts=num_experts,
            )
        )

        n = self.w2_weight_fp8[0].size(1)
        down_input_fp8 = (down_input, down_input_scale)
        down_output = torch.empty((num_groups, m, n), device=down_input.device, dtype=torch.bfloat16)

        # NOTE need to change according to DeepEP src code
        # TODO if DeepGEMM does not use all SM then let DeepEP use more
        deepep_num_sms = 32
        deepgemm_num_sms_upper_bound = torch.cuda.get_device_properties(device='cuda').multi_processor_count - deepep_num_sms
        actual_deepgemm_num_sms = {
            # temp hardcoded
            # 4: 120, 48: 116, # normal
            4: 119, 48: 120, # with the "specially treat first expert"
        }[num_ranks]

        down_output_signals = torch.zeros((num_local_experts,), dtype=torch.uint32, device=down_input.device)

        expert_slice = slice(0, 1)
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            _pick_expert_fp8(down_input_fp8, expert_slice),
            _pick_expert_fp8(self.w2_weight_fp8, expert_slice),
            down_output[expert_slice, :, :],
            masked_m[expert_slice],
            expected_m,
            recipe=(1, 128, 128),
        )

        self.hack_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.hack_stream):
            with configure_deep_gemm_num_sms(deepgemm_num_sms_upper_bound):
                # for local_expert_idx in range(num_local_experts):
                #     deep_gemm.fp8_m_grouped_gemm_nt_masked(
                #         _pick_expert_fp8(down_input_fp8, local_expert_idx=local_expert_idx),
                #         _pick_expert_fp8(w2_weight_fp8, local_expert_idx=local_expert_idx),
                #         _pick_expert(down_output, local_expert_idx=local_expert_idx),
                #         masked_m[local_expert_idx:local_expert_idx+1],
                #         expected_m,
                #         recipe=(1, 128, 128),
                #     )
                #     buffer.runtime.notify_src_signals(down_output_signals, local_expert_idx)

                # print("hi call fp8_m_grouped_gemm_nt_masked", flush=True)
                # deepgemm_out = deep_gemm.fp8_m_grouped_gemm_nt_masked(
                #     down_input_fp8, w2_weight_fp8, down_output, masked_m, expected_m, recipe=(1, 128, 128),
                #     d_signals=down_output_signals,
                # )
                # actual_deepgemm_num_sms = deepgemm_out["num_sms"]

                expert_slice = slice(1, num_local_experts)
                deepgemm_out = deep_gemm.fp8_m_grouped_gemm_nt_masked(
                    _pick_expert_fp8(down_input_fp8, expert_slice),
                    _pick_expert_fp8(self.w2_weight_fp8, expert_slice),
                    down_output[expert_slice, :, :],
                    masked_m[expert_slice],
                    expected_m,
                    recipe=(1, 128, 128),
                    d_signals=down_output_signals[expert_slice],
                )
                assert deepgemm_out["num_sms"] == actual_deepgemm_num_sms, f"{deepgemm_out=} {actual_deepgemm_num_sms=}"

                shared_experts_output = self.shared_experts(hidden_states)

        # sometimes DeepGEMM choose to use *LESS* sms, we need to consider this
        src_signal_expect_value = actual_deepgemm_num_sms
        # print(f"{deepgemm_num_sms_upper_bound=} {actual_deepgemm_num_sms=}", flush=True)

        # print('hi call low_latency_combine', flush=True)
        combined_x, combine_event, combine_hook = buffer.low_latency_combine(
            down_output, topk_idx, topk_weights, comm_handle,
            return_recv_hook=True,
            # async_finish=True, # NOTE
            src_signals=down_output_signals,
            src_signal_expect_value=src_signal_expect_value,
        )

        # # ------------------------------------
        # deepgemm_num_sms_upper_bound = 30  # very small
        # with torch.cuda.stream(deepgemm_stream):
        #     with configure_deep_gemm_num_sms(deepgemm_num_sms_upper_bound):
        #         for _ in range(20):
        #             print("hi call fp8_m_grouped_gemm_nt_masked")
        #             deep_gemm.fp8_m_grouped_gemm_nt_masked(**deepgemm_kwargs)
        # raise Exception
        # # ------------------------------------

        # print(f'hi after a while {down_output_signals=}', flush=True)

        assert combine_event.event is None
        # combine_event.current_stream_wait()

        # hack
        # print(f'hi call sync', flush=True)
        # torch.cuda.synchronize()

        torch.cuda.current_stream().wait_stream(self.hack_stream)

        # print(f'hi call large_gemm', flush=True)
        large_gemm()
        # print(f'hi call combine_hook', flush=True)
        combine_hook()
        # print(f'hi END', flush=True)

        if 0:
            assert torch.all(down_output_signals == src_signal_expect_value), f"{down_output_signals=} {src_signal_expect_value=}"

        # print(f"hi forward_layer_overlap {combined_x=}")
        return combined_x, shared_experts_output


def _pick_expert_fp8(a, expert_slice):
    return a[0][expert_slice, :, :], a[1][expert_slice, :, :]


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


# COPIED from deepgemm
def deepgemm_generate_grouped_masked(num_groups: int, m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    from deep_gemm.utils.math import align, ceil_div, per_token_cast_to_fp8, per_block_cast_to_fp8

    a = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
    ref_d = torch.einsum('gmk,gnk->gmn', a, b)

    a_fp8 = (torch.empty_like(a, dtype=torch.float8_e4m3fn), torch.empty((num_groups, m, ceil_div(k, 128)), device='cuda', dtype=torch.float))
    b_fp8 = (torch.empty_like(b, dtype=torch.float8_e4m3fn), torch.empty((num_groups, ceil_div(n, 128), ceil_div(k, 128)), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        a_fp8[0][i], a_fp8[1][i] = per_token_cast_to_fp8(a[i])
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i])

    return a_fp8, b_fp8, d, ref_d

class CopyEngineTester:
    def __init__(self):
        self.alt_stream = torch.cuda.Stream()

        device_count = torch.cuda.device_count()
        assert device_count == 4

        src_device = torch.cuda.current_device()
        dst_device = (src_device + 1) % device_count

        size = 8 * 1024 ** 3

        self.src = torch.full((size,), 42, dtype=torch.uint8, device=f'cuda:{src_device}')
        self.dst = torch.full((size,), 42, dtype=torch.uint8, device=f'cuda:{dst_device}')


    def __call__(self):
        num_iter = 10

        self.alt_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.alt_stream):
            for i in range(num_iter):
                self.dst.copy_(self.src, non_blocking=True)


# ref: CUDAGraphRunner, https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
def capture_cuda_graph(run_once):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            run_once()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run_once()

    return graph


# --------------------------------------------- SGLANG -----------------------------------------------------


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    num_processes = int(os.getenv("DEEPEP_TEST_NUM_PROCESSES", "8"))
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)

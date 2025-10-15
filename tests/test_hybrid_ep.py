# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
import argparse
import time
import torch
import torch.distributed as dist
import os
import deep_ep

from utils import TorchRef, bench, bench_kineto, init_dist

HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 7168))
MAX_NUM_OF_TOKENS_PER_RANK = int(os.environ.get("MAX_NUM_OF_TOKENS_PER_RANK", 4096))
# NUM_TOKENS_PER_RANK should equal or less than MAX_NUM_OF_TOKENS_PER_RANK
NUM_TOKENS_PER_RANK = int(os.environ.get("NUM_TOKENS_PER_RANK", 4096))
NUM_LOCAL_EXPERTS = int(os.environ.get("NUM_LOCAL_EXPERTS", 8))
NUM_OF_RANKS_PER_NODE = int(os.environ.get("NUM_OF_RANKS_PER_NODE", 4))
TOPK = int(os.environ.get("TOPK", 8))
PAD_MULTIPLE = int(os.environ.get("PAD_MULTIPLE", 32))
NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE
ITERATIONS = int(os.environ.get("ITERATIONS", 100))
SEED = int(os.environ.get("SEED", 42))
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    topk_idx = torch.zeros(seq_len, topk, device="cuda", dtype=torch.int64)
    topk_weights = torch.zeros(seq_len, topk, device="cuda", dtype=torch.float32)
    scaling_factor = torch.randn(
        seq_len, hidden_dim // 128, device="cuda", dtype=torch.float32
    )

    routing_map = torch.zeros(seq_len, num_of_experts, device="cuda", dtype=torch.bool)

    for i in range(seq_len):
        selected_experts = torch.randperm(num_of_experts, device="cuda")[:topk]
        topk_idx[i, :] = selected_experts.to(torch.int64)
        topk_weights[i, :] = torch.rand(topk, device="cuda", dtype=torch.float32)
        # Force balanced routing for testing
        # selected_experts = [
        #     ((i * topk) % num_of_experts + val) % num_of_experts for val in range(topk)
        # ]
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

    # Dispatch correctness check
    for with_probs in [True, False]:
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
            hidden=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights if with_probs else None, num_of_experts=NUM_OF_EXPERTS,
        )
        assert bitwise_equal(dispatched_hidden_ref, dispatched_hidden)
        if dispatched_probs is not None and dispatched_probs_ref is not None:
            start, end = ref._local_expert_range()
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
        num_dispatched_tokens = num_dispatched_tokens.cpu()
        local_expert_routing_map = local_expert_routing_map[
            : num_dispatched_tokens.item()
        ]
        # Simulate the permute and expert and unpermute. The expert is identity op
        copy_times = local_expert_routing_map.sum(dim=1)
        dispatched_hidden = dispatched_hidden.to(torch.bfloat16)  
        # The combine only support bf16
        hidden_to_combine = dispatched_hidden * copy_times.unsqueeze(1)
        probs_to_combine = dispatched_probs

        # The check for the combine
        combined_hidden, combined_probs = buffer.combine(
            hidden_to_combine, probs_to_combine, handle
        )

        # The reconstucted value should be TOPK times larger than the input hidden
        combined_hidden = combined_hidden / TOPK

        assert torch.allclose(
            combined_hidden, hidden.to(torch.bfloat16), atol=2e-5, rtol=1e-2
        )
        if combined_probs is not None and probs is not None:
            assert torch.allclose(combined_probs, probs, atol=2e-5, rtol=1e-2)

    # Dispatch with permute correctness check
    for with_probs in [True, False]:
        # The check for the dispatch
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
        )
        _, _, _, num_dispatched_tokens_tensor, local_expert_routing_map, _, _, _ = (
            handle
        )
        num_dispatched_tokens_tensor = num_dispatched_tokens_tensor.cpu()
        local_expert_routing_map = local_expert_routing_map[
            : num_dispatched_tokens_tensor.item()
        ]
        # The out_token_num of permutation is the sum of the tokens_per_expert
        out_token_num = tokens_per_expert.sum().item()
        (
            dispatched_hidden_ref,
            dispatched_probs_ref,
            dispatched_scaling_factor_ref,
        ) = ref.dispatch(
            hidden,
            routing_map,
            probs if with_probs else None,
            scaling_factor,
            local_expert_routing_map=local_expert_routing_map,
            out_token_num=out_token_num,
            pad_multiple=PAD_MULTIPLE,
            enable_permute=True,
        )

        assert bitwise_equal(dispatched_hidden_ref, dispatched_hidden)
        if dispatched_probs is not None and dispatched_probs_ref is not None:
            assert bitwise_equal(dispatched_probs_ref, dispatched_probs)
        if (
            dispatched_scaling_factor is not None
            and dispatched_scaling_factor_ref is not None
        ):
            assert bitwise_equal(
                dispatched_scaling_factor_ref, dispatched_scaling_factor
            )

        # Simulate the permute and expert and unpermute. The expert is identity op
        copy_times = local_expert_routing_map.sum(dim=1)
        # The combine only support bf16
        dispatched_hidden = dispatched_hidden.to(torch.bfloat16)  
        hidden_to_combine = dispatched_hidden
        probs_to_combine = dispatched_probs

        # The check for the combine
        combined_hidden, combined_probs = buffer.combine_with_unpermute(
            hidden=hidden_to_combine,
            probs=probs_to_combine,
            handle=handle,
            num_dispatched_tokens=num_dispatched_tokens,
            pad_multiple=PAD_MULTIPLE,
        )

        # The reconstucted value should be TOPK times larger than the input hidden
        combined_hidden = combined_hidden / TOPK

        assert torch.allclose(
            combined_hidden, hidden.to(torch.bfloat16), atol=2e-5, rtol=1e-2
        )
        if combined_probs is not None and probs is not None:
            assert torch.allclose(combined_probs, probs, atol=2e-5, rtol=1e-2)

    print(f'[rank {torch.distributed.get_rank()}] Correctness check passed ({"FP8" if hidden.dtype == torch.uint8 else "BF16"})')


def test_hybrid_ep_benchmark(buffer: deep_ep.HybridEPBuffer, group: dist.ProcessGroup, use_fp8: bool, nsys_profile: bool):
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

    rank = torch.distributed.get_rank()
    fp8_factor = (1 + 4 / 128) / 2
    dispatch_bf16_nvl_recv_bytes = dispatched_hidden.numel() * 2
    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

    dispatch_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'topk_idx': topk_idx, 'topk_weights': topk_weights, 'num_of_experts': NUM_OF_EXPERTS}
    t = bench(lambda: buffer.dispatch(**dispatch_args))[0]
    nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if hidden.dtype == torch.uint8 else dispatch_bf16_nvl_recv_bytes
    print(f'[rank {rank}] HybridEP dispatch torch API ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): '
            f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, nvl_recv_bytes: {nvl_recv_bytes / 1e6:.2f} MB', flush=True)

    dispatched_hidden, dispatched_probs, _, handle= (
        buffer.dispatch(hidden=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights, num_of_experts=NUM_OF_EXPERTS)
    )
    dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)
    combine_args = {'hidden': dispatched_hidden_bf16, 'probs': dispatched_probs, 'handle': handle}
    t = bench(lambda: buffer.combine(**combine_args))[0]
    print(f'[rank {rank}] HybridEP combine torch API: '
            f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, combine_send_bytes: {combine_bf16_nvl_send_bytes / 1e6:.2f} MB', flush=True)

    '''
    Benchmark of the dispatch and combine with permute extension
    '''
    dispatch_with_permute_args = {'hidden': hidden, 'scaling_factor': scaling_factor, 'routing_map': routing_map, 'probs': probs, 'pad_multiple': PAD_MULTIPLE}
    t = bench(lambda: buffer.dispatch_with_permute(**dispatch_with_permute_args))[0]
    nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if hidden.dtype == torch.uint8 else dispatch_bf16_nvl_recv_bytes
    print(f'[rank {rank}] HybridEP dispatch+permute torch API ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): '
            f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, nvl_recv_bytes: {nvl_recv_bytes / 1e6:.2f} MB', flush=True)

    dispatched_hidden, dispatched_probs, _, _, handle= (
        buffer.dispatch_with_permute(hidden=hidden, scaling_factor=scaling_factor, routing_map=routing_map, probs=probs, pad_multiple=PAD_MULTIPLE)
    )
    dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)
    combine_with_unpermute_args = {'hidden': dispatched_hidden_bf16, 'probs': dispatched_probs, 'handle': handle, 'pad_multiple': PAD_MULTIPLE}
    t = bench(lambda: buffer.combine_with_unpermute(**combine_with_unpermute_args))[0]
    print(f'[rank {rank}] HybridEP combine+unpermute torch API: '
            f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, combine_send_bytes: {combine_bf16_nvl_send_bytes / 1e6:.2f} MB', flush=True)
            

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
        print(f'[rank {rank}] HybridEP dispatch kernel ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): {nvl_recv_bytes / 1e9 / dispatch_t:.2f} GB/s, avg_t={dispatch_t * 1e6:.2f} us | '
              f'HybridEP combine kernel: {combine_bf16_nvl_send_bytes / 1e9 / combine_t:.2f} GB/s, avg_t={combine_t * 1e6:.2f} us', flush=True)
    else:
        torch.cuda.profiler.start()
        with torch.cuda.nvtx.range(f"hybrid-ep dispatch ({"FP8" if hidden.dtype == torch.uint8 else "BF16"})"):
            if rank == 0:
                print(f"profile hybrid-ep dispatch ({"FP8" if hidden.dtype == torch.uint8 else "BF16"})", flush=True)
            dispatch_args = {'tensor': hidden, 'scaling_factor': scaling_factor, 'topk_idx': topk_idx, 'topk_weights': topk_weights, 'num_of_experts': NUM_OF_EXPERTS}
            bench(lambda: buffer.dispatch(**dispatch_args))
        with torch.cuda.nvtx.range("hybrid-ep combine"):
            if rank == 0:
                print(f"profile hybrid-ep combine", flush=True)
            combine_args = {'tensor': dispatched_hidden_bf16, 'probs': dispatched_probs, 'handle': handle}
            bench(lambda: buffer.combine(**combine_args))
        time.sleep(1)
        torch.cuda.profiler.stop()


def test_main(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    try:
        for use_fp8 in [True, False]:
            buffer = deep_ep.HybridEPBuffer(
                group=group,
                hidden_dim=HIDDEN_DIM,
                max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
                num_local_experts=NUM_LOCAL_EXPERTS,
                use_fp8=use_fp8,
            )
            
            ref = TorchRef(
                ep_group=group,
                num_of_experts=NUM_OF_EXPERTS,
                num_of_ranks_per_node=NUM_OF_RANKS_PER_NODE,
            )

            test_hybrid_ep_correctness(buffer, ref, use_fp8)
            test_hybrid_ep_benchmark(buffer, group, use_fp8, args.nsys_profile)

    finally:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')
    parser.add_argument('--num-processes', type=int, default=4,
                       help='Number of processes to spawn (default: 4)')
    parser.add_argument('--nsys-profile', action='store_true', default=False,
                       help='benchmark with nsys profile or not (default: False)')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_main, args=(args.num_processes, args), nprocs=args.num_processes)

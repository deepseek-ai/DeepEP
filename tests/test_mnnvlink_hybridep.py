import argparse
import time
import torch
import torch.distributed as dist
import os
import deep_ep

from utils import TorchRef, bench, bench_kineto

HIDDEN_DIM = 7168
MAX_NUM_OF_TOKENS_PER_RANK = 4096
# NUM_TOKENS_PER_RANK should equal or less than MAX_NUM_OF_TOKENS_PER_RANK
NUM_TOKENS_PER_RANK = 4096
NUM_LOCAL_EXPERTS = 8
NUM_OF_RANKS_PER_NODE = 32
TOPK = 8
NUM_OF_EXPERTS = NUM_LOCAL_EXPERTS * NUM_OF_RANKS_PER_NODE
ITERATIONS = 100
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    # local_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))

    # Call the init process.
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))

    dist.init_process_group(
        backend="nccl",
        init_method=f'tcp://{ip}:{port}',
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank,
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


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
        # selected_experts = [0,8,16,24,32,40,48,56] # force balanced routing for testing
        routing_map[i, selected_experts] = True
        probs[i, selected_experts] = topk_weights[i, :]

    return hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights


def test_intra_node_correctness(buffer: deep_ep.HybridEpBuffer, ref: TorchRef, use_fp8: bool):
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
            num_of_tokens_for_experts,
            local_expert_routing_map,
            handle,
        ) = buffer.dispatch(
            tensor=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights if with_probs else None
        )

        assert torch.allclose(dispatched_hidden_ref, dispatched_hidden)
        if dispatched_probs is not None and dispatched_probs_ref is not None:
            start, end = ref._local_expert_range()
            masked_probs = torch.zeros_like(dispatched_probs)
            masked_probs[:, start:end] = dispatched_probs[:, start:end]
            assert torch.allclose(dispatched_probs_ref, dispatched_probs[:, start:end])
            dispatched_probs = masked_probs
        if (
            dispatched_scaling_factor is not None
            and dispatched_scaling_factor_ref is not None
        ):
            assert torch.allclose(
                dispatched_scaling_factor_ref, dispatched_scaling_factor
            )

        # expert the local routing map from the local routing map
        num_of_tokens_for_experts = num_of_tokens_for_experts.cpu()
        local_expert_routing_map = local_expert_routing_map[
            : num_of_tokens_for_experts.item()
        ]
        # Simulate the permute and expert and unpermute. The expert is identity op
        copy_times = local_expert_routing_map.sum(dim=1)
        dispatched_hidden = dispatched_hidden.to(
            torch.bfloat16
        )  # The combine only support bf16
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

    if torch.distributed.get_rank() == 0:
        print("Correctness check passed")


def test_intra_node_benchmark(buffer: deep_ep.HybridEpBuffer, group: dist.ProcessGroup, use_fp8: bool, nsys_profile: bool):
    hidden, probs, scaling_factor, routing_map, topk_idx, topk_weights = init_tensor(
        hidden_dim=HIDDEN_DIM,
        seq_len=NUM_TOKENS_PER_RANK,
        topk=TOPK,
        num_of_experts=NUM_OF_EXPERTS,
        use_fp8=use_fp8,
    )

    # warmup
    for _ in range(10):
        dispatched_hidden, dispatched_probs, _, _, _, handle = (
            buffer.dispatch(tensor=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights)
        )
        if dispatched_hidden.dtype == torch.uint8:
            dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)
        else:
            dispatched_hidden_bf16 = dispatched_hidden
        dispatched_probs = None
        _, _ = buffer.combine(dispatched_hidden_bf16, dispatched_probs, handle)

    rank = torch.distributed.get_rank()
    fp8_factor = (1 + 4 / 128) / 2
    dispatch_bf16_nvl_recv_bytes = dispatched_hidden.numel() * 2
    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

    dispatch_args = {'tensor': hidden, 'scaling_factor': scaling_factor, 'topk_idx': topk_idx, 'topk_weights': topk_weights}
    t = bench(lambda: buffer.dispatch(**dispatch_args))[0]
    nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if hidden.dtype == torch.uint8 else dispatch_bf16_nvl_recv_bytes
    print(f'[rank {rank}] HybridEP dispatch torch API ({"FP8" if hidden.dtype == torch.uint8 else "BF16"}): '
            f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, nvl_recv_bytes: {nvl_recv_bytes / 1e6:.2f} MB', flush=True)

    dispatched_hidden, dispatched_probs, _, _, _, handle= (
        buffer.dispatch(tensor=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights)
    )
    combine_args = {'tensor': dispatched_hidden_bf16, 'probs': dispatched_probs, 'handle': handle}
    t = bench(lambda: buffer.combine(**combine_args))[0]
    print(f'[rank {rank}] HybridEP combine torch API: '
            f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), t: {t * 1e6:.2f} us, combine_send_bytes: {combine_bf16_nvl_send_bytes / 1e6:.2f} MB', flush=True)


    if not nsys_profile:
        # noinspection PyShadowingNames
        def test_func():
            dispatched_hidden, dispatched_probs, _, _, _, handle = (
                buffer.dispatch(tensor=hidden, scaling_factor=scaling_factor, topk_idx=topk_idx, topk_weights=topk_weights)
            )
            if dispatched_hidden.dtype == torch.uint8:
                dispatched_hidden_bf16 = dispatched_hidden.to(torch.bfloat16)
            else:
                dispatched_hidden_bf16 = dispatched_hidden
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
            dispatch_args = {'tensor': hidden, 'scaling_factor': scaling_factor, 'topk_idx': topk_idx, 'topk_weights': topk_weights}
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
    buffer = deep_ep.HybridEpBuffer(
        group=group,
        hidden_dim=HIDDEN_DIM,
        max_num_of_tokens_per_rank=MAX_NUM_OF_TOKENS_PER_RANK,
        num_local_experts=NUM_LOCAL_EXPERTS,
        num_of_experts=NUM_OF_EXPERTS,
        use_fp8=args.use_fp8,
        num_of_ranks_per_node=NUM_OF_RANKS_PER_NODE,
    )

    # Initialize the torchRef
    ref = TorchRef(
        ep_group=group,
        num_of_experts=NUM_OF_EXPERTS,
        num_of_ranks_per_node=NUM_OF_RANKS_PER_NODE,
    )

    # Test body
    test_intra_node_correctness(buffer, ref, args.use_fp8)
    test_intra_node_benchmark(buffer, group, args.use_fp8, args.nsys_profile)

    # Destroy
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')
    parser.add_argument('--num-processes', type=int, default=4,
                       help='Number of processes to spawn (default: 4)')
    parser.add_argument('--use-fp8', action='store_true', default=False,
                       help='Use fp8 in dispatch or not (default: False)')
    parser.add_argument('--nsys-profile', action='store_true', default=False,
                       help='benchmark with nsys profile or not (default: False)')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_main, args=(args.num_processes, args), nprocs=args.num_processes)

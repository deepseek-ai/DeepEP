import os
import sys

# Must pop before PyTorch imports
os.environ.pop('NCCL_MIN_NCHANNELS', None)
os.environ.pop('NCCL_MAX_NCHANNELS', None)
os.environ.pop('NCCL_NVLS_ENABLE', None)

import argparse
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.math import align
from deep_ep.utils.testing import bench_kineto, parse_num_bytes


def import_baseline():
    stochastic_round_bf16 = None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        stochastic_round_bf16 = tilelang_ops.stochastic_round_bf16
    except Exception as ex:
        dist_print(f'Failed to load bf16 stochastic rounding helper: {ex}, test does not guarantee bitwise alignment', once_in_node=True)
        dist_print(once_in_node=True)
    return stochastic_round_bf16


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_rdma_ranks, num_nvl_ranks = deep_ep.comm.get_physical_domain_size(group)
    assert num_rdma_ranks * num_nvl_ranks == num_ranks

    # Allocate
    allocator = deep_ep.BufferAllocator()
    src_0 = allocator.allocate((num_ranks * 4096,), torch.float32)
    src_1 = allocator.allocate((num_ranks * (3 * 1024 + 8),), torch.float32)
    perf_src = allocator.allocate((align(args.num_bytes // 4, num_ranks * 8),), torch.float32)
    buffer = deep_ep.BucketBuffer(group, allocator, explicitly_destroy=True)
    num_sms = buffer.get_theoretical_num_sms('reduce_scatter', group) if args.num_sms is None else args.num_sms

    # Print configs
    dist_print(f'Config:\n'
               f' > Ranks: {num_rdma_ranks} x {num_nvl_ranks}\n'
               f' > #SM: {num_sms}\n'
               f' > Communication precision: {args.comm_precision}\n',
               once_in_node=True)

    use_bf16 = num_rdma_ranks > 1 and num_nvl_ranks == 1 and args.comm_precision == 'bf16'
    stochastic_round_bf16 = import_baseline() if use_bf16 else None

    # Check correctness
    scale = args.scale
    reference_inputs = []

    for bucket_idx, src in enumerate((src_0, src_1)):
        torch.manual_seed(rank_idx * 17 + bucket_idx)
        src.normal_()
        reference_input = src.clone()
        if use_bf16 and stochastic_round_bf16 is not None:
            reference_input = stochastic_round_bf16(src, rank_idx, num_ranks)
        reference_inputs.append(reference_input)

    handle = buffer.reduce_scatter([src_0, src_1], num_sms=num_sms, group=group, scale=scale, comm_precision=args.comm_precision)
    dsts = handle.wait()
    assert 0 < handle.num_sms <= torch.cuda.get_device_properties('cuda').multi_processor_count
    for src, dst, reference_input in zip((src_0, src_1), dsts, reference_inputs, strict=True):
        assert dst.data_ptr() == src.data_ptr() + rank_idx * dst.nbytes

        reference = torch.zeros(src.numel() // num_ranks, dtype=torch.float32, device='cuda')
        if num_nvl_ranks > 1:
            dist.reduce_scatter_single(reference, reference_input, group=group)
        else:
            reference_input_all_to_all = torch.empty_like(reference_input)
            torch.distributed.all_to_all_single(reference_input_all_to_all, reference_input)
            for i in range(num_rdma_ranks):
                peer = (rank_idx - i) % num_rdma_ranks
                reference.add_(reference_input_all_to_all[peer * reference.numel(): (peer + 1) * reference.numel()])
        reference.mul_(scale)

        if num_nvl_ranks > 1:
            torch.testing.assert_close(dst, reference, rtol=1e-5, atol=1e-5)
        elif not use_bf16 or stochastic_round_bf16 is not None:
            torch.testing.assert_close(dst, reference, rtol=0, atol=0)
        else:
            torch.testing.assert_close(dst, reference, rtol=1e-1, atol=1e-1) # just a loose check, use tilelang for bitwise alignment for strict correctness

    # Perf NCCL
    torch.manual_seed(rank_idx)
    original = torch.randn_like(perf_src)
    nccl_input = original.clone()
    nccl_output = torch.empty(perf_src.numel() // num_ranks, dtype=torch.float32, device='cuda')
    nccl_duration = bench_kineto(
        lambda: dist.reduce_scatter_single(nccl_output, nccl_input, group=group),
        kernel_names='ncclDevKernel_ReduceScatter', barrier_comm_profiling=True,
        barrier=buffer.barrier)

    # Perf ours
    dist_print('Performance:', once_in_node=True)
    duration = bench_kineto(
        lambda: buffer.reduce_scatter(
            perf_src, num_sms=num_sms, group=group, scale=scale, comm_precision=args.comm_precision).wait(),
        kernel_names='reduce_scatter', barrier_comm_profiling=True,
        barrier=buffer.barrier)
    dist_print(
        f' > Perf ({num_sms=}): '
        f'{duration * 1e6:7.1f} us | '
        f'{perf_src.nbytes / duration / 1e9:5.1f} GB/s | '
        f'{nccl_duration / duration:.2f}x NCCL',
        once_in_node=True)

    # Destroy
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test reduce-scatter')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-bytes', type=parse_num_bytes, default=1 << 30)
    parser.add_argument('--num-sms', type=int)
    parser.add_argument('--comm-precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

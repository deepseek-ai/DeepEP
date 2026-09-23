import os

# Must pop before PyTorch imports
os.environ.pop('NCCL_MIN_NCHANNELS', None)
os.environ.pop('NCCL_MAX_NCHANNELS', None)
os.environ.pop('NCCL_NVLS_ENABLE', None)

import argparse
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.testing import bench_kineto, parse_num_bytes
from deep_ep.utils.math import calc_diff


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_rdma_ranks, num_nvl_ranks = deep_ep.comm.get_physical_domain_size(group)

    # Allocate (in-place all-reduce overwrites the sources)
    allocator = deep_ep.BufferAllocator()
    src_0 = allocator.allocate((num_ranks * 4096,), torch.float32)
    src_1 = allocator.allocate((num_ranks * (3 * 1024 + 8),), torch.float32)
    perf_src = allocator.allocate((args.num_bytes // 4,), torch.float32)
    buffer = deep_ep.BucketBuffer(group, allocator, explicitly_destroy=True)
    num_sms = args.num_sms if args.num_sms is not None else buffer.get_theoretical_num_sms('all_reduce', group)

    # Print configs
    dist_print(f'Config:\n'
               f' > Ranks: {num_rdma_ranks} x {num_nvl_ranks}\n'
               f' > #SM: {num_sms}\n',
               once_in_node=True)

    # Check correctness
    scale = 0.25
    refs = []
    for bucket_idx, src in enumerate((src_0, src_1)):
        torch.manual_seed(rank_idx * 17 + bucket_idx)
        src.normal_()
        ref = src.clone()
        dist.all_reduce(ref, group=group)
        ref.mul_(scale)
        refs.append(ref)

    dist_print('Testing correctness:', once_in_node=True)
    handle = buffer.all_reduce([src_0, src_1], group=group, scale=scale)
    dsts = handle.wait()
    correct = True
    for src, dst, ref in zip((src_0, src_1), dsts, refs, strict=True):
        assert dst.data_ptr() == src.data_ptr()
        if not torch.allclose(dst, ref, rtol=1e-5, atol=1e-5):
            correct = False
            num_mismatches = (~torch.isclose(dst, ref, rtol=1e-5, atol=1e-5)).sum().item()
            diff = calc_diff(dst, ref)
            dist_print(f' > Correctness warning: {num_mismatches}/{dst.numel()} mismatched, '
                       f'diff: {diff}', once_in_node=True)
            assert diff < 1e-8
    if correct:
        dist_print(' > Correctness check passed', once_in_node=True)
    dist_print(once_in_node=True)

    # Perf NCCL
    torch.manual_seed(rank_idx)
    nccl_input = torch.randn_like(perf_src)

    nccl_t = bench_kineto(
        lambda: dist.all_reduce(nccl_input, group=group),
        kernel_names='ncclDevKernel_AllReduce',
        num_tests=args.num_tests,
        barrier_comm_profiling=True,
        barrier=buffer.barrier)

    # Perf ours
    dist_print('Performance:', once_in_node=True)
    t = bench_kineto(
        lambda: buffer.all_reduce(perf_src, num_sms=num_sms).wait(),
        kernel_names='all_reduce',
        num_tests=args.num_tests,
        barrier_comm_profiling=True,
        barrier=buffer.barrier)
    dist_print(
        f' > Perf ({num_sms=}): '
        f'{t * 1e6:7.1f} us | '
        f'{perf_src.nbytes / t / 1e9:5.1f} GB/s | '
        f'{nccl_t / t:.2f}x NCCL',
        once_in_node=True)

    # Destroy
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test all-reduce')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-bytes', type=parse_num_bytes, default=1 << 30)
    parser.add_argument('--num-sms', type=int)
    parser.add_argument('--num-tests', type=int, default=30)
    args = parser.parse_args()

    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

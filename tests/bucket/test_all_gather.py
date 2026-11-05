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
from deep_ep.utils.testing import bench, bench_kineto, parse_num_bytes


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_rdma_ranks, num_nvl_ranks = deep_ep.comm.get_physical_domain_size(group)
    torch.manual_seed(0)

    # Allocate
    allocator = deep_ep.BufferAllocator()
    gathered_tensors = []
    for _ in range(20):
        shard_numel = int(torch.randint(1, 1 << 20, ()).item()) * 8
        gathered_tensors.append(allocator.allocate((num_ranks * shard_numel,), torch.float32))
    perf_gathered = allocator.allocate((args.num_bytes // 4,), torch.float32)
    buffer = deep_ep.BucketBuffer(group, allocator, explicitly_destroy=True)
    num_sms = args.num_sms

    # Print configs
    dist_print(f'Config:\n'
               f' > Ranks: {num_rdma_ranks} x {num_nvl_ranks}\n'
               f' > #SM: {num_sms}\n',
               once_in_node=True)

    # Check correctness
    for i in range(len(gathered_tensors)):
        references, srcs = [], []
        buckets = (gathered_tensors[i], gathered_tensors[(i + 1) % len(gathered_tensors)])
        for gathered in buckets:
            shard_numel = gathered.numel() // num_ranks
            src = gathered[rank_idx * shard_numel:(rank_idx + 1) * shard_numel].normal_().add_(rank_idx)
            srcs.append(src)

            reference = torch.empty_like(gathered)
            dist.all_gather_into_tensor(reference, src, group=group)
            references.append(reference)

        dsts = buffer.all_gather(srcs, group=group, num_sms=num_sms).wait()
        for dst, gathered, reference in zip(dsts, buckets, references, strict=True):
            assert dst.data_ptr() == gathered.data_ptr()
            assert torch.equal(dst, reference)

    # Check correctness with out-of-buffer inputs (pure NVLink only), mixed with an in-place one
    if num_rdma_ranks == 1:
        for i in range(len(gathered_tensors)):
            references, srcs = [], []
            buckets = (gathered_tensors[i], gathered_tensors[(i + 1) % len(gathered_tensors)])
            for j, gathered in enumerate(buckets):
                shard_numel = gathered.numel() // num_ranks
                if j == 0:
                    src = torch.randn(shard_numel, dtype=torch.float32, device='cuda').add_(rank_idx)
                else:
                    src = gathered[rank_idx * shard_numel:(rank_idx + 1) * shard_numel].normal_().add_(rank_idx)
                srcs.append(src)

                reference = torch.empty_like(gathered)
                dist.all_gather_into_tensor(reference, src, group=group)
                references.append(reference)

            # Drop the out-of-buffer input right away to check the lifetime guard
            handle = buffer.all_gather(srcs, dsts=buckets, group=group, num_sms=num_sms)
            del srcs
            dsts = handle.wait()
            for dst, gathered, reference in zip(dsts, buckets, references, strict=True):
                assert dst.data_ptr() == gathered.data_ptr()
                assert torch.equal(dst, reference)

    # Perf NCCL
    shard_numel = perf_gathered.numel() // num_ranks
    src = perf_gathered[rank_idx * shard_numel:(rank_idx + 1) * shard_numel].normal_().add_(rank_idx)
    nccl_dst = torch.empty(perf_gathered.numel(), dtype=torch.float32, device='cuda')
    dist.all_gather_into_tensor(nccl_dst, src, group=group)
    nccl_t = bench_kineto(
        lambda: dist.all_gather_into_tensor(nccl_dst, src, group=group),
        kernel_names='ncclDevKernel_AllGather', barrier_comm_profiling=True,
        barrier=buffer.barrier)

    # Perf ours
    t, _, _ = bench(lambda: buffer.all_gather(src, num_sms=num_sms).wait())

    dist_print('Performance:', once_in_node=True)
    dist_print(
        f' > Perf ({num_sms=}): '
        f'{t * 1e6:7.1f} us | '
        f'{perf_gathered.nbytes / t / 1e9:5.1f} GB/s | '
        f'{nccl_t / t:.2f}x NCCL',
        once_in_node=True)

    # Destroy
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test all-gather')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-bytes', type=parse_num_bytes, default=1 << 30,
                        help='Total gathered size, with an optional binary '
                             'suffix: 1G (default), 64M, 512K, or plain bytes')
    parser.add_argument('--num-sms', type=int, default=0)
    args = parser.parse_args()

    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

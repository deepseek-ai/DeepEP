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
from deep_ep.utils.math import align
from deep_ep.utils.testing import parse_num_bytes


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_rdma_ranks, num_nvl_ranks = deep_ep.comm.get_physical_domain_size(group)
    assert num_rdma_ranks == 1, 'All-reduce is only implemented for the NVLink domain'
    alignment = deep_ep.get_num_rdma_alignment()
    assert args.num_bytes >= alignment
    numel = align(args.num_bytes // 4, num_ranks * 4)

    # Reserve storage for the three collective operators.
    num_buffer_bytes = align(3 * align(numel * 4, alignment),
                             deep_ep.get_num_allocation_alignment())
    # Allocate
    buffer = deep_ep.BucketBuffer(group, num_buffer_bytes, explicitly_destroy=True)

    # Print configs
    dist_print(f'Config:\n'
               f' > Ranks: {num_rdma_ranks} x {num_nvl_ranks}\n'
               f' > #Bytes: {args.num_bytes}\n',
               once_in_node=True)

    # Check correctness against NCCL
    torch.manual_seed(rank_idx)

    # Construct all-gather src and dst buffer (multi-dimensional sources)
    all_gather_src = torch.randn(numel // num_ranks, dtype=torch.float32, device='cuda').view(-1, 4)
    all_gather_dst = torch.empty((num_ranks, *all_gather_src.shape), dtype=torch.float32, device='cuda')
    all_gather_ref = torch.empty_like(all_gather_dst)
    dist.all_gather_single(all_gather_ref.view(-1), all_gather_src.view(-1), group=group)

    # Construct reduce-scatter src and dst buffer
    reduce_scatter_src = torch.randn(numel, dtype=torch.float32, device='cuda').view(-1, 4)
    reduce_scatter_ref = torch.empty(numel // num_ranks, dtype=torch.float32, device='cuda')
    dist.reduce_scatter_single(reduce_scatter_ref, reduce_scatter_src.view(-1), group=group)

    # Construct all-reduce src and dst buffer
    all_reduce_src = torch.randn(numel, dtype=torch.float32, device='cuda')
    all_reduce_ref = all_reduce_src.clone()
    dist.all_reduce(all_reduce_ref, group=group)

    # Check correctness
    with buffer.session():
        all_gather_dst = buffer.all_gather(
            all_gather_src, dsts=all_gather_dst, group=group).wait()
        torch.testing.assert_close(all_gather_dst, all_gather_ref, rtol=0, atol=0)

        reduce_scatter_dst = buffer.reduce_scatter(reduce_scatter_src, group=group).wait()
        torch.testing.assert_close(reduce_scatter_dst, reduce_scatter_ref, rtol=1e-5, atol=1e-5)

        all_reduce_dst = buffer.all_reduce(all_reduce_src, group=group).wait()
        torch.testing.assert_close(all_reduce_dst, all_reduce_ref, rtol=1e-5, atol=1e-5)

    # Check correctness with inputs allocated in the buffer, calling the in-place variants
    with buffer.session() as session:
        all_gather_inplace = session.allocate((num_ranks, *all_gather_src.shape), group=group)
        all_gather_inplace[rank_idx].copy_(all_gather_src)
        all_gather_result = buffer.all_gather(all_gather_inplace[rank_idx], group=group).wait()
        torch.testing.assert_close(all_gather_result, all_gather_ref.view(-1), rtol=0, atol=0)

        reduce_scatter_inplace = session.allocate(reduce_scatter_src.shape, group=group)
        reduce_scatter_inplace.copy_(reduce_scatter_src)
        reduce_scatter_result = buffer.reduce_scatter(reduce_scatter_inplace, group=group).wait()
        torch.testing.assert_close(reduce_scatter_result, reduce_scatter_ref, rtol=1e-5, atol=1e-5)

        all_reduce_inplace = session.allocate(all_reduce_src.shape, group=group)
        all_reduce_inplace.copy_(all_reduce_src)
        all_reduce_result = buffer.all_reduce(all_reduce_inplace, group=group).wait()
        torch.testing.assert_close(all_reduce_result, all_reduce_ref, rtol=1e-5, atol=1e-5)

    dist_print('Correctness check passed.', once_in_node=True)

    # Destroy
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test BucketBuffer session')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-bytes', type=parse_num_bytes, default=1 << 20,
                        help='Total size of each collective, with an optional binary '
                             'suffix: 1M (default), 64M, 512K, or plain bytes')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

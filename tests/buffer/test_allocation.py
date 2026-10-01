import argparse

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int) -> None:
    rank_idx, num_ranks, group_0 = init_dist(local_rank, num_local_ranks)
    assert num_ranks % 2 == 0
    group_1 = None
    for rank_begin_idx in range(0, num_ranks, 2):
        ranks = list(range(rank_begin_idx, rank_begin_idx + 2))
        group = dist.new_group(ranks)
        if rank_idx in ranks:
            group_1 = group
    assert group_1 is not None

    plan = deep_ep.BufferAllocator()
    bucket_0 = plan.allocate((3, 5), torch.float32)
    bucket_1 = plan.allocate((7,), torch.bfloat16)
    assert bucket_0.is_meta and bucket_1.is_meta

    buffer = deep_ep.BucketBuffer(
        group=[group_0, group_1], allocation_plan_or_num_bytes=plan, explicitly_destroy=True)
    assert bucket_0.is_cuda and bucket_1.is_cuda
    assert buffer.storage.dtype == torch.uint8
    assert buffer.storage.numel() == plan.num_bytes
    assert len(buffer.contexts) == 2
    assert bucket_0.data_ptr() == buffer.storage.data_ptr()

    bucket_0.fill_(rank_idx)
    bucket_1.fill_(rank_idx + 1)
    assert (bucket_0 == rank_idx).all()
    assert (bucket_1 == rank_idx + 1).all()
    buffer.destroy()

    buffer = deep_ep.BucketBuffer(group_0, plan.num_bytes, explicitly_destroy=True)
    assert buffer.storage.is_cuda and buffer.storage.dtype == torch.uint8
    assert buffer.storage.numel() == plan.num_bytes
    buffer.destroy()

    dist_print('Bucket allocation: passed', once_in_node=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test BucketBuffer allocation and group registration')
    parser.add_argument('--num-processes', type=int, default=4)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes,), nprocs=args.num_processes)

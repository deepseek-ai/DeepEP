import os

# Must pop before PyTorch imports
os.environ.pop('NCCL_MIN_NCHANNELS', None)
os.environ.pop('NCCL_MAX_NCHANNELS', None)

import argparse
import random
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.testing import bench, parse_num_bytes
from deep_ep.utils.math import align


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_rdma_ranks, num_nvl_ranks = deep_ep.comm.get_physical_domain_size(group)
    assert num_rdma_ranks * num_nvl_ranks == num_ranks
    alignment = deep_ep.get_num_rdma_alignment()
    assert args.num_bytes % alignment == 0

    dist_print(f'Config:\n'
               f' > Ranks: {num_rdma_ranks} x {num_nvl_ranks}\n'
               f' > #Bytes: {args.num_bytes}\n'
               f' > Pattern: {args.pattern}\n',
               once_in_node=True)

    # Allocate: [send region | recv region]
    allocator = deep_ep.BufferAllocator()
    send_tensor = allocator.allocate((args.num_bytes, ), torch.uint8)
    recv_tensor = allocator.allocate((args.num_bytes, ), torch.uint8)
    buffer = deep_ep.BucketBuffer(group, allocator, explicitly_destroy=True)
    reference = torch.empty(args.num_bytes, dtype=torch.uint8, device='cuda')
    num_sms = buffer.get_theoretical_num_sms('send_recv', group)
    send_offset = send_tensor.data_ptr() - buffer.storage.data_ptr()
    recv_offset = recv_tensor.data_ptr() - buffer.storage.data_ptr()

    # Fill the send region with rank-distinct data
    random.seed(args.seed)
    torch.manual_seed(args.seed + rank_idx)
    send_tensor.copy_(torch.randint(0, 256, (args.num_bytes, ), dtype=torch.uint8, device='cuda'))

    def make_plan(pattern):
        if pattern == 'ring':
            full_plan = [(src_rank_idx, (src_rank_idx + 1) % num_ranks, send_offset, recv_offset, args.num_bytes)
                         for src_rank_idx in range(num_ranks)]
        else:
            full_plan = []
            num_max_send_recv_entries = 500
            num_send_entries_per_rank = [0] * num_ranks
            num_recv_bytes_per_rank = [0] * num_ranks

            for _ in range(random.randint(1, num_ranks * num_ranks * 32)):
                src_rank_idx = random.randrange(num_ranks)
                dst_rank_idx = random.randrange(num_ranks)

                # Must not exceed the send entry number limit
                if num_send_entries_per_rank[src_rank_idx] >= num_max_send_recv_entries:
                    continue

                # Get the number of bytes to send
                num_bytes = align(random.randint(alignment, args.num_bytes), alignment)
                num_bytes = min(num_bytes, args.num_bytes - num_recv_bytes_per_rank[dst_rank_idx])
                if num_bytes == 0:
                    continue

                # Record the send op
                src_offset = send_offset + random.randrange(0, args.num_bytes - num_bytes + 1, alignment)
                dst_offset = recv_offset + random.randrange(
                    num_recv_bytes_per_rank[dst_rank_idx], args.num_bytes - num_bytes + 1, alignment)
                num_send_entries_per_rank[src_rank_idx] += 1
                num_recv_bytes_per_rank[dst_rank_idx] = dst_offset - recv_offset + num_bytes
                full_plan.append((src_rank_idx, dst_rank_idx, src_offset, dst_offset, num_bytes))

        # Construct the plan from the local send/recv ops
        send_ops = [(buffer.storage[src_offset: src_offset + num_bytes], dst_rank_idx)
                    for src_rank_idx, dst_rank_idx, src_offset, _, num_bytes in full_plan if src_rank_idx == rank_idx]
        recv_ops = [(buffer.storage[dst_offset: dst_offset + num_bytes], src_rank_idx)
                    for src_rank_idx, dst_rank_idx, _, dst_offset, num_bytes in full_plan if dst_rank_idx == rank_idx]
        plan = [(buffer.storage[src_offset: src_offset + num_bytes], dst_rank_idx, dst_offset)
                for src_rank_idx, dst_rank_idx, src_offset, dst_offset, num_bytes in full_plan if src_rank_idx == rank_idx]
        return plan, send_ops, recv_ops

    plan, send_ops, recv_ops = make_plan(args.pattern)

    def run_nccl():
        ops = [dist.P2POp(dist.isend, tensor, dst_rank_idx, group) for tensor, dst_rank_idx in send_ops]
        for tensor, src_rank_idx in recv_ops:
            offset = tensor.data_ptr() - recv_tensor.data_ptr()
            dst_tensor = reference[offset: offset + tensor.numel()]
            ops.append(dist.P2POp(dist.irecv, dst_tensor, src_rank_idx, group))
        # NOTES: Ranks unrelated to the plan do not participate in the NCCL batch
        if ops:
            for op in dist.batch_isend_irecv(ops):
                op.wait()

    def check_correctness(num_sms):
        for tensor, src_rank_idx in recv_ops:
            offset = tensor.data_ptr() - recv_tensor.data_ptr()
            assert torch.equal(tensor, reference[offset: offset + tensor.numel()]), \
                f'{rank_idx=}, {src_rank_idx=}, {num_sms=}, {offset=}'

    # Correctness
    run_nccl()
    assert buffer.send_recv(plan, num_sms=num_sms, group=group).wait() is None
    check_correctness(num_sms)

    dist.barrier()
    dist_print('Correctness check passed.', once_in_node=True)

    # Performance
    dist_print('Performance:', once_in_node=True)
    nccl_duration = bench(run_nccl)[0]
    duration = bench(lambda: buffer.send_recv(plan, num_sms=num_sms, group=group).wait())[0]
    num_send_bytes = sum(tensor.numel() for tensor, _ in send_ops)
    num_recv_bytes = sum(tensor.numel() for tensor, _ in recv_ops)
    dist_print(
        f' > Perf (Rank {rank_idx}): '
        f'{duration * 1e6:7.1f} us | '
        f'Send {num_send_bytes / duration / 1e9:4.0f} GB/s | '
        f'Recv {num_recv_bytes / duration / 1e9:4.0f} GB/s | '
        f'{nccl_duration / duration:.2f}x NCCL',
        once_in_node=False)

    # Pressure tests
    num_device_sms = torch.cuda.get_device_properties(local_rank).multi_processor_count
    for seed in range(int(1e9) if args.do_pressure_test else 0):
        random.seed(args.seed + seed)
        torch.manual_seed(args.seed + seed + rank_idx)
        num_sms = random.randint(2, num_device_sms) if num_rdma_ranks > 1 and num_nvl_ranks > 1 else 0
        dist_print(f'Testing with {seed=}, {num_sms=} ...', once_in_node=True)
        send_tensor.random_(0, 256)
        plan, send_ops, recv_ops = make_plan('rand')
        run_nccl()
        recv_tensor.copy_(reference.bitwise_not())
        assert buffer.send_recv(plan, num_sms=num_sms, group=group).wait() is None
        check_correctness(num_sms)
        dist.barrier()

    # Destroy
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test BucketBuffer send-recv against NCCL')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-bytes', type=parse_num_bytes, default=1 << 30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pattern', type=str, choices=('rand', 'ring'), default='rand')
    parser.add_argument('--do-pressure-test', action='store_true', help='Whether to do pressure test')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

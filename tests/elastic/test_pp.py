import argparse
import math
import os
import random
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist, dist_print, get_rdma_gbs
from deep_ep.utils.testing import bench_kineto


def generate_stress_ops(rank_idx: int, num_ranks: int, num_sends: int, shape: tuple):
    send_times = {(s, d): [] for s in range(num_ranks) for d in range(num_ranks) if s != d}
    recv_times = {(s, d): [] for s in range(num_ranks) for d in range(num_ranks) if s != d}

    for _ in range(num_sends):
        src_rank_idx = random.randint(0, num_ranks - 1)
        dst_rank_idx = (src_rank_idx + (1 if random.randint(0, 1) else -1)) % num_ranks
        st = random.randint(0, 10 ** 8)
        rt = st + random.randint(1, 3 * 10 ** 6)
        send_times[(src_rank_idx, dst_rank_idx)].append(st)
        recv_times[(src_rank_idx, dst_rank_idx)].append(rt)

    ops = []
    for (src_rank_idx, dst_rank_idx) in send_times:
        n = len(send_times[(src_rank_idx, dst_rank_idx)])
        sorted_send = sorted(send_times[(src_rank_idx, dst_rank_idx)])
        sorted_recv = sorted(recv_times[(src_rank_idx, dst_rank_idx)])
        for i in range(n):
            tensor = torch.randn(shape, dtype=torch.bfloat16, device='cuda')
            if src_rank_idx == rank_idx:
                ops.append(('send', sorted_send[i], dst_rank_idx, i, tensor))
            if dst_rank_idx == rank_idx:
                ops.append(('recv', sorted_recv[i], src_rank_idx, i, tensor))
    ops.sort(key=lambda x: (x[1], x[3]))
    return ops


# noinspection PyShadowingNames
@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    shape = (args.num_tokens, args.hidden)
    num_max_tensor_bytes = math.prod(shape) * 2
    num_max_inflight_tensors = args.num_max_inflight_tensors
    buffer = deep_ep.ElasticBuffer(
        group, explicitly_destroy=True, allow_hybrid_mode=False,
        num_bytes=deep_ep.ElasticBuffer.get_pp_buffer_size_hint(
            num_max_tensor_bytes, num_max_inflight_tensors))
    requested_num_qps = args.num_qps
    buffer.pp_set_config(num_max_tensor_bytes, num_max_inflight_tensors, num_qps=requested_num_qps)
    num_qps = buffer.num_pp_qps

    # Print configs
    assert num_ranks > 1
    dist_print(f'Config:\n'
               f' > Ranks: {num_ranks}\n'
               f' > Shape: {shape}\n'
               f' > QPs: {num_qps}/{buffer.num_allocated_qps}\n'
               f' > Max inflight tensors: {num_max_inflight_tensors}\n',
               once_in_node=True)

    # Run stress tests
    dist_print('Running stress tests:', once_in_node=True)
    for seed in range(args.num_stress_iterations):
        dist_print(f' > Testing with {seed=} ...', once_in_node=True)
        torch.manual_seed(42 + seed)
        random.seed(42 + seed)
        ops = generate_stress_ops(rank_idx, num_ranks, args.num_sends, shape)

        prev = 0
        for j, (op, timestamp, peer, _, tensor) in enumerate(ops):
            if op == 'send':
                buffer.pp_send(tensor, peer)
            else:
                result = torch.empty_like(tensor)
                buffer.pp_recv(result, peer)
                assert torch.equal(result, tensor), \
                    f'Rank {rank_idx}: mismatch at op {j}'
            if timestamp > prev:
                torch.cuda._sleep(int((timestamp - prev) / 10 ** 8 * args.num_sleep_cycles))
            prev = timestamp
    dist_print(' > All stress tests passed', once_in_node=True)
    dist_print(once_in_node=True)

    # Profiling
    dist_print('Profiling PP send and recv:', once_in_node=True)
    num_approx_rdma_cycles = int(num_max_tensor_bytes * 2 / get_rdma_gbs() * 1.5)

    def get_trace_path(prefix: str):
        return (None if not args.dump_profile_traces
                else f'{args.dump_profile_traces}/{prefix}_rank{rank_idx}.json')

    for hide_rdma_latency in (True, False):
        for num_concurrent in (1, 2, 3):
            send_tensors = [torch.randn(shape, dtype=torch.bfloat16, device='cuda') for _ in range(num_concurrent)]
            recv_tensors = [torch.empty(shape, dtype=torch.bfloat16, device='cuda') for _ in range(num_concurrent)]

            def loop(_hide_rdma_latency=hide_rdma_latency):
                torch.zeros((131072, 32768), dtype=torch.int, device='cuda')
                for t in send_tensors:
                    buffer.pp_send(t, (rank_idx + 1) % num_ranks)
                if _hide_rdma_latency:
                    torch.cuda._sleep(num_approx_rdma_cycles * num_concurrent)
                for t in recv_tensors:
                    buffer.pp_recv(t, (rank_idx - 1) % num_ranks)

            send_t, recv_t = bench_kineto(
                loop, kernel_names=('send_impl', 'recv_impl'),
                barrier_comm_profiling=True, barrier=buffer.barrier,
                trace_path=get_trace_path(f'pp_{num_concurrent}_{hide_rdma_latency}'))
            dist_print(
                f' > EP: {rank_idx:3}/{num_ranks:3} | '
                f'hide={int(hide_rdma_latency)}, concurrent={num_concurrent} | '
                f'send: {send_t * 1e6:.3f} us, '
                f'{2 * num_max_tensor_bytes / send_t / 1e9:.3f} GB/s | '
                f'recv: {recv_t * 1e6:.3f} us, '
                f'{(2 if hide_rdma_latency else 1) * num_max_tensor_bytes / recv_t / 1e9:.3f} GB/s')
    dist_print(once_in_node=True)

    # Destroy the runtime and communication group
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PP send/recv kernels')
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--num-tokens', type=int, default=4096)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-max-inflight-tensors', type=int, default=4)
    parser.add_argument('--num_qps', type=int, default=0)
    parser.add_argument('--num-stress-iterations', type=int, default=4)
    parser.add_argument('--num-sends', type=int, default=128)
    parser.add_argument('--num-sleep-cycles', type=int, default=10 ** 7)
    parser.add_argument('--dump-profile-traces', type=str, default='')
    args = parser.parse_args()

    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

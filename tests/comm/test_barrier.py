import argparse
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist, dist_print
from deep_ep.utils.testing import bench_kineto


def test_barrier(buffer: deep_ep.EPBuffer, args: argparse.Namespace):
    dist_print('Profiling barrier:', once_in_node=True)
    num_scaleout_ranks, num_scaleup_ranks = buffer.get_logical_domain_size()
    dist_print(f'Config:\n'
               f' > Ranks: {num_scaleout_ranks} x {num_scaleup_ranks}\n'
               f' > #QPs: {buffer.num_allocated_qps}\n',
               once_in_node=True)

    # Test barrier time
    def loop_barrier(num_tests=1000):
        for i in range(num_tests):
            buffer.barrier()

    t = bench_kineto(lambda: loop_barrier(), 'barrier', barrier_comm_profiling=True, barrier=buffer.barrier)
    dist_print(f' > EP: {buffer.rank_idx:3}/{buffer.num_ranks:3}, '
               f'barrier time: {t * 1e6:.3f} us')
    dist_print(once_in_node=True)


# noinspection PyShadowingNames
@torch.inference_mode()
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_rdma_ranks, num_nvl_ranks = deep_ep.comm.get_physical_domain_size(group)
    num_scaleout_ranks, num_scaleup_ranks = deep_ep.comm.get_logical_domain_size(group, args.allow_hybrid_mode)
    assert num_rdma_ranks * num_nvl_ranks == num_ranks
    assert num_scaleout_ranks * num_scaleup_ranks == num_ranks

    do_pressure_test = args.do_pressure_test
    for i in range(int(1e9) if do_pressure_test else 1):
        buffer = deep_ep.EPBuffer(
            group, num_bytes=2 ** 30,
            allow_hybrid_mode=args.allow_hybrid_mode,
            num_allocated_qps=args.num_allocated_qps,
            explicitly_destroy=True
        )

        # Test barrier
        test_barrier(buffer, args)

        # Destroy the runtime and communication group
        buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test EP barrier performance')

    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--allow-hybrid-mode', type=int, default=1, help='Whether to allow hybrid mode')
    parser.add_argument('--num-allocated-qps', type=int, default=8, help='Number of QPs to use (0 means auto)')
    parser.add_argument('--do-pressure-test', action='store_true', help='Whether to do pressure test')
    args = parser.parse_args()

    # Launch test processes
    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)

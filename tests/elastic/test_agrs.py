import argparse
import math
import random
import torch
import torch.distributed as dist
import numpy as np

import deep_ep
from deep_ep.utils.envs import init_dist, dist_print


def all_gather_ref(shape: tuple, rank_idx: int, num_ranks: int, round_idx: int = 0):
    ref_list = []
    for i in range(num_ranks):
        torch.manual_seed(42 + round_idx * 43 + i)
        ref_list.append(torch.randn(shape, dtype=torch.bfloat16, device='cuda'))
    return ref_list[rank_idx], torch.stack(ref_list, dim=0)


def generate_stress_ops(
    num_ops: int,
    num_max_inflight_agrs: int,
    shape: tuple,
    rank_idx: int,
    num_ranks: int,
) -> tuple[list[tuple], tuple[torch.Tensor], tuple[torch.Tensor]]:
    tensors, refs = zip(*(all_gather_ref(shape, rank_idx, num_ranks, round_idx=i) for i in range(num_ops)), strict=True)
    unprocessed = random.sample(range(num_ops), num_ops)
    inflight, ops = [], [('create_session', (-1,))]
    limit = num_max_inflight_agrs
    while unprocessed or inflight:
        max_g = min(len(unprocessed), limit)
        choices = []
        if max_g > 0:
            choices.append('ag')
        if inflight:
            choices.append('fetch')
        else:
            choices.append('destroy')
        op = random.choice(choices)
        if op == 'ag':
            b = tuple(unprocessed[-random.randint(1, max_g):])
            limit -= len(b)
            del unprocessed[-len(b):]
            inflight.append(b)
            ops.append(('ag', b))
        elif op == 'fetch':
            ops.append(('fetch', inflight.pop(random.randrange(len(inflight)))))
        else:
            ops.extend([('destroy_session', (-1,)), ('create_session', (-1,))])
            limit = num_max_inflight_agrs

    ops.append(('destroy_session', (-1,)))
    return ops, tensors, refs


def do_all_gather(buffer: deep_ep.ElasticBuffer,
                  is_inplace: bool, is_batched: bool,
                  tensors: tuple[torch.Tensor, ...],
                  start_event: torch.cuda.Event | None = None):
    # Copy into buffer if inplace
    if is_inplace:
        ag_tensors = buffer.agrs_get_inplace_tensor(tuple(t.shape for t in tensors), torch.bfloat16)
        for x, y in zip(ag_tensors, tensors, strict=True):
            x.copy_(y)
    else:
        ag_tensors = tensors

    # Record event
    if start_event is not None:
        torch.zeros(int(256e6 // 4), dtype=torch.int, device='cuda')  # flush L2 cache
        start_event.record()

    # Do all-gather
    if is_batched:
        *out_tensors, handle = buffer.all_gather(ag_tensors)
        return out_tensors, [handle]
    else:
        out_tensors, handles = [], []
        for t in ag_tensors:
            out_tensor, handle = buffer.all_gather(t)
            out_tensors.append(out_tensor)
            handles.append(handle)
        return out_tensors, handles


# noinspection PyTypeChecker,PyCallingNonCallable,PyShadowingNames
@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    # Print configs
    shape = (32, 64, 2048)
    num_max_inflight_agrs = args.num_max_inflight_agrs
    num_max_session_bytes = deep_ep.ElasticBuffer.get_agrs_num_max_session_bytes(
        group,
        [shape for _ in range(num_max_inflight_agrs)],
        torch.bfloat16
    )
    num_max_session_bytes = deep_ep.ElasticBuffer.get_agrs_buffer_size_hint(
        group, num_max_session_bytes)
    dist_print(f'Config:\n'
               f' > Ranks: {num_ranks}\n'
               f' > Shape: {shape}\n'
               f' > Max inflight AGRS: {num_max_inflight_agrs}\n',
               once_in_node=True)

    # Create buffer
    buffer = deep_ep.ElasticBuffer(group, explicitly_destroy=True, num_bytes=num_max_session_bytes)
    buffer.agrs_set_config(num_max_session_bytes, num_max_inflight_agrs)

    # Run stress tests
    dist_print('Running stress tests:', once_in_node=True)
    for seed in range(args.num_stress_iterations):
        random.seed(42 + seed)
        num_ops = 128
        ops, tensors, refs = generate_stress_ops(num_ops, num_max_inflight_agrs, shape, rank_idx, num_ranks)
        results = [None] * num_ops
        handles = dict()
        torch.cuda.synchronize()
        for op, indices in ops:
            if op == 'create_session':
                buffer.create_agrs_session()
            elif op == 'destroy_session':
                buffer.destroy_agrs_session()
            elif op == 'ag':
                is_inplace, is_batched = random.random() < 0.5, random.random() < 0.8
                handles[indices] = do_all_gather(buffer, is_inplace, is_batched, tuple(tensors[i] for i in indices))
            elif op == 'fetch':
                out_tensors, wait_handles = handles[indices]
                for h in wait_handles:
                    h()
                for out, idx in zip(out_tensors, indices, strict=True):
                    results[idx] = out.clone()

        for i in range(num_ops):
            assert results[i] is not None and torch.equal(results[i], refs[i]), \
                f'Rank {rank_idx}: stress mismatch at seed={seed}, op={i}'
        dist_print(f' > Seed {seed} passed ({num_ops} ops)', once_in_node=True)
    dist_print(once_in_node=True)

    # Destroy the buffer
    dist_print(f'Profiling all-gather:', once_in_node=True)
    buffer.destroy()

    # Profiling
    num_max_session_bytes = deep_ep.ElasticBuffer.get_agrs_num_max_session_bytes(
        group,
        [(2 ** 26,) for _ in range(num_max_inflight_agrs)],
        torch.bfloat16
    )
    num_max_session_bytes = deep_ep.ElasticBuffer.get_agrs_buffer_size_hint(
        group, num_max_session_bytes)
    buffer = deep_ep.ElasticBuffer(group, explicitly_destroy=True, num_bytes=num_max_session_bytes)
    buffer.agrs_set_config(num_max_session_bytes, num_max_inflight_agrs)
    for num_bytes in (2 ** p for p in range(20, 27)):
        # Create tensors
        shape = (num_bytes // 2, )
        tensors = tuple(torch.randn(shape, dtype=torch.bfloat16, device='cuda') for _ in range(num_max_inflight_agrs))

        # Tests
        for is_inplace in (False, True):
            for is_batched in (False, True):
                num_tests = 50
                start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
                end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
                torch.cuda.synchronize()

                for i in range(num_tests):
                    with buffer.agrs_new_session():
                        _, wait_handles = do_all_gather(buffer, is_inplace, is_batched, tensors, start_event=start_events[i])
                        for h in wait_handles:
                            h()
                    end_events[i].record()
                torch.cuda.synchronize()

                times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events, strict=True)])[1:]
                avg_t = np.average(times)
                unit = ('MB', 1e6) if num_bytes >= 1e6 else ('KB', 1e3)
                bandwidth_info = f', {num_bytes * num_ranks * num_max_inflight_agrs / avg_t / 1e9:.3f} GB/s' if num_ranks > 1 else ''

                dist_print(
                    f' > Rank: {rank_idx:3}/{num_ranks:3} | '
                    f'{num_ranks} x {(num_bytes / unit[1]):.0f} {unit[0]} | '
                    f'avg: {avg_t / num_max_inflight_agrs * 1e6:.3f} us'
                    f'{bandwidth_info}'
                    f' (inplace={int(is_inplace)}, batched={int(is_batched)})')
    dist_print(once_in_node=True)

    # Destroy the runtime and communication group
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test all_gather kernels')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-max-inflight-agrs', type=int, default=4)
    parser.add_argument('--num-stress-iterations', type=int, default=4)
    args = parser.parse_args()

    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

"""Compare valid-prefix sorting and early count copies with native dispatch paths."""

import argparse
from contextlib import nullcontext
import hashlib
import inspect
from itertools import product
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.buffers.ep import EPHandle
from deep_ep.utils.envs import init_dist
from deep_ep.utils.refs import dispatch as reference_dispatch

ARMS = ('late_full', 'late_prefix', 'early_full', 'early_prefix',
        'upstream_syncfree', 'upstream_defer')
TESTED_SOURCE = '93eb6eb238127e96c6d7a4a625a6dad158348509'


def emit(kind, **values):
    print(kind + ' ' + json.dumps(values, sort_keys=True), flush=True)


def make_inputs(rank, world, tokens, hidden, pattern):
    rows = tokens - rank % min(tokens, world)
    row = torch.arange(rows, device='cuda')
    col = torch.arange(hidden, device='cuda')
    x = ((row[:, None] + col[None, :] + 3 * rank) % 17 - 8).bfloat16()
    destination = (row + rank) % world
    if pattern == 'skewed':
        destination = torch.zeros_like(destination)
    second = destination if pattern in (
        'duplicate', 'skewed') else (destination + 1) % world
    routes = torch.stack((destination * 2, second * 2 + 1),
                         dim=1).to(deep_ep.topk_idx_t)
    weights = torch.ones((rows, 2), dtype=torch.float32, device='cuda')
    if pattern == 'masked':
        routes[row % 3 == 0, 1] = -1
        weights[row % 3 == 0, 1] = 0
    return x, routes, weights


def fingerprint(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().cpu().contiguous().view(
            torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class SavedReceive(torch.autograd.Function):

    @staticmethod
    def forward(ctx, received, weight):
        ctx.save_for_backward(received)
        return (received.float() * weight).sum()

    @staticmethod
    def backward(ctx, gradient):
        received, = ctx.saved_tensors
        return None, received.float().sum(dim=0) * gradient


class Runner:

    def __init__(self,
                 buffer,
                 x,
                 routes,
                 weights,
                 capacity,
                 allocate,
                 profile=False):
        self.buffer = buffer
        self.x, self.routes, self.weights = x, routes, weights
        self.capacity, self.allocate = capacity, allocate
        self.count_stream = torch.cuda.Stream()
        self.profile = profile
        # The independent compute island is identical in every arm.
        self.gemm_a = torch.ones((2048, 2048),
                                 dtype=torch.bfloat16,
                                 device='cuda')
        self.gemm_b = torch.ones_like(self.gemm_a)
        self.gemm_out = torch.empty_like(self.gemm_a)

    def region(self, name):
        return torch.profiler.record_function(
            name) if self.profile else nullcontext()

    def queue_counts(self, event, handle):
        expert = handle.psum_num_recv_tokens_per_expert
        ranks = handle.psum_num_recv_tokens_per_scaleup_rank
        assert expert.dtype == ranks.dtype == torch.int32
        host = torch.empty(expert.numel() + 1,
                           dtype=expert.dtype,
                           device='cpu',
                           pin_memory=True)
        done = torch.cuda.Event()
        with torch.cuda.stream(
                self.count_stream), self.region('count_copy_enqueue'):
            # Wait only on native completion, never consume the sorting hook on
            # this auxiliary stream. The owning result remains alive throughout.
            assert event.event is not None
            event.event.current_stream_wait()
            host[:-1].copy_(expert, non_blocking=True)
            host[-1:].copy_(ranks[-1:], non_blocking=True)
            done.record()
        return host, done

    def island(self, repetitions):
        with self.region('independent_gemm_island'):
            for _ in range(repetitions):
                torch.mm(self.gemm_a, self.gemm_b, out=self.gemm_out)

    def invoke(self, arm, island_repetitions=0):
        assert arm in ARMS
        deferred = arm == 'upstream_defer'
        with self.region('dispatch_launch'):
            result = self.buffer.dispatch(
                self.x,
                topk_idx=self.routes,
                topk_weights=self.weights,
                num_experts=self.buffer.num_ranks * 2,
                num_max_tokens_per_rank=self.capacity,
                num_sms=8,
                do_cpu_sync=deferred,
                do_expand=False,
                async_with_compute_stream=True,
                allocate_on_comm_stream=self.allocate,
                defer_epilogue=deferred)
        if deferred:
            self.island(island_repetitions)
            with self.region('native_deferred_epilogue'):
                received, indices, weights, handle = result.current_stream_wait(
                )
            total = received.shape[0]
            counts = handle.num_recv_tokens_per_expert_list
            return (received, indices, weights, handle, {
                'arm': arm,
                'allocated_rows': total,
                'valid_rows': total,
                'sorted_rows': total,
                'counts': counts,
                'count_prefetch': False
            })

        received, indices, weights, handle, event = result
        allocated_rows = received.shape[0]
        assert allocated_rows == self.capacity * self.buffer.num_ranks
        assert handle.channel_linked_list is None, 'This first prototype is single-domain compact BF16 only'
        original_hook = event.hook_after_wait
        assert getattr(original_hook, 'func', None).__self__ is handle
        assert getattr(original_hook.func, '__name__',
                       None) == 'deterministic_sort'
        assert len(
            original_hook.args) == 7 and original_hook.args[:2] == (False,
                                                                    False)
        assert original_hook.args[3] is None and original_hook.args[6] is None
        factorial = arm not in ('upstream_syncfree', 'upstream_defer')
        if factorial:
            event.hook_after_wait = None
        early = arm.startswith('early_')
        copied_counts = self.queue_counts(event, handle) if early else None
        self.island(island_repetitions)
        with self.region('dispatch_wait_and_native_hook'):
            event.current_stream_wait()
        if copied_counts is None:
            copied_counts = self.queue_counts(event, handle)
        host, done = copied_counts
        with self.region('count_host_wait'):
            done.synchronize()
        total = int(host[-1])
        expert_psum = host[:-1]
        counts = torch.diff(expert_psum,
                            prepend=expert_psum.new_zeros(1)).tolist()
        assert 0 <= total <= allocated_rows and all(count >= 0
                                                    for count in counts)
        prefix = arm.endswith('_prefix')
        if factorial:
            with self.region('prefix_sort' if prefix else 'capacity_sort'):
                if prefix:
                    handle.recv_src_metadata = handle.recv_src_metadata[:total]
                    # Only sort's shape interpretation changes. Native source,
                    # output capacity, dispatch and copy epilogue are unchanged.
                    handle.deterministic_sort(True, False, received[:total],
                                              None, indices[:total],
                                              weights[:total], None)
                else:
                    original_hook()
        # Every arm delivers the same exact-prefix consumer contract. Handle
        # trimming happens after sorting in full arms and before it in prefix arms.
        handle.recv_src_metadata = handle.recv_src_metadata[:total]
        handle.cached_recv_src_metadata_before_sort = handle.cached_recv_src_metadata_before_sort[:
                                                                                                  total]
        handle.num_recv_tokens = total
        handle.num_expanded_tokens = int(expert_psum[-1])
        handle.num_recv_tokens_per_expert_list = counts
        return (received[:total], indices[:total], weights[:total], handle, {
            'arm': arm,
            'allocated_rows': allocated_rows,
            'valid_rows': total,
            'sorted_rows': total if prefix else allocated_rows,
            'counts': counts,
            'count_prefetch': early
        })


def check_case(runner, arm, expected, case, island=0):
    received, indices, weights, handle, truth = runner.invoke(arm, island)
    torch.cuda.synchronize()
    for actual, reference in zip(
        (received, indices, weights, handle.recv_src_metadata[:, 0]),
            expected[:4]):
        assert torch.equal(actual, reference), (arm, case, 'dispatch')
    assert truth['counts'] == torch.bincount(
        indices[indices >= 0].long(),
        minlength=2).tolist(), (arm, case, 'expert counts')
    weight = torch.ones(received.shape[1],
                        device='cuda',
                        dtype=torch.float32,
                        requires_grad=True)
    loss = SavedReceive.apply(received, weight)
    payload = received * (indices >= 0).sum(dim=1, keepdim=True)
    combined, _, event = runner.buffer.combine(
        payload.to(torch.bfloat16),
        handle,
        async_with_compute_stream=True,
        allocate_on_comm_stream=runner.allocate)
    event.current_stream_wait()
    expected_combine = runner.x * (runner.routes >= 0).sum(dim=1, keepdim=True)
    assert torch.equal(combined, expected_combine), (arm, case, 'combine')
    later, later_indices, _, _, later_event = runner.buffer.dispatch(
        runner.x * 2,
        handle=handle,
        num_sms=8,
        async_with_compute_stream=True,
        allocate_on_comm_stream=runner.allocate)
    later_event.current_stream_wait()
    assert torch.equal(later, expected[0] * 2), (arm, case, 'cached payload')
    assert torch.equal(later_indices,
                       expected[1]), (arm, case, 'cached indices')
    loss.backward()
    torch.cuda.synchronize()
    assert torch.equal(weight.grad,
                       expected[0].float().sum(dim=0)), (arm, case,
                                                         'saved backward')
    emit('PREFIX_CORRECT_PASS', rank=dist.get_rank(), case=case, **truth)


def make_buffer(group, capacity, hidden):
    buffer = deep_ep.EPBuffer(group,
                              num_max_tokens_per_rank=capacity,
                              hidden=hidden,
                              num_topk=2,
                              deterministic=True,
                              allow_hybrid_mode=False,
                              explicitly_destroy=True)
    assert buffer.get_physical_domain_size()[0] == 1
    return buffer


def correctness(rank, world, group):
    for factor in (1, 4):
        capacity = 32 * factor
        buffer = make_buffer(group, capacity, 512)
        try:
            for allocate, pattern in product(
                (False, True), ('balanced', 'duplicate', 'skewed', 'masked')):
                tensors = make_inputs(rank, world, 32, 512, pattern)
                expected = reference_dispatch(*tensors, capacity, world * 2)
                runner = Runner(buffer, *tensors, capacity, allocate)
                case = f'correctness-cap{factor}-allocate{int(allocate)}-{pattern}'
                emit('INPUT_IDENTITY',
                     rank=rank,
                     case=case,
                     sha256=fingerprint(tensors))
                for arm in ARMS:
                    check_case(runner, arm, expected, case)
                    dist.barrier(group=group)
        finally:
            torch.cuda.synchronize()
            buffer.destroy()


def measured_call(runner, arm, island, group):
    # Synchronize rank entry and GPU completion outside/at the timing boundaries.
    # No profiler, reductions, validation or scalar logging inside the interval.
    dist.barrier(group=group)
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    packet = runner.invoke(arm, island)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return elapsed_ms, packet[-1]


def performance(rank, world, group, args):
    assert args.blocks % 2 == 0
    for factor, pattern, island in product((1, 4), ('balanced', 'skewed'),
                                           (0, 8)):
        capacity = args.tokens * factor
        buffer = make_buffer(group, capacity, args.hidden)
        try:
            tensors = make_inputs(rank, world, args.tokens, args.hidden,
                                  pattern)
            expected = reference_dispatch(*tensors, capacity, world * 2)
            runner = Runner(buffer, *tensors, capacity, False)
            case = f'perf-t{args.tokens}-h{args.hidden}-cap{factor}-{pattern}-gemm{island}'
            emit('INPUT_IDENTITY',
                 rank=rank,
                 case=case,
                 sha256=fingerprint(tensors))
            for arm in ARMS:
                check_case(runner, arm, expected, case, island)
                for _ in range(args.warmup):
                    runner.invoke(arm, island)
                    torch.cuda.synchronize()
            for block in range(args.blocks):
                offset = (block // 2) % len(ARMS)
                order = ARMS[offset:] + ARMS[:offset]
                if block % 2:
                    order = order[::-1]
                for position, arm in enumerate(order):
                    torch.cuda.reset_peak_memory_stats()
                    local_samples = []
                    for _ in range(args.samples):
                        elapsed, truth = measured_call(runner, arm, island,
                                                       group)
                        local_samples.append(elapsed)
                    gathered = [None] * world
                    memory = {
                        'peak_allocated':
                        torch.cuda.max_memory_allocated(),
                        'reserved':
                        torch.cuda.memory_reserved(),
                        'allocator_retries':
                        torch.cuda.memory_stats()['num_alloc_retries']
                    }
                    dist.all_gather_object(gathered, {
                        'samples_ms': local_samples,
                        'truth': truth,
                        'memory': memory
                    },
                                           group=group)
                    if rank == 0:
                        maximum = [
                            max(row['samples_ms'][index] for row in gathered)
                            for index in range(args.samples)
                        ]
                        emit('PREFIX_PERF_BLOCK',
                             case=case,
                             block=block,
                             position=position,
                             arm=arm,
                             maximum_rank_ms=maximum,
                             median_ms=statistics.median(maximum),
                             per_rank=gathered)
        finally:
            torch.cuda.synchronize()
            buffer.destroy()


def profile(rank, world, group, args):
    capacity = args.tokens * 4
    buffer = make_buffer(group, capacity, args.hidden)
    try:
        runner = Runner(buffer,
                        *make_inputs(rank, world, args.tokens, args.hidden,
                                     'balanced'),
                        capacity,
                        False,
                        profile=True)
        for arm in ARMS:
            for _ in range(3):
                runner.invoke(arm, 8)
                torch.cuda.synchronize()
            dist.barrier(group=group)
            if rank == 0:
                with torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA
                ]) as profiler:
                    for _ in range(3):
                        with torch.profiler.record_function(arm):
                            runner.invoke(arm, 8)
                        torch.cuda.synchronize()
                profiler.export_chrome_trace(
                    str(args.output / f'{arm}-rank0.trace.json.gz'))
            else:
                for _ in range(3):
                    runner.invoke(arm, 8)
                    torch.cuda.synchronize()
            dist.barrier(group=group)
    finally:
        torch.cuda.synchronize()
        buffer.destroy()


def worker(local_rank, args):
    rank, world, group = init_dist(local_rank, args.num_processes, seed=42)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    emit(
        'PREFIX_IMPLEMENTATION',
        rank=rank,
        tested_source=TESTED_SOURCE,
        package=deep_ep.__file__,
        public_sort_sha256=hashlib.sha256(
            inspect.getsource(
                EPHandle.deterministic_sort).encode()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_policy=os.environ['EP_AVOID_RECORD_STREAM'],
        arms=ARMS,
        metric=
        'unprofiled synchronized wall ms: dispatch + fixed island + count waits + sort to ready'
    )
    try:
        if args.phase in ('correctness', 'smoke'):
            correctness(rank, world, group)
        if args.phase in ('performance', 'smoke'):
            performance(rank, world, group, args)
        if args.phase == 'profile':
            profile(rank, world, group, args)
        emit('PREFIX_WORKER_COMPLETE', rank=rank, phase=args.phase)
    finally:
        deep_ep.comm.destroy_all_managed_nccl_comm()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',
                        choices=('correctness', 'performance', 'smoke',
                                 'profile'),
                        required=True)
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.tokens > 0 and args.hidden > 0 and args.blocks > 0 and args.samples > 0
    args.output.mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.spawn(worker,
                                args=(args, ),
                                nprocs=args.num_processes)

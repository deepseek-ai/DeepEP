import os

# Must pop before PyTorch imports
os.environ.pop('NCCL_MIN_NCHANNELS', None)
os.environ.pop('NCCL_MAX_NCHANNELS', None)
os.environ.pop('NCCL_NVLS_ENABLE', None)

import argparse
import contextlib
import gc
import math
import random
import weakref
from typing import List, Tuple

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.math import align
from deep_ep.utils.testing import parse_num_bytes

OPS = ('all_reduce', 'reduce_scatter', 'all_gather', 'rs_ag')

GUARD_VALUE = 12345
GUARD_PREFIX_ELEMS = 8
GUARD_SUFFIX_ELEMS = 24
GUARD_BYTES = (GUARD_PREFIX_ELEMS + GUARD_SUFFIX_ELEMS) * torch.float32.itemsize
ROUND_SEED_STRIDE = 1009
RANK_SEED_STRIDE = 1000003
SLEEP_CYCLES = 200001
DESTRUCTION_PERIOD = 2
SESSION_PERIOD = 3
EXACT_PERIOD = 7
EXTERNAL_INPUT_PERIOD = 4
VERIFY_PERIOD = 4
BARRIER_PERIOD = 7
BARRIER_VARIANT_PERIOD = 3
PACKED_PERIOD = 2
HANDLE_RELEASE_PERIOD = 2


def make_groups(world: dist.ProcessGroup, nvl: int) -> List[dist.ProcessGroup]:
    # Contiguous, equal-sized partitions of the world
    sizes = []
    half = world.size() // 2
    if world.size() % 2 == 0 and half > 1 and (half % nvl == 0 or nvl % half == 0):
        sizes.append(half)
    if 1 < nvl < world.size() and nvl not in sizes:
        sizes.append(nvl)
    groups = [world]
    for size in sizes:
        for start in range(0, world.size(), size):
            ranks = list(range(start, start + size))
            group = dist.new_group(ranks)
            if world.rank() in ranks:
                groups.append(group)
    # Pure RDMA rail groups
    if 1 < nvl < world.size():
        for rail in range(nvl):
            ranks = list(range(rail, world.size(), nvl))
            group = dist.new_group(ranks)
            if world.rank() in ranks:
                groups.append(group)
    return groups


def supported(op: str, physical: Tuple[int, int], precision: str) -> bool:
    rdma, nvl = physical
    if op == 'all_gather':
        return True
    if rdma * nvl == 1:
        return False
    if op == 'all_reduce':
        return True
    if precision == 'bf16':
        return rdma > 1 and nvl == 1
    return True


def make_sizes(groups: List[dist.ProcessGroup], args: argparse.Namespace, rng: random.Random) -> List[int]:
    quantum = math.lcm(*(group.size() * 32 for group in groups))
    sizes = [align(min(x * groups[0].size(), args.num_bytes), quantum) for x in (32, 4064, 32736)]
    sizes += [align(rng.randrange(1, args.num_bytes + 1), quantum) for _ in range(3)]
    sizes += [align(args.num_bytes, quantum)]
    sizes += [align(min((i + 1) * quantum, args.num_bytes), quantum) for i in range(max(0, args.num_buckets - len(sizes)))]
    return sizes


def make_schedule(physicals: List[Tuple[int, int]], args: argparse.Namespace, log_skips: bool) -> List[Tuple[int, str, str]]:
    schedule = []
    for group_idx, physical in enumerate(physicals):
        for op in args.ops:
            precisions = args.comm_precisions if op in ('reduce_scatter', 'rs_ag') else ['fp32']
            for precision in precisions:
                if supported(op, physical, precision):
                    schedule.append((group_idx, op, precision))
                elif log_skips:
                    print(f'  > SKIP: group={group_idx} ({physical[0]}x{physical[1]}), {op}/{precision} is not implemented', flush=True)
    assert schedule, 'No supported operations for the selected groups'
    return schedule


def make_inputs(arenas: List[torch.Tensor], group: dist.ProcessGroup, op: str, dtype: torch.dtype, exact: bool,
                use_session: bool) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    inputs, templates, guards = [], [], []
    for arena in arenas:
        payload = arena[GUARD_PREFIX_ELEMS:-GUARD_SUFFIX_ELEMS]
        if use_session:
            payload = torch.empty_like(payload)
        payload = payload.view(dtype)
        if op == 'all_gather':
            payload = payload.view(group.size(), -1)[group.rank()]
        if exact:
            template = torch.randint(0 if dtype == torch.uint8 else -2, 3, payload.shape, dtype=torch.int32, device='cuda').to(dtype)
        else:
            template = torch.randn_like(payload)
        inputs.append(payload)
        templates.append(template)
        if not use_session:
            guards += [arena[:GUARD_PREFIX_ELEMS], arena[-GUARD_SUFFIX_ELEMS:]]
    return inputs, templates, guards


def nccl_reference(op: str, inputs: List[torch.Tensor], group: dist.ProcessGroup, scale: float) -> List[torch.Tensor]:
    outputs = []
    for src in inputs:
        if op in ('all_reduce', 'rs_ag'):
            dst = src.clone()
            dist.all_reduce(dst, group=group)
            dst.mul_(scale)
        elif op == 'reduce_scatter':
            dst = torch.empty(src.numel() // group.size(), dtype=src.dtype, device='cuda')
            dist.reduce_scatter_single(dst, src.view(-1), group=group)
            dst.mul_(scale)
        else:
            dst = torch.empty(src.numel() * group.size(), dtype=src.dtype, device='cuda')
            dist.all_gather_single(dst, src.view(-1), group=group)
        outputs.append(dst)
    return outputs


def verify(pending: List[Tuple[str, torch.cuda.Event, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], bool]]) -> None:
    for case, done, snapshots, references, guards, exact in pending:
        try:
            done.synchronize()
            for actual, expected in zip(snapshots, references, strict=True):
                torch.testing.assert_close(actual.view(-1), expected.view(-1), rtol=0 if exact else 2e-4, atol=0 if exact else 2e-4)
            for guard in guards:
                torch.testing.assert_close(guard, torch.full_like(guard, GUARD_VALUE), rtol=0, atol=0)
        except Exception as error:
            raise AssertionError(f'Failed case {case}: {error}') from error
    pending.clear()


def run_round(round_idx: int, groups: List[dist.ProcessGroup], physicals: List[Tuple[int, int]], args: argparse.Namespace) -> None:
    rank = groups[0].rank()
    rng = random.Random(args.seed + round_idx * ROUND_SEED_STRIDE) # Rank agnostic rng (Must be used in the same way by all ranks)
    stream_rng = random.Random(args.seed + round_idx * ROUND_SEED_STRIDE + rank * RANK_SEED_STRIDE) # Rank specific rng
    alignment = deep_ep.get_num_allocation_alignment()
    sizes = make_sizes(groups, args, rng) # Bucket sizes

    explicit = round_idx % DESTRUCTION_PERIOD == 0 # Alternate destruction modes
    use_session = round_idx % SESSION_PERIOD == SESSION_PERIOD - 1 # Every third round uses session staging

    streams = [torch.cuda.default_stream(), torch.cuda.Stream(), torch.cuda.Stream(priority=-1), torch.cuda.Stream()]
    constructor = stream_rng.choice(streams)
    constructor.wait_stream(torch.cuda.current_stream())

    # Reserve guarded arenas plus similarly sized headroom for session staging
    total = 2 * sum(align(size + GUARD_BYTES, alignment) for size in sizes) + args.num_buckets * alignment
    arenas = []
    with torch.cuda.stream(constructor):
        buffer = deep_ep.BucketBuffer(groups, align(total, alignment), explicitly_destroy=explicit)
        offset = 0
        for size in sizes:
            arena = buffer.storage.narrow(0, offset, size + GUARD_BYTES).view(torch.float32)
            arena.fill_(GUARD_VALUE)
            arenas.append(arena)
            offset += align(size + GUARD_BYTES, alignment)
        ready = torch.cuda.Event()
        ready.record()
    torch.cuda.current_stream().wait_event(ready)

    if rank == 0:
        print(f' > Round {round_idx + 1}/{args.num_rounds}', flush=True)
    # Shuffle supported combinations identically on all ranks, then sample repeats
    schedule = make_schedule(physicals, args, log_skips=rank == 0 and round_idx == 0)
    rng.shuffle(schedule)
    pending, handles = [], []
    for step in range(args.num_tests):
        group_idx, op, precision = schedule[step] if step < len(schedule) else rng.choice(schedule)
        group, physical = groups[group_idx], physicals[group_idx]
        # Strict checks on BF16, which uses small ints to avoid quant error, all_gather and a subset of FP32 cases
        exact = precision == 'bf16' or op == 'all_gather' or (step + round_idx) % EXACT_PERIOD == 0
        # Exercise identity, zero and signed power-of-two scaling
        scale = rng.choice([1.0, 0.0, 0.25, -0.5, 2.0]) if op != 'all_gather' else 1.0
        # All_gather requires num_sms=0
        num_sms = 0 if op == 'all_gather' else rng.choice(args.num_sms)
        # Start with the largest batch and cycle through bucket counts on later steps
        count = args.num_buckets if step == 0 else 1 + step % args.num_buckets
        selected_arenas = rng.sample(arenas, count)
        # Alternate input forms across bucket cycles, including even batch limits
        packed = count == 1 and bool((step // args.num_buckets + round_idx) % PACKED_PERIOD)
        # Vary producer, launch and consumer streams independently on each rank
        producer, launch, consumer = [stream_rng.choice(streams) for _ in range(3)]
        # Exercise several dtypes for all_gather while keeping reduction inputs FP32
        dtype = rng.choice([torch.float32, torch.bfloat16, torch.int32, torch.uint8]) if op == 'all_gather' else torch.float32

        # Prepare independent reference data before the producer writes the inputs
        inputs, templates, guard_views = make_inputs(selected_arenas, group, op, dtype, exact, use_session)
        # Periodically test direct external inputs on the supported pure NVLink all_gather path
        external_input = op == 'all_gather' and not use_session and physical[0] == 1 and step % EXTERNAL_INPUT_PERIOD == 1
        case_details = (f'group={group_idx} ({physical[0]}x{physical[1]}), {op}/{precision}, dtype={dtype}, '
                        f'bytes={[t.nbytes for t in templates]}, SM={num_sms}, scale={scale}, exact={exact}, packed={packed}, '
                        f'session={use_session}, external_input={external_input}, explicit_destroy={explicit}, '
                        f'streams={[streams.index(s) for s in (producer, launch, consumer)]}')
        case = f'round={round_idx + 1}/{args.num_rounds}, case={step + 1}/{args.num_tests}: {case_details}'
        if rank == 0:
            print(f'  > Case {step + 1}/{args.num_tests}: {case_details}', flush=True)

        references = nccl_reference(op, templates, group, scale)
        inputs_ready = torch.cuda.Event()
        inputs_ready.record()

        # Order previous consumption -> input writes -> launch, even when streams differ
        with torch.cuda.stream(producer):
            producer.wait_event(ready)
            producer.wait_event(inputs_ready)
            torch.cuda._sleep(stream_rng.randrange(SLEEP_CYCLES))
            for tensor, template in zip(inputs, templates, strict=True):
                tensor.copy_(template)
            produced = torch.cuda.Event()
            produced.record()
        launch.wait_event(produced)

        sources = inputs[0] if packed else inputs
        session = buffer.session() if use_session else contextlib.nullcontext()
        with session:
            with torch.cuda.stream(launch):
                if op == 'all_reduce':
                    handle = buffer.all_reduce(sources, group=group, num_sms=num_sms, scale=scale)
                elif op in ('reduce_scatter', 'rs_ag'):
                    handle = buffer.reduce_scatter(sources, group=group, num_sms=num_sms, scale=scale, comm_precision=precision)
                elif op == 'all_gather':
                    if use_session:
                        targets = [torch.empty(t.numel() * group.size(), dtype=t.dtype, device='cuda') for t in inputs]
                        handle = buffer.all_gather(sources, dsts=targets[0] if packed else targets, group=group, num_sms=0)
                    elif external_input:
                        external = [t.clone() for t in inputs]
                        targets = [arena[GUARD_PREFIX_ELEMS:-GUARD_SUFFIX_ELEMS].view(dtype) for arena in selected_arenas]
                        handle = buffer.all_gather(external[0] if packed else external,
                                                   dsts=targets[0] if packed else targets,
                                                   group=group,
                                                   num_sms=0)
                        del external
                    else:
                        handle = buffer.all_gather(sources, group=group, num_sms=0)
                if op != 'all_gather':
                    reduction_sms = handle.num_sms
                if op == 'rs_ag':
                    shards = handle.wait()
                    handles.append(handle)
                    handle = buffer.all_gather(shards, dsts=sources if use_session else None, group=group)
            handles.append(handle)

            with torch.cuda.stream(consumer):
                # Alternate retaining and releasing the event after enqueueing the consumer wait
                result = handle.current_stream_wait(release_handle=bool(step % HANDLE_RELEASE_PERIOD))
                results = [result] if packed else result
                assert len(results) == len(inputs), case
                if op != 'all_gather':
                    reduction_op = 'reduce_scatter' if op == 'rs_ag' else op
                    expected_sms = buffer.get_theoretical_num_sms(reduction_op, group) if num_sms is None else num_sms
                    assert reduction_sms == expected_sms, case
                if not use_session:
                    for tensor, src, arena in zip(results, inputs, selected_arenas, strict=True):
                        pointer = arena[GUARD_PREFIX_ELEMS:-GUARD_SUFFIX_ELEMS].data_ptr() if op == 'all_gather' else src.data_ptr()
                        if op == 'reduce_scatter':
                            pointer += group.rank() * tensor.nbytes
                        assert tensor.data_ptr() == pointer, case
                # Snapshot immediately after wait so deferred CPU checks cannot hide missing dependencies
                snapshots = [tensor.clone() for tensor in results]
                guards = [guard.clone() for guard in guard_views]
                ready = torch.cuda.Event()
                ready.record()

        torch.cuda.current_stream().wait_event(ready)
        pending.append((case, ready, snapshots, references, guards, exact))
        # Validate snapshots in batches and leave the final batch for teardown
        if (step + 1) % VERIFY_PERIOD == 0 and step + 1 < args.num_tests:
            verify(pending)
            handles.clear()
        # Periodically vary barrier synchronization and sequencing
        if step % BARRIER_PERIOD == BARRIER_PERIOD - 1:
            with torch.cuda.stream(consumer):
                buffer.barrier(group=group, wait_comm_stream=True,
                               with_cpu_sync=step % BARRIER_VARIANT_PERIOD == 0,
                               sequential=bool(step % BARRIER_VARIANT_PERIOD))
                ready.record()
            torch.cuda.current_stream().wait_event(ready)

    # Keep the last snapshots pending across destruction with all dependencies recorded
    handles.clear()
    if explicit:
        buffer.destroy()
    buffer_ref = weakref.ref(buffer)
    del session, handle, buffer
    gc.collect()
    assert buffer_ref() is None, 'BucketBuffer was retained after teardown'
    verify(pending)
    if rank == 0:
        print(f' > Round {round_idx + 1}/{args.num_rounds} passed: {args.num_tests} cases per rank', flush=True)


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, _, world = init_dist(local_rank, num_local_ranks, args.seed)
    physical = deep_ep.comm.get_physical_domain_size(world)
    max_sms = torch.cuda.get_device_properties('cuda').multi_processor_count
    assert all(sm is None or sm <= max_sms for sm in args.num_sms), 'Requested SM count exceeds the device limit'
    dist_print(f'Config:\n'
               f' > Ranks: {physical[0]} x {physical[1]}\n'
               f' > Rounds: {args.num_rounds}\n'
               f' > Cases per round: {args.num_tests}\n'
               f' > #SM: {args.num_sms}\n'
               f' > Spill check: {args.spill_check}\n'
               f' > Seed: {args.seed}',
               once_in_node=True)
    dist_print('Testing correctness:', once_in_node=True)
    for round_idx in range(args.num_rounds):
        groups = make_groups(world, physical[1])
        physicals = [deep_ep.comm.get_physical_domain_size(group) for group in groups]
        os.environ['EP_AVOID_RECORD_STREAM'] = str(round_idx % 2)
        run_round(round_idx, groups, physicals, args)
        deep_ep.comm.destroy_all_managed_nccl_comm()
        for group in reversed(groups[1:]):
            dist.destroy_process_group(group)
        torch.cuda.empty_cache()
    dist_print(f' > Correctness check passed: {args.num_rounds * args.num_tests} cases per rank', once_in_node=True)
    deep_ep.comm.destroy_all_managed_nccl_comm()
    dist.destroy_process_group(world)
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description='Test mixed BucketBuffer collectives')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--ops', nargs='+', choices=OPS, default=list(OPS))
    parser.add_argument('--num-bytes',
                        type=parse_num_bytes,
                        default=1 << 30,
                        help='Maximum full-message bytes per bucket, before alignment')
    parser.add_argument('--num-sms', nargs='+', default=['auto', '6', '8', '12'])
    parser.add_argument('--num-buckets', type=int, default=8, help='Maximum buckets per case')
    parser.add_argument('--num-tests', type=int, default=48, help='Cases per round')
    parser.add_argument('--num-rounds', type=int, default=12, help='Buffer creation/destruction rounds')
    parser.add_argument('--comm-precisions', nargs='+', choices=('fp32', 'bf16'), default=['fp32', 'bf16'])
    parser.add_argument('--spill-check', choices=('on', 'off'), default='on')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if min(args.num_processes, args.num_tests, args.num_rounds, args.num_bytes, args.num_buckets) < 1 or args.num_buckets > 64:
        parser.error('Counts and sizes must be positive; num-buckets must be 1..64')
    if any(sm != 'auto' and (not sm.isdecimal() or int(sm) < 2) for sm in args.num_sms):
        parser.error('num-sms must be integers >= 2 if not auto')
    args.num_sms = [None if sm == 'auto' else int(sm) for sm in args.num_sms]
    os.environ['DJ_JIT_CHECK_NO_SPILLS'] = str(int(args.spill_check == 'on'))
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == '__main__':
    main()

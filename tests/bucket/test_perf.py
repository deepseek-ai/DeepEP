import os

# Must pop before PyTorch imports
# Spawned workers must retain the selected NCCL profile
if __name__ == '__main__':
    os.environ.pop('NCCL_MIN_NCHANNELS', None)
    os.environ.pop('NCCL_MAX_NCHANNELS', None)
    os.environ.pop('NCCL_NVLS_ENABLE', None)

import argparse
import contextlib
import json
import random
import statistics
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.math import align
from deep_ep.utils.testing import parse_num_bytes

OPS = ('all_reduce', 'reduce_scatter', 'all_gather')
NUM_WARMUPS = 5
NUM_FLUSH_BYTES = 256 << 20
NUM_SLEEP_CYCLES = 2000000
SHARD_ALIGNMENT = 32
MAX_BUCKETS = 64
RAGGED_SIZE_PERIOD = 7
DEFAULT_NUM_BYTES = [1 << 20, 64 << 20, 1 << 30]
NCCL_CONFIGS = {
    'auto': {},
    'cta32': {'NCCL_MIN_CTAS': '32', 'NCCL_MAX_CTAS': '32'},
    'cta64': {'NCCL_MIN_CTAS': '64', 'NCCL_MAX_CTAS': '64'},
    'ring': {'NCCL_ALGO': 'Ring'},
    'tree': {'NCCL_ALGO': 'allreduce:Tree'},
    'nvls': {'NCCL_ALGO': 'allreduce:NVLS', 'NCCL_NVLS_ENABLE': '1'},
    'nvls-tree': {'NCCL_ALGO': 'allreduce:NVLSTree', 'NCCL_NVLS_ENABLE': '1'},
}
NCCL_TUNING_VARS = (
    'NCCL_ALGO', 'NCCL_PROTO', 'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS',
    'NCCL_MIN_CTAS', 'NCCL_MAX_CTAS', 'NCCL_NVLS_ENABLE', 'NCCL_COLLNET_ENABLE',
)


def measure(fn: Callable[[], Any], group: dist.ProcessGroup, flush: torch.Tensor, num_tests: int) -> float:
    # Warmup
    for _ in range(NUM_WARMUPS):
        fn()
    torch.cuda.synchronize()

    # Keep cache flushing and rank synchronization outside the measured interval
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for start, end in zip(start_events, end_events, strict=True):
        flush.zero_()
        dist.barrier(group=group)
        torch.cuda._sleep(NUM_SLEEP_CYCLES)
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()

    # Take the median of the slowest rank's duration for each sample, in microseconds
    times_us = torch.tensor([s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events, strict=True)],
                            dtype=torch.float64, device='cuda')
    dist.all_reduce(times_us, op=dist.ReduceOp.MAX, group=group)
    return statistics.median(times_us.cpu().tolist())


def nccl_collective(op: str, inputs: List[torch.Tensor], outputs: List[torch.Tensor], group: dist.ProcessGroup, coalesced: bool) -> None:
    with dist._coalescing_manager(group=group) if coalesced else contextlib.nullcontext():
        for src, dst in zip(inputs, outputs, strict=True):
            if op == 'all_reduce':
                dist.all_reduce(src, group=group)
            elif op == 'reduce_scatter':
                dist.reduce_scatter_single(dst, src, group=group)
            else:
                dist.all_gather_single(dst, src[:src.numel() // group.size()], group=group)


def get_nccl_sm_budget(profile: str) -> str:
    return NCCL_CONFIGS[profile].get('NCCL_MAX_CTAS', 'auto')


def make_sizes(num_bytes: int, num_buckets: int, num_ranks: int, ragged: bool, seed: int) -> List[int]:
    alignment = num_ranks * SHARD_ALIGNMENT
    sizes = [align((num_bytes + num_buckets - 1) // num_buckets, alignment)] * num_buckets
    if ragged and num_buckets > 1:
        sizes = [alignment * (i % RAGGED_SIZE_PERIOD + 1) for i in range(num_buckets - 1)]
        sizes.append(align(max(alignment, num_bytes - sum(sizes)), alignment))
        random.Random(seed + num_bytes + num_buckets).shuffle(sizes)
    return sizes


def get_skip_reason(op: str, num_rdma_ranks: int, num_nvl_ranks: int, comm_precision: str) -> Optional[str]:
    if num_rdma_ranks * num_nvl_ranks == 1 and op != 'all_gather':
        return 'single-rank reduction is not implemented'
    if op == 'reduce_scatter' and comm_precision == 'bf16' and not (num_rdma_ranks > 1 and num_nvl_ranks == 1):
        return 'BF16 communication requires pure RDMA'
    return None


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace, profile: str, result_path: str) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks, args.seed)
    num_rdma_ranks, num_nvl_ranks = deep_ep.comm.get_physical_domain_size(group)
    num_device_sms = torch.cuda.get_device_properties('cuda').multi_processor_count
    assert all(num_sms is None or 2 <= num_sms <= num_device_sms for num_sms in args.num_sms)

    nccl_sm_budget = get_nccl_sm_budget(profile)

    if profile == 'auto':
        dist_print(f'Config:\n'
                   f' > Ranks: {num_rdma_ranks} x {num_nvl_ranks}\n'
                   f' > Operators: {", ".join(args.ops)}\n'
                   f' > Requested bytes: {args.num_bytes}\n'
                   f' > Buckets: {args.num_buckets}\n'
                   f' > DeepEP #SM budget candidates: {["auto" if n is None else n for n in args.num_sms]}\n'
                   f' > NCCL profiles: {", ".join(args.nccl_profiles)} (auto always runs first)\n'
                   f' > Timing samples: {args.num_tests}\n'
                   f' > Communication precision: {args.comm_precision}\n'
                   f' > Spill check: {args.spill_check}\n'
                   f' > Seed: {args.seed}\n',
                   once_in_node=True)
        dist_print('Performance (DeepEP + NCCL auto):', once_in_node=True)
    else:
        dist_print(f'\nNCCL tuning only (profile={profile}, #SM budget={nccl_sm_budget}):', once_in_node=True)

    # Build and shuffle the cases identically on all ranks
    applicable = True
    if profile == 'nvls':
        applicable = num_rdma_ranks == 1 and num_nvl_ranks > 1
    elif profile == 'nvls-tree':
        applicable = num_rdma_ranks > 1 and num_nvl_ranks > 1
    if not applicable and rank_idx == 0:
        print(f' > SKIP NCCL {profile}: incompatible topology', flush=True)
    cases = [(op, size, count) for op in args.ops for size in args.num_bytes for count in args.num_buckets
             if applicable and (profile not in ('tree', 'nvls', 'nvls-tree') or op == 'all_reduce')]
    random.Random(args.seed).shuffle(cases)
    flush = torch.empty(NUM_FLUSH_BYTES, dtype=torch.uint8, device='cuda')
    rows = []

    for op, num_bytes, num_buckets in cases:
        sizes = make_sizes(num_bytes, num_buckets, num_ranks, args.ragged, args.seed)
        precision = args.comm_precision if op == 'reduce_scatter' else 'fp32'
        if rank_idx == 0:
            print(f' > Case: {op}, requested={num_bytes}, bucket_bytes={sizes}, precision={precision}', flush=True)

        # Perf NCCL
        inputs = [torch.zeros(size // torch.float32.itemsize, dtype=torch.float32, device='cuda') for size in sizes]
        outputs = [torch.empty_like(src) if op != 'reduce_scatter' else
                   torch.empty(src.numel() // num_ranks, dtype=src.dtype, device='cuda') for src in inputs]
        baselines = []
        for coalesced in ([False, True] if num_buckets > 1 else [False]):
            fn = partial(nccl_collective, op, inputs, outputs, group, coalesced)
            duration_us = measure(fn, group, flush, args.num_tests)
            baselines.append((duration_us, f'{profile}/{"coalesced" if coalesced else "loop"}'))
        nccl_us, nccl_name = min(baselines)
        if rank_idx == 0:
            label = f'NCCL ({nccl_name}, #SM budget={nccl_sm_budget}):'
            print(f'  > {label:<44} '
                  f'{nccl_us:7.1f} us | {sum(sizes) / nccl_us / 1000:5.1f} GB/s', flush=True)

        del fn, inputs, outputs
        torch.cuda.empty_cache()

        # Perf DeepEP only in the auto profile
        deep_ep_us, best_num_sms = None, None
        reason = get_skip_reason(op, num_rdma_ranks, num_nvl_ranks, args.comm_precision)
        if profile == 'auto' and reason:
            if rank_idx == 0:
                print(f'  > SKIP DeepEP {op}: {reason}', flush=True)
        elif profile == 'auto':
            allocator = deep_ep.BufferAllocator()
            tensors = [allocator.allocate((size // torch.float32.itemsize,), torch.float32) for size in sizes]
            buffer = deep_ep.BucketBuffer(group, allocator, explicitly_destroy=True)
            for tensor in tensors:
                tensor.zero_()
            sources = [tensor.view(num_ranks, -1)[rank_idx] for tensor in tensors] if op == 'all_gather' else tensors
            candidates = [0] if op == 'all_gather' else list(args.num_sms)
            random.Random(args.seed + num_bytes + num_buckets).shuffle(candidates)
            timings = []
            for num_sms in candidates:
                selected_num_sms = buffer.get_theoretical_num_sms(op, group) if num_sms is None else num_sms
                if op == 'all_reduce':
                    fn = partial(buffer.all_reduce, sources, group=group, num_sms=num_sms)
                elif op == 'reduce_scatter':
                    fn = partial(buffer.reduce_scatter, sources, group=group, num_sms=num_sms, comm_precision=args.comm_precision)
                else:
                    fn = partial(buffer.all_gather, sources, group=group, num_sms=num_sms)
                duration_us = measure(lambda fn=fn: fn().wait(), group, flush, args.num_tests)
                timings.append((duration_us, selected_num_sms))
                if rank_idx == 0:
                    label = f'DeepEP (#SM budget={selected_num_sms}):'
                    print(f'  > {label:<44} '
                          f'{duration_us:7.1f} us | {sum(sizes) / duration_us / 1000:5.1f} GB/s | '
                          f'{nccl_us / duration_us:.2f}x NCCL', flush=True)
            deep_ep_us, best_num_sms = min(timings)
            buffer.destroy()
            del fn, buffer, tensors, allocator, sources
        if rank_idx == 0:
            rows.append([op, num_bytes, sizes, deep_ep_us, best_num_sms, nccl_us, nccl_name])

    if rank_idx == 0:
        Path(result_path).write_text(json.dumps(rows))

    deep_ep.comm.destroy_all_managed_nccl_comm()
    dist.destroy_process_group(group)
    dist.destroy_process_group()


def run_profiles(args: argparse.Namespace) -> None:
    # Always run the baseline first, even when auto is omitted from the CLI
    args.nccl_profiles = list(dict.fromkeys(['auto', *args.nccl_profiles]))
    results = {}
    for profile in args.nccl_profiles:
        for key in NCCL_TUNING_VARS:
            os.environ.pop(key, None)
        os.environ.update(NCCL_CONFIGS[profile])

        # Fresh workers inherit this profile before importing PyTorch
        with tempfile.TemporaryDirectory() as directory:
            result_path = str(Path(directory) / 'timings.json')
            torch.multiprocessing.spawn(test, args=(args.num_processes, args, profile, result_path), nprocs=args.num_processes)
            if int(os.getenv('RANK', '0')) != 0:
                continue
            for op, num_bytes, sizes, deep_ep_us, num_sms, nccl_us, nccl_name in json.loads(Path(result_path).read_text()):
                key = (op, num_bytes, tuple(sizes))
                if key not in results:
                    results[key] = [deep_ep_us, num_sms, nccl_us, nccl_name]
                elif nccl_us < results[key][2]:
                    results[key][2:] = [nccl_us, nccl_name]

    if int(os.getenv('RANK', '0')) == 0:
        print('\nBest performance (DeepEP vs best measured NCCL):', flush=True)
        for (op, num_bytes, sizes), (deep_ep_us, num_sms, nccl_us, nccl_name) in results.items():
            if deep_ep_us is not None:
                nccl_sm_budget = get_nccl_sm_budget(nccl_name.split('/')[0])
                print(f' > {op}: requested={num_bytes}, bytes={sum(sizes)}, buckets={len(sizes)} | '
                      f'DeepEP (#SM budget={num_sms}) {deep_ep_us:7.1f} us | '
                      f'NCCL ({nccl_name}, #SM budget={nccl_sm_budget}) {nccl_us:7.1f} us | '
                      f'{nccl_us / deep_ep_us:.2f}x NCCL', flush=True)
        print(' > Performance test passed (supported cases)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test BucketBuffer performance')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--ops', nargs='+', choices=OPS, default=list(OPS))
    parser.add_argument('--num-bytes', nargs='+', type=parse_num_bytes, default=DEFAULT_NUM_BYTES)
    parser.add_argument('--num-sms', nargs='+', default=['auto', '6', '8', '12'])
    parser.add_argument('--num-buckets', nargs='+', type=int, default=[1, 8])
    parser.add_argument('--num-tests', type=int, default=30)
    parser.add_argument('--comm-precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--nccl-profiles', nargs='+', choices=tuple(NCCL_CONFIGS), default=['auto', 'cta32', 'cta64'],
                        help='NCCL tuning profiles, auto is always included and runs first')
    parser.add_argument('--ragged', action='store_true', help='Mix one large bucket with small buckets')
    parser.add_argument('--spill-check', choices=('on', 'off'), default='on')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.num_processes < 1 or args.num_tests < 1 or min(args.num_bytes) < 1 or any(n < 1 or n > MAX_BUCKETS for n in args.num_buckets):
        parser.error(f'Require positive sizes/processes, num-tests >= 1 and 1..{MAX_BUCKETS} buckets')
    if any(sm != 'auto' and (not sm.isdecimal() or int(sm) < 2) for sm in args.num_sms):
        parser.error('num-sms must be auto or integers >= 2')
    args.num_sms = [None if sm == 'auto' else int(sm) for sm in args.num_sms]
    os.environ['DJ_JIT_CHECK_NO_SPILLS'] = str(int(args.spill_check == 'on'))
    run_profiles(args)

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.distributed as dist


def flush_l2_cache(enabled: bool = True):
    """
    Flush the GPU L2 cache by writing a large zero-initialized tensor.

    Arguments:
        enabled: if `False`, does nothing.
    """
    l2_flush_cache_size = 256e6
    if enabled:
        torch.empty(int(l2_flush_cache_size // 4), dtype=torch.int, device='cuda').zero_()


def bench(fn, num_warmups: int = 30, num_tests: int = 30,
          post_fn: Optional[Callable] = None, flush_l2: bool = True):
    """
    Benchmark a function using CUDA events.

    Arguments:
        fn: the function to benchmark.
        num_warmups: the number of warmup iterations.
        num_tests: the number of measurement iterations.
        post_fn: an optional function to call after each test iteration.
        flush_l2: whether to flush the L2 cache before each iteration.

    Returns:
        avg: the average execution time in seconds.
        min: the minimum execution time in seconds.
        max: the maximum execution time in seconds.
    """
    torch.cuda.synchronize()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        flush_l2_cache(flush_l2)
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times)


class empty_suppress:

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    """
    Context manager to suppress stdout and stderr output.
    """

    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(fn,
                 kernel_names: Union[str, tuple],
                 num_tests: int = 30,
                 suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None,
                 flush_l2: bool = True,
                 barrier_comm_profiling: bool = False,
                 num_kernels_per_period: int = 1,
                 barrier: Optional[Callable] = None):
    """
    Benchmark a function using the PyTorch profiler (kineto) to get per-kernel timing.

    Arguments:
        fn: the function to benchmark.
        kernel_names: the CUDA kernel name(s) to profile.
        num_tests: the number of test iterations.
        suppress_kineto_output: whether to suppress profiler output.
        trace_path: the path to save the Chrome trace (`None` to skip).
        flush_l2: whether to flush the L2 cache before each iteration.
        barrier_comm_profiling: whether to insert a barrier before each iteration to reduce
            unbalanced CPU launch overhead.
        num_kernels_per_period: the number of kernels launched per test period.
        barrier: a custom barrier function to use instead of `dist.all_reduce`.

    Returns:
        durations: the average kernel duration(s) in seconds.
    """
    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)

    # Skip profiling
    # Conflict with Nsight Systems, Nsight Compute and Compute Sanitizer
    if int(os.environ.get('EP_USE_NVIDIA_TOOLS', 0)):
        return (1, ) * len(kernel_names) if is_tuple else 1

    # For some auto-tuning kernels with prints
    fn()
    torch.cuda.synchronize()

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    barrier_comm_profiling &= int(os.environ.get('EP_DISABLE_BARRIER_PROFILING', 0)) == 0
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule, acc_events=True)
        dummy = torch.ones(1, dtype=torch.float, device='cuda')
        with profiler:
            for i in range(2):
                for _ in range(num_tests):
                    # Flush L2 cache
                    flush_l2_cache(flush_l2)

                    # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                    if barrier_comm_profiling:
                        torch.cuda._sleep(int(2e7)) # ~10ms

                        # Some network may have ring-based implement, so be careful to use `all_reduce`
                        if barrier is None:
                            dist.all_reduce(dummy)
                        else:
                            barrier()
                    fn()
                torch.cuda.synchronize()
                profiler.step()

    # Parse the profiling table
    prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) <= 1, f'Errors of the kernel {name} in the profiling table: {prof_lines}'

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel durations
    units = {'ms': 1e3, 'us': 1e6}
    kernel_durations = []
    for name in kernel_names:
        total_time = 0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += float(time_str.replace(unit, '')) / scale * int(num_str)
                        total_num += int(num_str)
                        break
        kernel_durations.append(total_time / total_num if total_num > 0 else 0)

    # Expand the kernels by periods
    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            profiler.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())

        for i, kernel_name in enumerate(kernel_names):
            events = [event for event in profile_data['traceEvents'] if f'::{kernel_name}' in event['name']]
            events = sorted(events, key=lambda event: event['ts'])
            durations = [event['dur'] / 1e6 for event in events]
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations[i] = [sum(durations[j::num_kernels_per_period]) / num_kernel_patterns for j in range(num_kernels_per_period)]

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]


def parse_num_bytes(text: str) -> int:
    """Parse a byte count, optionally with a binary suffix: `1G`, `64M`, `512k`.
    """
    match = re.fullmatch(r'\s*([0-9.]+)\s*([kmgt])?(?:i?b)?\s*', text, re.IGNORECASE)
    if match is None:
        raise ValueError(
            f'{text!r} is not a byte count; expected e.g. 1G, 64M, 512K, 1048576')
    scale = {None: 1, 'k': 1 << 10, 'm': 1 << 20, 'g': 1 << 30, 't': 1 << 40}
    suffix = match.group(2)
    try:
        value = int(float(match.group(1)) * scale[suffix.lower() if suffix else None])
    except ValueError:
        raise ValueError(f'{text!r} has an unparseable number')
    if value <= 0:
        raise ValueError(f'byte count must be positive, got {value}')
    return value

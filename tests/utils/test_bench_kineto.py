import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import torch


_TESTING_PATH = Path(__file__).parents[2] / 'deep_ep' / 'utils' / 'testing.py'
_TESTING_SPEC = importlib.util.spec_from_file_location('deep_ep_testing', _TESTING_PATH)
testing = importlib.util.module_from_spec(_TESTING_SPEC)
_TESTING_SPEC.loader.exec_module(testing)

class _CpuProfiler(torch.profiler.profile):

    instances = []

    def __init__(self, **kwargs):
        kwargs['activities'] = [torch.profiler.ProfilerActivity.CPU]
        super().__init__(**kwargs)
        self.events = None
        self.instances.append(self)

    def key_averages(self):
        self.events = super().key_averages()

        def cpu_table(**kwargs):
            kwargs['sort_by'] = 'cpu_time_total'
            return self.events.table(**kwargs)

        return SimpleNamespace(table=cpu_table)


class BenchKinetoTest(unittest.TestCase):

    def test_kineto_warmup_budget_and_recorded_samples(self):
        real_ones = torch.ones
        calls, barrier = Mock(), Mock()

        def fn():
            with torch.profiler.record_function('kineto_test_kernel'):
                calls()

        with patch.object(torch.profiler, 'profile', _CpuProfiler), \
             patch.object(torch, 'ones', side_effect=lambda *args, **kwargs: real_ones(
                 *args, **{**kwargs, 'device': 'cpu'})), \
             patch.object(torch.cuda, 'synchronize'), \
             patch.object(torch.cuda, '_sleep', lambda _: None, create=True), \
             patch.object(testing, 'flush_l2_cache') as flush_cache, \
             patch.dict(os.environ, {'EP_USE_NVIDIA_TOOLS': '0', 'EP_DISABLE_BARRIER_PROFILING': '0'}):
            for num_tests in (30, 1):
                with self.subTest(num_tests=num_tests):
                    calls.reset_mock()
                    flush_cache.reset_mock()
                    barrier.reset_mock()
                    _CpuProfiler.instances.clear()

                    testing.bench_kineto(
                        fn,
                        'kineto_test_kernel',
                        num_tests=num_tests,
                        flush_l2=True,
                        barrier_comm_profiling=True,
                        barrier=barrier,
                    )

                    self.assertEqual(calls.call_count, num_tests + 2)
                    self.assertEqual(flush_cache.call_args_list, [call(True)] * (num_tests + 1))
                    self.assertEqual(barrier.call_count, num_tests + 1)
                    event_counts = [event.count for event in _CpuProfiler.instances[-1].events
                                    if event.key == 'kineto_test_kernel']
                    self.assertEqual(event_counts, [num_tests])


if __name__ == '__main__':
    unittest.main()

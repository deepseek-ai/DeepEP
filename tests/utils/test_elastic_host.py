import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def add_module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def load_elastic():
    module_names = (
        'deep_ep',
        'deep_ep.buffers',
        'deep_ep._C',
        'deep_ep.utils',
        'deep_ep.utils.event',
        'deep_ep.utils.math',
        'deep_ep.utils.semantic',
        'deep_ep.utils.envs',
        'deep_ep.utils.comm',
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        package = add_module('deep_ep')
        package.__path__ = []
        buffers = add_module('deep_ep.buffers')
        buffers.__path__ = []
        utils = add_module('deep_ep.utils')
        utils.__path__ = []

        add_module('deep_ep._C', EventHandle=type('EventHandle', (), {}))
        add_module('deep_ep.utils.event', EventOverlap=type('EventOverlap', (), {}))
        add_module('deep_ep.utils.math', align=lambda x, y: math.ceil(x / y) * y)
        add_module(
            'deep_ep.utils.semantic',
            value_or=lambda value, default: default if value is None else value,
            weak_lru=lambda *args, **kwargs: lambda function: function,
        )
        add_module(
            'deep_ep.utils.envs',
            check_fast_rdma_atomic_support=lambda: True,
            check_nvlink_connections=lambda group: None,
            check_torch_deterministic=lambda: None,
            get_nvlink_gbs=lambda: 0,
            get_rdma_gbs=lambda: 0,
        )
        add_module('deep_ep.utils.comm', get_nccl_comm_handle=lambda *args, **kwargs: None)

        path = Path(__file__).parents[2] / 'deep_ep' / 'buffers' / 'elastic.py'
        spec = importlib.util.spec_from_file_location('deep_ep.buffers.elastic', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, saved_module in saved_modules.items():
            if saved_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved_module


def make_hybrid_buffer(elastic):
    """Build the minimal ElasticBuffer topology used by the host-only test."""
    buffer = elastic.ElasticBuffer.__new__(elastic.ElasticBuffer)
    buffer.num_rdma_ranks = 2
    buffer.num_nvlink_ranks = 8
    buffer.num_scaleout_ranks = 2
    buffer.num_scaleup_ranks = 8
    buffer.num_ranks = 16
    buffer.prefer_overlap_with_compute = True
    return buffer


class TheoreticalSMTests(unittest.TestCase):
    def test_missing_hybrid_bandwidth_falls_back_to_device_sm_count(self):
        elastic = load_elastic()
        buffer = make_hybrid_buffer(elastic)

        properties = types.SimpleNamespace(multi_processor_count=132)
        with patch.object(torch.cuda, 'get_device_properties', return_value=properties):
            num_sms = buffer.get_theoretical_num_sms(64, 8)

        self.assertEqual(num_sms, 132)


if __name__ == '__main__':
    unittest.main()

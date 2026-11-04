import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def envs(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    package = types.ModuleType('deep_ep')
    package.__path__ = [str(root / 'deep_ep')]
    utils = types.ModuleType('deep_ep.utils')
    utils.__path__ = [str(root / 'deep_ep/utils')]
    monkeypatch.setitem(sys.modules, 'deep_ep', package)
    monkeypatch.setitem(sys.modules, 'deep_ep.utils', utils)
    monkeypatch.setitem(sys.modules, 'deep_ep._C', types.ModuleType('deep_ep._C'))
    monkeypatch.setitem(sys.modules, 'deep_ep.comm', types.ModuleType('deep_ep.comm'))
    spec = importlib.util.spec_from_file_location('deep_ep.utils.envs', root / 'deep_ep/utils/envs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    'output,nic,rate,fast',
    [
        (("CA 'mlx5_0'\n    Number of ports: 0\n"
          "CA 'mlx5_1'\n    CA type: MT4131\n    Port 1:\n        Rate: 400\n"), 'mlx5_0', 0, False),
        (("CA 'mlx5_0'\n    CA type: MT4119\n    Port 1:\n        Rate: 200\n"
          "CA 'mlx5_1'\n    CA type: MT4131\n    Port 1:\n        Rate: 400\n"), 'mlx5_0', 25, False),
        (("CA 'mlx5_0'\n    CA type: MT4119\n    Port 1:\n        Rate: 200\n"
          "CA 'mlx5_1'\n    CA type: MT4131\n    Port 1:\n        Rate: 400\n"), 'mlx5_1', 50, True),
        ("CA 'mlx5X0'\n    CA type: MT4131\n    Port 1:\n        Rate: 400\n", 'mlx5.0', 0, False),
        ("CA 'mlx5_1'\n    CA type: MT4131\n    Port 1:\n        Rate: 400\n", 'mlx5_0', 0, False),
    ],
)
def test_ibstat_does_not_borrow_fields_from_other_ca(envs, monkeypatch, output, nic, rate, fast):
    monkeypatch.setattr(envs.subprocess, 'run', lambda *args, **kwargs: types.SimpleNamespace(stdout=output))
    assert envs.get_rdma_gbs(nic) == rate
    assert envs.check_fast_rdma_atomic_support(nic) is fast

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


def test_uses_current_logical_device_and_caches_each_device(envs, monkeypatch):
    # The UUIDs and link rates are synthetic; logical device 0 need not be
    # physical GPU 0 when CUDA_VISIBLE_DEVICES is set.
    uuids = ['00000000-0000-0000-0000-000000000001',
             '00000000-0000-0000-0000-000000000002']
    rates = {f'GPU-{uuids[0]}': 100, f'GPU-{uuids[1]}': 40}
    current = [1]
    calls = []
    monkeypatch.setattr(envs.torch.cuda, 'current_device', lambda: current[0])
    monkeypatch.setattr(envs.torch.cuda, 'get_device_properties',
                        lambda index: types.SimpleNamespace(uuid=uuids[index]))

    def run(argv, **kwargs):
        calls.append(argv)
        if '-i' in argv:
            rate = rates[argv[argv.index('-i') + 1]]
            output = f'GPU 0: synthetic\n  Link 0: {rate} GB/s\n'
        else:
            output = 'GPU 0: synthetic\n  Link 0: 100 GB/s\nGPU 1: synthetic\n  Link 0: 40 GB/s\n'
        return types.SimpleNamespace(stdout=output)

    monkeypatch.setattr(envs.subprocess, 'run', run)
    assert envs.get_nvlink_gbs() == pytest.approx(36)
    current[0] = 0
    assert envs.get_nvlink_gbs() == pytest.approx(90)
    current[0] = 1
    assert envs.get_nvlink_gbs() == pytest.approx(36)
    assert len(calls) == 2
    assert envs.get_nvlink_gbs(0.5) == pytest.approx(20)
    assert len(calls) == 3
    assert envs.get_nvlink_gbs.cache_info().currsize == 3
    envs.get_nvlink_gbs.cache_clear()
    assert envs.get_nvlink_gbs() == pytest.approx(36)
    assert len(calls) == 4


def test_failure_to_identify_device_returns_zero(envs, monkeypatch):
    def fail():
        raise RuntimeError('no CUDA device')

    monkeypatch.setattr(envs.torch.cuda, 'current_device', fail)
    monkeypatch.setattr(envs.subprocess, 'run', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    assert envs.get_nvlink_gbs() == 0


def test_failed_query_does_not_log_uuid(envs, monkeypatch, capsys):
    uuid = '00000000-0000-0000-0000-000000000001'
    monkeypatch.setattr(envs.torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(envs.torch.cuda, 'get_device_properties',
                        lambda index: types.SimpleNamespace(uuid=uuid))

    def fail(argv, **kwargs):
        raise envs.subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(envs.subprocess, 'run', fail)
    assert envs.get_nvlink_gbs() == 0
    assert uuid not in capsys.readouterr().out

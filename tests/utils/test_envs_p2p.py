import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def envs(monkeypatch):
    """Exercise the actual preflight without requiring torch or the CUDA extension."""
    root = Path(__file__).parents[2] / 'deep_ep'
    package = ModuleType('deep_ep')
    package.__path__ = [str(root)]
    utils = ModuleType('deep_ep.utils')
    utils.__path__ = [str(root / 'utils')]
    comm = ModuleType('deep_ep.utils.comm')
    comm.get_nccl_comm_handle = Mock()
    torch = ModuleType('torch')
    dist = ModuleType('torch.distributed')
    dist.ProcessGroup = object
    dist.all_gather_object = lambda output, obj, group: output.__setitem__(slice(None), group.gather(obj))
    torch.distributed = dist
    torch.cuda = SimpleNamespace(current_device=Mock(return_value=0),
                                 device_count=Mock(return_value=2),
                                 get_device_properties=Mock(side_effect=lambda device: SimpleNamespace(uuid=f'GPU-{device}')),
                                 can_device_access_peer=Mock(return_value=True))
    for name, module in [('deep_ep', package), ('deep_ep.utils', utils), ('deep_ep._C', ModuleType('deep_ep._C')),
                         ('deep_ep.utils.comm', comm), ('torch', torch), ('torch.distributed', dist)]:
        monkeypatch.setitem(sys.modules, name, module)
    for name in ('p2p', 'envs'):
        spec = importlib.util.spec_from_file_location(f'deep_ep.utils.{name}', root / 'utils' / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module, '_get_physical_node_id', Mock(return_value='node-a'))
    return module


@pytest.fixture(params=['torch', 'mpi'])
def make_group(request):

    def make(rank, replies):
        calls = []

        def gather(obj):
            calls.append(obj)
            return replies[len(calls) - 1]

        if request.param == 'mpi':
            group = SimpleNamespace(Get_rank=lambda: rank, allgather=gather)
        else:
            group = SimpleNamespace(rank=lambda: rank, size=lambda: len(replies[0]), gather=gather)
        return group, calls

    return make


@pytest.mark.parametrize('failure', ['current_device', 'device_uuid', 'node_id'])
def test_identity_failure_is_gathered_before_any_rank_raises(envs, make_group, failure):
    error = RuntimeError('identity unavailable')
    query = {
        'current_device': envs.torch.cuda.current_device,
        'device_uuid': envs.torch.cuda.get_device_properties,
        'node_id': envs._get_physical_node_id
    }[failure]
    query.side_effect = error
    identity_error = 'rank 0: RuntimeError: identity unavailable'
    replies = [[(None, identity_error), (('node-a', 'GPU-1'), None)]]
    group, calls = make_group(0, replies)

    with pytest.raises(RuntimeError, match=f'could not identify the physical topology: {identity_error}'):
        envs.check_nvlink_connections(group)

    assert calls == [(None, identity_error)]
    envs.torch.cuda.can_device_access_peer.assert_not_called()


def test_healthy_rank_reports_remote_identity_failure_without_querying_peers(envs, make_group):
    identity_error = 'rank 1: RuntimeError: UUID unavailable'
    group, calls = make_group(0, [[(('node-a', 'GPU-0'), None), (None, identity_error)]])

    with pytest.raises(RuntimeError, match=f'could not identify the physical topology: {identity_error}'):
        envs.check_nvlink_connections(group)

    assert calls == [(('node-a', 'GPU-0'), None)]
    envs.torch.cuda.device_count.assert_not_called()


@pytest.mark.parametrize('properties', [SimpleNamespace(), SimpleNamespace(uuid=None)])
def test_missing_uuid_is_reported_collectively(envs, make_group, properties):
    envs.torch.cuda.get_device_properties.side_effect = None
    envs.torch.cuda.get_device_properties.return_value = properties
    error = 'rank 0: RuntimeError: PyTorch did not expose a UUID for the current CUDA device'
    group, calls = make_group(0, [[(None, error), (('node-a', 'GPU-1'), None)]])

    with pytest.raises(RuntimeError, match=error):
        envs.check_nvlink_connections(group)

    assert calls == [(None, error)]


def test_multiple_identity_failures_have_rank_order_on_every_rank(envs, make_group):
    errors = [f'rank {rank}: RuntimeError: identity {rank}' for rank in range(2)]
    messages = []
    for rank in range(2):
        envs.torch.cuda.current_device.side_effect = RuntimeError(f'identity {rank}')
        group, calls = make_group(rank, [[(None, error) for error in errors]])
        with pytest.raises(RuntimeError) as exc:
            envs.check_nvlink_connections(group)
        messages.append(str(exc.value))
        assert calls == [(None, errors[rank])]
    assert messages == ['DeepEP P2P preflight could not identify the physical topology: ' + '; '.join(errors)] * 2


def test_preflight_success_uses_two_collectives(envs, make_group):
    group, calls = make_group(0, [[(('node-a', 'GPU-0'), None), (('node-a', 'GPU-1'), None)], [([(1, True)], None), ([(0, True)], None)]])

    envs.check_nvlink_connections(group)

    assert calls == [(('node-a', 'GPU-0'), None), ([(1, True)], None)]
    envs.torch.cuda.can_device_access_peer.assert_called_once_with(0, 1)


def test_query_failure_is_gathered_and_reported_on_every_rank(envs, make_group):
    identity = [(('node-a', 'GPU-0'), None), (('node-a', 'GPU-1'), None)]
    error = 'rank 1: RuntimeError: peer query failed'
    messages = []
    for rank in range(2):
        envs.torch.cuda.current_device.return_value = rank
        envs.torch.cuda.can_device_access_peer.side_effect = RuntimeError('peer query failed') if rank else None
        group, calls = make_group(rank, [identity, [([(1, True)], None), ([], error)]])
        with pytest.raises(RuntimeError) as exc:
            envs.check_nvlink_connections(group)
        messages.append(str(exc.value))
        assert calls[-1] == (([], error) if rank else ([(1, True)], None))
    assert messages == ['DeepEP P2P preflight could not query the physical topology: ' + error] * 2


def test_unsupported_pairs_are_identical_on_every_rank(envs, make_group):
    identity = [(('node-a', 'GPU-0'), None), (('node-a', 'GPU-1'), None)]
    envs.torch.cuda.can_device_access_peer.return_value = False
    messages = []
    for rank in range(2):
        envs.torch.cuda.current_device.return_value = rank
        group, calls = make_group(rank, [identity, [([(1, False)], None), ([(0, False)], None)]])
        with pytest.raises(RuntimeError, match='2/2 directed pairs') as exc:
            envs.check_nvlink_connections(group)
        messages.append(str(exc.value))
        assert calls[-1] == ([(1 - rank, False)], None)
    assert messages[0] == messages[1]

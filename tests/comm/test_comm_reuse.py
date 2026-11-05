import os
from unittest import mock

import torch.distributed as dist

import deep_ep.comm.handle as comm_mod
from deep_ep.comm.handle import get_nccl_comm_handle

# Every collective `get_nccl_comm_handle` could reach. Each call is recorded, so the
# tests can tell exactly which collectives a path runs.
_COLLECTIVES = ('all_gather_object', 'all_gather', 'all_reduce', 'barrier', 'broadcast', 'broadcast_object_list')


class _FakeBackend:

    def __init__(self, comm_ptr: int):
        self._ptr = comm_ptr

    def _comm_ptr(self) -> int:
        return self._ptr


class _NoCommPtrBackend:
    """A backend from a PyTorch build without `_comm_ptr()`."""


class _FakeGroup:
    """Stands in for a 2-rank ProcessGroup; collectives are monkeypatched."""

    def __init__(self, backend: _FakeBackend):
        self._backend = backend

    def _get_backend(self, _device):
        return self._backend

    def size(self) -> int:
        return 2

    def rank(self) -> int:
        return 0

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


def _call(group, calls, reuse: str = '1', **kwargs):
    """Call get_nccl_comm_handle with every collective recorded into `calls`."""

    def recorder(name):

        def collective(*args, **_kwargs):
            if name == 'all_gather_object':
                # The unique-id exchange on the create-our-own path
                out_list, obj = args[0], args[1]
                calls.append((name, obj))
                out_list[:] = [b'fake-unique-id'] * len(out_list)
            else:
                calls.append((name, None))

        return collective

    patches = [mock.patch.dict(os.environ, {'EP_REUSE_NCCL_COMM': reuse})]
    patches += [mock.patch.object(dist, name, recorder(name)) for name in _COLLECTIVES]
    patches += [
        mock.patch.object(comm_mod._C, 'get_local_nccl_unique_id', create=True, return_value=b'fake-unique-id'),
        mock.patch.object(comm_mod._C, 'create_nccl_comm', create=True, return_value=0xDEADBEEF),
        mock.patch.object(comm_mod._C, 'destroy_nccl_comm', create=True, return_value=None),
    ]
    for p in patches:
        p.start()
    try:
        return get_nccl_comm_handle(group, **kwargs)
    finally:
        for p in reversed(patches):
            p.stop()


def _run(local_ptr: int, reuse: str = '1'):
    """Return the handle, the collectives it ran, and the group (for a second, cache-hitting call)."""
    comm_mod._storage.clear()
    group = _FakeGroup(_FakeBackend(local_ptr))
    calls = []
    handle = _call(group, calls, reuse=reuse)
    return handle, calls, group


def test_live_comm_is_reused_without_a_collective():
    handle, calls, _ = _run(local_ptr=0x1234)
    assert handle.get() == 0x1234
    assert not handle.managed
    # No collective at all: a rank that reaches this alone (e.g. through a size or
    # topology helper) returns instead of waiting for its peers.
    assert calls == []


def test_null_comm_creates_managed_comm():
    handle, calls, _ = _run(local_ptr=0)
    assert handle.get() == 0xDEADBEEF, 'a null `_comm_ptr()` must never be reused'
    assert handle.managed
    # The only collective is the unique-id exchange of the create-our-own path.
    assert calls == [('all_gather_object', b'fake-unique-id')]


def test_env_hatch_disables_reuse_even_when_comm_is_live():
    # `EP_REUSE_NCCL_COMM=0` must win even when a live pointer is available.
    handle, calls, _ = _run(local_ptr=0x1234, reuse='0')
    assert handle.managed, 'EP_REUSE_NCCL_COMM=0 must force a DeepEP-managed comm'
    assert handle.get() != 0x1234, 'must not borrow PyTorch\'s comm when reuse is off'
    assert calls == [('all_gather_object', b'fake-unique-id')]


def test_cache_hit_runs_no_collective():
    # The cache check sits above the unique-id exchange, so a second call on the
    # same group returns the cached managed comm without any collective.
    first, calls1, group = _run(local_ptr=0)
    assert calls1 == [('all_gather_object', b'fake-unique-id')]

    calls2 = []
    second = _call(group, calls2)
    assert second is first, 'the cache must return the same handle'
    assert calls2 == []


def test_force_new_comm_never_reuses_or_hits_the_cache():
    cached, _, group = _run(local_ptr=0x1234)
    calls = []
    fresh = _call(group, calls, force_new_comm=True)
    assert fresh is not cached and fresh.managed
    assert calls == [('all_gather_object', b'fake-unique-id')]
    assert comm_mod._storage[group] is cached, 'a forced comm must not replace the cached one'


def test_backend_without_comm_ptr_creates_managed_comm():
    comm_mod._storage.clear()
    calls = []
    handle = _call(_FakeGroup(_NoCommPtrBackend()), calls)
    assert handle.managed
    assert calls == [('all_gather_object', b'fake-unique-id')]


if __name__ == '__main__':
    test_live_comm_is_reused_without_a_collective()
    test_null_comm_creates_managed_comm()
    test_env_hatch_disables_reuse_even_when_comm_is_live()
    test_cache_hit_runs_no_collective()
    test_force_new_comm_never_reuses_or_hits_the_cache()
    test_backend_without_comm_ptr_creates_managed_comm()
    print('All comm-reuse tests passed')

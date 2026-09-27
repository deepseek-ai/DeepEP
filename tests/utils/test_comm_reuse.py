import os
from unittest import mock

import torch.distributed as dist

import deep_ep.utils.comm as comm_mod
from deep_ep.utils.comm import get_nccl_comm_handle


class _FakeBackend:

    def __init__(self, comm_ptr: int):
        self._ptr = comm_ptr

    def _comm_ptr(self) -> int:
        return self._ptr


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


def _run(local_ptr: int, gathered_flags, reuse: str = '1'):
    """Drive get_nccl_comm_handle with a faked group.

    `gathered_flags` is what the decision all_gather returns (one bool per rank). Returns the
    handle, the objects contributed to all_gather_object, and the group, so a caller can make a
    second (cache-hitting) call on the same group.
    """
    comm_mod._storage.clear()
    os.environ['EP_REUSE_NCCL_COMM'] = reuse
    group = _FakeGroup(_FakeBackend(local_ptr))
    calls = []

    def fake_all_gather_object(out_list, obj, _group):
        calls.append(obj)
        if isinstance(obj, bool):
            out_list[:] = gathered_flags
        else:
            # unique-id gather on the create-our-own path
            out_list[:] = [b'fake-unique-id'] * len(out_list)

    with mock.patch.object(dist, 'all_gather_object', fake_all_gather_object), \
         mock.patch.object(comm_mod._C, 'get_local_nccl_unique_id', create=True, return_value=b'fake-unique-id'), \
         mock.patch.object(comm_mod._C, 'create_nccl_comm', create=True, return_value=0xDEADBEEF), \
         mock.patch.object(comm_mod._C, 'destroy_nccl_comm', create=True, return_value=None):
        handle = get_nccl_comm_handle(group)
    return handle, calls, group


def test_all_ranks_have_comm_reuses():
    handle, calls, _ = _run(local_ptr=0x1234, gathered_flags=[True, True])
    assert handle.get() == 0x1234
    assert not handle.managed
    # Exactly one collective: the decision gather (no unique-id exchange).
    assert calls == [True]


def test_null_local_ptr_creates_managed_comm():
    handle, calls, _ = _run(local_ptr=0, gathered_flags=[False, False])
    assert handle.get() == 0xDEADBEEF
    assert handle.managed
    # Decision gather ran first (reached even with a null local pointer), then the
    # unique-id gather on the fall-through path.
    assert calls[0] is False
    assert len(calls) == 2


def test_rank_skew_takes_fallthrough_on_every_rank():
    # This rank holds a live comm, but a peer does not: must NOT reuse, must proceed to the
    # same create-our-own path the null rank takes.
    handle, calls, _ = _run(local_ptr=0x1234, gathered_flags=[True, False])
    assert handle.get() == 0xDEADBEEF
    assert handle.managed
    assert calls[0] is True
    assert len(calls) == 2


def test_env_hatch_disables_reuse_even_when_all_ranks_live():
    # `EP_REUSE_NCCL_COMM=0` is the documented escape hatch for the reuse path, and it must win
    # even when every rank reports a live pointer. Without this case, replacing the getenv lookup
    # with a literal `True` leaves the rest of this file green: the hatch had no coverage at all.
    try:
        handle, calls, _ = _run(local_ptr=0x1234, gathered_flags=[True, True], reuse='0')
        assert handle.managed, 'EP_REUSE_NCCL_COMM=0 must force a DeepEP-managed comm'
        assert handle.get() != 0x1234, 'must not borrow PyTorch\'s comm when reuse is off'
        # The reuse-disabled path skips the nullness decision entirely: only the unique-id
        # gather runs, so no boolean is ever contributed.
        assert all(not isinstance(c, bool) for c in calls)
    finally:
        os.environ['EP_REUSE_NCCL_COMM'] = '1'


def test_cache_hit_bypasses_the_decision_collective():
    # The cache check sits above the decision collective, so a cache hit runs no collective at
    # all. That is correct when every rank calls this helper uniformly, and a hang if they do
    # not, so the guarantee is pinned here rather than left implied.
    first, calls1, group = _run(local_ptr=0x1234, gathered_flags=[True, True])
    assert calls1 == [True]

    calls2 = []

    def counting_gather(out_list, obj, _group):
        calls2.append(obj)
        out_list[:] = [True] * len(out_list)

    with mock.patch.object(dist, 'all_gather_object', counting_gather):
        second = get_nccl_comm_handle(group)
    assert second is first, 'the cache must return the same handle'
    assert calls2 == [], 'a cache hit runs no collective, so callers must be group-uniform'


if __name__ == '__main__':
    test_all_ranks_have_comm_reuses()
    test_null_local_ptr_creates_managed_comm()
    test_rank_skew_takes_fallthrough_on_every_rank()
    test_env_hatch_disables_reuse_even_when_all_ranks_live()
    test_cache_hit_bypasses_the_decision_collective()
    print('All comm-reuse tests passed')

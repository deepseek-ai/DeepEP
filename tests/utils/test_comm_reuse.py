import os
import unittest
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


class TestCommReuseDecision(unittest.TestCase):
    """The PyTorch-comm reuse decision must be group-uniform: reuse only when
    ALL ranks report a live `_comm_ptr()`, and the decision collective must be
    reached by every rank regardless of its local pointer value."""

    def setUp(self):
        comm_mod._storage.clear()
        os.environ['EP_REUSE_NCCL_COMM'] = '1'

    def _run(self, local_ptr: int, gathered_flags):
        """Drive get_nccl_comm_handle with a faked group; `gathered_flags` is
        what the decision all_gather returns (one bool per rank)."""
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
        return handle, calls

    def test_all_ranks_have_comm_reuses(self):
        handle, calls = self._run(local_ptr=0x1234, gathered_flags=[True, True])
        self.assertEqual(handle.get(), 0x1234)
        self.assertFalse(handle.managed)
        # Exactly one collective: the decision gather (no unique-id exchange).
        self.assertEqual(calls, [True])

    def test_null_local_ptr_creates_managed_comm(self):
        handle, calls = self._run(local_ptr=0, gathered_flags=[False, False])
        self.assertEqual(handle.get(), 0xDEADBEEF)
        self.assertTrue(handle.managed)
        # Decision gather ran first (reached even with a null local pointer),
        # then the unique-id gather on the fall-through path.
        self.assertEqual(calls[0], False)
        self.assertEqual(len(calls), 2)

    def test_rank_skew_takes_fallthrough_on_every_rank(self):
        # This rank holds a live comm, but a peer does not: must NOT reuse,
        # must proceed to the same create-our-own path the null rank takes.
        handle, calls = self._run(local_ptr=0x1234, gathered_flags=[True, False])
        self.assertEqual(handle.get(), 0xDEADBEEF)
        self.assertTrue(handle.managed)
        self.assertEqual(calls[0], True)
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()

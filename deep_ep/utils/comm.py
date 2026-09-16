import os
import time

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep._C as _C


class NCCLCommHandle:
    """
    A wrapper around a raw NCCL communicator. Manages the lifecycle of the communicator if created by DeepEP,
    or simply wraps an existing one if obtained from PyTorch.

    Attributes:
        nccl_comm: the raw NCCL communicator.
        managed: whether the communicator was created by DeepEP and should be destroyed when this handle is dropped.
    """

    def __init__(self, nccl_comm: int, managed: bool):
        self.nccl_comm = nccl_comm
        self.managed = managed
        self.destroy = _C.destroy_nccl_comm

    def __del__(self):
        if self.managed:
            self.destroy(self.nccl_comm)

    def get(self) -> int:
        """
        Get the raw NCCL communicator.

        Returns:
            nccl_comm: the raw NCCL communicator.
        """
        return self.nccl_comm

_storage = dict()


def get_nccl_comm_handle(group: dist.ProcessGroup, force_new_comm: bool = False) -> NCCLCommHandle:
    """
    Get or create an NCCL communicator handle for the given process group.
    Results are cached, so subsequent calls with the same group return the same handle.

    Reuse of PyTorch's communicator only happens when every rank in the group already
    holds one (the decision is agreed across the group, so all ranks take the same
    branch). If any rank's communicator is not materialized yet, all ranks create a
    DeepEP-managed comm instead — and that self-built comm stays cached for the
    lifetime of the group entry (an extra NCCL communicator and its buffers), even if
    PyTorch's own communicator materializes later. Call `destroy_all_managed_nccl_comm`
    to release cached comms.

    Arguments:
        group: the communication group.
        force_new_comm: if set, never reuse PyTorch's communicator and never hit the cache; always
            create a fresh DeepEP-managed comm.

    Returns:
        handle: the NCCL communicator handle.
    """
    # Check cache hit
    global _storage
    if not force_new_comm and group in _storage:
        return _storage[group]

    # New PyTorch has such API
    backend = group._get_backend(torch.device('cuda'))
    if not force_new_comm and hasattr(backend, '_comm_ptr') and int(os.getenv('EP_REUSE_NCCL_COMM', '1')):
        # PyTorch creates NCCL communicators lazily: `_comm_ptr()` is a passive
        # accessor that returns 0 when the group+device has no communicator yet
        # (eager creation only happens when `init_process_group` received
        # `device_id=...`). Reusing a null handle would crash later in C++
        # (e.g. `ncclTeamWorld` dereferences `comm->nRanks`).
        # The reuse decision must be group-uniform: ranks can disagree on
        # `_comm_ptr()` (e.g. one rank materialized its communicator via an
        # earlier point-to-point op, or is on a different current device), and
        # a per-rank branch here would send some ranks into the group-wide
        # `all_gather_object` below while others return early — a hang, not an
        # error. So every rank gathers every rank's nullness first (this
        # collective is reached unconditionally), and all ranks then take the
        # same branch: reuse only if ALL ranks hold a real communicator.
        comm_ptr = backend._comm_ptr()
        have_comm = [None, ] * group.size()
        dist.all_gather_object(have_comm, comm_ptr != 0, group)
        if all(have_comm):
            _storage[group] = NCCLCommHandle(comm_ptr, False)
            return _storage[group]

    # For old PyTorch, we have to recreate a NCCL comm
    nccl_unique_ids = [None, ] * group.size()
    dist.all_gather_object(nccl_unique_ids, _C.get_local_nccl_unique_id(), group)
    root_unique_id = nccl_unique_ids[0]

    # Create a new communicator
    key = time.time_ns() if force_new_comm else group
    _storage[key] = NCCLCommHandle(
        _C.create_nccl_comm(root_unique_id, group.size(), group.rank()), True)
    return _storage[key]


def destroy_all_managed_nccl_comm() -> None:
    """
    Destroy all cached NCCL communicator handles and clear the cache.

    """
    _storage.clear()

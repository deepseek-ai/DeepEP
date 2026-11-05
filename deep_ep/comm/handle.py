import os
import time
from typing import Any, Optional, Tuple

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

    PyTorch's communicator is reused only if it already exists on the current device.
    Otherwise a DeepEP-managed comm is created instead, and it stays cached for the
    lifetime of the group entry (an extra NCCL communicator and its buffers), even if
    PyTorch's own communicator materializes later. Call `destroy_all_managed_nccl_comm`
    to release cached comms. Set `EP_REUSE_NCCL_COMM=0` to never reuse PyTorch's comm.

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
        # PyTorch creates NCCL communicators lazily: `_comm_ptr()` returns 0 until a
        # collective has run on this group and device (or `init_process_group` got
        # `device_id=...`), and a null handle would crash in C++ (`ncclTeamWorld`).
        # No collective is needed to agree on this: PyTorch creates a group's
        # communicator collectively, so ranks whose current device is the one PyTorch
        # used for the group all see the same result. That fails if the group's first
        # op ran on another device (e.g. `barrier()` without `device_ids`); use
        # `EP_REUSE_NCCL_COMM=0` there. Staying collective-free keeps calls such as
        # `get_physical_domain_size(group)` local when PyTorch's comm exists.
        comm_ptr = backend._comm_ptr()
        if comm_ptr != 0:
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


def get_physical_domain_size(group_or_buffer: Any) -> Tuple[int, int]:
    """Return physical RDMA and NVLink domain sizes for a process group or buffer."""
    context = getattr(group_or_buffer, 'context', None)
    if context is not None:
        return context.get_physical_domain_size()
    return _C.get_physical_domain_size(get_nccl_comm_handle(group_or_buffer).get())


def get_logical_domain_size(group_or_buffer: Any,
                            allow_hybrid_mode: Optional[bool] = None) -> Tuple[int, int]:
    """Return logical scaleout and scaleup domain sizes for a process group or buffer."""
    context = getattr(group_or_buffer, 'context', None)
    if context is not None:
        return context.get_logical_domain_size()
    allow_hybrid_mode = True if allow_hybrid_mode is None else allow_hybrid_mode
    return _C.get_logical_domain_size(get_nccl_comm_handle(group_or_buffer).get(), allow_hybrid_mode)


def destroy_all_managed_nccl_comm() -> None:
    """
    Destroy all cached NCCL communicator handles and clear the cache.

    """
    _storage.clear()

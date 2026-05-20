import os

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


def get_nccl_comm_handle(group: dist.ProcessGroup) -> NCCLCommHandle:
    """
    Get or create an NCCL communicator handle for the given process group.
    Results are cached, so subsequent calls with the same group return the same handle.

    Arguments:
        group: the communication group.

    Returns:
        handle: the NCCL communicator handle.
    """
    # Check cache hit
    global _storage
    if group in _storage:
        return _storage[group]

    # New PyTorch has such API
    backend = group._get_backend(torch.device('cuda'))
    if hasattr(backend, '_comm_ptr') and int(os.getenv('EP_REUSE_NCCL_COMM', '1')):
        _storage[group] = NCCLCommHandle(backend._comm_ptr(), False)
        return _storage[group]

    # For old PyTorch, we have to recreate a NCCL comm
    nccl_unique_ids = [None, ] * group.size()
    dist.all_gather_object(nccl_unique_ids, _C.get_local_nccl_unique_id(), group)
    root_unique_id = nccl_unique_ids[0]

    # Create a new communicator
    _storage[group] = NCCLCommHandle(
        _C.create_nccl_comm(root_unique_id, group.size(), group.rank()), True)
    return _storage[group]


def destroy_all_managed_nccl_comm() -> None:
    """
    Destroy all cached NCCL communicator handles and clear the cache.

    """
    _storage.clear()

import torch
import torch.distributed as dist
from typing import Optional

# noinspection PyUnresolvedReferences
import deep_ep._C as _C

from .base import BufferBase
from .. import comm


class PPBuffer(BufferBase):
    """Own contiguous symmetric storage for pipeline-parallel send/recv."""

    def __init__(self,
                 group: dist.ProcessGroup,
                 num_max_tensor_bytes: int,
                 num_max_inflight_tensors: int,
                 sl_idx: Optional[int] = None,
                 num_gpu_timeout_secs: int = 100,
                 explicitly_destroy: bool = False):
        """Create the storage and its communication group.

        Arguments:
            group: the communication group.
            num_max_tensor_bytes: the maximum tensor size in bytes per send/recv operation.
            num_max_inflight_tensors: the maximum number of in-flight tensors at once.
            sl_idx: the service level index for RDMA traffic. ``None`` uses the default.
            num_gpu_timeout_secs: the GPU-side timeout in seconds.
            explicitly_destroy: whether the user must call ``destroy()`` explicitly.
        """
        self.group = group
        super().__init__(explicitly_destroy)
        self.nccl_comm_handle = comm.get_nccl_comm_handle(group)

        # Create CPP runtime
        self.runtime = _C.PPBuffer(
            group.rank(), group.size(), self.nccl_comm_handle.get(),
            num_max_tensor_bytes, num_max_inflight_tensors,
            sl_idx, num_gpu_timeout_secs, explicitly_destroy)
        self.storage = self.runtime.storage

        # Call a barrier to ensure initialization visibility for all peers
        torch.cuda.synchronize()
        group.barrier()
        torch.cuda.synchronize()

    def send(self, x: torch.Tensor, dst_rank_idx: int, num_sms: Optional[int] = None) -> None:
        """Send a tensor to an adjacent rank in the pipeline-parallel ring.

        Arguments:
            x: the contiguous CUDA tensor to send; its size must not exceed ``num_max_tensor_bytes``.
            dst_rank_idx: the destination rank, which must be the previous or next rank in the ring.
            num_sms: the number of SMs to use. ``None`` selects all SMs.
        """
        self.runtime.send(x, dst_rank_idx, 0 if num_sms is None else num_sms)

    def recv(self, x: torch.Tensor, src_rank_idx: int, num_sms: Optional[int] = None) -> None:
        """Receive a tensor from an adjacent rank in the pipeline-parallel ring.

        Arguments:
            x: the contiguous CUDA output tensor to receive into; its size must not exceed the
                configured ``num_max_tensor_bytes``.
            src_rank_idx: the source rank, which must be the previous or next rank in the ring.
            num_sms: the number of SMs to use. ``None`` selects all SMs.
        """
        self.runtime.recv(x, src_rank_idx, 0 if num_sms is None else num_sms)

    def destroy(self) -> None:
        super().destroy()
        self.storage = None
        self.nccl_comm_handle = None

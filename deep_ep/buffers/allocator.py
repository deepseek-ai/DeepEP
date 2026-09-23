import torch
from typing import List, Tuple

# noinspection PyUnresolvedReferences
import deep_ep._C as _C

from ..utils.math import align


class BufferAllocator:
    """Plan contiguous, aligned tensors before allocating their CUDA storage."""

    def __init__(self):
        self.alignment = _C.get_num_allocation_alignment()
        self.allocations: List[Tuple[torch.Tensor, int]] = []
        self.num_bytes = 0
        self.materialized = False

    def allocate(self, shape, dtype: torch.dtype) -> torch.Tensor:
        """Append an aligned tensor allocation and return its meta placeholder."""
        assert not self.materialized, 'A materialized allocation plan cannot be changed'
        tensor = torch.empty(shape, dtype=dtype, device='meta')
        offset = align(self.num_bytes, self.alignment)
        self.allocations.append((tensor, offset))
        self.num_bytes = align(offset + tensor.nbytes, self.alignment)
        return tensor

    def materialize(self, storage: torch.Tensor) -> None:
        assert not self.materialized, 'An allocation plan can only be materialized once'
        assert storage.is_cuda and storage.is_contiguous() and storage.dtype == torch.uint8
        assert storage.numel() == self.num_bytes

        for tensor, offset in self.allocations:
            value = storage.narrow(0, offset, tensor.nbytes).view(tensor.dtype).view(tensor.shape)
            torch.utils.swap_tensors(tensor, value)
        self.materialized = True

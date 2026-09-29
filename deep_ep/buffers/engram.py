import functools
import os
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep._C as _C

from .base import BufferBase
from .. import comm
from ..utils.envs import EnvOverride, check_nvlink_connections, check_fast_rdma_atomic_support
from ..utils.math import align


# TODO: remove this
def _accept_legacy_storage_size(init):
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        # TODO: Remove the legacy keyword alias once callers migrate to num_rdma_storage_bytes.
        if 'num_rdma_storage_size' in kwargs:
            if 'num_rdma_storage_bytes' in kwargs:
                raise TypeError('Specify only one of num_rdma_storage_bytes and num_rdma_storage_size')
            kwargs['num_rdma_storage_bytes'] = kwargs.pop('num_rdma_storage_size')
        return init(self, *args, **kwargs)

    return wrapper


class EngramBuffer(BufferBase):
    """GPU fetch buffer with separate CPU or GPU storage shards registered for RDMA."""

    # Common communication functions; refer to `deep_ep.comm` for usage.
    barrier = comm.barrier
    get_comm_stream = comm.get_comm_stream
    get_physical_domain_size = comm.get_physical_domain_size
    get_logical_domain_size = comm.get_logical_domain_size

    @_accept_legacy_storage_size
    def __init__(self,
                 group: dist.ProcessGroup,
                 num_gpu_bytes: int,
                 num_rdma_storage_bytes: int,
                 use_cpu_rdma_storage: bool,
                 num_allocated_qps: int,
                 qp_depth: int,
                 restrict_rd_atomic: bool = False,
                 allow_hybrid_mode: bool = True,
                 sl_idx: Optional[int] = None,
                 num_gpu_timeout_secs: int = 100,
                 explicitly_destroy: bool = False):
        self.group = group
        self.rank_idx = group.rank()
        self.num_ranks = group.size()
        self.num_gpu_bytes = num_gpu_bytes
        self.num_rdma_storage_bytes = num_rdma_storage_bytes
        self.use_cpu_rdma_storage = use_cpu_rdma_storage
        self.allow_hybrid_mode = allow_hybrid_mode

        num_max_local_ranks = int(os.getenv('EP_NUM_MAX_LOCAL_RANKS', 16)) if allow_hybrid_mode else 1
        num_registered_bytes = num_gpu_bytes + num_rdma_storage_bytes * num_max_local_ranks + (1 << 32)
        num_total_gpu_bytes = torch.cuda.get_device_properties('cuda').total_memory
        if num_registered_bytes > num_total_gpu_bytes:
            os.environ['NCCL_WIN_STRIDE'] = str(align(num_registered_bytes, 1 << 32))

        check_nvlink_connections(group)

        assert num_allocated_qps > 0 and qp_depth > 0, \
            'pass `num_allocated_qps` and `qp_depth` from `get_theoretical_config`'
        self.num_allocated_qps = num_allocated_qps

        shared_comm = []
        if allow_hybrid_mode:
            pid, fd = _C.EngramBuffer.create_shared_handle(num_rdma_storage_bytes, use_cpu_rdma_storage)
            shared_comm = [None] * self.num_ranks
            dist.all_gather_object(shared_comm, (pid, fd), group)

        super().__init__(explicitly_destroy)
        # register only GDAKI backend to accelerate init
        with EnvOverride([
            ('NCCL_GIN_TYPE', '3', True),
            # TODO: Configure RD atomics through NCCL devComm config when supported.
            ('NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC', '8', restrict_rd_atomic),
        ]):
            self.nccl_comm_handle = comm.get_nccl_comm_handle(group, force_new_comm=True)
            self.runtime = _C.EngramBuffer(
                self.rank_idx, self.num_ranks,
                self.nccl_comm_handle.get(), shared_comm,
                num_gpu_bytes, num_rdma_storage_bytes,
                use_cpu_rdma_storage, allow_hybrid_mode,
                sl_idx, num_allocated_qps, qp_depth,
                num_gpu_timeout_secs,
                explicitly_destroy)
        self.context = self.runtime.context

        self.num_scaleout_ranks, self.num_scaleup_ranks = self.get_logical_domain_size()
        self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks
        self.scaleup_rank_idx = self.rank_idx % self.num_scaleup_ranks
        self.num_rdma_ranks, self.num_nvlink_ranks = self.get_physical_domain_size()

        torch.cuda.synchronize()
        group.barrier()
        torch.cuda.synchronize()

    def destroy(self) -> None:
        """Destroy the runtime. Requires ``explicitly_destroy=True`` at construction."""
        super().destroy()
        self.context = None
        self.nccl_comm_handle = None

    @staticmethod
    def get_storage_size_hint(group: dist.ProcessGroup,
                              num_entries_per_layer: List[int], hidden: int,
                              num_max_tokens: int, num_entries_per_token: int,
                              dtype: torch.dtype = torch.bfloat16,
                              num_sf_packs: int = 0) -> Tuple[int, int]:
        """
        Return the aligned main GPU buffer and per-rank RDMA storage sizes.

        The GPU size covers the fetch receive area plus this rank's shard of the FP8
        scaling-factor table; pass ``num_sf_packs=0`` for BF16 storage.
        """
        return _C.EngramBuffer.calculate_storage_size(
            comm.get_nccl_comm_handle(group).get(),
            num_entries_per_layer, hidden, dtype.itemsize,
            num_max_tokens, num_entries_per_token, num_sf_packs)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_theoretical_config(group: dist.ProcessGroup,
                               num_layers: int,
                               num_max_tokens: int,
                               num_entries_per_token: int,
                               num_max_read_requests_per_qp: int = 16) -> Tuple[int, int, bool]:
        """
        Return the recommended `(num_qps, qp_depth, restrict_rd_atomic)`.
        Pass `restrict_rd_atomic` to the constructor to limit requester read atomics.
        """
        nic_read_mpps = 90 if check_fast_rdma_atomic_support() else 45
        return _C.EngramBuffer.get_theoretical_config(
            comm.get_nccl_comm_handle(group).get(),
            num_layers, num_max_tokens, num_entries_per_token,
            nic_read_mpps, num_max_read_requests_per_qp)

    def set_config(self, num_entries_per_layer: List[int], hidden: int,
                   num_max_tokens: int, num_entries_per_token: int,
                   dtype: torch.dtype = torch.bfloat16,
                   num_sf_packs: int = 0) -> None:
        self.runtime.set_config(num_entries_per_layer, hidden, dtype.itemsize,
                                num_max_tokens, num_entries_per_token, num_sf_packs)

    def write(self, storages: List[torch.Tensor],
              sfs: Optional[List[torch.Tensor]] = None) -> None:
        self.runtime.write(storages, sfs)

    def fetch(self, indices: torch.Tensor, num_qps: int = 0,
              use_tma_aligned_col_major_sf: bool = True) -> List[Callable]:
        """
        Issue the RDMA gets for all layers and return one wait hook per layer.

        Arguments:
            indices: `[num_layers, num_tokens, num_entries_per_token]` entry indices to fetch.
            num_qps: the number of GIN contexts (QPs) to drive, 0 for all of the allocated ones.
            use_tma_aligned_col_major_sf: gather the fetched factors column-major and TMA-aligned.
        """
        num_qps = self.num_allocated_qps if num_qps == 0 else num_qps
        assert num_qps <= self.num_allocated_qps, 'Allocated QPs are not enough'
        return self.runtime.fetch(indices, num_qps, use_tma_aligned_col_major_sf)

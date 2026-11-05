import functools
import os
import math
import torch
import torch.distributed as dist
from typing import Optional, Sequence, Tuple, Union

# noinspection PyUnresolvedReferences
import deep_ep._C as _C
# noinspection PyUnresolvedReferences
from deep_ep._C import EventHandle

from .allocator import BufferAllocator
from .base import BufferBase
from .. import comm
from ..utils.event import EventOverlap
from ..utils.math import align
from ..utils.semantic import value_or, weak_lru
from ..utils.envs import (
    check_fast_rdma_atomic_support,
    check_nvlink_connections, check_torch_deterministic,
    get_nvlink_gbs, get_rdma_gbs,
    get_sm_read_gbs, get_sm_write_gbs
)


class EPHandle:
    """
    Communication handle returned by `EPBuffer.dispatch`.
    Can be reused as a cached handle in subsequent `EPBuffer.dispatch` calls to skip layout recomputation,
    and is consumed by `EPBuffer.combine` to reverse the token routing.

    Attributes:
        do_expand: whether the expanding (one-token-per-expert-slot) layout is used.
        num_experts: the number of all experts.
        expert_alignment: align the number of tokens received by each local expert to this variable.
        num_max_tokens_per_rank: the maximum number of tokens per rank, all the ranks must hold the same value.
        num_sms: the SM count used during dispatch (reused in combine).
        topk_idx: original top-k expert indices from dispatch, `[num_tokens, num_topk]`.
            Must not be modified while this handle is in use. PyTorch version counters detect ordinary
            in-place changes, including through views, but not writes through `.data` or raw pointers.
            Tensors created in inference mode have no version counter and cannot be checked.
        psum_num_recv_tokens_per_scaleup_rank: inclusive prefix sum of deduplicated received token counts
            per scaleup rank, shape `[num_scaleup_ranks]`. A token is counted once per rank even if
            multiple of its top-k experts land on the same rank. The last element equals the total number
            of received tokens.
        psum_num_recv_tokens_per_expert: prefix sum of alignment-padded received token counts per local
            expert, shape `[num_local_experts]`. Each expert's count is padded to `expert_alignment`.
            In non-expand mode, this is the inclusive prefix sum. In expand mode, `psum[i]` equals
            the aligned cumulative count of experts before `i` plus the actual (unaligned) token count
            of expert `i` — so `psum[i] - align(psum[i-1], expert_alignment)` recovers the real
            count for expert `i`, and `align(psum[i], expert_alignment)` gives expert `i+1`'s
            starting offset.
        num_recv_tokens_per_expert_list: Python list of per-expert received token counts (CPU-side).
        num_unaligned_recv_tokens_per_expert: the actual (unaligned) number of tokens received per local
            expert, shape `[num_local_experts]` with `torch.int`. Only populated in expand mode.
        recv_src_metadata: source token indices and buffer slot indices.
        dst_buffer_slot_idx: destination buffer slot indices from dispatch.
        token_metadata_at_forward: per-channel forwarded token metadata (hybrid mode only).
        channel_linked_list: per-channel per-scaleup-peer linked list (hybrid mode only).
        num_recv_tokens: the total number of received tokens.
    """

    def __init__(self,
                 do_expand: bool,
                 num_experts: int, expert_alignment: int,
                 num_max_tokens_per_rank: int,
                 num_sms: int,
                 topk_idx: torch.Tensor,
                 num_recv_tokens: int,
                 num_expanded_tokens: int,
                 num_recv_tokens_per_expert_list: list,
                 psum_num_recv_tokens_per_scaleup_rank: torch.Tensor,
                 psum_num_recv_tokens_per_expert: torch.Tensor,
                 num_unaligned_recv_tokens_per_expert: torch.Tensor,
                 recv_src_metadata: torch.Tensor,
                 dst_buffer_slot_idx: torch.Tensor,
                 token_metadata_at_forward: Optional[torch.Tensor],
                 channel_linked_list: Optional[torch.Tensor]):
        assert topk_idx is not None

        self.do_expand = do_expand
        self.num_experts = num_experts
        self.expert_alignment = expert_alignment
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_sms = num_sms
        self.psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert
        self.num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
        self.recv_src_metadata = recv_src_metadata
        self.dst_buffer_slot_idx = dst_buffer_slot_idx
        self.token_metadata_at_forward = token_metadata_at_forward
        self.channel_linked_list = channel_linked_list

        # May not be accurate without CPU sync
        self.num_recv_tokens = num_recv_tokens
        self.num_expanded_tokens = num_expanded_tokens

        # For deterministic features
        self.cached_recv_src_metadata_before_sort = None

        # `topk_idx` cannot change in the handle
        self._topk_idx = topk_idx
        self._topk_idx_version = None if topk_idx.is_inference() else topk_idx._version

    @property
    def topk_idx(self) -> torch.Tensor:
        assert self._topk_idx_version is None or self._topk_idx._version == self._topk_idx_version, \
            '`topk_idx` must not be modified while the EP handle is in use'
        return self._topk_idx

    def deterministic_sort(self,
                           do_cpu_sync: bool,
                           is_cached_dispatch: bool,
                           recv_x: torch.Tensor,
                           recv_sf: Optional[torch.Tensor],
                           recv_topk_idx: torch.Tensor,
                           recv_topk_weights: torch.Tensor,
                           channel_linked_list: Optional[torch.Tensor]):
        """
        Sort received tokens to guarantee deterministic dispatch output.
        The principle:
          - Non-expand mode: sort everything that depends on the receive order, including
            `recv_x`, `recv_sf`, `recv_topk_weights`, `recv_topk_idx`, and `self.recv_src_metadata`
            (`recv_src_metadata` is sorted only for non-cached dispatch, since it is not regenerated in cached mode).
          - Expand mode: only sort the expanded arrays — `recv_x`, `recv_sf`, and `recv_topk_weights`.
            The slot pointers in `self.recv_src_metadata[:, 2:]` are updated to reflect the new positions, but `self.recv_src_metadata` itself is not permuted.
        """

        # NOTE: `self.recv_src_metadata` is generated once during non-cached dispatch and is not
        # regenerated during cached dispatch (applies to both expand and non-expand mode). So we:
        #  1. Cache it for later sorting
        #  2. Only permute `self.recv_src_metadata` in non-cached mode
        if not is_cached_dispatch:
            self.cached_recv_src_metadata_before_sort = self.recv_src_metadata.clone()
        assert self.cached_recv_src_metadata_before_sort is not None
        sort_keys = self.cached_recv_src_metadata_before_sort[:, 0]

        # Ignore trailing tokens by setting their `sort_keys` to max
        num_recv_tokens = self.psum_num_recv_tokens_per_scaleup_rank[-1] if not do_cpu_sync else self.recv_src_metadata.shape[0]
        if not do_cpu_sync:
            oob_tokens_mask = torch.arange(0, self.recv_src_metadata.shape[0], device=self.recv_src_metadata.device) >= num_recv_tokens
            sort_keys = sort_keys.clone()
            sort_keys[oob_tokens_mask] = torch.iinfo(sort_keys.dtype).max
        orig_indices = torch.sort(sort_keys).indices

        def get_reverse_permutation(perm: torch.Tensor) -> torch.Tensor:
            assert perm.dim() == 1
            result = torch.empty_like(perm)
            result[perm] = torch.arange(0, perm.shape[0], dtype=perm.dtype, device=perm.device)
            return result

        def permute(tensor: Optional[torch.Tensor], orig_indices: torch.Tensor):
            if tensor is not None:
                tmp = tensor[orig_indices]
                tensor.copy_(tmp)

        if not self.do_expand:
            # Non-expand mode
            # If cached dispatch is enabled, the `dispatch` kernel stores values according to `dst_buffer_slot_idx`, and the `dispatch_copy_epilogue_impl` kernel writes the info of token i into the i-th slot
            permute(recv_x, orig_indices)
            permute(recv_sf, orig_indices)
            permute(recv_topk_weights, orig_indices)
            permute(recv_topk_idx, orig_indices)
            if not is_cached_dispatch:
                permute(self.recv_src_metadata, orig_indices)

            if not is_cached_dispatch and channel_linked_list is not None:
                valid_mask = (channel_linked_list >= 0) & (channel_linked_list < num_recv_tokens)
                to_indices = get_reverse_permutation(orig_indices)
                channel_linked_list[valid_mask] = to_indices[channel_linked_list[valid_mask]].to(channel_linked_list.dtype)

        elif not is_cached_dispatch:
            # Expand mode. In cached mode the copy epilogue places tokens according to
            # `self.recv_src_metadata[:, 2:]`, so we only need to permute when `is_cached_dispatch` is `False`.
            # In expand mode, `recv_x`, `recv_sf`, and `recv_topk_weights` are grouped by expert ID, possibly with padding (expert alignment). We permute tokens within each expert and update `self.recv_src_metadata[:, 2:]` accordingly.

            # Now we're going to construct the sorting key, which is:
            #  - `expert_idx*src_token_global_index_max_x2 + (-src_token_global_index_max) + src_token_global_idx`, for valid tokens
            #  - `expert_idx * src_token_global_index_max_x2`, for padding slots
            # This guarantees a two-key sort: first by expert, then by order within each expert.
            # Valid tokens precede padding tokens, and valid tokens are sorted by `src_token_global_idx`.
            src_token_global_index_max_x2 = 10000000000    # 1e10
            tensor_dim0_after_expand = recv_x.shape[0]

            expert_token_idx_start = self.psum_num_recv_tokens_per_expert - self.num_unaligned_recv_tokens_per_expert
            token_idx2expert_idx = torch.bucketize(torch.arange(tensor_dim0_after_expand, device='cuda'),
                                                   expert_token_idx_start[1:], right=True, out_int32=False)
            sort_keys_for_expanded_tensors = token_idx2expert_idx * src_token_global_index_max_x2

            slots = self.cached_recv_src_metadata_before_sort[:, 2:]    # [num_recv_tokens, topk]
            src_global_idx = self.cached_recv_src_metadata_before_sort[:, 0]
            valid_mask = slots >= 0
            if not do_cpu_sync:
                valid_mask[oob_tokens_mask] = False
            sort_keys_for_expanded_tensors.scatter_add_(0, slots[valid_mask], -src_token_global_index_max_x2//2 + src_global_idx.unsqueeze(1).expand_as(slots)[valid_mask].to(torch.int64))

            orig_indices_for_expanded_tensors = torch.sort(sort_keys_for_expanded_tensors, stable=True).indices.to(torch.int32)
            permute(recv_x, orig_indices_for_expanded_tensors)
            permute(recv_sf, orig_indices_for_expanded_tensors)
            permute(recv_topk_weights, orig_indices_for_expanded_tensors)

            to_indices_for_expanded_tensors = get_reverse_permutation(orig_indices_for_expanded_tensors)
            self.recv_src_metadata[:, 2:][valid_mask] = to_indices_for_expanded_tensors[self.recv_src_metadata[:, 2:][valid_mask]]


class EPBuffer(BufferBase):
    """
    The EP communication buffer, which supports:
        - high-throughput expert-parallel all-to-all (dispatch and combine, using NVLink and/or RDMA)

    Attributes:
        group: the communication group.
        rank_idx: the rank index.
        num_ranks: the number of ranks in the group.
        allow_hybrid_mode: whether to enable hybrid mode for multi-node communication. Hybrid mode uses
            hierarchical RDMA + NVLink communication to achieve higher bandwidth, and is more friendly
            to multi-plane/multi-rail networks.
        allow_multiple_reduction: whether to allow multiple reductions in combine. If disabled,
            only one reduction will be done in the combine epilogue for best precision,
            but it may increase data transfer size.
        prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            If enabled, we tend to use fewer SMs.
        num_bytes: the total buffer size in bytes.
        num_max_tokens_per_rank: the default maximum tokens per rank.
        num_scaleout_ranks: the number of scaleout ranks.
        num_scaleup_ranks: the number of scaleup ranks.
        scaleout_rank_idx: the scaleout rank index of this rank.
        scaleup_rank_idx: the scaleup rank index of this rank.
        num_rdma_ranks: the number of physical RDMA ranks.
        num_nvlink_ranks: the number of physical NVLink ranks.
        runtime: the C++ runtime.
    """

    # Common communication functions; refer to `deep_ep.comm` for usage.
    barrier = comm.barrier
    get_comm_stream = comm.get_comm_stream
    get_physical_domain_size = comm.get_physical_domain_size
    get_logical_domain_size = comm.get_logical_domain_size

    def __init__(self,
                 group: dist.ProcessGroup,
                 # Provide `num_bytes` (excludes workspace)
                 num_bytes: Optional[int] = None,
                 # Or provide MoE settings (BF16 by default)
                 num_max_tokens_per_rank: int = 0,
                 hidden: int = 0,
                 num_topk: int = 0,
                 use_fp8_dispatch: bool = False,
                 # Load balance configs
                 lb_allocation_plan_or_num_bytes: Union[BufferAllocator, int] = 0,
                 # Configs
                 deterministic: bool = False,
                 allow_hybrid_mode: bool = True,
                 allow_multiple_reduction: bool = True,
                 prefer_overlap_with_compute: bool = True,
                 sl_idx: Optional[int] = None,
                 num_allocated_qps: int = 0,
                 num_cpu_timeout_secs: int = 300, num_gpu_timeout_secs: int = 100,
                 explicitly_destroy: bool = False):
        """
        Initialize the EP communication buffer.

        Arguments:
            group: the communication group.
            num_bytes: the total buffer size in bytes (excludes workspace), if set, overrides MoE-based calculation.
                Must be aligned to ``deep_ep.get_num_allocation_alignment()`` bytes.
            num_max_tokens_per_rank: the maximum number of tokens per rank, used for buffer size calculation.
            hidden: the hidden dimension of each token.
            num_topk: the number of top-k experts per token.
            use_fp8_dispatch: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling factors.
            lb_allocation_plan_or_num_bytes: allocation plan or byte count for the LB region, separate
                from `num_bytes`. Zero disables the LB region. Byte counts must be aligned to
                ``deep_ep.get_num_allocation_alignment()`` bytes. When passing a plan, all ranks in
                an LSA domain must use the same tensor shapes and allocation order.
            deterministic: whether to use deterministic routing algorithms.
            allow_hybrid_mode: whether to enable hybrid mode.
            allow_multiple_reduction: whether to allow multiple reductions in combine.
            prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            sl_idx: the optional RDMA service level index. It overrides `EP_DEFAULT_RDMA_SL` and can be overridden by
                `EP_OVERRIDE_RDMA_SL`.
            num_allocated_qps: the number of QPs to allocate for RDMA (0 for automatic).
            num_cpu_timeout_secs: CPU-side timeout in seconds for CPU sync.
            num_gpu_timeout_secs: GPU-side timeout in seconds for GPU operations.
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
        """
        # Some useful utilities
        self.group = group
        self.rank_idx = group.rank()
        self.num_ranks = group.size()
        self.allow_hybrid_mode = allow_hybrid_mode
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        self.deterministic = deterministic
        
        # Create NCCL comm handle
        self.nccl_comm_handle = comm.get_nccl_comm_handle(group)

        # Calculate buffer size (already 2 MB-aligned from hint functions / calculate_ep_buffer_size)
        if num_bytes is None:
            # NOTES: we allow `num_topk == 0`, as the buffer size can also be calculated by number of ranks (maybe bigger though)
            num_bytes = _C.calculate_ep_buffer_size(
                self.nccl_comm_handle.get(),
                num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,
                allow_hybrid_mode, allow_multiple_reduction)

        if os.environ.get('EP_BUFFER_DEBUG', 0):
            print(f'Initializing EP buffer with {num_bytes} bytes at rank EP {group.rank()}/{group.size()}')
        self.num_bytes = num_bytes

        # Load balance configs
        lb_allocation_plan = lb_allocation_plan_or_num_bytes if isinstance(lb_allocation_plan_or_num_bytes, BufferAllocator) else None
        num_lb_bytes = lb_allocation_plan_or_num_bytes if lb_allocation_plan is None else lb_allocation_plan.num_bytes
        if lb_allocation_plan is not None:
            assert not lb_allocation_plan.materialized
        assert isinstance(num_lb_bytes, int) and num_lb_bytes >= 0
        self.num_lb_bytes = num_lb_bytes

        # Store default values
        self.num_max_tokens_per_rank = num_max_tokens_per_rank

        # Check PCIe GPUs
        check_nvlink_connections(group)

        # Automatic maximum QP count allowed
        if num_allocated_qps == 0:
            # Hybrid mode will consume more QPs
            # The extra QP is for notify warps
            if self.allow_hybrid_mode:
                num_allocated_qps = 65 if check_fast_rdma_atomic_support() else 129
            else:
                num_allocated_qps = 17
        self.num_allocated_qps = num_allocated_qps

        # Create CPP handle
        super().__init__(explicitly_destroy)
        self.runtime = _C.EPBuffer(
            self.rank_idx, self.num_ranks,
            self.nccl_comm_handle.get(), num_bytes, num_lb_bytes,
            allow_hybrid_mode, allow_multiple_reduction, prefer_overlap_with_compute,
            sl_idx, num_allocated_qps,
            num_cpu_timeout_secs, num_gpu_timeout_secs,
            self.explicitly_destroy)
        self.context = self.runtime.context

        # Materialize LB allocation plan
        if lb_allocation_plan is not None:
            lb_allocation_plan.materialize(self.runtime.lb_storage)

        # Logical rank indices
        self.num_scaleout_ranks, self.num_scaleup_ranks = self.get_logical_domain_size()
        self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks
        self.scaleup_rank_idx = self.rank_idx % self.num_scaleup_ranks

        # Physical rank indices
        self.num_rdma_ranks, self.num_nvlink_ranks = self.get_physical_domain_size()

        # Call a barrier to ensure initialization visibility for all peers
        torch.cuda.synchronize()
        group.barrier()
        torch.cuda.synchronize()

    def destroy(self) -> None:
        """
        Destroy the C++ runtime and release resources. Requires `explicitly_destroy=True` at construction.
        """
        super().destroy()
        self.context = None
        self.nccl_comm_handle = None

    @staticmethod
    def get_buffer_size_hint(group: dist.ProcessGroup,
                             num_max_tokens_per_rank: int, hidden: int,
                             num_topk: int = 0, use_fp8_dispatch: bool = False,
                             allow_hybrid_mode: bool = True,
                             allow_multiple_reduction: bool = True) -> int:
        """
        Get a recommended buffer size (in bytes) for the given MoE settings, without constructing the buffer.
        The returned value is aligned to 2 MB.

        Arguments:
            group: the communication group.
            num_max_tokens_per_rank: the maximum number of tokens per rank.
            hidden: the hidden dimension of each token.
            num_topk: the number of top-k experts per token.
            use_fp8_dispatch: whether to use FP8 for dispatch.
            allow_hybrid_mode: whether to enable hybrid mode.
            allow_multiple_reduction: whether to allow multiple reductions in combine.

        Returns:
            size: the recommended buffer size in bytes (2 MB-aligned).
        """
        # NOTES: calculate_ep_buffer_size already returns 2 MB-aligned values
        return _C.calculate_ep_buffer_size(
            comm.get_nccl_comm_handle(group).get(),
            num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,
            allow_hybrid_mode, allow_multiple_reduction)

    @staticmethod
    def _unpack_handle(handle: Optional[EPHandle] = None) \
        -> Tuple[Optional[int], Optional[int], Optional[list],
                 Optional[torch.Tensor], Optional[torch.Tensor],
                 Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
                 Optional[torch.Tensor], Optional[torch.Tensor]]:
        if handle is None:
            return None, None, None, None, None, None, None, None, None, None
        return (handle.num_recv_tokens,
                handle.num_expanded_tokens,
                handle.num_recv_tokens_per_expert_list,
                handle.psum_num_recv_tokens_per_scaleup_rank,
                handle.psum_num_recv_tokens_per_expert,
                handle.num_unaligned_recv_tokens_per_expert,
                handle.dst_buffer_slot_idx,
                handle.token_metadata_at_forward,
                handle.recv_src_metadata,
                handle.channel_linked_list)

    @staticmethod
    def capture() -> EventHandle:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event_handle: the captured event handle.
        """
        return EventHandle()

    @weak_lru(maxsize=None)
    def get_theoretical_num_sms(self, num_experts: int, num_topk: int,
                                num_scaleout_topk: int = 0,
                                rdma_gbs: float = 0, nvlink_gbs: float = 0,
                                sm_read_gbs: float = 0, sm_write_gbs: float = 0) -> int:
        """
        Estimate the optimal number of SMs for dispatch/combine kernels based on bandwidth modeling.
        The result is cached. This assumes a balanced gate distribution.

        Arguments:
            num_experts: the number of all experts.
            num_topk: the number of top-k experts per token.
            num_scaleout_topk: reserved for balanced gate (must be 0 currently).
            rdma_gbs: the RDMA bandwidth in GB/s (0 for auto-detect).
            nvlink_gbs: the NVLink bandwidth in GB/s (0 for auto-detect).
            sm_read_gbs: the per-SM HBM read bandwidth in GB/s (0 for the default).
            sm_write_gbs: the per-SM HBM write bandwidth in GB/s (0 for the default).

        Returns:
            num_sms: the recommended SM count (even, at least 4).
        """
        # TODO: support `do_expand` and `allow_multiple_reduction`

        # The `1` in this function means scale-up traffic
        # i.e. the HBM read volume of the dispatch copy epilogue, equals to "the number of tokens" * "num_expected_topk" * "data size per token"
        # NOTES: this is for balanced gate
        # For V3.0's group-limited gate, please do not use this function
        # TODO: support this
        assert num_scaleout_topk == 0

        # Get bandwidth
        sm_read_gbs = get_sm_read_gbs() if sm_read_gbs == 0 else sm_read_gbs
        sm_write_gbs = get_sm_write_gbs() if sm_write_gbs == 0 else sm_write_gbs
        if rdma_gbs == 0 and self.num_rdma_ranks > 1:
            rdma_gbs = get_rdma_gbs()
        if nvlink_gbs == 0:
            nvlink_gbs = get_nvlink_gbs()

        # Initial count
        # NOTES: we don't count HBM traffic
        sm_read, sm_write = 0, 0
        rdma_traffic, nvlink_traffic = 0, 0

        def get_expected_topk(num_groups: int) -> float:
            assert num_experts % num_groups == 0
            return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, num_topk) / math.comb(num_experts, num_topk))

        # Expected top-k scale-out ranks
        num_expected_scaleout_topk = get_expected_topk(self.num_scaleout_ranks) if self.num_scaleout_ranks > 1 else 0

        # Expected top-k scale-up ranks
        num_expected_topk = get_expected_topk(self.num_ranks)

        # Read tokens
        sm_read += 1 / num_expected_topk

        # NOTES: we don't consider the skip-send-buffer cases (all selections fall in the local)
        if self.num_scaleout_ranks > 1:
            # Scaleout warps: write send buffer
            sm_write += 1 / num_expected_topk

            # Scaleout traffic
            sm_write += (1 / num_expected_topk) * (num_expected_scaleout_topk / self.num_scaleout_ranks)  # Local bypass
            rdma_traffic += (1 / num_expected_topk) * (num_expected_scaleout_topk * (1 - 1 / self.num_scaleout_ranks))

            # Forward warps
            sm_read += num_expected_scaleout_topk / num_expected_topk
            sm_write += 1  # Issue scaleup
            nvlink_traffic += 1 - (1 / self.num_scaleup_ranks)
        else:
            # Write send buffer
            if self.num_rdma_ranks > 1:
                sm_write += 1 / num_expected_topk

            # Issue NVLink
            sm_write += self.num_nvlink_ranks / self.num_ranks

            # NVLink and RDMA traffic
            nvlink_traffic += self.num_nvlink_ranks / self.num_ranks * (1 - 1 / self.num_nvlink_ranks)  # Except local bypass
            rdma_traffic += (self.num_ranks - self.num_nvlink_ranks) / self.num_ranks

        # Found the bounded one
        if self.num_scaleout_ranks > 1 and (rdma_traffic / rdma_gbs) > (nvlink_traffic / nvlink_gbs):
            bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs
        else:
            bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs

        # Calculate SM count
        # NOTES: will try to use more SMs if not overlap with compute
        num_device_sms = torch.cuda.get_device_properties('cuda').multi_processor_count
        num_sms = num_device_sms  # No traffic, e.g., EP=1
        if bounded_traffic > 0:
            num_sms = max(
                # Dispatch
                bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
                bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
                # Combine
                bounded_gbs / bounded_traffic * (sm_write - nvlink_traffic) / sm_read_gbs,
                bounded_gbs / bounded_traffic * (sm_read + nvlink_traffic) / sm_write_gbs,
            )
        num_sms = align(max(4, math.ceil(num_sms * 1.3)), 4)
        num_sms = num_sms if self.prefer_overlap_with_compute else max(num_sms, 64)
        num_sms = min(num_sms, num_device_sms // 2 * 2)

        # Summary
        if os.environ.get('EP_BUFFER_DEBUG', 0):
            print(f'EP SM approximation: '
                  f'{sm_read=}, {sm_write=}, {rdma_traffic=}, {nvlink_traffic=}, '
                  f'{rdma_gbs=}, {nvlink_gbs=}, '
                  f'{num_expected_scaleout_topk=}, {num_expected_topk=}, '
                  f'{bounded_traffic=}, {bounded_gbs=}, {num_sms=}')
        return num_sms

    def get_theoretical_num_qps(self, num_sms: int) -> int:
        """
        Estimate the optimal number of RDMA QPs based on SM count and mode.

        Arguments:
            num_sms: the number of SMs used for the dispatch/combine kernel.

        Returns:
            num_qps: the recommended QP count, capped by `num_allocated_qps`.
        """
        # For direct mode, we encourage less QPs to reduce DB ringing overhead
        num_qps = min(num_sms, 8 + 1)

        # For hybrid mode, we encourage every channel (and notify) to have an independent QP
        if self.allow_hybrid_mode:
            num_qps = num_sms * 16 + 1

        return min(num_qps, self.num_allocated_qps)

    def dispatch(self,
                 x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 topk_idx: Optional[torch.Tensor] = None,
                 topk_weights: Optional[torch.Tensor] = None,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 num_experts: Optional[int] = None,
                 num_max_tokens_per_rank: Optional[int] = None,
                 expert_alignment: Optional[int] = None,
                 num_sms: int = 0, num_qps: int = 0,
                 previous_event: Optional[EventHandle] = None,
                 async_with_compute_stream: bool = False,
                 allocate_on_comm_stream: bool = False,
                 handle: Optional[EPHandle] = None,
                 do_handle_copy: bool = False,
                 do_cpu_sync: Optional[bool] = None,
                 do_expand: bool = False,
                 do_zero_padding: bool = False,
                 use_tma_aligned_col_major_sf: bool = False,
                 defer_epilogue: bool = False) \
            -> Union[Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           Optional[torch.Tensor], Optional[torch.Tensor],
                           EPHandle, EventOverlap], EventOverlap]:
        """
        Dispatch tokens to different ranks. Supports both single-node and multi-node settings.
            SM and QP counts are automatically determined if not specified.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be
                `[num_tokens, hidden]`, and type must be `torch.bfloat16`; for the second type (FP8 mode),
                the first element of the tuple must be `[num_tokens, hidden]` with type `torch.float8_e4m3fn`,
                the second is the scale factors.
            topk_idx: `[num_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert
                indices selected by each token, `-1` means no selections.
                Stored by reference in the handle; must not be modified until all uses of the handle complete.
                Must be `None` if `handle` is provided.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
                Must be `None` if `handle` is provided.
            cumulative_local_expert_recv_stats: `[num_local_experts]` with `torch.int`, a cumulative expert count
                tensor for statistics, useful for online EP load balance monitoring.
            num_experts: the number of all experts. Inferred from `handle` if provided.
            num_max_tokens_per_rank: the maximum number of tokens per rank. Inferred from constructor default
                or `handle` if provided.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_sms: the number of SMs to use (0 for automatic via `get_theoretical_num_sms`).
            num_qps: the number of RDMA QPs to use (0 for automatic via `get_theoretical_num_qps`).
            previous_event: the event to wait before actually executing the kernel.
                If set, `allocate_on_comm_stream` must also be `True`.
            async_with_compute_stream: the current stream will not wait for the communication kernels to be
                finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the
                communication stream.
            handle: an optional cached `EPHandle` from a previous dispatch, if set, the CPU will reuse the layout
                information to save some time. `topk_idx` must be `None` (reused from handle).
                `topk_weights` can be optionally provided (e.g. for backward pass with cached expand).
            do_handle_copy: retained for compatibility; must be `False`.
            do_cpu_sync: whether to synchronize with CPU to get exact received token counts.
                `None` defaults to `True` unless `handle` is provided.
            do_expand: whether to use the expanding layout (one slot per expert per token).
            do_zero_padding: whether to zero out the alignment padding slots in the expanded output.
                Only valid when `do_expand` is True. Ensures alignment gaps between experts are zeroed.
            use_tma_aligned_col_major_sf: whether to use TMA-aligned column-major layout for scale factors.
            defer_epilogue: whether to defer the CPU receive-count wait and copy epilogue until
                `event.current_stream_wait()` is called. This requires `async_with_compute_stream=True`.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`.
                Only returned when `defer_epilogue=False`.
            recv_topk_idx: received expert indices. Only returned when `defer_epilogue=False`.
            recv_topk_weights: received expert weights (`None` if `topk_weights` was not provided).
                Only returned when `defer_epilogue=False`.
            handle: the returned communication handle. Only returned when `defer_epilogue=False`.
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
                With `defer_epilogue=True`, this function returns the `EventOverlap` object directly instead
                of the five-item tuple. Call `event.current_stream_wait()` to run the copy epilogue and obtain
                `(recv_x, recv_topk_idx, recv_topk_weights, handle)`.
        """
        assert not do_handle_copy, '`do_handle_copy` must be False; handle copying is no longer supported'
        check_torch_deterministic()

        # Automatic decide SM and QP count
        num_topk = (handle.topk_idx if topk_idx is None else topk_idx).shape[1]
        num_sms = self.get_theoretical_num_sms(num_experts, num_topk) if num_sms == 0 else align(num_sms, 2)
        num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps
        assert num_qps <= self.num_allocated_qps, f'Allocated QPs are not enough'

        # Unpack SF
        x, sf = x if isinstance(x, tuple) else (x, None)

        # Unpack handles
        # Reuse some values if possible
        if handle is not None:
            assert topk_idx is None
            assert do_cpu_sync is None or not do_cpu_sync, 'Cannot do CPU sync with cached handle'
            topk_idx = handle.topk_idx
            num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, handle.num_max_tokens_per_rank)
            num_experts = value_or(num_experts, handle.num_experts)
            expert_alignment = value_or(expert_alignment, handle.expert_alignment)
            do_cpu_sync = False

            # Should be aligned with the handle context
            assert (num_experts, expert_alignment, num_max_tokens_per_rank) == \
                   (handle.num_experts, handle.expert_alignment, handle.num_max_tokens_per_rank)
        (cached_num_recv_tokens, cached_num_expanded_tokens,
         cached_num_recv_tokens_per_expert_list,
         cached_psum_num_recv_tokens_per_scaleup_rank, cached_psum_num_recv_tokens_per_expert,
         cached_num_unaligned_recv_tokens_per_expert,
         cached_dst_buffer_slot_idx,
         cached_token_metadata_at_forward,
         cached_recv_src_metadata,
         cached_channel_linked_list) = self._unpack_handle(handle)

        # Some default values
        num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, self.num_max_tokens_per_rank)
        expert_alignment = value_or(expert_alignment, 1)
        do_cpu_sync = value_or(do_cpu_sync, True)

        # Do dispatch
        result, event, deferred_epilogue = self.runtime.dispatch(x, sf, topk_idx, topk_weights,
                                                                 cumulative_local_expert_recv_stats,
                                                                 cached_num_recv_tokens,
                                                                 cached_num_expanded_tokens,
                                                                 cached_num_recv_tokens_per_expert_list,
                                                                 cached_psum_num_recv_tokens_per_scaleup_rank,
                                                                 cached_psum_num_recv_tokens_per_expert,
                                                                 cached_num_unaligned_recv_tokens_per_expert,
                                                                 cached_dst_buffer_slot_idx,
                                                                 cached_token_metadata_at_forward,
                                                                 cached_recv_src_metadata,
                                                                 cached_channel_linked_list,
                                                                 num_max_tokens_per_rank,
                                                                 num_experts, expert_alignment,
                                                                 num_sms, num_qps,
                                                                 previous_event,
                                                                 async_with_compute_stream, allocate_on_comm_stream,
                                                                 do_cpu_sync, do_expand,
                                                                 do_zero_padding,
                                                                 use_tma_aligned_col_major_sf,
                                                                 defer_epilogue)
        event_overlap = EventOverlap(event)

        def finalize_dispatch(dispatch_result: tuple, deterministic_by_hook: bool):
            (recv_x, recv_sf,
             recv_topk_idx, recv_topk_weights,
             num_recv_tokens, num_expanded_tokens,
             num_recv_tokens_per_expert_list,
             psum_num_recv_tokens_per_scaleup_rank,
             psum_num_recv_tokens_per_expert,
             num_unaligned_recv_tokens_per_expert,
             recv_src_metadata,
             dst_buffer_slot_idx,
             token_metadata_at_forward,
             channel_linked_list) = dispatch_result

            # Create handle if not cached
            nonlocal handle
            is_cached_dispatch = handle is not None
            handle = EPHandle(do_expand,
                              num_experts, expert_alignment,
                              num_max_tokens_per_rank,
                              num_sms,
                              topk_idx,
                              num_recv_tokens, num_expanded_tokens,
                              num_recv_tokens_per_expert_list,
                              psum_num_recv_tokens_per_scaleup_rank,
                              psum_num_recv_tokens_per_expert,
                              num_unaligned_recv_tokens_per_expert,
                              recv_src_metadata,
                              dst_buffer_slot_idx,
                              token_metadata_at_forward,
                              channel_linked_list) if handle is None else handle

            # Do deterministic
            if self.deterministic:
                deterministic_epilogue = functools.partial(
                    handle.deterministic_sort,
                    do_cpu_sync, is_cached_dispatch,
                    recv_x, recv_sf, recv_topk_idx, recv_topk_weights, channel_linked_list
                )
                if deterministic_by_hook:
                    event_overlap.register_hook_after_wait(deterministic_epilogue)
                else:
                    deterministic_epilogue()

            # Return values
            recv_x = (recv_x, recv_sf) if recv_sf is not None else recv_x
            return recv_x, recv_topk_idx, recv_topk_weights, handle

        # Just launch the dispatch
        if deferred_epilogue is not None:
            event_overlap.register_hook_after_wait(
                lambda: finalize_dispatch(deferred_epilogue(), False))
            return event_overlap

        # Do epilogue ASAP
        assert result is not None
        return *finalize_dispatch(result, async_with_compute_stream), event_overlap

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) \
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    def combine(self,
                x: torch.Tensor,
                handle: EPHandle,
                topk_weights: Optional[torch.Tensor] = None,
                bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                num_sms: int = 0, num_qps: int = 0,
                previous_event: EventHandle = None,
                async_with_compute_stream: bool = False,
                allocate_on_comm_stream: bool = False,
                defer_epilogue: bool = False) \
            -> Union[Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap], EventOverlap]:
        """
        Combine (reduce) tokens from different ranks back to their original ranks.
        Supports both single-node and multi-node settings.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the `dispatch` function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float` for non-expand mode, or
                `[num_tokens]` 1D for expand mode. The tokens' top-k weights for reducing to
                its original ranks.
            bias: 0, 1 or 2 `[num_combined_tokens, hidden]` with `torch.bfloat16` final bias to the output.
            num_sms: the number of SMs to use (0 to reuse the SM count from the dispatch handle).
            num_qps: the number of RDMA QPs to use (0 for automatic via `get_theoretical_num_qps`).
            previous_event: the event to wait before actually executing the kernel.
                If set, `allocate_on_comm_stream` must also be `True`.
            async_with_compute_stream: the current stream will not wait for the communication kernels to be
                finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the
                communication stream.
            defer_epilogue: whether to defer the reduce epilogue until `event.current_stream_wait()` is called.
                This requires `async_with_compute_stream=True`.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
                Only returned when `defer_epilogue=False`.
            combined_topk_weights: the reduced top-k weights, with shape `[num_combined_tokens, num_topk]`
                and type `torch.float`. Only returned when `defer_epilogue=False`.
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
                With `defer_epilogue=True`, this function returns the `EventOverlap` object directly instead
                of the three-item tuple. Call `event.current_stream_wait()` to run the reduce epilogue and
                obtain `(combined_x, combined_topk_weights)`.
        """
        check_torch_deterministic()

        # Automatic decide SM and QP count
        num_sms = handle.num_sms if num_sms == 0 else align(num_sms, 2)
        num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps
        assert num_qps <= self.num_allocated_qps, f'Allocated QPs are not enough'

        bias_0, bias_1 = EPBuffer._unpack_bias(bias)
        result, event, deferred_epilogue = self.runtime.combine(x, topk_weights,
                                                                bias_0, bias_1,
                                                                handle.recv_src_metadata,
                                                                handle.topk_idx,
                                                                handle.psum_num_recv_tokens_per_scaleup_rank,
                                                                handle.token_metadata_at_forward,
                                                                handle.channel_linked_list,
                                                                handle.num_experts,
                                                                handle.num_max_tokens_per_rank,
                                                                num_sms, num_qps,
                                                                previous_event,
                                                                async_with_compute_stream,
                                                                allocate_on_comm_stream,
                                                                handle.do_expand,
                                                                defer_epilogue)
        event_overlap = EventOverlap(event)
        if deferred_epilogue is not None:
            event_overlap.register_hook_after_wait(deferred_epilogue)
            return event_overlap

        assert result is not None
        combined_x, combined_topk_weights = result
        return combined_x, combined_topk_weights, event_overlap

    @weak_lru(maxsize=None)
    def lb_get_theoretical_num_sms(self) -> int:
        """Estimate the LB SM count from bandwidths, rounded up to a multiple of 2."""
        # Budget both reading expert weights from HBM and writing them onto NVLink.
        # Use the bandwidth budget independently of the redundancy mapping: ring traffic
        # or rank skew can hit a link bottleneck before using all of this SM capacity.
        nvlink_gbs = get_nvlink_gbs()
        return align(math.ceil(max(nvlink_gbs / get_sm_read_gbs(), nvlink_gbs / get_sm_write_gbs())), 2)

    def lb_prefetch_weights(self,
                            redundant_expert_weights: Sequence[torch.Tensor] | torch.Tensor,
                            expert_weights: Sequence[torch.Tensor] | torch.Tensor,
                            redundancy_mapping: torch.Tensor,
                            num_sms: int = 0,
                            previous_event: Optional[EventHandle] = None) -> EventOverlap:
        """Push this rank's expert weights into the redundant expert slots requested by peers.

        Collective: every rank in the LSA domain must call it, and the fused barriers make the redundant weights
        visible to their users by the time the returned event is waited on.
        Tensor sequences must be nonempty and contain no `None` entries.

        Arguments:
            redundant_expert_weights: each tensor is a contiguous `[num_redundant_experts, *shape]`
                destination and must reside in this buffer's LB region.
            expert_weights: each tensor is a contiguous `[num_local_experts, *shape]` source with at least
                one dimension and a positive `num_local_experts`. Bytes per expert must match its
                destination and be a multiple of `deep_ep.get_num_tma_alignment()`; the trailing shapes may differ.
            redundancy_mapping: `[num_nvlink_ranks, num_redundant_experts]` int32, identical on every rank in the LSA domain;
                entry `[r, c]` is the expert id assigned to redundant slot `c` on LSA rank `r`, or -1 if empty.
                Expert ids are local to the LSA domain, in [0, num_nvlink_ranks * num_local_experts).
            num_sms: the number of SMs to use; 0 uses `lb_get_theoretical_num_sms()`.
            previous_event: the event to wait for before communication; defaults to waiting for the current stream
        """
        redundant_expert_weights = ([redundant_expert_weights] if isinstance(redundant_expert_weights, torch.Tensor)
                                    else list(redundant_expert_weights))
        expert_weights = [expert_weights] if isinstance(expert_weights, torch.Tensor) else list(expert_weights)

        num_sms = self.lb_get_theoretical_num_sms() if num_sms == 0 else align(num_sms, 2)
        return EventOverlap(self.runtime.lb_prefetch_weights(
            redundant_expert_weights, expert_weights, redundancy_mapping, num_sms, previous_event))

    def lb_reduce_grads(self,
                        redundant_expert_grads: torch.Tensor,
                        expert_grads: torch.Tensor,
                        redundancy_mapping: torch.Tensor,
                        num_sms: int = 0,
                        previous_event: Optional[EventHandle] = None) -> EventOverlap:
        """Accumulate redundant expert gradients from peers into this rank's `expert_grads`.

        The mirror of `lb_prefetch_weights`: use the same redundancy mapping to return gradients.
        Collective, and it adds rather than overwrites, so seed `expert_grads` before calling.

        Arguments:
            redundant_expert_grads: `[num_redundant_experts, hidden]` fp32 redundant gradients that must reside in this buffer's LB region.
            expert_grads: `[num_local_experts, hidden]` fp32 destination, accumulated into.
            redundancy_mapping: see `lb_prefetch_weights`.
            num_sms: the number of SMs to use; 0 uses `lb_get_theoretical_num_sms()`.
            previous_event: the event to wait for before communication; defaults to waiting for the current stream
        """
        num_sms = self.lb_get_theoretical_num_sms() if num_sms == 0 else align(num_sms, 2)
        return EventOverlap(self.runtime.lb_reduce_grads(
            redundant_expert_grads, expert_grads, redundancy_mapping, num_sms, previous_event))

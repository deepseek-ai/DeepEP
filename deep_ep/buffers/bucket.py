import inspect
import math
import torch
import torch.distributed as dist
from functools import partial
from typing import Any, Optional, Sequence, Tuple, Union

# noinspection PyUnresolvedReferences
import deep_ep._C as _C

from .allocator import BufferAllocator
from .base import BufferBase
from .. import comm
from ..utils.envs import get_nvlink_gbs, get_rdma_gbs, get_sm_read_gbs, get_sm_write_gbs
from ..utils.event import EventOverlap
from ..utils.math import align


class BucketSession:
    """A session in which collective methods can accept unregistered tensors"""

    def __init__(self, buffer: 'BucketBuffer'):
        self.buffer = buffer
        self.alignment = _C.get_num_rdma_alignment()
        self.num_allocated_bytes = 0
        self.context_idx: Optional[int] = None
        self.collective_methods = ('all_gather', 'reduce_scatter', 'all_reduce')

    def __enter__(self) -> 'BucketSession':
        # NOTES: Users must ensure themselves all the allocated bytes in the buffer can be
        # safely reused before entering this session.
        assert not any(name in self.buffer.__dict__ for name in self.collective_methods), \
            'Nested sessions are not supported'
        self.num_allocated_bytes = 0
        self.context_idx = None
        for name in self.collective_methods:
            method = partial(self._collective_func, name)
            setattr(self.buffer, name, method)
        self.buffer.ring_send_recv = self._ring_send_recv

        def send_recv(*args, **kwargs):
            raise RuntimeError('send_recv is not supported in a session')

        self.buffer.send_recv = send_recv
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for name in self.collective_methods:
            delattr(self.buffer, name)
        del self.buffer.ring_send_recv
        del self.buffer.send_recv

    def _call_original(self, name: str, *args, **kwargs):
        return getattr(type(self.buffer), name)(self.buffer, *args, **kwargs)

    def _parse_args(self, name: str, *args, **kwargs) -> dict:
        """Bind call kwargs to the original collective's signature."""
        signature = inspect.signature(getattr(type(self.buffer), name))
        bound = signature.bind(self.buffer, *args, **kwargs)
        bound.apply_defaults()
        kwargs = bound.arguments
        kwargs.pop('self')
        return kwargs

    def _allocate(self, num_bytes: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        assert num_bytes > 0
        allocated_bytes = align(num_bytes, self.alignment)
        offset = self.num_allocated_bytes
        if offset + allocated_bytes > self.buffer.num_bytes:
            raise RuntimeError(f'BucketBuffer session out of storage, which has only {self.buffer.num_bytes}')
        self.num_allocated_bytes += allocated_bytes
        return self.buffer.storage.narrow(0, offset, num_bytes).view(dtype)

    def _check_group(self, group: Optional[dist.ProcessGroup]) -> None:
        # Check only one group is used in one session.
        context_idx = self.buffer.get_context_idx(group)
        assert self.context_idx in (None, context_idx), \
            'All collectives in a session must use the same group'
        self.context_idx = context_idx

    @staticmethod
    def _wrap_handle(handle: EventOverlap,
                     dsts: Sequence[torch.Tensor],
                     result: Any) -> EventOverlap:
        """Copy staged results to their external dsts and return ``result``."""
        hook = handle.hook_after_wait
        handle.hook_after_wait = None

        def hook_after_wait():
            dst_buffers = hook()
            copy_pairs = [(dst, dst_buffer) for dst_buffer, dst in zip(dst_buffers, dsts, strict=True)
                          if dst.data_ptr() != dst_buffer.data_ptr()]
            if copy_pairs:
                torch._foreach_copy_([dst.view(-1) for dst, _ in copy_pairs],
                                     [dst_buffer.view(dst.dtype).view(-1) for dst, dst_buffer in copy_pairs])
            return result
        handle.register_hook_after_wait(hook_after_wait)
        return handle

    def allocate(self,
                 shapes: Union[Sequence[int], Sequence[Sequence[int]]],
                 dtype: torch.dtype = torch.float32,
                 group: Optional[dist.ProcessGroup] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Allocate tensors in registered buffer."""
        group = self.buffer.groups[self.buffer.get_context_idx(group)]
        self._check_group(group)

        is_batched = isinstance(shapes[0], (tuple, torch.Size, list))
        shapes = shapes if is_batched else (shapes, )
        tensors = [self._allocate(math.prod(shape) * dtype.itemsize, dtype).view(shape) for shape in shapes]
        return tuple(tensors) if is_batched else tensors[0]

    def _collective_func(self, name, *args, **kwargs) -> EventOverlap:
        """Collective methods accepting unregistered sources."""

        # Parse args
        kwargs = self._parse_args(name, *args, **kwargs)
        group = kwargs.get('group')
        group = self.buffer.groups[self.buffer.get_context_idx(group)]
        self._check_group(group)
        rank_idx, num_ranks = group.rank(), group.size()

        # Extract srcs and dsts
        num_shards = {
            'all_gather': (num_ranks, 1),
            'reduce_scatter': (1, num_ranks),
            'all_reduce': (1, 1),
        }[name]
        srcs = kwargs.pop('srcs')
        dsts = kwargs.pop('dsts', None)
        dst_exists = dsts is not None
        accumulate = name == 'reduce_scatter' and dst_exists
        if not dst_exists:
            dsts = srcs
        is_packed, srcs, dsts = BucketBuffer._pack_tensors(srcs, dsts)
        num_rdma_ranks = self.buffer.contexts[self.context_idx].get_physical_domain_size()[0]
        accept_unregistered_src = name == 'all_gather' and num_rdma_ranks == 1

        # Construct src & dst buffers
        storage_begin = self.buffer.storage.data_ptr()
        storage_end = storage_begin + self.buffer.num_bytes
        src_buffers = []
        dst_buffers = []
        copy_tasks = []
        for src, dst in zip(srcs, dsts, strict=True):
            # Check configs
            if dst_exists:
                assert src.dtype == dst.dtype
                assert dst.is_contiguous(), f'{name} output must be contiguous'
                assert dst.nbytes * num_shards[1] == src.nbytes * num_shards[0]

            # Find or allocate the registered buffer
            is_src_registered = storage_begin <= src.data_ptr() < storage_end
            is_dst_registered = storage_begin <= dst.data_ptr() < storage_end if dst_exists else False
            num_buffer_bytes = src.nbytes * num_shards[0]
            if is_src_registered:
                assert src.is_contiguous(), f'{name} registered source must be contiguous'
                offset = src.data_ptr() - storage_begin - (rank_idx % num_shards[0]) * src.nbytes
                buffer = self.buffer.storage.narrow(0, offset, num_buffer_bytes).view(src.dtype)
            elif is_dst_registered and not accumulate:
                offset = dst.data_ptr() - storage_begin - (rank_idx % num_shards[1]) * dst.nbytes
                buffer = self.buffer.storage.narrow(0, offset, num_buffer_bytes).view(dst.dtype)
            else:
                buffer = self._allocate(num_buffer_bytes, src.dtype)

            src_buffer = buffer.view(num_shards[0], -1)[rank_idx % num_shards[0]].view(src.shape)
            dst_buffer = dst if accumulate else buffer.view(num_shards[1], -1)[rank_idx % num_shards[1]]
            if dst_exists:
                dst_buffer = dst_buffer.view(dst.shape)

            if not is_src_registered and not accept_unregistered_src:
                copy_tasks.append((src_buffer, src))

            src_buffers.append(src_buffer)
            dst_buffers.append(dst_buffer)

        # In-place calls output into the registered buffers
        if not dst_exists:
            dsts = dst_buffers

        # Copy srcs
        if copy_tasks:
            torch._foreach_copy_(*zip(*copy_tasks))

        # Launch
        if accumulate:
            handle = self._call_original(name, src_buffers, dsts=dsts, **kwargs)
        elif accept_unregistered_src:
            handle = self._call_original(name, srcs, dsts=dst_buffers, **kwargs)
        else:
            handle = self._call_original(name, src_buffers, **kwargs)
        return self._wrap_handle(handle, dsts, dsts[0] if is_packed else dsts)

    def _ring_send_recv(self, *args, **kwargs) -> EventOverlap:
        """Ring send-recv accepting unregistered sources and destinations."""
        kwargs = self._parse_args('ring_send_recv', *args, **kwargs)
        group = self.buffer.groups[self.buffer.get_context_idx(kwargs.get('group'))]
        self._check_group(group)
        rank_idx, num_ranks = group.rank(), group.size()

        srcs = kwargs.pop('srcs')
        dsts = kwargs.pop('dsts')
        dst_exists = dsts is not None
        has_dst = kwargs['cyclic'] or (0 <= rank_idx + kwargs['shift'] < num_ranks)
        has_src = kwargs['cyclic'] or (0 <= rank_idx - kwargs['shift'] < num_ranks)
        is_packed, srcs, dsts = BucketBuffer._pack_tensors(srcs, dsts if dst_exists else srcs)
        accept_unregistered_src = self.buffer.contexts[self.context_idx].get_physical_domain_size()[0] == 1

        storage_begin = self.buffer.storage.data_ptr()
        storage_end = storage_begin + self.buffer.num_bytes
        src_buffers, dst_buffers, copy_tasks = [], [], []
        for src, dst in zip(srcs, dsts, strict=True):
            if dst_exists:
                assert src.dtype == dst.dtype
                assert dst.is_contiguous(), 'ring_send_recv output must be contiguous'
                assert dst.nbytes == src.nbytes

            is_src_registered = storage_begin <= src.data_ptr() < storage_end
            is_dst_registered = dst_exists and (storage_begin <= dst.data_ptr() < storage_end)
            if is_src_registered:
                assert src.is_contiguous(), 'ring_send_recv registered source must be contiguous'
                src_buffer = src
            elif accept_unregistered_src:
                src_buffer = src.contiguous()
            else:
                src_buffer = self._allocate(src.nbytes, src.dtype).view(src.shape)
                if has_dst:
                    copy_tasks.append((src_buffer, src))
            dst_buffer = dst if is_dst_registered else self._allocate(src.nbytes, src.dtype).view(dst.shape)
            src_buffers.append(src_buffer)
            dst_buffers.append(dst_buffer)

        if not dst_exists:
            dsts = dst_buffers
        if copy_tasks:
            torch._foreach_copy_(*zip(*copy_tasks))

        handle = self._call_original('ring_send_recv', src_buffers, dsts=dst_buffers, **kwargs)
        return self._wrap_handle(handle, dsts if has_src else dst_buffers, dsts[0] if is_packed else dsts)


class BucketBuffer(BufferBase):
    """Own contiguous symmetric storage registered with one or more communication groups.

    Stream usage:
        Communication runs on the comm stream; epilogues run on the current stream.
    """

    @staticmethod
    def _pack_tensors(*tensors):
        # `None` is an optional list, packed as `[]`
        is_packed = isinstance(tensors[0], torch.Tensor)
        tensors = tuple([] if item is None else [item] if isinstance(item, torch.Tensor) else list(item)
                        for item in tensors)
        assert all(len(item) in (0, len(tensors[0])) for item in tensors)
        return is_packed, *tensors

    def __init__(self,
                 group: Union[dist.ProcessGroup, Sequence[dist.ProcessGroup]],
                 allocation_plan_or_num_bytes: Union[BufferAllocator, int],
                 sl_idx: Optional[int] = None,
                 num_gpu_timeout_secs: int = 100,
                 explicitly_destroy: bool = False):
        # Provide plan or bytes
        allocation_plan = allocation_plan_or_num_bytes if isinstance(allocation_plan_or_num_bytes, BufferAllocator) else None
        num_bytes = allocation_plan_or_num_bytes if allocation_plan is None else allocation_plan.num_bytes
        if allocation_plan is not None:
            assert not allocation_plan.materialized
        assert isinstance(num_bytes, int) and num_bytes > 0

        # Configs
        self.num_bytes = num_bytes
        self.groups = [group] if isinstance(group, dist.ProcessGroup) else list(group)
        super().__init__(explicitly_destroy)
        self.nccl_comm_handles = [comm.get_nccl_comm_handle(item) for item in self.groups]

        # Bandwidths for planning the hybrid pipelines
        nvl_gbs, rdma_gbs = get_nvlink_gbs(), get_rdma_gbs()
        assert nvl_gbs > 0 and rdma_gbs > 0, 'Failed to detect the NVLink or RDMA bandwidth'

        # Create CPP runtime
        self.runtime = _C.BucketBuffer(
            [item.rank() for item in self.groups],
            [item.size() for item in self.groups],
            [item.get() for item in self.nccl_comm_handles],
            num_bytes,
            sl_idx,
            num_gpu_timeout_secs,
            nvl_gbs, rdma_gbs,
            explicitly_destroy)
        self.contexts = self.runtime.contexts
        self.storage = self.runtime.storage

        # Materialize
        if allocation_plan is not None:
            allocation_plan.materialize(self.storage)

    def get_theoretical_num_sms(self, op: str, group: Optional[dist.ProcessGroup] = None) -> int:
        context_idx = self.get_context_idx(group)
        context = self.contexts[context_idx]
        num_rdma_ranks, num_nvl_ranks = context.get_physical_domain_size()

        if op == 'send_recv':
            if num_rdma_ranks > 1 and num_nvl_ranks > 1:
                # Send recv has no assumption on the communication domain
                dst_gbs = max(get_nvlink_gbs(), get_rdma_gbs())
                num_sms = max(dst_gbs / get_sm_read_gbs(), dst_gbs / get_sm_write_gbs())
                return align(math.ceil(num_sms), 2)
            else:
                return 0

        if op == 'reduce_scatter':
            # NVLink
            if num_rdma_ranks == 1 and num_nvl_ranks > 1:
                dst_gbs = get_nvlink_gbs() / num_nvl_ranks
                num_sms = max(dst_gbs / get_sm_read_gbs(), dst_gbs / get_sm_write_gbs())
                return align(max(6, math.ceil(num_sms)), 2)

            # Hybrid
            if num_nvl_ranks > 1 and num_rdma_ranks > 1:
                dst_gbs = min(get_nvlink_gbs() / num_nvl_ranks,
                              get_rdma_gbs() * num_rdma_ranks / (num_rdma_ranks - 1))
                # Multimem and local reduction each require one read and one write.
                num_sms = max(2 * dst_gbs / get_sm_read_gbs(),
                              2 * dst_gbs / get_sm_write_gbs())
                return align(max(8, math.ceil(num_sms * 1.2)), 2)

            # RDMA
            if num_nvl_ranks == 1 and num_rdma_ranks > 1:
                algorithm_gbs = get_rdma_gbs() * num_rdma_ranks / (num_rdma_ranks - 1)
                num_sms = max(algorithm_gbs / get_sm_read_gbs(),
                              algorithm_gbs / num_rdma_ranks / get_sm_write_gbs())
                return align(max(6, math.ceil(num_sms)), 2)

        if op == 'all_reduce':
            # Hybrid
            if num_rdma_ranks > 1 and num_nvl_ranks > 1:
                dst_gbs = min(get_nvlink_gbs() / num_nvl_ranks,
                              get_rdma_gbs() * num_rdma_ranks / (2 * (num_rdma_ranks - 1)))
                # Multimem reduction, RDMA reduction and broadcast each require one read and one write.
                num_sms = max(3 * dst_gbs / get_sm_read_gbs(),
                              3 * dst_gbs / get_sm_write_gbs())
                return align(max(6, math.ceil(num_sms)), 2)

            # NVLink
            if num_rdma_ranks == 1 and num_nvl_ranks > 1:
                dst_gbs = get_nvlink_gbs() / num_nvl_ranks
                num_sms = max(dst_gbs / get_sm_read_gbs(), dst_gbs / get_sm_write_gbs())
                return align(max(4, math.ceil(num_sms)), 2)

            # RDMA
            if num_nvl_ranks == 1 and num_rdma_ranks > 1:
                rdma_gbs = get_rdma_gbs()
                # Local reduction requires one read and one write
                num_sms = max(rdma_gbs / 2 / get_sm_read_gbs(),
                              rdma_gbs / 2 / get_sm_write_gbs())
                return align(max(6, math.ceil(num_sms)), 2)

            raise NotImplementedError

        if op in ('all_gather', 'ring_send_recv'):
            return 0

        raise NotImplementedError

    def get_context_idx(self, group: Optional[dist.ProcessGroup]):
        assert len(self.groups) == 1 or group is not None, '`group` is required when the buffer has multiple groups'
        return 0 if group is None else self.groups.index(group)

    def session(self) -> BucketSession:
        """Open a session in which collectives accept unregistered tensors.
        """
        return BucketSession(self)

    def barrier(self,
                wait_comm_stream: bool = True,
                with_cpu_sync: bool = False,
                sequential: bool = True,
                group: Optional[dist.ProcessGroup] = None) -> None:
        """Run a barrier on the selected communication group."""
        context_idx = self.get_context_idx(group)
        comm.barrier(self, wait_comm_stream, with_cpu_sync, sequential, context_idx)

    def send_recv(self,
                  send_ops: list[tuple[torch.Tensor, int, int]],
                  num_sms: Optional[int] = None,
                  group: Optional[dist.ProcessGroup] = None) -> EventOverlap:
        """Launch batched send-recv without a transfer plan.

        Arguments:
            send_ops: (tensor, dst_rank_idx, dst_offset) entries. Rank indices are
                relative to group; dst_offset is a byte offset into the destination
                buffer's storage. Tensors must be contiguous views of this buffer.
                Empty tensors are ignored.
            num_sms: advised num sms to use. None selects the SM count automatically.
            group: group to communicate.

        Returns:
            An async handle; .wait() returns None. Received data is written directly
            into the destination buffer's storage.

        Note:
            Each rank in the group must participate even if it has nothing to send or recv.
        """
        context_idx = self.get_context_idx(group)
        num_sms = self.get_theoretical_num_sms('send_recv', self.groups[context_idx]) if num_sms is None else num_sms
        event, epilogue, num_sms = self.runtime.send_recv(send_ops, context_idx, num_sms)

        # Create handle
        handle = EventOverlap(event)
        handle.register_hook_after_wait(epilogue)
        setattr(handle, 'num_sms', num_sms)
        return handle

    def reduce_scatter(self,
                       srcs: Union[torch.Tensor, Sequence[torch.Tensor]],
                       comm_precision: str = 'fp32',
                       scale: float = 1.0,
                       group: Optional[dist.ProcessGroup] = None,
                       num_sms: Optional[int] = None,
                       *,
                       dsts: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None) -> EventOverlap:
        """Launch batched reduce-scatter; ``None`` selects the SM count automatically.
        Accumulate into ``dsts`` when provided; otherwise overwrite the local shards in-place.
        """
        is_packed, srcs, dsts = self._pack_tensors(srcs, dsts)

        # Launch
        context_idx = self.get_context_idx(group)
        num_sms = self.get_theoretical_num_sms('reduce_scatter', self.groups[context_idx]) if num_sms is None else num_sms
        dsts, event, num_sms = self.runtime.reduce_scatter(
            srcs, context_idx, num_sms, comm_precision, scale, dsts)

        # Create handle
        handle = EventOverlap(event)
        handle.register_hook_after_wait(lambda: dsts[0] if is_packed else dsts)
        setattr(handle, 'num_sms', num_sms)
        return handle

    def all_reduce(self,
                   srcs: Union[torch.Tensor, Sequence[torch.Tensor]],
                   scale: float = 1.0,
                   group: Optional[dist.ProcessGroup] = None,
                   num_sms: Optional[int] = None) -> EventOverlap:
        """Launch batched in-place all-reduce; ``None`` selects the SM count automatically.

        Arguments:
            srcs: source tensors to reduce; must belong to this ``BucketBuffer`` and are
                overwritten in-place with the reduced results.
            scale: the scaling factor applied to the reduced results.
            group: the communication group to use. Required when this buffer owns multiple groups.
            num_sms: the number of SMs to use. ``None`` selects the SM count automatically.
        """
        is_packed, srcs = self._pack_tensors(srcs)

        # Launch
        context_idx = self.get_context_idx(group)
        num_sms = self.get_theoretical_num_sms('all_reduce', self.groups[context_idx]) if num_sms is None else num_sms
        dsts, event, num_sms = self.runtime.all_reduce(srcs, context_idx, num_sms, scale)

        # Create handle
        handle = EventOverlap(event)
        handle.register_hook_after_wait(lambda: dsts[0] if is_packed else dsts)
        setattr(handle, 'num_sms', num_sms)
        return handle

    def all_gather(self,
                   srcs: Union[torch.Tensor, Sequence[torch.Tensor]],
                   dsts: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
                   group: Optional[dist.ProcessGroup] = None,
                   num_sms: Optional[int] = None) -> EventOverlap:
        """Launch batched all-gather.

        Arguments:
            srcs: the local shards. Without ``dsts``, each must belong to this ``BucketBuffer`` and be the
                ``rank_idx``-th shard of the gathered tensor, which is gathered in-place. With ``dsts``, each can be
                any contiguous CUDA tensor (out-of-buffer inputs are only supported by the pure NVLink path).
            dsts: the gathered tensors, each must belong to this ``BucketBuffer`` with
                ``dst.nbytes == group.size() * src.nbytes``.
            group: the communication group to use. Required when this buffer owns multiple groups.
            num_sms: the number of SMs to use. Must be ``0`` or ``None`` (copy-engine driven).

        Returns:
            a handle whose ``wait()`` returns the gathered tensor(s) as 1-D views over the storage with the dtype
            of ``srcs``.
        """
        is_packed, srcs, dsts = self._pack_tensors(srcs, dsts)

        # Launch
        context_idx = self.get_context_idx(group)
        num_sms = self.get_theoretical_num_sms('all_gather', self.groups[context_idx]) if num_sms is None else num_sms
        event, epilogue = self.runtime.all_gather(srcs, dsts, context_idx, num_sms)

        # Create handle
        handle = EventOverlap(event)
        handle.register_hook_after_wait(lambda: epilogue()[0] if is_packed else epilogue())
        setattr(handle, 'num_sms', num_sms)
        return handle

    def ring_send_recv(self,
                       srcs: Union[torch.Tensor, Sequence[torch.Tensor]],
                       dsts: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
                       shift: int = 1,
                       cyclic: bool = False,
                       group: Optional[dist.ProcessGroup] = None,
                       num_sms: Optional[int] = None) -> EventOverlap:
        """Launch batched ring send-recv where each rank sends to ``rank + shift``.

        Arguments:
            srcs: source tensor(s) to send; all ranks must pass matching shapes and dtypes.
                Out-of-buffer inputs are only supported by the pure NVLink path.
            dsts: destination tensor(s) belonging to this ``BucketBuffer``, non-overlapping with ``srcs``,
                and allocated at symmetric offsets across all ranks in ``group``.
            shift: rank offset to send to (e.g., ``1`` sends ``rank -> rank + 1``, ``-1`` sends ``rank -> rank - 1``).
            cyclic: if ``False`` (default), boundary ranks whose destination/source is out of ``[0, group.size())``
                do not send/receive; if ``True``, wrap around modulo ``group.size()``.
            group: the communication group to use. Required when this buffer owns multiple groups.
            num_sms: the number of SMs to use. Must be ``0`` or ``None``.

        Returns:
            a handle whose ``wait()`` returns the destination tensor(s).
        """
        assert dsts is not None, '`dsts` is required when calling `ring_send_recv` outside a session'
        is_packed, srcs, dsts = self._pack_tensors(srcs, dsts)

        # Launch
        context_idx = self.get_context_idx(group)
        num_sms = self.get_theoretical_num_sms('ring_send_recv', self.groups[context_idx]) if num_sms is None else num_sms
        event, epilogue = self.runtime.ring_send_recv(srcs, dsts, shift, cyclic, context_idx, num_sms)

        # Create handle
        handle = EventOverlap(event)
        handle.register_hook_after_wait(lambda: epilogue()[0] if is_packed else epilogue())
        setattr(handle, 'num_sms', num_sms)
        return handle

    def destroy(self) -> None:
        super().destroy()
        self.contexts = None
        self.storage = None
        self.nccl_comm_handles = None

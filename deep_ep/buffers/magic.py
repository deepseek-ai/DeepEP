from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import deep_ep._C as _C

from ..comm import create_nccl_comm_handle
from ..utils.event import EventOverlap


class MagicBuffer:
    """Dispatch and combine tokens with caller-allocated CUDA output tensors.

    All ranks must issue matching operations in the same order. Each dispatch must
    be followed by exactly one combine before another dispatch on this buffer.
    Call ``destroy()`` explicitly on every rank before shutting down the group.
    """

    def __init__(self, group: dist.ProcessGroup, max_tokens_per_rank: int, workspace_size: int = 0) -> None:
        """Collectively create a buffer on the current CUDA device.

        Arguments:
            group: the communication group; all ranks must construct the buffer together.
            max_tokens_per_rank: the maximum number of input tokens on each rank.
                Must be positive and identical across the group.
            workspace_size: the workspace size in bytes for each of the send and receive
                workspaces on each rank. Zero selects 256 MiB. Must be identical across
                the group and large enough for the configured token count and row size.
        """
        world_size = group.size()
        workspace_size = workspace_size or (256 << 20)
        if world_size not in (2, 4, 8, 16, 32, 64):
            raise ValueError('world_size must be 2, 4, 8, 16, 32 or 64')
        self._nccl_comm_handle = create_nccl_comm_handle(group)
        self.runtime = _C.MagicBuffer(group.rank(), world_size, self._nccl_comm_handle.get(), max_tokens_per_rank, workspace_size)
        dist.barrier(group=group)

    def destroy(self) -> None:
        if self.runtime is not None:
            runtime, self.runtime = self.runtime, None
            runtime.destroy()
        self._nccl_comm_handle = None

    def dispatch(self,
                 x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 topk_ids: torch.Tensor,
                 topk_weights: torch.Tensor,
                 num_local_experts: int,
                 recv_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 recv_topk_ids: torch.Tensor,
                 recv_topk_weights: torch.Tensor,
                 async_with_compute_stream: bool = False,
                 psum_recv_per_rank: Optional[torch.Tensor] = None) -> Optional[EventOverlap]:
        """Collectively route tokens into preallocated receive tensors.

        All tensors must be contiguous and on this buffer's CUDA device. Ranks must
        use matching layouts, top-k widths and expert counts. Tokens are deduplicated
        per destination rank. Rows from the same source peer are contiguous, but
        the peer blocks are not sorted by rank.

        Arguments:
            x: payload ``[num_tokens, hidden]`` or a ``(payload, scales)`` pair, with
                scales shaped ``[num_tokens, num_scale_values]``. Requires
                ``num_tokens <= max_tokens_per_rank`` and row sizes that are multiples
                of 16 bytes.
            topk_ids: int32 global expert indices ``[num_tokens, topk]``, with
                ``1 <= topk <= 32``. Negative indices are ignored.
            topk_weights: float32 weights matching ``topk_ids``.
            num_local_experts: the positive number of experts per rank.
            recv_x: matching output payload ``[capacity, hidden]`` or a payload/scale pair.
                Payload, scale and top-k outputs must share sufficient receive capacity.
            recv_topk_ids: int32 output with shape ``[capacity, topk]``.
            recv_topk_weights: float32 output with shape ``[capacity, topk]``.
            async_with_compute_stream: if true, call ``event.current_stream_wait()``
                before using outputs; otherwise the stream wait is enqueued internally.
            psum_recv_per_rank: optional int32 output ``[group.size()]`` containing
                inclusive prefix sums of receive counts in peer arrival order. After
                completion, its last element gives the valid row count. Unused receive
                rows are not cleared.

        Returns:
            ``EventOverlap`` if asynchronous, otherwise ``None``. Keep tensors and the
            event alive until completion. Call combine once with the same receive
            capacity and row order before dispatching again.
        """
        x, x_scale = x if isinstance(x, tuple) else (x, None)
        recv_x, recv_x_scale = recv_x if isinstance(recv_x, tuple) else (recv_x, None)
        event = self.runtime.dispatch(x,
                                      x_scale,
                                      topk_ids,
                                      topk_weights,
                                      recv_x,
                                      recv_x_scale,
                                      recv_topk_ids,
                                      recv_topk_weights,
                                      num_local_experts,
                                      async_with_compute_stream,
                                      psum_recv_per_rank)
        return EventOverlap(event) if async_with_compute_stream else None

    def combine(self,
                x: torch.Tensor,
                out: torch.Tensor,
                async_with_compute_stream: bool = False,
                combine_num_sf_packs: int = 0) -> Optional[EventOverlap]:
        """Collectively sum routed results using the preceding dispatch's routing.

        Call this method exactly once after each dispatch on every rank. All tensors
        must be contiguous CUDA tensors on this buffer's device. Preserve the dispatched
        receive row order and capacity; do not slice ``x`` to the valid row count.
        Only valid received rows are read.

        Arguments:
            x: results in dispatch receive order: bfloat16 ``[capacity, hidden]`` when
                ``combine_num_sf_packs=0``, otherwise uint8 rows packing ``hidden`` int8
                values, float32 scales and alignment padding. Rows must occupy a
                multiple of 16 bytes.
            out: bfloat16 ``[num_tokens, hidden]``, with the preceding dispatch's input
                token count.
            async_with_compute_stream: if true, call ``event.current_stream_wait()``
                before using ``out``; otherwise the stream wait is enqueued internally.
            combine_num_sf_packs: the number of scales per quantized row; zero selects bfloat16.

        Returns:
            An ``EventOverlap`` when asynchronous, otherwise ``None``. Keep tensors
            and the returned event alive until the operation finishes.
        """
        event = self.runtime.combine(x, out, async_with_compute_stream, combine_num_sf_packs)
        return EventOverlap(event) if async_with_compute_stream else None

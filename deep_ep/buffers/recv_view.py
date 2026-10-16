import torch
from typing import Optional

from .. import comm


class DispatchRecvView:
    """
    Forward-only receive storage of a borrowed compact dispatch: logical row `r` is `slab[row_indices[r]]`.

    For asynchronous dispatch, wait the returned dispatch event first. Consumers on other streams call `wait()`
    before reading. After enqueueing all consumers, call `release()` with every consuming stream; raw slab
    references are invalid afterwards.
    """

    def __init__(self, owner, slab: torch.Tensor, row_indices: torch.Tensor) -> None:
        self._owner = owner
        self._slab = slab
        self._row_indices = row_indices
        self._ready_event: Optional[torch.cuda.Event] = None

    @property
    def slab(self) -> torch.Tensor:
        self._check_ready()
        return self._slab

    @property
    def row_indices(self) -> torch.Tensor:
        self._check_ready()
        return self._row_indices

    def wait(self) -> None:
        """
        The current stream waits for the dispatch and its epilogue, for consumers on other streams.
        """
        self._check_ready()
        torch.cuda.current_stream(self._slab.device).wait_event(self._ready_event)

    def release(self, *consumer_streams: torch.cuda.Stream) -> None:
        """
        Order later communication after the consumers on the given streams (the current stream by default).
        """
        self._check_ready()
        consumer_streams = consumer_streams or (torch.cuda.current_stream(self._slab.device), )
        comm_stream = comm.get_comm_stream(self._owner)
        for stream in consumer_streams:
            assert stream.device == self._slab.device, 'Consumer streams must be on the receive-view device'
            stream.wait_event(self._ready_event)
            self._row_indices.record_stream(stream)
            comm_stream.wait_stream(stream)
        self._owner._active_recv_view = None
        self._owner = self._slab = self._row_indices = self._ready_event = None

    def _check_ready(self) -> None:
        if self._slab is None:
            raise RuntimeError('The dispatch receive view has been released')
        if self._ready_event is None:
            raise RuntimeError('Wait the dispatch event before consuming the receive view')

    def _mark_ready(self) -> None:
        self._ready_event = torch.cuda.Event()
        self._ready_event.record(torch.cuda.current_stream(self._slab.device))

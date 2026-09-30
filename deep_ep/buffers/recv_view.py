"""An exclusive, forward-only view of compact BF16 dispatch payloads."""
import torch

from .. import comm


class DispatchRecvView:
    """Read logical row ``r`` as ``slab[row_indices[r]]`` after dispatch wait.

    The payload belongs to the communication buffer. Keep host calls serialized
    and release this view after enqueueing all consumers, passing every consuming
    CUDA stream. Copies of ``slab`` references are invalid after release. Saving
    this view for backward is unsupported; use a materialized cached replay.
    """

    def __init__(self, owner, slab, row_indices):
        self._owner = owner
        self._slab = slab
        self._row_indices = row_indices
        self._ready_event = None
        self._released = False

    def _mark_ready(self):
        self._ready_event = torch.cuda.Event()
        self._ready_event.record(torch.cuda.current_stream(self._slab.device))

    def wait(self):
        if self._released:
            raise RuntimeError('The dispatch receive view has been released')
        if self._ready_event is None:
            raise RuntimeError('Wait the dispatch event before consuming the receive view')
        torch.cuda.current_stream(self._slab.device).wait_event(self._ready_event)

    @property
    def slab(self):
        self.wait()
        return self._slab

    @property
    def row_indices(self):
        self.wait()
        return self._row_indices

    def release(self, *consumer_streams):
        self.wait()
        if not consumer_streams:
            consumer_streams = (torch.cuda.current_stream(self._slab.device),)
        comm_stream = comm.get_comm_stream(self._owner)
        for stream in consumer_streams:
            if stream.device != self._slab.device:
                raise ValueError('Consumer streams must belong to the receive-view device')
            stream.wait_event(self._ready_event)
            self._row_indices.record_stream(stream)
            comm_stream.wait_stream(stream)
        self._owner._active_recv_view = None
        self._released = True
        self._owner = self._slab = self._row_indices = self._ready_event = None

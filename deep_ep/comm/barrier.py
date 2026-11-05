from typing import Any, Optional

import deep_ep._C as _C


def barrier(buffer: Any, wait_comm_stream: bool = True,
            with_cpu_sync: bool = False, sequential: bool = True,
            context_idx: Optional[int] = None) -> None:
    """Run a barrier on the current stream, optionally waiting for the communication stream."""
    context = buffer.context if context_idx is None else buffer.contexts[context_idx]
    _C.barrier(context, with_cpu_sync, sequential, wait_comm_stream)

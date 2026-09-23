from typing import Any

import torch

# noinspection PyUnresolvedReferences
import deep_ep._C as _C


def get_comm_stream(buffer: Any) -> torch.Stream:
    """Return DeepEP's communication stream."""
    ts: torch.Stream = _C.get_comm_stream()
    return torch.cuda.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)

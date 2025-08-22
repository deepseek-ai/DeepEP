import os
import subprocess
import torch
import torch.distributed as dist
from typing import Any, Optional, Tuple

# noinspection PyUnresolvedReferences
from deep_ep_cpp import Config, EventHandle


class EventOverlap:
    """
    A wrapper class to manage CUDA events, also for better overlapping convenience.

    Attributes:
        event: the CUDA event captured.
        extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
    """

    def __init__(self, event: Optional[EventHandle] = None,
                 extra_tensors: Optional[Tuple[torch.Tensor]] = None) -> None:
        """
        Initialize the class.

        Arguments:
            event: the CUDA event captured.
            extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
        """
        self.event = event

        # NOTES: we use extra tensors to achieve stream recording, otherwise,
        # stream recording will be incompatible with CUDA graph.
        self.extra_tensors = extra_tensors

    def current_stream_wait(self) -> None:
        """
        The current stream `torch.cuda.current_stream()` waits for the event to be finished.
        """
        assert self.event is not None
        self.event.current_stream_wait()

    def __enter__(self) -> Any:
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        with event_overlap():
            do_something_on_current_stream()
        # After exiting the `with` scope, the current stream with wait the event to be finished.
        ```
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.
        """
        if self.event is not None:
            self.event.current_stream_wait()


def check_nvlink_connections(group: dist.ProcessGroup, 
                              allow_nvlink_for_normal_mode: bool = True,
                              allow_nvlink_for_low_latency_mode: bool = True,
                              low_latency_mode: bool = False) -> None:
    """
    Check NVLink requirements based on the mode and configuration.
    
    Arguments:
        group: the communication group.
        allow_nvlink_for_normal_mode: whether NVLink is allowed for normal mode.
        allow_nvlink_for_low_latency_mode: whether NVLink is allowed for low-latency mode.
        low_latency_mode: whether running in low-latency mode.
    """
    # Determine which setting to check
    allow_nvlink = allow_nvlink_for_low_latency_mode if low_latency_mode else allow_nvlink_for_normal_mode
    if not allow_nvlink:
        return 
    
    # noinspection PyUnresolvedReferences
    import pynvml
    
    pynvml.nvmlInit()
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').strip(',').split(',')
    physical_device_indices = [0] * group.size()
    physical_device_indices[group.rank()] = int(devices[torch.cuda.current_device()])
    dist.all_gather_object(physical_device_indices, physical_device_indices[group.rank()], group)
    
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_indices]
    for i in range(len(handles)):
        for j in range(i + 1, len(handles)):
            status = pynvml.nvmlDeviceGetP2PStatus(handles[i], handles[j], pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
            assert status == pynvml.NVML_P2P_STATUS_OK, \
                f'No NVLink connection between GPU {physical_device_indices[i]} and GPU {physical_device_indices[j]}, ' \
                f'but allow_nvlink_for_{"low_latency" if low_latency_mode else "normal"}_mode=True'
    
    pynvml.nvmlShutdown()



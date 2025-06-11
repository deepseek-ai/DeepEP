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


def check_nvlink_connections(group: dist.ProcessGroup):
    """
    Check NVLink connection between every pair of GPUs.

    Arguments:
        group: the communication group.
    """
    # Check NVLink connection
    # NOTES: some A100 PCIE GPUs only have pairwise NVLink connection, so that we can only use EP2
    if 'PCIE' in torch.cuda.get_device_name():
        assert group.size() <= 2, 'No NVLink connection between all GPUs'
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').strip(',').split(',')
        physical_device_idx = int(devices[torch.cuda.current_device()])
        physical_device_indices = [0, ] * group.size()
        dist.all_gather_object(physical_device_indices, physical_device_idx, group)

        # Get connection matrix from `nvidia-smi`
        lines = subprocess.check_output(['nvidia-smi', 'topo', '-p2p', 'n']).decode('utf-8').split('\n')
        for line in lines:
            if line.lstrip().startswith(f'GPU{physical_device_idx}') and 'X' in line:
                status = line.strip().lstrip(f'GPU{physical_device_idx}').split()
                for dst_gpu_rank in physical_device_indices:
                    assert status[dst_gpu_rank] in ('X', 'OK'), f'No NVLink connection between GPU {physical_device_idx} and GPU {dst_gpu_rank}'

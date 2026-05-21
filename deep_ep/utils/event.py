import torch
from typing import Any, Optional, Tuple, Callable

# noinspection PyUnresolvedReferences
from deep_ep._C import EventHandle


class EventOverlap:
    """
    A wrapper class to manage CUDA events, also for better overlapping convenience.

    Attributes:
        event: the CUDA event captured.
        extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
    """

    def __init__(self, event: Optional[EventHandle] = None, extra_tensors: Optional[Tuple[torch.Tensor]] = None) -> None:
        """
        Initialize the class.

        Arguments:
            event: the CUDA event captured.
            extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
        """
        self.event = event

        # NOTES: we use extra tensors to achieve stream recording, otherwise,
        # stream recording will be incompatible with CUDA graph.
        # TODO: `extra_tensors` is not longer useful for EPv2, as objects are stored in `self.event`
        self.extra_tensors = extra_tensors

        # A wrapper for `with event_overlap(release_handle=True)`
        self._release_handle_by_call = False

        # A hook that will be triggered after `current_stream_wait()`
        # Useful for deterministic dispatch, which requires a sort (on the current stream) after `self.current_stream_wait()` is invoked
        self.hook_after_wait: Optional[Callable] = None

    def current_stream_wait(self, release_handle: bool = False) -> None:
        """
        The current stream `torch.cuda.current_stream()` waits for the event to be finished.
        """
        assert self.event is not None
        self.event.current_stream_wait()

        if self.hook_after_wait is not None:
            self.hook_after_wait()
            self.hook_after_wait = None

        # In `self.event`, we also have some V2 APIs storing tensors to record in it,
        # So, after waiting the current stream, those tensors can be released by deleting `self.event`
        # However, you better do it by yourself (to be compatible with multi-stream waits)
        if release_handle:
            self.event = None

    def register_hook_after_wait(self, hook_after_wait: Callable) -> None:
        """
        Register a hook, which will be invoked after `self.current_stream_wait()`
        """
        assert self.hook_after_wait is None, "A hook is already registered on this `EventOverlap`"
        self.hook_after_wait = hook_after_wait

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        """
        Configures the 'release_handle' behavior for the upcoming context manager usage.
        Usage:
            with event_overlap(release_handle=True):
                ...
        Returns `self` to ensure no new wrapper object is created, keeping the reference count of the underlying event unchanged/managed solely by this instance.
        """
        self._release_handle_by_call = release_handle
        return self

    def __enter__(self) -> Any:
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        with event_overlap:
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
            self.current_stream_wait(release_handle=self._release_handle_by_call)
        self._release_handle_by_call = False

from abc import ABC, abstractmethod


class BufferBase(ABC):
    """Common lifecycle for communication buffers."""

    def __init__(self, explicitly_destroy: bool = False):
        self.explicitly_destroy = explicitly_destroy
        self.runtime = None

    @property
    def main_context(self):
        """The first communication context, or ``None`` after destruction."""
        return self.runtime.main_context if self.runtime is not None else None

    def print_memory_usage(self) -> None:
        """Print this rank's workspace, communication buffer, and RDMA storage sizes."""
        assert self.runtime is not None
        self.runtime.print_memory_usage()

    @abstractmethod
    def destroy(self) -> None:
        """Destroy the runtime. Requires ``explicitly_destroy=True`` at construction."""
        assert self.explicitly_destroy
        if self.runtime is not None:
            self.runtime.destroy()
            self.runtime = None

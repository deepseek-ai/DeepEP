from enum import Enum, auto
import os

class BackendType(Enum):
    NVSHMEM = auto()
    NCCL_GIN = auto()
    AUTO = auto()

    @classmethod
    def from_str(cls, backend_str: str) -> 'BackendType':
        backend_str = backend_str.lower()
        if backend_str == 'nvshmem':
            return cls.NVSHMEM
        elif backend_str in ('nccl_gin', 'nccl', 'gin'):
            return cls.NCCL_GIN
        elif backend_str == 'auto':
            return cls.AUTO
        else:
            raise ValueError(f"Unknown backend type: {backend_str}")

    @classmethod
    def detect(cls) -> 'BackendType':
        # Check environment variable
        env_backend = os.getenv('DEEP_EP_BACKEND')
        if env_backend:
            try:
                return cls.from_str(env_backend)
            except ValueError as e:
                print(f"Warning: Invalid DEEP_EP_BACKEND value: {env_backend}, falling back to auto-detection")
        
        # Auto-detection logic: Default to NVSHMEM for backward compatibility
        return cls.NVSHMEM

    def __str__(self) -> str:
        if self == BackendType.NVSHMEM:
            return "NVSHMEM"
        elif self == BackendType.NCCL_GIN:
            return "NCCL_GIN"
        elif self == BackendType.AUTO:
            return "AUTO"
        else:
            return "UNKNOWN" 
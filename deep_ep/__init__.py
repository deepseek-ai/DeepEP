import filecmp
import functools
import glob
import os
import subprocess
import sysconfig
from pathlib import Path

import torch

from .utils.find_pkgs import find_nccl_root


def find_nccl_location():
    """Return the NCCL root and library directory for package or Ubuntu installs."""
    try:
        nccl_root = Path(find_nccl_root())
    except AssertionError:
        if not Path('/usr/include/nccl.h').is_file():
            raise
        multiarch = sysconfig.get_config_var('MULTIARCH')
        lib_dirs = []
        if multiarch:
            lib_dirs.extend([
                Path('/usr/lib') / multiarch,
                Path('/lib') / multiarch,
            ])
        lib_dirs.extend([Path('/usr/lib'), Path('/usr/local/lib')])
        for lib_dir in lib_dirs:
            candidates = [lib_dir / 'libnccl.so', *sorted(lib_dir.glob('libnccl.so.*'))]
            if any(candidate.is_file() for candidate in candidates):
                return Path('/usr'), lib_dir
        raise
    return nccl_root, nccl_root / 'lib'

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Initialize
@functools.lru_cache()
def find_cuda_home() -> str:
    """
    Find the CUDA installation directory, cached.

    Returns:
        cuda_home: the CUDA installation path.
    """
    # TODO: reuse PyTorch API later
    # For some PyTorch versions, the original `_find_cuda_home` will initialize CUDA, which is incompatible with process forks
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # noinspection PyBroadException
        try:
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output(['which', 'nvcc'], stderr=devnull).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    assert cuda_home is not None
    return cuda_home


def check_nccl_so():
    """
    Verify that the NCCL library loaded at runtime matches the linked version.
    Aborts if duplicate NCCL libraries are found or if versions mismatch.
    """
    if int(os.environ.get('EP_SUPPRESS_NCCL_CHECK', 0)):
        return

    # PyTorch may load another NCCL library, which is different to the linked one
    with open('/proc/self/maps', 'r') as f:
        loaded_nccl_so = None
        for so in [line.strip().split(' ')[-1] for line in f if 'libnccl' in line]:
            loaded_nccl_so = so if loaded_nccl_so is None else loaded_nccl_so
            assert so == loaded_nccl_so, f'Duplicate NCCL runtime found in the current system: {so} and {loaded_nccl_so}'
    nccl_root, nccl_lib_dir = find_nccl_location()
    linked_nccl_so_candidates = sorted(glob.glob(f'{nccl_lib_dir}/libnccl.so*'))
    assert linked_nccl_so_candidates, f'No libnccl.so found in {nccl_lib_dir}/'
    linked_nccl_so = linked_nccl_so_candidates[0]

    # So checking binary-level equalness is necessary
    # noinspection PyTypeChecker
    assert filecmp.cmp(loaded_nccl_so, linked_nccl_so, shallow=False), \
        (f'Invalid NCCL versions: {loaded_nccl_so} (loaded) v.s. {linked_nccl_so} (expected), '
         f'please contact Chenggang or Shangyan to upgrade PyTorch NCCL version')


def init_jit():
    """
    Initialize the JIT compilation runtime. Sets up CUDA and NCCL root paths for the JIT compiler.
    """
    # noinspection PyUnresolvedReferences
    import deep_ep._C as _C
    library_root_path = os.path.dirname(os.path.abspath(__file__))
    nccl_root, _ = find_nccl_location()
    _C.init_jit(library_root_path,  # Library root directory path
                find_cuda_home(),   # CUDA home
                str(nccl_root))     # NCCL root

# Run initialization
check_nccl_so()
init_jit()


# Import APIs after initialization
from .buffers.legacy import Buffer
from .buffers.elastic import ElasticBuffer, EPHandle
# noinspection PyUnresolvedReferences
from .utils.event import EventOverlap, EventHandle
from .utils.envs import get_physical_domain_size, get_logical_domain_size

# noinspection PyUnresolvedReferences
from deep_ep._C import Config, topk_idx_t

__version__ = '2.1.0'

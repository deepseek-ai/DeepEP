import functools
import os
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import Optional


def find_pkg_root(name: str, lib_name: Optional[str] = None, optional: bool = False):
    """
    Find the root directory of an installed NVIDIA package by inspecting Python package metadata.
    Checks environment variables `EP_{NAME}_ROOT_DIR` and `{NAME}_DIR` first.

    Arguments:
        name: the package name (e.g., `'nccl'`).
        lib_name: the library filename to search for within the package files.
        optional: if ``False``, raises an assertion error when the package is not found.

    Returns:
        root: the package root directory, or `None` if not found and optional.
    """
    upper = name.upper()
    for env_name in (f'EP_{upper}_ROOT_DIR', f'{upper}_DIR'):
        if env_name in os.environ:
            return os.environ[env_name]

    path_priority = {p: i for i, p in enumerate(sys.path)}
    best, best_priority = None, len(sys.path)

    for dist in distributions():
        dist_name = (dist.metadata['Name'] or '').lower()
        if f'nvidia-{name}' not in dist_name and f'nvidia_{name}' not in dist_name:
            continue

        dist_site = str(dist._path.parent)
        priority = path_priority.get(dist_site, len(sys.path))
        if priority > best_priority:
            continue

        if lib_name is not None:
            for f in (dist.files or []):
                if lib_name in str(f):
                    lib_dir = os.path.dirname(str(f.locate()))
                    root = os.path.dirname(lib_dir) if os.path.basename(lib_dir) == 'lib' else lib_dir
                    best, best_priority = root, priority
                    break
        else:
            pkg_dir = os.path.join(dist_site, 'nvidia', name)
            if os.path.isdir(pkg_dir):
                best, best_priority = pkg_dir, priority

    # Raise error if not optional
    if not optional:
        assert best is not None, f'Cannot find package: {name}'
    return best


@functools.lru_cache()
def find_nccl_root(optional: bool = False):
    """
    Find the NCCL installation root directory, cached.

    Arguments:
        optional: if `False`, raises an assertion error when NCCL is not found.

    Returns:
        root: the NCCL root directory.
    """
    return find_pkg_root('nccl', lib_name='libnccl.so', optional=optional)


def get_nccl_lib_name(root: str) -> str:
    """Resolve the unversioned library or the SONAME-only library shipped in pip wheels."""
    lib_dir = Path(root) / 'lib'
    for path in (lib_dir / 'libnccl.so', *sorted(lib_dir.glob('libnccl.so.*'))):
        if path.is_file():
            return path.name
    raise ModuleNotFoundError(f'libnccl.so not found under {lib_dir}')

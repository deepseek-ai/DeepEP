import ctypes
import re
from pathlib import Path


_VERSION_MACROS = ("NCCL_MAJOR", "NCCL_MINOR", "NCCL_PATCH")


def read_nccl_header_version(header_path) -> int:
    header_path = Path(header_path)
    contents = header_path.read_text(encoding="utf-8")
    values = {}
    for name in _VERSION_MACROS:
        match = re.search(rf"^\s*#\s*define\s+{name}\s+(\d+)\s*$", contents, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing {name} in NCCL header: {header_path}")
        values[name] = int(match.group(1))
    return values["NCCL_MAJOR"] * 10000 + values["NCCL_MINOR"] * 100 + values["NCCL_PATCH"]


def get_nccl_runtime_version(library_path: str) -> int:
    library = ctypes.CDLL(library_path)
    try:
        get_version = library.ncclGetVersion
    except AttributeError as exc:
        raise RuntimeError(f"{library_path} does not export ncclGetVersion") from exc
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    return_code = get_version(ctypes.byref(version))
    if return_code != 0:
        raise RuntimeError(f"ncclGetVersion failed for {library_path} with return code {return_code}")
    return version.value


def format_nccl_version(version: int) -> str:
    major = version // 10000
    minor = (version // 100) % 100
    patch = version % 100
    return f"{major}.{minor}.{patch}"

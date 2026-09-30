import ctypes
import importlib.util
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[2] / "deep_ep" / "utils" / "nccl.py"


def load_nccl_module():
    spec = importlib.util.spec_from_file_location("nccl_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_setup_module(monkeypatch):
    cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    cpp_extension.BuildExtension = object
    cpp_extension.CUDAExtension = lambda *args, **kwargs: None
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.cpp_extension = cpp_extension
    torch = types.ModuleType("torch")
    torch.utils = torch_utils
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)

    setup_path = Path(__file__).parents[2] / "setup.py"
    spec = importlib.util.spec_from_file_location("setup_under_test", setup_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reads_version_from_nccl_header(tmp_path):
    module = load_nccl_module()
    header = tmp_path / "nccl.h"
    header.write_text(
        "#define NCCL_MAJOR 2\n#define NCCL_MINOR 30\n#define NCCL_PATCH 4\n",
        encoding="utf-8",
    )

    assert module.read_nccl_header_version(header) == 23004


def test_rejects_header_with_missing_version_macro(tmp_path):
    module = load_nccl_module()
    header = tmp_path / "nccl.h"
    header.write_text(
        "#define NCCL_MAJOR 2\n#define NCCL_MINOR 30\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="NCCL_PATCH"):
        module.read_nccl_header_version(header)


@pytest.mark.parametrize(
    ("version", "expected"),
    [(23004, "2.30.4"), (22809, "2.28.9"), (30000, "3.0.0")],
)
def test_formats_nccl_version(version, expected):
    module = load_nccl_module()

    assert module.format_nccl_version(version) == expected


class FakeNcclGetVersion:
    def __init__(self, version, return_code=0):
        self.version = version
        self.return_code = return_code

    def __call__(self, version_ptr):
        ctypes.cast(version_ptr, ctypes.POINTER(ctypes.c_int)).contents.value = self.version
        return self.return_code


class FakeLibrary:
    def __init__(self, version, return_code=0):
        self.ncclGetVersion = FakeNcclGetVersion(version, return_code)


def test_reads_runtime_version_with_nccl_api(monkeypatch):
    module = load_nccl_module()
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: FakeLibrary(22809))

    assert module.get_nccl_runtime_version("/runtime/libnccl.so.2") == 22809


def test_reports_nccl_api_failure(monkeypatch):
    module = load_nccl_module()
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: FakeLibrary(0, 7))

    with pytest.raises(RuntimeError, match=r"libnccl\.so\.2.*return code 7"):
        module.get_nccl_runtime_version("/runtime/libnccl.so.2")


def test_reports_missing_nccl_version_symbol(monkeypatch):
    module = load_nccl_module()
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: object())

    with pytest.raises(RuntimeError, match=r"libnccl\.so\.2.*does not export ncclGetVersion"):
        module.get_nccl_runtime_version("/runtime/libnccl.so.2")


def test_build_metadata_records_nccl_header_version(tmp_path, monkeypatch):
    module = load_setup_module(monkeypatch)
    nccl_root = tmp_path / "nccl"
    include_dir = nccl_root / "include"
    include_dir.mkdir(parents=True)
    (include_dir / "nccl.h").write_text(
        "#define NCCL_MAJOR 2\n#define NCCL_MINOR 30\n#define NCCL_PATCH 4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module.find_pkgs, "find_nccl_root", lambda: str(nccl_root))
    build_lib = tmp_path / "build"
    builder = types.SimpleNamespace(build_lib=str(build_lib))

    module.CustomBuildPy.generate_default_envs(builder)

    generated = (build_lib / "deep_ep" / "envs.py").read_text(encoding="utf-8")
    assert "built_nccl_version = 23004" in generated

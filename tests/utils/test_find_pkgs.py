import os
import sys
import tempfile
from pathlib import Path
from unittest import mock
import pytest

# Mock torch and deep_ep._C before importing deep_ep so tests can run in any environment
sys.modules.setdefault("torch", mock.MagicMock())
sys.modules.setdefault("torch.distributed", mock.MagicMock())
sys.modules.setdefault("deep_ep._C", mock.MagicMock())

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deep_ep.utils.find_pkgs import find_pkg_root, get_nccl_lib_name
from deep_ep.utils.envs import EnvOverride


def test_get_nccl_lib_name_standard_lib():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        lib_dir = root / "lib"
        lib_dir.mkdir()
        (lib_dir / "libnccl.so.2.20.5").touch()

        name = get_nccl_lib_name(str(root))
        assert name == "libnccl.so.2.20.5"


def test_get_nccl_lib_name_lib64():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        lib64_dir = root / "lib64"
        lib64_dir.mkdir()
        (lib64_dir / "libnccl.so").touch()

        name = get_nccl_lib_name(str(root))
        assert name == "libnccl.so"


def test_get_nccl_lib_name_direct_dir():
    # When root is already the library directory itself (e.g. /usr/lib64 or /usr/lib/x86_64-linux-gnu)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "libnccl.so.2").touch()

        name = get_nccl_lib_name(str(root))
        assert name == "libnccl.so.2"


def test_get_nccl_lib_name_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ModuleNotFoundError, match="libnccl.so not found under"):
            get_nccl_lib_name(tmpdir)


def test_find_pkg_root_env_override():
    with tempfile.TemporaryDirectory() as tmpdir:
        with EnvOverride([("EP_NCCL_ROOT_DIR", tmpdir, True)]):
            res = find_pkg_root("nccl")
            assert res == tmpdir


def test_find_pkg_root_empty_env_ignored():
    # Empty environment variable should not be returned as a valid directory
    with EnvOverride([("EP_NCCL_ROOT_DIR", "", True), ("NCCL_DIR", "", True)]):
        res = find_pkg_root("nccl", optional=True)
        assert res != ""


def test_find_pkg_root_optional_missing():
    with EnvOverride([("EP_NONEXISTENT_PKG_ROOT_DIR", "", True), ("NONEXISTENT_PKG_DIR", "", True)]):
        res = find_pkg_root("nonexistent_pkg", optional=True)
        assert res is None


def test_env_override_exit_safety():
    # Verify EnvOverride handles deleted keys when originally not present
    if "TEST_DEEP_EP_VAR_NEW" in os.environ:
        del os.environ["TEST_DEEP_EP_VAR_NEW"]
    with EnvOverride([("TEST_DEEP_EP_VAR_NEW", "modified", True)]):
        assert os.environ["TEST_DEEP_EP_VAR_NEW"] == "modified"
        del os.environ["TEST_DEEP_EP_VAR_NEW"]  # manually deleted inside context
    # On exit, os.environ.pop() is called without raising KeyError
    assert "TEST_DEEP_EP_VAR_NEW" not in os.environ

import builtins
import io
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


INIT_PATH = Path(__file__).parents[1] / "deep_ep" / "__init__.py"


def module(name, **attributes):
    value = types.ModuleType(name)
    for key, item in attributes.items():
        setattr(value, key, item)
    return value


def load_deep_ep(
    monkeypatch,
    *,
    runtime_version=22809,
    built_version=23004,
    suppress=False,
    loaded_nccl_path="/runtime/libnccl.so.2",
):
    package_name = "deep_ep_guard_test"
    calls = {"runtime_queries": [], "init_jit": 0}

    package = module(package_name)
    package.__path__ = [str(INIT_PATH.parent)]
    utils_package = module(f"{package_name}.utils")
    utils_package.__path__ = [str(INIT_PATH.parent / "utils")]
    buffers_package = module(f"{package_name}.buffers")
    buffers_package.__path__ = [str(INIT_PATH.parent / "buffers")]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, f"{package_name}.utils", utils_package)
    monkeypatch.setitem(sys.modules, f"{package_name}.buffers", buffers_package)

    monkeypatch.setitem(sys.modules, "torch", module("torch"))
    monkeypatch.setitem(
        sys.modules,
        f"{package_name}.utils.find_pkgs",
        module(f"{package_name}.utils.find_pkgs", find_nccl_root=lambda: "/expected"),
    )

    def get_runtime_version(path):
        calls["runtime_queries"].append(path)
        return runtime_version

    monkeypatch.setitem(
        sys.modules,
        f"{package_name}.utils.nccl",
        module(
            f"{package_name}.utils.nccl",
            get_nccl_runtime_version=get_runtime_version,
            format_nccl_version=lambda version: f"{version // 10000}.{(version // 100) % 100}.{version % 100}",
        ),
    )

    generated_envs = module(
        f"{package_name}.envs",
        persistent_envs={"EP_TEST_PERSISTENT": "from-wheel"},
    )
    if built_version is not None:
        generated_envs.built_nccl_version = built_version
    monkeypatch.setitem(sys.modules, f"{package_name}.envs", generated_envs)

    def init_jit(*args):
        calls["init_jit"] += 1

    extension = module(
        f"{package_name}._C",
        init_jit=init_jit,
        Config=object,
        topk_idx_t=object,
    )
    monkeypatch.setitem(sys.modules, f"{package_name}._C", extension)
    canonical_package = module("deep_ep")
    canonical_package.__path__ = [str(INIT_PATH.parent)]
    monkeypatch.setitem(sys.modules, "deep_ep", canonical_package)
    monkeypatch.setitem(sys.modules, "deep_ep._C", extension)
    monkeypatch.setitem(
        sys.modules,
        f"{package_name}.buffers.legacy",
        module(f"{package_name}.buffers.legacy", Buffer=object),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{package_name}.buffers.elastic",
        module(f"{package_name}.buffers.elastic", ElasticBuffer=object, EPHandle=object),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{package_name}.utils.event",
        module(f"{package_name}.utils.event", EventOverlap=object, EventHandle=object),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{package_name}.utils.envs",
        module(
            f"{package_name}.utils.envs",
            get_physical_domain_size=lambda: 1,
            get_logical_domain_size=lambda: 1,
        ),
    )

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path) == "/proc/self/maps":
            line = f"7f000 r-xp 0000 00:00 0 {loaded_nccl_path}\n" if loaded_nccl_path else ""
            return io.StringIO(line)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr("glob.glob", lambda pattern: ["/expected/lib/libnccl.so.2"])
    monkeypatch.setattr("filecmp.cmp", lambda left, right, shallow=False: True)
    if suppress:
        monkeypatch.setenv("EP_SUPPRESS_NCCL_CHECK", "1")
    else:
        monkeypatch.delenv("EP_SUPPRESS_NCCL_CHECK", raising=False)
    monkeypatch.setenv("CUDA_HOME", "/fake/cuda")

    spec = importlib.util.spec_from_file_location(
        package_name,
        INIT_PATH,
        submodule_search_locations=[str(INIT_PATH.parent)],
    )
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, loaded)
    spec.loader.exec_module(loaded)
    calls["persistent_env"] = os.environ.get("EP_TEST_PERSISTENT")
    return calls


def test_rejects_runtime_older_than_build_before_extension_init(monkeypatch):
    with pytest.raises(AssertionError, match=r"2\.28\.9.*2\.30\.4.*upgrade NCCL"):
        load_deep_ep(monkeypatch, runtime_version=22809, built_version=23004)


@pytest.mark.parametrize("runtime_version", [23004, 23100])
def test_accepts_runtime_equal_to_or_newer_than_build(monkeypatch, runtime_version):
    calls = load_deep_ep(
        monkeypatch,
        runtime_version=runtime_version,
        built_version=23004,
    )

    assert calls["runtime_queries"] == ["/runtime/libnccl.so.2"]
    assert calls["init_jit"] == 1


def test_missing_build_metadata_keeps_existing_behavior(monkeypatch):
    calls = load_deep_ep(monkeypatch, runtime_version=22809, built_version=None)

    assert calls["runtime_queries"] == []
    assert calls["init_jit"] == 1
    assert calls["persistent_env"] == "from-wheel"


def test_suppression_skips_runtime_version_check(monkeypatch):
    calls = load_deep_ep(
        monkeypatch,
        runtime_version=22809,
        built_version=23004,
        suppress=True,
    )

    assert calls["runtime_queries"] == []
    assert calls["init_jit"] == 1


def test_reports_when_no_nccl_runtime_is_loaded(monkeypatch):
    with pytest.raises(AssertionError, match="No NCCL runtime is loaded"):
        load_deep_ep(monkeypatch, loaded_nccl_path=None)

# SPDX-License-Identifier: MIT

import ast
import importlib
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


def build_namespace(cuda_home):
    path = Path(__file__).resolve().parents[1] / "setup.py"
    names = {
        "get_nvtx_library",
        "get_extension_hybrid_ep_cpp",
        "get_extension_deep_ep_cpp",
        "to_nvcc_gencode",
    }
    tree = ast.parse(path.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "__file__": str(path),
        "CUDA_HOME": cuda_home,
        "Path": Path,
        "os": os,
        "re": re,
        "importlib": importlib,
        "shutil": SimpleNamespace(
            copytree=lambda *a, **kw: None, copy=lambda *a, **kw: None
        ),
        "CUDAExtension": lambda *args, **kwargs: kwargs,
    }
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("directory", ["lib64", "lib"])
@pytest.mark.parametrize("suffix", [".so", ".a"])
@pytest.mark.parametrize("interop", [False, True])
def test_hybrid_ep_links_available_nvtx_library(
    tmp_path, monkeypatch, directory, suffix, interop
):
    lib = tmp_path / directory
    lib.mkdir()
    (lib / ("libnvToolsExt" + suffix)).touch()
    if interop:
        (lib / ("libnvtx3interop" + suffix)).touch()
    monkeypatch.setenv("HYBRID_EP_MULTINODE", "0")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0")
    namespace = build_namespace(str(tmp_path))
    extension = namespace["get_extension_hybrid_ep_cpp"]()
    assert extension["libraries"] == [
        "cuda",
        "nvtx3interop" if interop else "nvToolsExt",
    ]


@pytest.mark.parametrize("cuda_home", [None, "/nonexistent-cuda-toolkit"])
def test_missing_toolkit_preserves_existing_nvtx_linker_requirement(cuda_home):
    assert build_namespace(cuda_home)["get_nvtx_library"]() == "nvtx3interop"


@pytest.mark.parametrize("cccl_available", [False, True])
def test_deep_ep_discovers_cccl_from_selected_toolkit(
    tmp_path, monkeypatch, cccl_available
):
    cccl = tmp_path / "include" / "cccl"
    if cccl_available:
        cccl.mkdir(parents=True)
    nvshmem = tmp_path / "nvshmem"
    nvshmem.mkdir()
    monkeypatch.setenv("NVSHMEM_DIR", str(nvshmem))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0")
    monkeypatch.setenv("DISABLE_SM90_FEATURES", "0")
    monkeypatch.setenv("DISABLE_AGGRESSIVE_PTX_INSTRS", "1")
    namespace = build_namespace(str(tmp_path))
    extension = namespace["get_extension_deep_ep_cpp"]()
    assert (str(cccl) in extension["include_dirs"]) is cccl_available
    assert extension["include_dirs"][0] == "csrc/"
    assert str(nvshmem / "include") in extension["include_dirs"]
    assert "-lnvshmem_device" in extension["extra_compile_args"]["nvcc_dlink"]

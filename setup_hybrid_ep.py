# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
import subprocess
import setuptools
import importlib
import shutil
import re

from pathlib import Path
from paddle.utils.cpp_extension import BuildExtension, CUDAExtension, _get_cuda_arch_flags
from paddle.utils.cpp_extension.extension_utils import (
    add_compile_flag,
)

def collect_package_files(package: str, relative_dir: str):
    base_path = Path(package) / relative_dir
    if not base_path.exists():
        return []
    return [
        str(path.relative_to(package))
        for path in base_path.rglob('*')
        if path.is_file()
    ]


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')

def to_nvcc_gencode(s: str) -> str:
    flags = []
    for part in re.split(r'[,\s;]+', s.strip()):
        if not part:
            continue
        m = re.fullmatch(r'(\d+)\.(\d+)([A-Za-z]?)', part)
        if not m:
            raise ValueError(f"Invalid entry: {part}")
        major, minor, suf = m.groups()
        arch = f"{int(major)}{int(minor)}{suf.lower()}"
        flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    return " ".join(flags)


def get_extension_hybrid_ep_cpp():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    enable_multinode = os.getenv("HYBRID_EP_MULTINODE", "0").strip().lower() in {"1", "true", "t", "yes", "y", "on"}

    # Default to Blackwell series
    os.environ['PADDLE_CUDA_ARCH_LIST'] = os.getenv('PADDLE_CUDA_ARCH_LIST', '10.0')

    # Basic compile arguments
    compile_args = {
        "nvcc": [
            "-std=c++17",
            "-Xcompiler",
            "-fPIC",
            "--expt-relaxed-constexpr",
            "-O3",
            "--shared",
        ],
    }

    sources = [
        "csrc/hybrid_ep/hybrid_ep.cu",
        "csrc/hybrid_ep/allocator/allocator.cu",
        "csrc/hybrid_ep/jit/compiler.cu",
        "csrc/hybrid_ep/executor/executor.cu",
        "csrc/hybrid_ep/extension/permute.cu",
        "csrc/hybrid_ep/extension/allgather.cu",
        "csrc/hybrid_ep/pybind_hybrid_ep.cu",
    ]
    include_dirs = [
        os.path.join(current_dir, "csrc/hybrid_ep/"),
        os.path.join(current_dir, "csrc/hybrid_ep/backend/"),
    ]
    # include_dirs.append("/root/paddlejob/share-storage/gpfs/system-public/lzy/eb5_tool/miniconda3/envs/lzy/lib/python3.10/site-packages/paddle/include/")
    # include_dirs.append("/root/paddlejob/share-storage/gpfs/system-public/lzy/eb5_tool/miniconda3/envs/lzy/lib/python3.10/site-packages/paddle/include/paddle/phi/include")
    library_dirs = []
    libraries = ["cuda", "nvtx3interop"]
    extra_objects = []
    runtime_library_dirs = []
    nvcc_dlink = ['-dlink']
    extra_link_args = ["-lcuda"]

    if len(nvcc_dlink) > 0:
        nvcc_dlink = nvcc_dlink + _get_cuda_arch_flags()
        compile_args['nvcc_dlink'] = nvcc_dlink


    # Add dependency for jit
    compile_args["nvcc"].extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])
    compile_args["nvcc"].append(f'-DSM_ARCH="{os.environ["PADDLE_CUDA_ARCH_LIST"]}"')
    # Copy the hybrid backend code to python package for JIT compilation
    shutil.copytree(
        os.path.join(current_dir, "csrc/hybrid_ep/backend/"),
        os.path.join(current_dir, "deep_ep/backend/"),
        dirs_exist_ok=True
    )
    # Add inter-node dependency 
    if enable_multinode:
        sources.extend(["csrc/hybrid_ep/internode.cu"])
        rdma_core_dir = os.getenv("RDMA_CORE_HOME", "")
        nccl_dir = os.path.join(current_dir, "third-party/nccl")        
        compile_args["nvcc"].append("-DHYBRID_EP_BUILD_MULTINODE_ENABLE")
        compile_args["nvcc"].append(f"-DRDMA_CORE_HOME=\"{rdma_core_dir}\"")
        extra_link_args.append(f"-l:libnvidia-ml.so.1")

        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=current_dir)
        # Generate the inter-node dependency to the python package for JIT compilation
        subprocess.run(["make", "-j", "src.build", f"NVCC_GENCODE={to_nvcc_gencode(os.environ['PADDLE_CUDA_ARCH_LIST'])}"], cwd=nccl_dir, check=True)
        # Add third-party dependency 
        include_dirs.append(os.path.join(nccl_dir, "src/transport/net_ib/gdaki/doca-gpunetio/include"))
        include_dirs.append(os.path.join(rdma_core_dir, "include"))
        library_dirs.append(os.path.join(rdma_core_dir, "lib"))
        runtime_library_dirs.append(os.path.join(rdma_core_dir, "lib"))
        libraries.append("mlx5")
        libraries.append("ibverbs")
        # Copy the inter-node dependency to python package
        shutil.copytree(
            os.path.join(nccl_dir, "src/transport/net_ib/gdaki/doca-gpunetio/include"),
            os.path.join(current_dir, "deep_ep/backend/nccl/include"),
            dirs_exist_ok=True
        )
        shutil.copytree(
            os.path.join(nccl_dir, "build/obj/transport/net_ib/gdaki/doca-gpunetio"),
            os.path.join(current_dir, "deep_ep/backend/nccl/obj"),
            dirs_exist_ok=True
        )
        # Set the extra objects
        DOCA_OBJ_PATH = os.path.join(current_dir, "deep_ep/backend/nccl/obj")
        extra_objects = [
            os.path.join(DOCA_OBJ_PATH, "doca_gpunetio.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_gpunetio_high_level.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_cuda_wrapper.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_device_attr.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_ibv_wrapper.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_mlx5dv_wrapper.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_qp.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_cq.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_srq.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_uar.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_verbs_umem.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_gpunetio_gdrcopy.o"),
            os.path.join(DOCA_OBJ_PATH, "doca_gpunetio_log.o"),
        ]


    print(f'Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {libraries}')
    print(f' > Library dirs: {library_dirs}')
    print(f' > Extra link args: {extra_link_args}')
    print(f' > Compilation flags: {compile_args}')
    print(f' > Extra objects: {extra_objects}')
    print(f' > Runtime library dirs: {runtime_library_dirs}')
    print(f' > Arch list: {os.environ["PADDLE_CUDA_ARCH_LIST"]}')
    print()

    add_compile_flag(compile_args, ['-DPADDLE_WITH_CUDA'])
    add_compile_flag(compile_args, ['-DWITH_DISTRIBUTE'])
    add_compile_flag(compile_args, ['-DWITH_NVSHMEM'])
    add_compile_flag(compile_args, ['-DWITH_GPU'])
    add_compile_flag(compile_args, ['-DWITH_FLUID_ONLY'])

    extension_hybrid_ep_cpp = CUDAExtension(
        name="hybrid_ep_cpp",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_objects=extra_objects,
        runtime_library_dirs=runtime_library_dirs,
        extra_link_args=extra_link_args,
    )

    return extension_hybrid_ep_cpp

if __name__ == '__main__':
    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(
        name='deep_ep',
        version='1.2.1' + revision,
        packages=setuptools.find_packages(
            include=['deep_ep']
        ),
        install_requires=[
            'pynvml',
        ],
        ext_modules=[
            get_extension_hybrid_ep_cpp(),
        ],
        cmdclass={
            'build_ext': BuildExtension
        },
        package_data={
            'deep_ep': collect_package_files('deep_ep', 'backend'),
        },
        include_package_data=True
    )

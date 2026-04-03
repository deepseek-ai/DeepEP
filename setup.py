# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
import subprocess
import setuptools
import importlib
import shutil
import re

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


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
    use_nixl = os.getenv("USE_NIXL", "0").strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    # Fallback when env may not propagate (e.g. pip build isolation): .use_nixl in project root
    if enable_multinode and not use_nixl and os.path.isfile(os.path.join(current_dir, ".use_nixl")):
        use_nixl = True
        print("Using NIXL (found .use_nixl; DOCA/NCCL build unavailable)")

    # Default to Blackwell series
    os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '10.0')

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
        "csrc/hybrid_ep/buffer/intranode.cu",
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
    library_dirs = []
    libraries = ["cuda", "nvtx3interop"]
    extra_objects = []
    runtime_library_dirs = []
    extra_link_args = []

    # Add dependency for jit
    compile_args["nvcc"].append(f'-DSM_ARCH="{os.environ["TORCH_CUDA_ARCH_LIST"]}"')
    # Copy the hybrid backend code to python package for JIT compilation
    shutil.copytree(
        os.path.join(current_dir, "csrc/hybrid_ep/backend/"),
        os.path.join(current_dir, "deep_ep/backend/"),
        dirs_exist_ok=True
    )
    # Copy the utils.cuh
    shutil.copy(
        os.path.join(current_dir, "csrc/hybrid_ep/utils.cuh"),
        os.path.join(current_dir, "deep_ep/backend/utils.cuh")
    )
    # Add inter-node dependency 
    if enable_multinode:
        compile_args["nvcc"].append("-DHYBRID_EP_BUILD_MULTINODE_ENABLE")
        print(f'Multinode enabled: use_nixl={use_nixl} (USE_NIXL={os.getenv("USE_NIXL", "0")})')
        if use_nixl:
            # NIXL path: use NIXL connector instead of DOCA
            print('  -> NIXL path: skipping NCCL/DOCA build')
            compile_args["nvcc"].append("-DUSE_NIXL")
            sources.extend([
                "csrc/hybrid_ep/buffer/internode_nixl.cu",
                "csrc/hybrid_ep/buffer/nixl_connector.cu",
            ])
            nixl_home = os.getenv("NIXL_HOME", "/usr/local/nixl")
            ucx_home = os.getenv("UCX_HOME", "/usr")
            nixl_include = os.path.join(nixl_home, "include")
            nixl_gpu_include = os.path.join(nixl_home, "include/gpu/ucx")
            import platform
            machine = platform.machine()
            if machine == "aarch64":
                nixl_lib_suffix = "lib/aarch64-linux-gnu"
            else:
                nixl_lib_suffix = "lib/x86_64-linux-gnu"
            nixl_lib = os.path.join(nixl_home, nixl_lib_suffix)
            include_dirs.extend([nixl_include, nixl_gpu_include, os.path.join(ucx_home, "include")])
            library_dirs.append(nixl_lib)
            runtime_library_dirs.append(nixl_lib)
            libraries.extend(["nixl", "nixl_build", "nixl_common"])
            extra_link_args.extend([f"-Wl,-rpath,{nixl_lib}"])
            extra_link_args.append("-l:libnvidia-ml.so.1")
            libraries.extend(["mlx5", "ibverbs"])
            doca_home = os.getenv("DOCA_HOME", "")
            if doca_home:
                include_dirs.append(os.path.join(doca_home, "include"))
            rdma_core_dir = os.getenv("RDMA_CORE_HOME", "")
            if rdma_core_dir:
                include_dirs.append(os.path.join(rdma_core_dir, "include"))
                library_dirs.append(os.path.join(rdma_core_dir, "lib"))
        else:
            # DOCA path: use RDMA coordinator (requires NCCL submodule + DOCA)
            print('  -> DOCA path: building NCCL/DOCA')
            sources.extend(["csrc/hybrid_ep/buffer/internode_doca.cu"])
            rdma_core_dir = os.getenv("RDMA_CORE_HOME", "")
            nccl_dir = os.path.join(current_dir, "third-party/nccl")
            compile_args["nvcc"].append(f"-DRDMA_CORE_HOME=\"{rdma_core_dir}\"")
            extra_link_args.append(f"-l:libnvidia-ml.so.1")

            subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=current_dir)
            nccl_make = subprocess.run(
                ["make", "-j", "src.build", f"NVCC_GENCODE={to_nvcc_gencode(os.environ['TORCH_CUDA_ARCH_LIST'])}"],
                cwd=nccl_dir,
            )
            if nccl_make.returncode != 0:
                raise SystemExit(
                    "NCCL/DOCA build failed (missing doca_gpunetio_device.h or DOCA SDK).\n"
                    "Use NIXL instead: export USE_NIXL=1 HYBRID_EP_MULTINODE=1 && pip install .\n"
                    "Or create .use_nixl in the project root and rebuild."
                )
            include_dirs.append(os.path.join(nccl_dir, "src/transport/net_ib/gdaki/doca-gpunetio/include"))
            include_dirs.append(os.path.join(rdma_core_dir, "include"))
            library_dirs.append(os.path.join(rdma_core_dir, "lib"))
            runtime_library_dirs.append(os.path.join(rdma_core_dir, "lib"))
            libraries.append("mlx5")
            libraries.append("ibverbs")
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
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print()

    extension_hybrid_ep_cpp = CUDAExtension(
        "hybrid_ep_cpp",
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

def get_extension_deep_ep_cpp():
    disable_nvshmem = False
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    nvshmem_host_lib = 'libnvshmem_host.so'
    if nvshmem_dir is None:
        try:
            nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
            nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
            import nvidia.nvshmem as nvshmem
        except (ModuleNotFoundError, AttributeError, IndexError):
            print('Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. All internode and low-latency features are disabled\n')
            disable_nvshmem = True
    else:
        disable_nvshmem = False

    if not disable_nvshmem:
        assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'

    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable',
                 '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    sources = ['csrc/deep_ep.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu']
    include_dirs = ['csrc/']
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = ['-lcuda']

    # NVSHMEM flags
    if disable_nvshmem:
        cxx_flags.append('-DDISABLE_NVSHMEM')
        nvcc_flags.append('-DDISABLE_NVSHMEM')
    else:
        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu', 'csrc/kernels/pcie.cu'])
        include_dirs.extend([f'{nvshmem_dir}/include'])
        library_dirs.extend([f'{nvshmem_dir}/lib'])
        nvcc_dlink.extend(['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem_device'])
        extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_dir}/lib'])

    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')

        # Disable internode and low-latency kernels
        assert disable_nvshmem
    else:
        # Prefer H800 series
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')

        # CUDA 12 flags
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])
        
        # Ensure device linking and CUDA device runtime when RDC is enabled
        if '-rdc=true' in nvcc_flags and '-dlink' not in nvcc_dlink:
            nvcc_dlink.append('-dlink')

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Summary
    print(f'Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f' > NVSHMEM path: {nvshmem_dir}')
    print()

    extension_deep_ep_cpp = CUDAExtension(
        name='deep_ep_cpp',
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        sources=sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )

    return extension_deep_ep_cpp

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
            get_extension_deep_ep_cpp(),
            get_extension_hybrid_ep_cpp()
        ],
        cmdclass={
            'build_ext': BuildExtension
        },
        package_data={
            'deep_ep': collect_package_files('deep_ep', 'backend'),
        },
        include_package_data=True
    )

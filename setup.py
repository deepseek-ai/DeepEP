# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
import subprocess
import setuptools
import importlib
import shutil
import re

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


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
        "csrc/hybrid_ep/allocator/allocator.cu",
        "csrc/hybrid_ep/jit/compiler.cu",
        "csrc/hybrid_ep/executor/executor.cu",
        "csrc/hybrid_ep/extension/permute.cu",
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

    # Add dependency for jit
    compile_args["nvcc"].append(f'-DSM_ARCH="{os.environ["TORCH_CUDA_ARCH_LIST"]}"')
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

        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=current_dir)
        # Generate the inter-node dependency to the python package for JIT compilation
        subprocess.run(["make", "-j", "src.build", f"NVCC_GENCODE={to_nvcc_gencode(os.environ['TORCH_CUDA_ARCH_LIST'])}"], cwd=nccl_dir, check=True)
        # Add third-party dependency 
        include_dirs.append(os.path.join(nccl_dir, "src/transport/gdaki/doca-gpunetio/include"))
        include_dirs.append(os.path.join(rdma_core_dir, "include"))
        library_dirs.append(os.path.join(rdma_core_dir, "lib"))
        runtime_library_dirs.append(os.path.join(rdma_core_dir, "lib"))
        libraries.append("mlx5")
        libraries.append("ibverbs")
        # Copy the inter-node dependency to python package
        shutil.copytree(
            os.path.join(nccl_dir, "src/transport/gdaki/doca-gpunetio/include"),
            os.path.join(current_dir, "deep_ep/backend/nccl/include"),
            dirs_exist_ok=True
        )
        shutil.copytree(
            os.path.join(nccl_dir, "build/obj/transport/gdaki/doca-gpunetio"),
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
    extra_link_args = []

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
            'deep_ep': ['backend/*'],
        },
        include_package_data=True
    )

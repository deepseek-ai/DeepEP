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
        os.environ['PADDLE_CUDA_ARCH_LIST'] = os.getenv('PADDLE_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')

        # Disable internode and low-latency kernels
        assert disable_nvshmem
    else:
        # Prefer H800 series
        os.environ['PADDLE_CUDA_ARCH_LIST'] = os.getenv('PADDLE_CUDA_ARCH_LIST', '9.0')

        # CUDA 12 flags
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])
        
        # Ensure device linking and CUDA device runtime when RDC is enabled
        if '-rdc=true' in nvcc_flags and '-dlink' not in nvcc_dlink:
            nvcc_dlink.append('-dlink')

    # CUDA 12 flags
    nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['PADDLE_CUDA_ARCH_LIST'].strip() != '9.0':
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
        nvcc_dlink = nvcc_dlink + _get_cuda_arch_flags()
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Summary
    print(f'Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["PADDLE_CUDA_ARCH_LIST"]}')
    print(f' > NVSHMEM path: {nvshmem_dir}')
    print()

    add_compile_flag(extra_compile_args, ['-DPADDLE_WITH_CUDA'])
    add_compile_flag(extra_compile_args, ['-DWITH_DISTRIBUTE'])
    add_compile_flag(extra_compile_args, ['-DWITH_NVSHMEM'])
    add_compile_flag(extra_compile_args, ['-DWITH_GPU'])
    add_compile_flag(extra_compile_args, ['-DWITH_FLUID_ONLY'])

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
        ],
        cmdclass={
            'build_ext': BuildExtension
        },
        package_data={
            'deep_ep': collect_package_files('deep_ep', 'backend'),
        },
        include_package_data=True
    )

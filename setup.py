import os
import subprocess
import setuptools
import importlib
import re
from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_nvcc_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)
        output = output.decode("utf-8")
        match = re.search(r"release\s+(\d+\.\d+)", output)
        if match:
            return match.group(1)
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


if __name__ == '__main__':
    libraries = []

    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable',
                 '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    sources = ['csrc/deep_ep.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu']
    include_dirs = ['csrc/']
    library_dirs = ['/usr/local/lib64']
    nvcc_dlink = []
    extra_link_args = []

    # This workaround is needed because NVCC 12.9 has a bug (https://nvbugspro.nvidia.com/bug/5595631 for details) which fails UCX compilation
    nvcc_version = get_nvcc_version()
    assert nvcc_version is not None, 'Could not get NVCC version'
    if nvcc_version == '12.9':
        cxx_flags.append('-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE')
        nvcc_flags.append('-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE')

    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')

        # Disable internode and low-latency kernels
        assert True
    else:
        # Prefer H800 series
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')

        # CUDA 12 flags
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    nvcc_dlink_libs = []
    extra_link_libs = []

    if os.getenv('ENABLE_DEBUG_LOGS', '0') == '1':
        print('Debug logs enabled')
        cxx_flags.append('-DENABLE_DEBUG_LOGS')
        nvcc_flags.append('-DENABLE_DEBUG_LOGS')

    # Get the library path for NIXL
    nixl_lib_path = os.getenv('NIXL_LIB_PATH', None)
    assert nixl_lib_path and os.path.exists(nixl_lib_path), 'NIXL_LIB_PATH must be set and valid'
    print(f'NIXL library path: {nixl_lib_path}')

    # Get colon-separated include paths
    nixl_include_paths_str = os.getenv('NIXL_INCLUDE_PATHS', '')
    nixl_include_paths = [p for p in nixl_include_paths_str.split(':') if p]
    print(f'NIXL include paths: {nixl_include_paths}')
    include_dirs.extend(nixl_include_paths)

    library_dirs.append(nixl_lib_path)
    library_dirs.append(os.path.join(nixl_lib_path, 'core'))
    library_dirs.append(os.path.join(nixl_lib_path, 'plugins'))

    nvcc_dlink_libs.extend(['-lnixl'])
    extra_link_libs.extend(['-lnixl'])
    libraries.append('nixl')

    sources.extend(['csrc/kernels/internode_ll.cu', 'csrc/kernels/internode.cu'])

    # Create linker flags from all library directories
    link_flags = [f'-L{lib_dir}' for lib_dir in library_dirs]
    rpath_flags = [f'-Wl,-rpath,{lib_dir}' for lib_dir in library_dirs]

    nvcc_dlink = ['-dlink'] + link_flags + nvcc_dlink_libs
    extra_link_args = extra_link_libs + rpath_flags

    # Debug build flags (host and device)
    debug_env = os.getenv('DEEPEP_DEBUG', os.getenv('DEBUG', '0'))
    if str(debug_env).lower() in ('1', 'true', 'yes', 'on'):
        print('Debug build enabled: adding host/device debug flags and disabling optimizations')
        # Remove optimization flags if present
        cxx_flags = [flag for flag in cxx_flags if flag != '-O3']
        nvcc_flags = [flag for flag in nvcc_flags if flag != '-O3']
        # Enable host debug info and keep frame pointers
        cxx_flags.extend(['-g', '-O0', '-fno-omit-frame-pointer'])
        # Enable device debug info; also pass host flags through NVCC
        nvcc_flags.extend(['-G', '-g', '-lineinfo', '-Xcompiler', '-g', '-Xcompiler', '-fno-omit-frame-pointer'])

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
    print()

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(
        name='deep_ep',
        version='1.1.0' + revision,
        packages=setuptools.find_packages(
            include=['deep_ep']
        ),
        ext_modules=[
            CUDAExtension(
                name='deep_ep_cpp',
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                libraries=libraries,
                sources=sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )

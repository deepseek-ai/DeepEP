import ast
import re
import os
import subprocess
import setuptools
import importlib

from pathlib import Path
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_env_names = ('EP_JIT_CACHE_DIR', 'EP_JIT_PRINT_COMPILER_COMMAND', 'EP_NUM_TOPK_IDX_BITS', 'EP_NCCL_ROOT_DIR')

# Load discover module without triggering `deep_ep.__init__`
find_pkgs_spec = importlib.util.spec_from_file_location('find_pkgs', os.path.join(current_dir, 'deep_ep', 'utils', 'find_pkgs.py'))
find_pkgs = importlib.util.module_from_spec(find_pkgs_spec)
find_pkgs_spec.loader.exec_module(find_pkgs)


# Wheel specific: NVIDIA pip wheels (nvidia-nvshmem-cu12, nvidia-nccl-cu12)
# only ship the SO name of the host library, e.g. `libnvshmem_host.so.3`,
# without the unversioned `libnvshmem_host.so` symlink. So `-l:libnvshmem_host.so`
# (exact-name link) cannot resolve. Resolve the real file name at build time
# and pass it through to the linker instead.
def _find_versioned_so(base_dir, prefix):
    """Return the real filename of the first ``{prefix}.so*`` under ``base_dir/lib``.

    Prefers an unversioned ``{prefix}.so`` symlink when present so we keep
    behaving identically to the Tarball install. Falls back to the SONAME
    file (``{prefix}.so.X``) shipped by pip wheels.
    """
    lib_dir = Path(base_dir).joinpath('lib')
    unversioned = lib_dir / f'{prefix}.so'
    if unversioned.exists():
        return unversioned.name
    for file in sorted(lib_dir.rglob(f'{prefix}.so.*')):
        return file.name
    raise ModuleNotFoundError(f'{prefix}.so not found under {lib_dir}')


def get_nvshmem_host_lib_name(base_dir):
    return _find_versioned_so(base_dir, 'libnvshmem_host')


def get_nccl_lib_name(base_dir):
    return _find_versioned_so(base_dir, 'libnccl')


def get_package_version():
    with open(Path(current_dir) / 'deep_ep' / '__init__.py', 'r') as f:
        version_match = re.search(r'^__version__\s*=\s*(.*)$', f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))

    # noinspection PyBroadException
    try:
        status_cmd = ['git', 'status', '--porcelain']
        status_output = subprocess.check_output(status_cmd).decode('ascii').strip()
        if status_output:
            print(f'Warning: Git working directory is not clean. Uncommitted changes:\n{status_output}')
            assert False, 'Git working directory is not clean'

        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except:
        revision = '+local'
    return f'{public_version}{revision}'


class CustomBuildPy(build_py):
    def run(self):
        # Make clusters' cache setting default into `envs.py`
        self.generate_default_envs()

        # Finally, run the regular build
        build_py.run(self)

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        # noinspection PyShadowingNames
        for name in persistent_env_names:
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''

        # Create temporary build directory
        build_include_dir = os.path.join(self.build_lib, 'deep_ep')
        os.makedirs(build_include_dir, exist_ok=True)
        with open(os.path.join(self.build_lib, 'deep_ep', 'envs.py'), 'w') as f:
            f.write(code)


if __name__ == '__main__':
    # TODO: make NVSHMEM and legacy optional
    nvshmem_root_dir = find_pkgs.find_nvshmem_root()
    nccl_root_dir = find_pkgs.find_nccl_root()

    # `128,2417` is used to suppress warnings of `fmt`
    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3', '--extended-lambda', '--diag-suppress=128,2417']
    sources = ['csrc/python_api.cpp', 'csrc/kernels/legacy/layout.cu', 'csrc/kernels/legacy/intranode.cu']
    include_dirs = [f'{current_dir}/deep_ep/include',
                    f'{current_dir}/third-party/fmt/include',
                    '/usr/local/cuda/include/cccl']
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = ['-lcuda']

    # NVSHMEM flags. Use the real on-disk file name (which may be SONAME-only
    # like ``libnvshmem_host.so.3`` when NVSHMEM came from a pip wheel) so
    # that ``-l:NAME`` can resolve. The static device library always ships
    # under its canonical name, so it stays hard-coded.
    sources.extend(['csrc/kernels/legacy/internode.cu', 'csrc/kernels/legacy/internode_ll.cu', 'csrc/kernels/backend/nvshmem.cu'])
    include_dirs.extend([f'{nvshmem_root_dir}/include'])
    library_dirs.extend([f'{nvshmem_root_dir}/lib'])
    nvcc_dlink.extend(['-dlink', f'-L{nvshmem_root_dir}/lib', '-lnvshmem_device'])
    nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_root_dir)
    extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_root_dir}/lib'])

    # NCCL flags. Same story as NVSHMEM above — pip wheels ship
    # ``libnccl.so.2`` only, so resolve the real name dynamically.
    sources.extend(['csrc/kernels/backend/nccl.cu'])
    include_dirs.extend([f'{nccl_root_dir}/include'])
    nccl_lib = get_nccl_lib_name(nccl_root_dir)
    extra_link_args.extend([f'-l:{nccl_lib}', f'-Wl,-rpath,{nccl_root_dir}/lib'])

    # CUDA driver sources
    sources.extend(['csrc/kernels/backend/cuda_driver.cu'])

    # TODO: remove these
    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')

        # Disable internode and low-latency kernels
        assert False, 'Not implemented'
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

    # Legacy environment name
    if 'TOPK_IDX_BITS' in os.environ:
        assert 'EP_NUM_TOPK_IDX_BITS' not in os.environ
        os.environ['EP_NUM_TOPK_IDX_BITS'] = os.environ['TOPK_IDX_BITS']

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if 'EP_NUM_TOPK_IDX_BITS' in os.environ:
        num_topk_idx_bits = int(os.environ['EP_NUM_TOPK_IDX_BITS'])
        cxx_flags.append(f'-DEP_NUM_TOPK_IDX_BITS={num_topk_idx_bits}')
        nvcc_flags.append(f'-DEP_NUM_TOPK_IDX_BITS={num_topk_idx_bits}')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f' > NVSHMEM path: {nvshmem_root_dir}')
    print(f' > NCCL path: {nccl_root_dir}')
    # Print persistent env variables
    persistent_envs = []
    for name in persistent_env_names:
        if name in os.environ:
            persistent_envs.append((name, os.environ[name]))
    if len(persistent_envs) > 0:
        print(f' > Persistent envs:')
        for k, v in persistent_envs:
            print(f'   > {k}: {v}')
    print()

    setuptools.setup(
        name='deep_ep',
        version=get_package_version(),
        packages=setuptools.find_packages(include=['deep_ep', 'deep_ep.*']),
        package_data={
            'deep_ep': [
                'include/deep_ep/**/*',
            ]
        },
        ext_modules=[
            CUDAExtension(name='deep_ep._C',
                          include_dirs=include_dirs,
                          library_dirs=library_dirs,
                          sources=sources,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args)
        ],
        cmdclass={
            'build_ext': BuildExtension,
            'build_py': CustomBuildPy
        }
    )

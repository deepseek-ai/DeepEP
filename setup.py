import ast
import re
import os
import shutil
import subprocess
import setuptools
import importlib

from pathlib import Path
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_env_names = ('EP_JIT_CACHE_DIR', 'EP_JIT_PRINT_COMPILER_COMMAND', 'EP_JIT_CPP_STANDARD',
                        'EP_NUM_TOPK_IDX_BITS', 'EP_NCCL_ROOT_DIR', 'EP_DEFAULT_RDMA_SL', 'EP_OVERRIDE_RDMA_SL')

# Load discover module without triggering `deep_ep.__init__`
find_pkgs_spec = importlib.util.spec_from_file_location('find_pkgs', os.path.join(current_dir, 'deep_ep', 'utils', 'find_pkgs.py'))
find_pkgs = importlib.util.module_from_spec(find_pkgs_spec)
find_pkgs_spec.loader.exec_module(find_pkgs)


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

        # Then, copy csrc into the wheel for agent-side error lookup
        self.prepare_agent_files()

        # Finally, run the regular build
        build_py.run(self)

    def prepare_agent_files(self):
        # Copy csrc into the wheel for agent-side error lookup
        package_dir = os.path.join(self.build_lib, 'deep_ep')
        for name in ('csrc', ):
            dst = os.path.join(package_dir, name)
            shutil.rmtree(dst, ignore_errors=True)
            ignore = shutil.ignore_patterns('cmake-build-*') if name == 'csrc' else None
            shutil.copytree(os.path.join(current_dir, name), dst, ignore=ignore)

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
    nccl_root_dir = find_pkgs.find_nccl_root()

    cxx_flags = ['-std=c++20', '-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    sources = ['csrc/python_api.cpp',
               'csrc/kernels/comm/context.cpp',
               'csrc/kernels/driver/driver.cpp']
    include_dirs = [f'{current_dir}/deep_ep/include',
                    f'{current_dir}/third-party/deep_jit/include',
                    '/usr/local/cuda/include/cccl']
    library_dirs = []
    extra_link_args = []

    # NCCL flags
    include_dirs.extend([f'{nccl_root_dir}/include'])
    library_dirs.extend([f'{nccl_root_dir}/lib'])
    extra_link_args.extend([f'-l:{find_pkgs.get_nccl_lib_name(nccl_root_dir)}', f'-Wl,-rpath,{nccl_root_dir}/lib'])

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if 'EP_NUM_TOPK_IDX_BITS' in os.environ:
        num_topk_idx_bits = int(os.environ['EP_NUM_TOPK_IDX_BITS'])
        cxx_flags.append(f'-DEP_NUM_TOPK_IDX_BITS={num_topk_idx_bits}')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
    }

    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
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

    # noinspection bad-argument-type
    setuptools.setup(
        name='deep_ep',
        version=get_package_version(),
        packages=setuptools.find_packages(include=['deep_ep', 'deep_ep.*']),
        package_data={'deep_ep': ['include/deep_ep/**/*']},
        ext_modules=[
            CUDAExtension(name='deep_ep._C',
                          include_dirs=include_dirs,
                          library_dirs=library_dirs,
                          sources=sources,
                          extra_compile_args=extra_compile_args,
                          extra_link_args=extra_link_args)
        ],
        cmdclass={'build_ext': BuildExtension, 'build_py': CustomBuildPy}
    )

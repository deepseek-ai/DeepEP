import os
import sys
import subprocess
import setuptools
import importlib

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


if __name__ == '__main__':
    disable_nvshmem_and_nccl = False

    # detect NVSHMEM 
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    nvshmem_host_lib = 'libnvshmem_host.so'

    nccl_home = os.getenv('NCCL_HOME', None)

    if nvshmem_dir is None and nccl_home is None:
        try:
            nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
            nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
            import nvidia.nvshmem as nvshmem  # noqa: F401
        except (ModuleNotFoundError, AttributeError, IndexError):
            print(
                'Warning: `NVSHMEM_DIR` and `NCCL_HOME` are not specified, and the NVSHMEM module is not installed. All internode and low-latency features are disabled\n'
            )
            disable_nvshmem_and_nccl = True
    else:
        disable_nvshmem_and_nccl = False

    if not disable_nvshmem_and_nccl:
        if nvshmem_dir is not None:
            assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'
        if nccl_home is not None:
            assert os.path.exists(nccl_home), f'The specified NCCL directory does not exist: {nccl_home}'

    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    nvcc_flags.extend([ # Allow half and bfloat16 operators for NVSHMEM and internode.cu compatibility PyTorch automatically defines these NO_* flags but we need the operators enabled
        '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__'
    ])
    sources = [
        'csrc/deep_ep.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu',
        'csrc/kernels/backend_factory.cpp'
    ]
    include_dirs = ['csrc/']
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = ['-lcuda']

    # NCCL Enable
    if os.getenv('ENABLE_NCCL', '0') == '1' and nccl_home is not None:
    #if nccl_home is not None and nvshmem_dir is None:
        cxx_flags.append('-DENABLE_NCCL')
        nvcc_flags.append('-DENABLE_NCCL')
        print('NCCL communication backend enabled')

        # Add build/include for nccl.h
        include_dirs.append(f'{nccl_home}/build/include')
        #print(f'Added NCCL build include: {nccl_build_include}')
        include_dirs.append(f'{nccl_home}/src/include')
        #print(f'Added NCCL src include: {nccl_src_include}')

        # Add NCCL library path
        library_dirs.append(f'{nccl_home}/build/lib')
        #print(f'Added NCCL library directory: {nccl_lib}')
        extra_link_args.extend([f'-Wl,-rpath,{nccl_home}/build/lib'])
        #print(f'Added NCCL rpath: {nccl_lib}')

        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu', 'csrc/kernels/nccl_gin_backend.cu'])  
        
        # Device link flags for RDC (relocatable device code)
        nvcc_dlink.extend(['-dlink'])
        
        # Add NCCL linking - dynamic only for external applications
        extra_link_args.extend([f'-L{nccl_home}/build/lib', '-lnccl'])
        #print('Added NCCL library linking: -lnccl (dynamic)')

    # NVSHMEM flags
    if disable_nvshmem_and_nccl:
        cxx_flags.append('-DDISABLE_NVSHMEM_AND_NCCL')
        nvcc_flags.append('-DDISABLE_NVSHMEM_AND_NCCL')

    if os.getenv('ENABLE_NCCL', '0') == '0' and nvshmem_dir is not None:
        cxx_flags.append('-DENABLE_NVSHMEM')
        nvcc_flags.append('-DENABLE_NVSHMEM')
        print('NVSHMEM communication backend enabled')
        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu'])
        include_dirs.extend([f'{nvshmem_dir}/include'])
        library_dirs.extend([f'{nvshmem_dir}/lib'])
        nvcc_dlink.extend(['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem_device'])
        # Use --no-as-needed to ensure NVSHMEM host library is always linked
        extra_link_args.extend(
            ['-Wl,--no-as-needed', f'-l:{nvshmem_host_lib}', '-Wl,--as-needed', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_dir}/lib'])

    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')

        # Disable internode and low-latency kernels
        assert disable_nvshmem_and_nccl
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

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
        nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

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
    print(f' > NVSHMEM path: {nvshmem_dir}')
    print()

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(name='deep_ep',
                     version='1.2.1' + revision,
                     packages=setuptools.find_packages(include=['deep_ep']),
                     ext_modules=[
                         CUDAExtension(name='deep_ep_cpp',
                                       include_dirs=include_dirs,
                                       library_dirs=library_dirs,
                                       sources=sources,
                                       extra_compile_args=extra_compile_args,
                                       extra_link_args=extra_link_args)
                     ],
                     cmdclass={'build_ext': BuildExtension})

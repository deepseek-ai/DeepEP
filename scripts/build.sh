#!/bin/bash

source /workspace/set_env.sh
export TORCH_CUDA_ARCH_LIST=""
export NIXL_INCLUDE_PATHS=$NIXL_HOME/src/api/gpu/ucx:$NIXL_HOME/src/api/cpp/:$UCX_HOME/include
export NIXL_LIB_PATH=$NIXL_HOME/build/src
export NVSHMEM_DIR=/workspace/nvshmem_src
#apt install bear -y
#bear -- python3 setup.py build > compile.log 2>&1
python3 setup.py build
ln -sf build/lib.linux-x86_64-3.10/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so
pip install -e .


#!/bin/bash

echo "Environment configured inside container. Please confirm ucx, rdma, doca paths."
# Base Directories
export CUDA_HOME=/usr/local/cuda
export RDMA_HOME=/workspace/rdma-core/build
export DOCA_HOME=/workspace/doca/build/install
export UCX_BASE=/workspace/ucx
export GDR_BASE=/workspace/gdrcopy
export NIXL_HOME=/workspace/nixl
export NVSHMEM_HOME=/workspace/nvshmem_src/build
export ETCD_LIB=/workspace/etcd-cpp-apiv3/build/src

export GDR=$GDR_BASE/include
export GDR_LIB=$GDR_BASE/lib
export UCX_HOME=$UCX_BASE/rfs
export NIXL_INCLUDE_PATHS=$NIXL_HOME/include
export NIXL_LIB_PATH=$NIXL_HOME/lib

# Create gdrcopy symlink if missing
mkdir -p /opt/mellanox
ln -sf "$(dirname "$GDR")" /opt/mellanox/gdrcopy

# Include Paths
export CUDA_INC=$CUDA_HOME/include
export CPATH="$CUDA_INC:$DOCA_HOME/include:$RDMA_HOME/include:$CPATH"
export INCLUDE="$NIXL_HOME/include:$DOCA_HOME/include:$INCLUDE"

# Library Paths
export CUDA_LIB64=$CUDA_HOME/lib64
export LD_LIBRARY_PATH="$ETCD_LIB:$NIXL_HOME/build/lib:$RDMA_HOME/lib:$DOCA_HOME/lib/x86_64-linux-gnu:$UCX_HOME/lib:$CUDA_LIB64:$LD_LIBRARY_PATH"
export LDFLAGS="-L$UCX_HOME/lib -L$RDMA_HOME/lib -L$GDR_LIB -L$DOCA_HOME/lib/x86_64-linux-gnu -L$CUDA_LIB64"

# Compilation Flags
export CFLAGS="-I$UCX_HOME/include -I$RDMA_HOME/include -I$GDR -I$DOCA_HOME/include -I$CUDA_INC -I/usr/local/include"
export CPPFLAGS="$CFLAGS -I/usr/local/cuda/include"
export CXXFLAGS="$CFLAGS"
export NVCC_FLAGS="-I$DOCA_HOME/include -I$RDMA_HOME/include -I$CUDA_INC"

# PKG Config Paths
export PKG_CONFIG_PATH="$RDMA_HOME/lib/pkgconfig:/usr/local/lib/pkgconfig:$UCX_BASE/build-release/lib/pkgconfig:$UCX_HOME/lib/pkgconfig:$DOCA_HOME/lib/x86_64-linux-gnu/pkgconfig"

# PATH for NIXL binaries
export PATH="$CUDA_HOME/bin:$NIXL_HOME/build/examples/cpp:$UCX_HOME/bin:$RDMA_HOME/bin:$PATH"

# nvshmem config
# Configure NVSHMEM build options
export NVSHMEM_IBGDA_SUPPORT=1
export NVSHMEM_SHMEM_SUPPORT=0
export NVSHMEM_UCX_SUPPORT=0
export NVSHMEM_USE_NCCL=0
export NVSHMEM_PMIX_SUPPORT=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
export NVSHMEM_USE_GDRCOPY=1
export NVSHMEM_IBRC_SUPPORT=0
export NVSHMEM_BUILD_TESTS=0
export NVSHMEM_BUILD_EXAMPLES=0
export NVSHMEM_MPI_SUPPORT=0
export NVSHMEM_BUILD_HYDRA_LAUNCHER=0
export NVSHMEM_BUILD_TXZ_PACKAGE=0

# Set NVSHMEM runtime variables
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
export NVSHMEM_QP_DEPTH=1024

# Update system library and binary paths
export NVSHMEM_DIR=$NVSHMEM_HOME
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export PATH=$NVSHMEM_HOME/bin:$PATH

# Configure DeepEP CUDA and build environment
unset TORCH_CUDA_ARCH_LIST
unset DISABLE_SM90_FEATURES
unset DISABLE_AGGRESSIVE_PTX_INSTRS


echo "�~\~E Environment configured."
echo "CUDA_HOME=$CUDA_HOME"
echo "RDMA_HOME=$RDMA_HOME"
echo "DOCA_HOME=$DOCA_HOME"
echo "UCX_HOME=$UCX_HOME"
echo "NIXL_HOME=$NIXL_HOME"
echo "NVSHMEM_HOME=$NVSHMEM_HOME"

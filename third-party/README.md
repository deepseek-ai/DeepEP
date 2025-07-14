# Install NVSHMEM

## Important notices

**This project is neither sponsored nor supported by NVIDIA.**

**Use of NVIDIA NVSHMEM is governed by the terms at [NVSHMEM Software License Agreement](https://docs.nvidia.com/nvshmem/api/sla.html).**

## Prerequisites

Hardware requirements:
   - GPUs inside one node needs to be connected by NVLink
   - GPUs across different nodes needs to be connected by RDMA devices, see [GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
   - InfiniBand GPUDirect Async (IBGDA) support, see [IBGDA Overview](https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/)
   - For more detailed requirements, see [NVSHMEM Hardware Specifications](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html#hardware-requirements)

## Installation procedure

### 1. Acquiring NVSHMEM source code

Download NVSHMEM source code from the [NVIDIA NVSHMEM OPEN SOURCE PACKAGES](https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_cuda12-all).

### 2. [Optional] apply our custom patch

**NOTE: After NVSHMEM v3.3.9, it is no longer necessary to apply our patch to achieve optimal performance.**

Navigate to your NVSHMEM source directory and apply our provided patch:

```bash
git apply /path/to/deep_ep/dir/third-party/nvshmem.patch
```

### 3. Configure NVIDIA driver (required by inter-node communication)

Enable IBGDA by modifying `/etc/modprobe.d/nvidia.conf`:

```bash
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
```

Update kernel configuration:

```bash
sudo update-initramfs -u
sudo reboot
```

For more detailed configurations, please refer to the [NVSHMEM Installation Guide](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html).

### 4. Build and installation

DeepEP uses NVLink for intra-node communication and IBGDA for inter-node communication. All the other features are disabled to reduce the dependencies.

```bash
export CUDA_HOME=/path/to/cuda
# disable all features except IBGDA
export NVSHMEM_IBGDA_SUPPORT=1

export NVSHMEM_SHMEM_SUPPORT=0
export NVSHMEM_UCX_SUPPORT=0
export NVSHMEM_USE_NCCL=0
export NVSHMEM_PMIX_SUPPORT=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_IBRC_SUPPORT=0
export NVSHMEM_BUILD_TESTS=0
export NVSHMEM_BUILD_EXAMPLES=0
export NVSHMEM_MPI_SUPPORT=0
export NVSHMEM_BUILD_HYDRA_LAUNCHER=0
export NVSHMEM_BUILD_TXZ_PACKAGE=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0

cmake -G Ninja -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/your/dir/to/install
cmake --build build/ --target install
```

## Post-installation configuration

Set environment variables in your shell configuration:

```bash
export NVSHMEM_DIR=/path/to/your/dir/to/install  # Use for DeepEP installation
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
```

## Verification

```bash
nvshmem-info -a # Should display details of nvshmem
```

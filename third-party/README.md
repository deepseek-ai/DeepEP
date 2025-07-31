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

Software requirements:
   - NVSHMEM v3.3.9 or later

## Installation procedure

### 1. Install NVSHMEM binaries

NVSHMEM 3.3.9 binaries are available in several formats:
   - Tarballs for  [x86_64](https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.3.9_cuda12-archive.tar.xz) and [aarch64](https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-sbsa/libnvshmem-linux-sbsa-3.3.9_cuda12-archive.tar.xz)
   - RPM and deb packages: instructions can be found on the [NVHSMEM installer page](https://developer.nvidia.com/nvshmem-downloads?target_os=Linux)
   - Conda packages through conda-forge
   - pip wheels through PyPI: `pip install nvidia-nvshmem-cu12`
DeepEP is compatible with upstream NVSHMEM 3.3.9 and later.


### 2. Enable NVSHMEM IBGDA support

NVSHMEM Supports two modes with different requirements. Either of the following methods can be used to enable IBGDA support.

#### 2.1 Configure NVIDIA driver

This configuration enables traditional IBGDA support.

Modify `/etc/modprobe.d/nvidia.conf`:

```bash
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
```

Update kernel configuration:

```bash
sudo update-initramfs -u
sudo reboot
```

#### 2.2 Install GDRCopy and load the gdrdrv kernel module

This configuration enables IBGDA through asynchronous post-send operations assisted by the CPU. More information about CPU-assisted IBGDA can be found in [this blog](https://developer.nvidia.com/blog/enhancing-application-portability-and-compatibility-across-new-platforms-using-nvidia-magnum-io-nvshmem-3-0/#cpu-assisted_infiniband_gpu_direct_async%C2%A0).
It comes with a small performance penalty, but can be used when modifying the driver regkeys is not an option.

Download GDRCopy
GDRCopy is available as prebuilt deb and rpm packages [here](https://developer.download.nvidia.com/compute/redist/gdrcopy/). or as source code on the [GDRCopy github repository](https://github.com/NVIDIA/gdrcopy).

Install GDRCopy following the instructions on the [GDRCopy github repository](https://github.com/NVIDIA/gdrcopy?tab=readme-ov-file#build-and-installation).

## Post-installation configuration

When not installing NVSHMEM from RPM or deb packages, set the following environment variables in your shell configuration:

```bash
export NVSHMEM_DIR=/path/to/your/dir/to/install  # Use for DeepEP installation
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
```

## Verification

```bash
nvshmem-info -a # Should display details of nvshmem
```

# Install NVSHMEM

## Important notices

**This project is neither sponsored nor supported by NVIDIA.**

**Use of NVIDIA NVSHMEM is governed by the terms at [NVSHMEM Software License Agreement](https://docs.nvidia.com/nvshmem/api/sla.html).**

## Prerequisites

1. [GDRCopy](https://github.com/NVIDIA/gdrcopy) (v2.4 and above recommended) is a low-latency GPU memory copy library based on NVIDIA GPUDirect RDMA technology, and *it requires kernel module installation with root privileges.*

2. Hardware requirements
   - GPUDirect RDMA capable devices, see [GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
   - InfiniBand GPUDirect Async (IBGDA) support, see [IBGDA Overview](https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/)
   - For more detailed requirements, see [NVSHMEM Hardware Specifications](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html#hardware-requirements)

## Installation procedure

### 1. Install GDRCopy

GDRCopy requires kernel module installation on the host system. Complete these steps on the bare-metal host before container deployment:

#### Build and installation

```bash
wget https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v2.4.4.tar.gz
cd gdrcopy-2.4.4/
make -j$(nproc)
sudo make prefix=/opt/gdrcopy install
```

#### Kernel module installation

After compiling the software, you need to install the appropriate packages based on your Linux distribution.   
For instance, using Ubuntu 22.04 and CUDA 12.3 as an example:

```bash
pushd packages
CUDA=/path/to/cuda ./build-deb-packages.sh
sudo dpkg -i gdrdrv-dkms_2.4.4_amd64.Ubuntu22_04.deb \
             libgdrapi_2.4.4_amd64.Ubuntu22_04.deb \
             gdrcopy-tests_2.4.4_amd64.Ubuntu22_04+cuda12.3.deb \
             gdrcopy_2.4.4_amd64.Ubuntu22_04.deb
popd
sudo ./insmod.sh  # Load kernel modules on the bare-metal system
```

#### Container environment notes  

For containerized environments:
1. Host: keep kernel modules loaded (`gdrdrv`)
2. Container: install DEB packages *without* rebuilding modules:
   ```bash
   sudo dpkg -i gdrcopy_2.4.4_amd64.Ubuntu22_04.deb \
                libgdrapi_2.4.4_amd64.Ubuntu22_04.deb \
                gdrcopy-tests_2.4.4_amd64.Ubuntu22_04+cuda12.3.deb
   ```

#### Verification

```bash
gdrcopy_copybw  # Should show bandwidth test results
```

### 2. Acquiring NVSHMEM source code

Download NVSHMEM v3.1.7 from the [NVIDIA NVSHMEM Archive](https://developer.nvidia.com/nvshmem-archive).

### 3. Apply our custom patch

Navigate to your NVSHMEM source directory and apply our provided patch:

```bash
git apply /path/to/deep_ep/dir/third-party/nvshmem.patch
```

### 4. Configure NVIDIA driver

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

### 5. Build and installation

The following example demonstrates building NVSHMEM with IBGDA support:

```bash
CUDA_HOME=/path/to/cuda \
GDRCOPY_HOME=/path/to/gdrcopy \
NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/path/to/your/dir/to/install

cd build
make -j$(nproc)
make install
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

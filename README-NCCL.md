# NCCL Setup and Usage Guide

For system requirements and dependencies, please refer to the main [README.md](README.md).

**Reference**: For a detailed description of DeepEP over NCCL with GPU-Initiated Networking (GIN) and performance evaluation, see [GPU-Initiated Networking for NCCL](https://arxiv.org/abs/2511.15076).

## NCCL Version Requirements

This guide requires NCCL with the Device API support including, in particular, GPU-Initiated Networking (GIN). The recommended supported version is:

- **NCCL 2.29.1** or later
- Repository: [https://github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)

## Step 1: Setup the environment and compile NCCL

```bash
# Setup CUDA environment
export CUDA_HOME=/usr/local/cuda-12.8
export NCCL_DIR=/path/to/nccl
cd $NCCL_DIR
make clean && make MPI_HOME=$MPI_DIR NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" src.build -j32 # adjust NVCC_GENCODE accordingly
```

## Step 2: Use Low Latency (LL) Kernels

### Low Latency Kernels Overview
LL (Low Latency) kernels are optimized for latency-sensitive inference decoding tasks. They leverage NVLink and RDMA to minimize delays and provide hook-based communication-computation overlapping without occupying SM resources.

### Enable NCCL compilation
```bash
export NCCL_DIR=/path/to/nccl
export LD_LIBRARY_PATH=$NCCL_DIR/build/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$NCCL_DIR/build/lib/libnccl.so.2 # Needed on systems where PyTorch uses RPATH to load its internal NCCL library. 
                                                   # Dynamic linker priority: LD_PRELOAD > RPATH > LD_LIBRARY_PATH > RUNPATH
export ENABLE_NCCL=1; python3 setup.py build_ext --inplace; pip install --no-build-isolation .
```

### Example 1: Run LL kernels with NCCL backend
```bash
DEEP_EP_BACKEND=nccl NCCL_GIN_TYPE=3 NCCL_NET_PLUGIN=none TORCH_DISTRIBUTED_BACKEND=gloo python3 tests/test_low_latency.py --num-processes 4 --disable-nvlink
```

### Enable NVSHMEM compilation
```bash
export NVSHMEM_DIR=/path/to/nvshmem
unset ENABLE_NCCL; python3 setup.py build_ext --inplace; pip install --no-build-isolation .
```

### Example 2: Run LL kernels with NVSHMEM backend
```bash
python3 tests/test_low_latency.py --num-processes 4 --disable-nvlink
```

### Example 3: Run LL kernels with custom parameters
```bash
python3 tests/test_low_latency.py --num-processes 8 --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256
```

## Step 3: Use High Throughput (HT) Kernels

### High Throughput Kernels Overview
HT (High Throughput) kernels are optimized for training and inference prefilling tasks. They use NVLink and RDMA forwarding for asymmetric-domain bandwidth forwarding, providing high throughput for normal kernels.

### Enable NCCL compilation
```bash
export ENABLE_NCCL=1; python3 setup.py build_ext --inplace; pip install --no-build-isolation .
```

### Example 1: Run HT kernels with NCCL backend
```bash
DEEP_EP_BACKEND=nccl NCCL_GIN_TYPE=3 NCCL_NET_PLUGIN=none TORCH_DISTRIBUTED_BACKEND=gloo python tests/test_internode.py;
```

### Enable NVSHMEM compilation
```bash
unset ENABLE_NCCL; python3 setup.py build_ext --inplace; pip install --no-build-isolation .
```

### Example 2: Run HT kernels with NVSHMEM backend
```bash
python tests/test_internode.py
```

### Example 3: Run HT kernels with custom parameters
```bash
python tests/test_internode.py --num-processes 8 --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 256
```

## Step 4: Clean (if needed)

```bash
rm -rf build/ dist/ *.egg-info/ *.so;
```

---

## Important Notes

- When switching between NCCL and NVSHMEM backends, always rebuild and reinstall
- Adjust paths according to your system configuration (CUDA_HOME, NCCL_DIR, etc.)
- The NVCC_GENCODE flag should match your GPU architecture (e.g., `-gencode=arch=compute_90,code=sm_90` for H100)
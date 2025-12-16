# NCCL Setup and Usage Guide

For system requirements and dependencies, please refer to the main [README.md](README.md).

**Reference**: For a detailed description of DeepEP over NCCL with GPU-Initiated Networking (GIN) and performance evaluation, see [GPU-Initiated Networking for NCCL](https://arxiv.org/abs/2511.15076).

## NCCL Version Requirements

This guide requires NCCL with the Device API support including, in particular, GPU-Initiated Networking (GIN). The recommended supported version is:

- **NCCL 2.28.9** or later
- Repository: [https://github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)

## Step 1: Setup the environment and compile NCCL

```bash
# Setup CUDA environment
export CUDA_HOME=/usr/local/cuda-12.8
export NCCL_SRC=/path/to/nccl
cd $NCCL_SRC
make clean && make MPI_HOME=$MPI_DIR NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" src.build -j32 # adjust NVCC_GENCODE accordingly
export NCCL_DIR=$NCCL_SRC/build  # NCCL_DIR points to the build directory containing lib/ and include/
```

## Step 2: Use Low Latency (LL) Kernels

### Low Latency Kernels Overview
LL (Low Latency) kernels are optimized for latency-sensitive inference decoding tasks. They leverage NVLink and RDMA to minimize delays and provide hook-based communication-computation overlapping without occupying SM resources.

### Enable NCCL compilation
```bash
export NCCL_DIR=/path/to/nccl/install/  # Points to build directory containing bin/, include/, and lib/
export LD_LIBRARY_PATH=$NCCL_DIR/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$NCCL_DIR/lib/libnccl.so.2 # Needed on systems where PyTorch uses RPATH to load its internal NCCL library. 
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

### GIN Backend Selection

The `NCCL_GIN_TYPE` environment variable selects the GPU-Initiated Networking (GIN) backend:

| Value | Backend | Description |
|:-----:|:-------:|:------------|
| `2`   | Proxy   | Uses CPU proxy for network operations |
| `3`   | GDAKI   | GPU Direct Async Kernel-Initiated - direct GPU-to-network communication |

Example usage:
```bash
# Use GDAKI backend (recommended for best performance)
NCCL_GIN_TYPE=3 python tests/test_low_latency.py ...

# Use Proxy backend
NCCL_GIN_TYPE=2 python tests/test_low_latency.py ...
```

---

## Performance

We benchmark DeepEP on H100 (900 GB/s NVLink bidirectional bandwidth) with 8×400 Gbit/s InfiniBand (~50 GB/s per NIC maximum bandwidth), comparing **NVSHMEM** and **NCCL** backends.

### Normal kernels with NVLink and RDMA forwarding

We follow the DeepSeek-V3/R1 pretraining setting (4096 tokens per batch, 7168 hidden, top-4 groups, top-8 experts, FP8 dispatching and BF16 combining).

**NVSHMEM**

|   Type    | Dispatch #EP | Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
|:---------:|:------------:|:--------------------:|:-----------:|:--------------------:|
| Internode |      16      |  79.7 GB/s (RDMA)    |     16      |  66.4 GB/s (RDMA)    |
| Internode |      32      |  62.9 GB/s (RDMA)    |     32      |  62.9 GB/s (RDMA)    |
| Internode |      64      |  53.5 GB/s (RDMA)    |     64      |  53.2 GB/s (RDMA)    |

**NCCL**

|   Type    | Dispatch #EP | Bottleneck bandwidth | Combine #EP | Bottleneck bandwidth |
|:---------:|:------------:|:--------------------:|:-----------:|:--------------------:|
| Internode |      16      |  76.9 GB/s (RDMA)    |     16      |  66.2 GB/s (RDMA)    |
| Internode |      32      |  61.7 GB/s (RDMA)    |     32      |  62.3 GB/s (RDMA)    |
| Internode |      64      |  52.7 GB/s (RDMA)    |     64      |  52.9 GB/s (RDMA)    |

### Low-latency kernels with pure RDMA

We follow a typical DeepSeek-V3/R1 production setting (128 tokens per batch, 7168 hidden, top-8 experts, FP8 dispatching and BF16 combining).

**NVSHMEM**

| Dispatch #EP | Latency  | RDMA bandwidth | Combine #EP | Latency  | RDMA bandwidth |
|:------------:|:--------:|:--------------:|:-----------:|:--------:|:--------------:|
|      8       | 160.7 μs |   46.8 GB/s    |      8      | 304.2 μs |   47.8 GB/s    |
|      16      | 182.3 μs |   41.4 GB/s    |     16      | 319.8 μs |   45.5 GB/s    |
|      32      | 188.7 μs |   40.0 GB/s    |     32      | 332.9 μs |   43.7 GB/s    |
|      64      | 225.1 μs |   34.8 GB/s    |     64      | 343.1 μs |   42.4 GB/s    |

**NCCL**

| Dispatch #EP | Latency  | RDMA bandwidth | Combine #EP | Latency  | RDMA bandwidth |
|:------------:|:--------:|:--------------:|:-----------:|:--------:|:--------------:|
|      8       | 160.8 μs |   47.0 GB/s    |      8      | 302.8 μs |   47.9 GB/s    |
|      16      | 178.6 μs |   42.2 GB/s    |     16      | 320.8 μs |   45.3 GB/s    |
|      32      | 190.1 μs |   39.8 GB/s    |     32      | 333.2 μs |   43.6 GB/s    |
|      64      | 218.9 μs |   34.5 GB/s    |     64      | 351.1 μs |   41.4 GB/s    |
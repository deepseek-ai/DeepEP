# NCCL GIN Setup and Usage Guide

For system requirements and dependencies, please refer to the main [README.md](README.md).

## Step 0: Let's first setup the environment

```bash
source /home/nvidia/ashafi/work/sw/hpcx-v2.23-gcc-doca_ofed-ubuntu22.04-cuda12-aarch64/hpcx-mt-init.sh
hpcx_load

export NCCL_GIN_HOME=/home/nvidia/ashafi/work/gin/nccl

export CUDA_HOME=/usr/local/cuda-12.8
export NVSHMEM_DIR=/home/nvidia/ashafi/work/a2a/upstream/libnvshmem-linux-sbsa-3.3.9_cuda12-archive
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$NCCL_GIN_HOME/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Step 1: Use Low Latency (LL) Kernels

### Low Latency Kernels Overview
LL (Low Latency) kernels are optimized for latency-sensitive inference decoding tasks. They leverage NVLink and RDMA to minimize delays and provide hook-based communication-computation overlapping without occupying SM resources.

### Enable NCCL GIN compilation
```bash
export ENABLE_NCCL_GIN=1; python3 setup.py build_ext --inplace; pip install --no-build-isolation .
```

### Example 1: Run LL kernels with NCCL GIN backend
```bash
DEEP_EP_BACKEND=nccl_gin NCCL_GIN_TYPE=3 NCCL_NET_PLUGIN=none TORCH_DISTRIBUTED_BACKEND=gloo python3 tests/test_low_latency.py --num-processes 4 --disable-nvlink
```

### Enable NVSHMEM compilation
```bash
unset ENABLE_NCCL_GIN; python3 setup.py build_ext --inplace; pip install --no-build-isolation .
```

### Example 2: Run LL kernels with NVSHMEM backend
```bash
python3 tests/test_low_latency.py --num-processes 4 --disable-nvlink
```

### Example 3: Run LL kernels with custom parameters
```bash
python3 tests/test_low_latency.py --num-processes 8 --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 256
```

## Step 2: Use High Throughput (HT) Kernels

### High Throughput Kernels Overview
HT (High Throughput) kernels are optimized for training and inference prefilling tasks. They use NVLink and RDMA forwarding for asymmetric-domain bandwidth forwarding, providing high throughput for normal kernels.

### Enable NCCL GIN compilation
```bash
export ENABLE_NCCL_GIN=1; python3 setup.py build_ext --inplace; pip install --no-build-isolation .
```

### Example 1: Run HT kernels with NCCL GIN backend
```bash
DEEP_EP_GIN_NUM_COMMS=12 LD_LIBRARY_PATH=$MPI_HOME/lib/:$HPCX_UCX_DIR/lib/:$NCCL_GIN_HOME/build/lib:$NCCL_GIN_HOME/src/gin/transport/gdaki/doca-gpunetio-lite/lib:$LD_LIBRARY_PATH DEEP_EP_BACKEND=nccl_gin NCCL_GIN_TYPE=3 NCCL_GIN_ENABLE=1 UCX_IB_DM_COUNT=0 NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_NET_PLUGIN=none NCCL_DEBUG=0 DOCA_GPUNETIO_LITE_DEBUG=0 python tests/test_internode.py
```

### Enable NVSHMEM compilation
```bash
unset ENABLE_NCCL_GIN; python3 setup.py build_ext --inplace; pip install --no-build-isolation .
```

### Example 2: Run HT kernels with NVSHMEM backend
```bash
export DEEP_EP_BACKEND=nvshmem
python tests/test_internode.py
```

### Example 3: Run HT kernels with custom parameters
```bash
python tests/test_internode.py --num-processes 8 --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 256
```

## Step 3: Clean (if needed)

```bash
rm -rf build/ dist/ *.egg-info/ *.so;
```

---

## Important Notes

- When switching between NCCL GIN and NVSHMEM backends, always rebuild and reinstall
- By default, DeepEP uses NCCL as the communication backend for PyTorch Distributed that is used as a launcher for dispatch/combine primitives. However, the NCCL backend in PyTorch Distributed requires NCCL_P2P_DISABLE=1 on EOS, which is only possible for running high throughput kernels. NCCL_P2P_DISABLE=1 must not be set when running low latency kernels with NVLink; for this case, the Gloo communication backend for PyTorch Distributed can be used. [FIXME: This needs investigation. There is no reason for NCCL_P2P_DISABLE=1 to be set for running PyTorch Distributed with NCCL backend.]

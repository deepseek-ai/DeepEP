# Hybrid-EP 

## Overview
This document introduces the Hybrid Expert Parallel (Hybrid-EP) implementation to the DeepEP library, developed by NVIDIA as an optimized solution for large-scale MoE (Mixture of Experts) model all-to-all communication. This implementation is specifically designed to leverage NVIDIA GPU hardware capabilities, significantly reducing Streaming Multiprocessor (SM) resource usage while dramatically improving communication efficiency and overall throughput. This implementation maintains full backward compatibility with DeepEP. Users can seamlessly integrate Hybrid-EP into existing workflows without code modifications.

### NIXL Integration (Experimental)

> **⚠️ Experimental**: NIXL-based inter-node communication is an experimental feature. **Performance is not final — this is an initial integration that brings NIXL into Hybrid-EP, and we are actively working to improve NIXL path performance toward parity with the DOCA/RDMA path.** We welcome feedback and contributions.

Hybrid-EP supports [NIXL](https://github.com/ai-dynamo/nixl) (NVIDIA Inter-node eXchange Library) as an alternative inter-node communication backend alongside the existing DOCA/RDMA path. NIXL uses UCX for GPU-to-GPU RDMA transfers and does not require the NCCL submodule or DOCA SDK at build time, simplifying deployment in environments where DOCA is unavailable.

**Key points:**
- Build with `USE_NIXL=1` to select the NIXL path; the default (`USE_NIXL=0`) preserves the DOCA path with no behavior change
- NIXL code is fully guarded by `#ifdef USE_NIXL`; no NIXL symbols are linked when building with DOCA
- A complete example Dockerfile is provided at [`docs/Dockerfile.nixl`](Dockerfile.nixl)

## 🎯 Design Goals

1. **Maximize Network Bandwidth Utilization** - Achieve optimal network bandwidth usage for large-scale distributed training
2. **Minimize SM Resource Consumption** - Preserve computational resources for core ML workloads
3. **Hardware-Aware Optimization** - Leverage NVIDIA NVLink, RDMA, and other advanced hardware features for maximum efficiency

## 🏗️ Core Architecture

### Communication Operators
- **Dispatch**: Efficiently distribute tokens to corresponding expert nodes
- **Combine**: Aggregate expert computation results with optimized reduction operations

### Hierarchical Communication Design
- **Inter-node Communication**: High-performance RDMA-based communication across nodes*
- **Intra-node Communication**: NVLink-optimized data transfer using Tensor Memory Accelerator (TMA) instructions

## 🔧 Implementation Features

### Hardware Optimizations
- **TMA Instructions**: Leverage Tensor Memory Accelerator instructions for minimal SM overhead
- **RDMA Integration**: High-efficiency inter-node communication
- **Pipeline Architecture**: Warp-level pipeline parallelism within execution blocks

### Supported Data Types
- ✅ **BF16** (Brain Floating Point 16-bit)
- ✅ **FP8** (8-bit Floating Point)

### CUDA Graph Integration
- Full CUDA Graph compatibility for reduced launch overhead
- Zero CPU-GPU synchronization requirements
- Dynamic block count configuration for optimal resource utilization

## 📊 Performance Results

### H100 Platform

**HybridEP Performance Results (IB Bandwidth in GB/s):**

**Test Configuration:**
- Device: H100
- Tokens: 4096
- Hidden Dimension: 7168
- TopK: 8
- Router: Random Uniform
- Local Experts: 8
- SM Count: 4/8/16
- Ranks: 16/32/64

**Note**: All bandwidth values represent algorithm bandwidth.

| Ranks | SM Count | Torch API ||| Kernel Only |||
|-------|----------|-----------|-----------|-----------|-----------|-----------|-----------|
|       |          | **Dispatch (FP8)** | **Dispatch (BF16)** | **Combine** | **Dispatch (FP8)** | **Dispatch (BF16)** | **Combine** |
| 16    | 4       | 28.09	| 37.08 |	42.47 |	34.00 |	44.40 |	52.00    |
|       | 8       | 44.87	| 57.74 |	56.96 |	62.00 |	76.80 |	68.00    |
|       | 16      | 48.26	| 54.47 |	53.35 |	68.48 |	71.71 |	62.95    |
| 32    | 4       | 32.58	| 44.88 |	43.60 |	38.50 |	52.60 |	51.00    |
|       | 8       | 41.23	| 46.54 |	50.68 |	51.30 |	54.50 |	56.40    |
|       | 16      | 42.10	| 47.36 |	52.53 |	55.35 |	57.69 |	57.46    |
| 64    | 4       | 30.42	| 40.63 |	41.00 |	37.50 |	48.00 |	46.00    |
|       | 8       | 35.71	| 41.46 |	47.68 |	46.63 |	50.55 |	51.03    |
|       | 16      | 35.01	| 41.24 |	46.57 |	46.52 |	49.97 |	49.77    |

### B200 Platform

**Test Configuration:**
- Device: B200
- Tokens: 4096
- Hidden Dimension: 7168
- TopK: 8
- Router: Random Uniform
- Local Experts: 8
- Ranks: 8

**Performance Comparison (Bandwidth in GB/s):**

| Implementation | Measurement Type | SM Count | Dispatch (FP8) | Dispatch (BF16) | Combine |
|----------------|------------------|----------|----------------|-----------------|---------|
| DeepEP         | Torch API        | 16       | 246            | 348             | 302     |
|                |                  | 24       | 349            | 494             | 420     |
|                |                  | 28       | 397            | 560             | 477     |
|                |                  | 32       | 443            | 619             | 524     |
|                |                  | 36       | 482            | 635             | 549     |
|                |                  | 40       | 519            | 629             | 570     |
|                |                  | 44       | 544            | 640             | 577     |
|                |                  | 48       | 554            | 646             | 586     |
| **HybridEP**   | Torch API        | 16       | **409.71**     | **535.94**      | **530.86** |
|                | Only Kernel Time | 16       | **599.27**     | **734.95**      | **673.84** |


### GB200 Platform

**Test Configuration:**
- Device: GB200
- Tokens: 4096
- Hidden Dimension: 7168
- TopK: 8
- Router: Random Uniform
- Local Experts: 8
- SM Count: 16/32
- Ranks: 8/16/24/32/36

**Note**: All bandwidth values represent algorithm bandwidth.

**HybridEP Performance Results (Bandwidth in GB/s):**

| Ranks | SM Count | Torch API ||| Kernel Only |||
|-------|----------|-----------|-----------|-----------|-----------|-----------|-----------|
|       |          | **Dispatch (FP8)** | **Dispatch (BF16)** | **Combine** | **Dispatch (FP8)** | **Dispatch (BF16)** | **Combine** |
| 8     | 16       | 421.67	| 550.10 |	538.44 |	620.98 |	750.15 |	684.27    |
|       | 32       | 455.35	| 545.71 |	568.94 |	713.98 |	764.03 |	737.13    |
| 16    | 16       | 397.33	| 472.84 |	474.48 |	577.17 |	661.93 |	600.75    |
|       | 32       | 444.67	| 523.48 |	521.55 |	650.48 |	706.95 |	666.26    |
| 24    | 16       | 281.73	| 441.89 |	444.40 |	360.12 |	637.80 |	565.53    |
|       | 32       | 403.20	| 507.32 |	483.76 |	577.96 |	665.97 |	639.80    |
| 32    | 16       | 236.33	| 485.50 |	423.19 |	286.93 |	629.79 |	547.25    |
|       | 32       | 392.70	| 484.22 |	464.54 |	538.86 |	642.23 |	605.15    |
| 36    | 16       | 215.36	| 469.96 |	418.27 | 	260.53 |	612.85 |	543.27    |
|       | 32       | 361.13	|	479.02 |	447.89 |  489.27 |	632.31 |	596.99	  |

**DeepEP Performance Results (Bandwidth in GB/s):**

| Ranks | SM Count | Torch API |||
|-------|----------|-----------|-----------|-----------|
|       |          | **Dispatch (FP8)** | **Dispatch (BF16)** | **Combine** |
| 8     | 16       | 248.86    | 362.01    | 310.21    |
|       | 24       | 350.97    | 512.72    | 425.95    |
|       | 32       | 447.76    | 615.78    | 519.57    |
| 16    | 16       | 242.51    | 328.80    | 278.34    |
|       | 24       | 338.87    | 442.47    | 378.32    |
|       | 32       | 393.72    | 520.76    | 442.51    |
| 24    | 16       | 258.33    | 324.64    | 126.53    |
|       | 24       | 351.05    | 450.22    | 163.62    |
|       | 32       | 405.04    | 502.84    | 207.10    |


## 🚀 Usage Guide

### Installation

#### Intra-node and MNNVL Installation
For intra-node communication and MNNVL support, you can install directly by specifying the GPU architecture:

```bash
export TORCH_CUDA_ARCH_LIST="9.0 10.0"  # Adjust based on your GPU architecture
pip install .
```

#### Multi-node NIXL Installation (recommended when NIXL is available)
For multi-node support with NIXL. NIXL replaces DOCA for inter-node GPU data transfers, so the DOCA SDK and NCCL submodule are not needed at build time. Note that NCCL may still be used at runtime by `torch.distributed` for collective metadata operations.

**Prerequisites:**
- **NIXL** ([ai-dynamo/nixl](https://github.com/ai-dynamo/nixl)) — GPU-aware inter-node communication library. Install from source; see the NIXL README for build instructions.
- **UCX** ([openucx/ucx](https://github.com/openucx/ucx)) — UCX v1.17+ is recommended. Pre-installed in NVIDIA NGC PyTorch containers (e.g. `nvcr.io/nvidia/pytorch:24.12-py3` and later), as well as the NGC TensorFlow and Triton inference containers. If your image does not include UCX, install from source or via `apt install libucx-dev`.

```bash
export HYBRID_EP_MULTINODE=1
export USE_NIXL=1
export NIXL_HOME=/usr/local/nixl  # Path to NIXL install prefix (contains include/ and lib/)
export UCX_HOME=/usr              # Path to UCX install prefix (contains include/ and lib/)
export TORCH_CUDA_ARCH_LIST="9.0 10.0"  # Adjust based on your GPU architecture
pip install .
```

**Dockerfile example:** A complete, ready-to-build Dockerfile (based on the NGC PyTorch 26.03 image) that builds UCX, etcd-cpp-apiv3, NIXL, rdma-core, and DeepEP from source is provided at [`docs/Dockerfile.nixl`](Dockerfile.nixl). To build:

```bash
docker build -f docs/Dockerfile.nixl -t deepep-nixl .
```

#### Multi-node RDMA (DOCA) Installation
For multi-node support with DOCA/RDMA, additional configuration is required, make sure RDMA core libraries are properly installed and the path points to the directory containing the RDMA headers and libraries.

```bash
export HYBRID_EP_MULTINODE=1
# Do NOT set USE_NIXL - DOCA path requires NCCL submodule + DOCA SDK
export RDMA_CORE_HOME=/path/to/rdma-core  # Path to your RDMA core installation
export TORCH_CUDA_ARCH_LIST="9.0 10.0"  # Adjust based on your GPU architecture
pip install .
```
 
> RDMA Core requirement: install `rdma-core` v60.0 ([reference](https://github.com/linux-rdma/rdma-core/tree/v60.0)), and the latest release is also recommended ([linux-rdma/rdma-core](https://github.com/linux-rdma/rdma-core.git)).

Example:
```bash
git clone https://github.com/linux-rdma/rdma-core.git
cd rdma-core
git checkout tags/v60.0
sh build.sh
export RDMA_CORE_HOME=/path/to/rdma-core/build
```

Hybrid EP’s RDMA topology probing relies on `libnvidia-ml.so.1`. During Dockerfile builds, compile against the NVML stubs (for example, those shipped in `libnvidia-ml-dev`), then at runtime launch the container with `--gpus all` or a Kubernetes device plugin so that the NVIDIA container runtime injects the host’s real NVML library and prevents driver/library mismatches.

Example:
```bash
WORKDIR /workspace
RUN git clone https://github.com/linux-rdma/rdma-core.git && \
    cd rdma-core && git checkout tags/v60.0 && sh build.sh
ENV RDMA_CORE_HOME=/workspace/rdma-core/build
RUN apt-get update && \
    apt-get install -y --no-install-recommends libnvidia-ml-dev
RUN git clone -b hybrid_ep https://github.com/deepseek-ai/DeepEP.git
ENV HYBRID_EP_MULTINODE=1
RUN cd DeepEP && \
    TORCH_CUDA_ARCH_LIST="9.0 10.0" MAX_JOBS=8 pip install --no-build-isolation . && \
    apt-get purge -y libnvidia-ml-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
```

### Running with NIXL

#### Starting etcd

NIXL uses [etcd](https://etcd.io/) as a distributed key-value store for metadata exchange between ranks. An etcd server must be running and reachable by all nodes before launching the job.

Start etcd on one of the nodes (or a dedicated service node) in a separate terminal or `screen`/`tmux` session:

```bash
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://$(hostname):2379
```

This binds etcd to all interfaces on port 2379 and advertises the machine's hostname so that remote nodes can connect.

Then, when launching the job, set `NIXL_ETCD_ENDPOINTS` on every rank to point at that machine. For example, if etcd is running on `node01`:

```bash
srun --export=ALL,NIXL_ETCD_ENDPOINTS=http://node01:2379 \
     python tests/test_hybrid_ep.py
```

> **Tip:** etcd only needs to be started once per allocation. You do not need to restart it between successive `srun` invocations — the run-ID mechanism (`SLURM_STEP_ID`) automatically prevents key collisions across runs.

### NIXL Runtime Configuration

When using the NIXL inter-node path (`USE_NIXL=1`), the following environment variables can be used to tune performance and reliability. All variables are optional and have sensible defaults.

#### Performance Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPEP_NIXL_GDA_NUM_CHANNELS` | `1` | Number of GPU Direct Async (GDA) channels per UCX endpoint. More channels can increase throughput by allowing the GPU to post more concurrent RDMA operations through the NIC. Start with 1 and increase (e.g., 2 or 4) while monitoring for diminishing returns. The optimal value depends on the NIC capabilities and number of remote peers. |

#### Connection & Metadata

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPEP_NIXL_RUN_ID` | *(auto)* | Unique identifier for this run, used to prevent etcd key collisions between successive invocations. If unset, falls back to `SLURM_STEP_ID` then `SLURM_JOB_ID` automatically. Only set this if you are not using Slurm and experience stale-metadata errors across runs. |
| `DEEPEP_NIXL_FETCH_RETRY_INTERVAL` | `200` | Number of 10 ms polling iterations before invalidating and re-fetching a remote agent's metadata. |
| `DEEPEP_NIXL_FETCH_MAX_RETRIES` | `50` | Maximum number of invalidate-and-re-fetch cycles when remote metadata is unavailable. |
| `DEEPEP_NIXL_WIREUP_MAX_RETRIES` | `2000` | Maximum retry iterations for `makeConnection` during UCX wire-up. Increase at large scale or on slow networks. |
| `DEEPEP_NIXL_WIREUP_RETRY_MS` | `10` | Sleep duration (ms) between `makeConnection` retries. |
| `DEEPEP_NIXL_PREPMV_MAX_RETRIES` | `5000` | Maximum retry iterations for `prepRemoteMemView`. |
| `DEEPEP_NIXL_PREPMV_RETRY_MS` | `20` | Sleep duration (ms) between `prepRemoteMemView` retries. |
| `DEEPEP_NIXL_PREPMV_INITIAL_DELAY_MS` | `0` | Optional initial delay (ms) before creating remote memory views. Useful as a debugging aid; not normally needed. |
| `NIXL_ETCD_ENDPOINTS` | `http://localhost:2379` | etcd endpoint(s) used by NIXL for metadata exchange. |

### Troubleshooting

**Error: `No rule to make target '.../doca_gpunetio_device.h', needed by 'lib'`**

This occurs when the DOCA path is used but the DOCA SDK or NCCL build is incomplete. To avoid this, use NIXL instead:

```bash
export HYBRID_EP_MULTINODE=1
export USE_NIXL=1
pip install .
```

If `USE_NIXL` is not propagated (e.g. with pip build isolation), create a marker file before building:

```bash
touch .use_nixl
HYBRID_EP_MULTINODE=1 pip install .
```

During build, you should see `Multinode enabled: use_nixl=True` and `-> NIXL path: skipping NCCL/DOCA build`. If you see `-> DOCA path` instead, `USE_NIXL` is not being passed (try the `.use_nixl` file).

### Quick Start

Refer to `tests/test_hybrid_ep.py` for comprehensive usage examples including:
- Multi-node configuration
- Intra-node testing scenarios
- Inter-node testing scenarios
- Performance benchmarking setups

For more information on the design and tuning details, please refer to the [Hybrid-EP Design Document](Hybrid-EP_Implementation.md).


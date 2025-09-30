# Hybrid-EP Intra-node Implementation

## Overview
This document introduces the Hybrid Expert Parallel (Hybrid-EP) implementation to the DeepEP library, developed by NVIDIA as an optimized solution for large-scale MoE (Mixture of Experts) model all-to-all communication. This implementation is specifically designed to leverage NVIDIA GPU hardware capabilities, significantly reducing Streaming Multiprocessor (SM) resource usage while dramatically improving communication efficiency and overall throughput.

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

*Note: RDMA functionality will be available in upcoming releases.

## 🔧 Implementation Features

### Hardware Optimizations
- **TMA Instructions**: Leverage Tensor Memory Accelerator instructions for minimal SM overhead
- **RDMA Integration**: High-efficiency inter-node communication (coming soon)*
- **Pipeline Architecture**: Warp-level pipeline parallelism within execution blocks

### Supported Data Types
- ✅ **BF16** (Brain Floating Point 16-bit)
- ✅ **FP8** (8-bit Floating Point)

### CUDA Graph Integration
- Full CUDA Graph compatibility for reduced launch overhead
- Zero CPU-GPU synchronization requirements
- Dynamic block count configuration for optimal resource utilization

*RDMA features are currently under final testing and will be released shortly.

## 📊 Performance Results

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


## 🏛️ Code Structure

### New Files
```
csrc/hybrid_ep/
├── hybrid_ep.cu                   # Main CUDA implementation
├── hybrid_ep.cuh                  # Header definitions
├── pybind_hybrid_ep.cu            # PyBind bindings
├── config.cuh                     # Config definitions required by hybrid-EP kernels
├── utils.cuh                      # Utility helpers and macros
├── allocator/                     # Allocator for memory accessible by remote ranks
├── backend/                       # Core Hybrid-EP kernel implementations
│   └── hybrid_ep_backend.cuh
├── executor/                      # Kernel runner
├── extension/                     # Useful extensions
└── jit/                           # JIT compiler
    
deep_ep/
├── hybrid_ep_buffer.py            # Python interface
└── buffer.py                      # Buffer management

tests/
└── test_hybrid_ep.py              # Hybrid-EP tests
```

### Build Instructions
Follow the same build process as the main branch. No additional dependencies required.

## 🚀 Usage Guide

### Quick Start
Refer to `tests/test_hybrid_ep.py` for comprehensive usage examples including:
- Multi-node configuration
- Intra-node testing scenarios
- Inter-node testing will come soon
- Performance benchmarking setups

### Important Configuration Note
Here are important parameter settings in `csrc/hybrid_ep/config.cuh`. You can modify these parameters via `HybridEpBuffer.init_config()` or by setting proper environment variables (see `deep_ep/hybrid_ep_buffer.py`) to achieve better performance/usability:

- HIDDEN_DIM  
  Hidden size (must match model hidden dimension).

- MAX_NUM_OF_TOKENS_PER_RANK   
  The largest sequence length for the input of the dispatch kernel.

- NUM_OF_EXPERTS_PER_RANK  
  Number of experts hosted by each rank.

- NUM_OF_NODES  
  **Number of NVLink domains**, not the number of OS nodes / containers.

- NUM_OF_RANKS_PER_NODE  
  Number of ranks within one NVLink domain.  

- NUM_THREADS_PER_BLOCK_PREPROCESSING_API  
  Thread-block width for the preprocessing kernel.

- NUM_OF_BLOCKS_PREPROCESSING_API  
  Grid size for the preprocessing kernel.

- NUM_OF_STAGES_DISPATCH_API  
  Pipeline depth for dispatch.  
  Larger → better occupancy, but shared-memory usage grows linearly.  
  Reduce this if `HIDDEN_DIM` is very large.

- NUM_OF_BLOCKS_DISPATCH_API  
  Number of CTAs to launch for dispatch; controls how many SMs are used.

- NUM_OF_STAGES_G2S_COMBINE_API  
  Pipeline depth for global-to-shared (G2S) in combine.  
  Same shared-memory trade-off as dispatch.

- NUM_OF_STAGES_S2G_COMBINE_API  
  Pipeline depth for shared-to-global (S2G) in combine.  
  Same shared-memory trade-off as above.

- NUM_OF_BLOCKS_COMBINE_API  
  Number of CTAs for combine kernels.

---

## 📋 Implementation Status & Roadmap

### ✅ Current Features
- Full compatibility with existing DeepEP codebase
- Optimized intra-node communication via NVLink
- Support for BF16 and FP8 data types
- CUDA Graph integration
- Comprehensive performance improvements

### 🚧 Upcoming Features
- **Low Latency Mode**: Enhanced performance for latency-critical workloads
- **RDMA Integration**: High-performance inter-node communication

### ⚠️ Current Limitations
- RDMA functionality not yet available (under final testing)

### 🎯 Migration Notes
This implementation maintains full backward compatibility with DeepEP. Users can seamlessly integrate Hybrid-EP into existing workflows without code modifications.


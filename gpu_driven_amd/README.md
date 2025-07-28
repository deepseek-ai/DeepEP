# GPU-Driven Communication Backend - HIP Version

This directory contains the HIP (ROCm) port of the CUDA-based GPU-driven communication backend. The code has been converted from NVIDIA CUDA to AMD HIP to enable running on AMD GPUs.

## Overview

This is a high-performance communication backend that enables GPU-driven RDMA operations with minimal CPU intervention. The original CUDA implementation has been manually ported to HIP with careful attention to preserving functionality while adapting to AMD GPU architecture.

## Directory Structure

```
gpu_driven/
├── include/           # Header files (converted to HIP)
│   ├── common.hpp     # Common definitions and HIP runtime
│   ├── gpu_kernel.hip.h # GPU kernel declarations  
│   ├── peer_copy.hip.h  # Peer copy operations
│   ├── proxy.hpp      # CPU proxy declarations
│   ├── rdma.hpp       # RDMA operations
│   └── ring_buffer.hip.h # Ring buffer data structures
├── src/               # Source files
│   ├── common.cpp     # Common utilities
│   ├── gpu_kernel.hip # GPU kernels (converted from CUDA)
│   ├── peer_copy.hip  # Peer copy implementation (converted)
│   ├── peer_copy_worker.cpp # Worker threads (HIP APIs)
│   ├── proxy.cpp      # CPU proxy implementation (HIP APIs)
│   └── rdma.cpp       # RDMA implementation (HIP APIs)
├── bench/             # Benchmark applications
│   ├── benchmark_local.hip  # Local benchmarks
│   └── benchmark_remote.hip # Remote benchmarks
├── tests/             # Test applications
│   ├── batched_gpu_to_cpu_bench.hip
│   ├── gpu_to_cpu_bench.hip
│   └── pcie_bench.hip
└── Makefile           # HIP-compatible build system
```

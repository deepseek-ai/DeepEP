# PCIe kernel Implementation

## Overview
This chapter introduces the PCIe kernel implementation to the DeepEP library, developed by NVIDIA. DeepEP was originally designed for high-end data center GPUs with NVLink connectivity. However, many GPU configurations lack NVLink support, such as RTX professional series cards and consumer GPUs. This PR extends DeepEP to support non-NVLink environments.

## 🏗️ Core Architecture

### Communication Operators
- **Dispatch**: Efficiently distribute tokens to corresponding expert nodes
- **Combine**: Aggregate expert computation results with optimized reduction operations

### 1D Communication Design
All communication (both intra-node and inter-node) is routed exclusively through the RDMA NIC, eliminating the complexity of PCIe/RDMA traffic splitting. 

### Bounce Buffer Strategy
To maintain predictable memory usage, we employ an intermediate buffer mechanism, just similar to current normal mode implementation. These buffers are pre-registered to the NIC to optimize RDMA transfer performance while preventing excessive memory consumption.

*Note: The current PCIe mode supports communication with up to 32 ranks.

## 📊 Performance Results

### H20 Platform

**Test Configuration:**
- Device: H20 with NVLink disabled
- NIC : 8 CX7, one GPU one NIC
- Tokens: 4096
- Hidden Dimension: 7168
- TopK: 8
- SMs : 24

**Performance Comparison (Bandwidth in GB/s):**

| **Dispatch EP**| **Dispatch (FP8)**| **Combine EP** | **Combine (BF16)** |
|----------------|-------------------|----------------|--------------------|
| 8              | 53.01 GB/s        | 8              | 53.67 GB/s         |
| 16             | 47.98 GB/s        | 16             | 49.48 GB/s         |

## 🏛️ Code Structure

### New Files
```
csrc/
└── kernels/
    └── pcie.cu        # pcie core implementation
    
deep_ep/
└── buffer.py          # Buffer management

tests/
└── test_normal_without_nvl.py       # Normal mode w/o NVLink testing 
```

## 🚀 Usage Guide

### Quick Start
Refer to `tests/test_normal_without_nvl.py`, the only change for user to use pcie kernel is to select transport path by setting 'allow_nvlink_for_normal_mode'.

## 📋 Future Plan
- Add TMA support
- Reduce SM usage by merging sender and receiver
- Optimization for CX8
- Hierarchical Path RDMA+PCIE support (under exploration)

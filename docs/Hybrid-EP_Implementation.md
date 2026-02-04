# Hybrid-EP Implementation Guide

## Table of Contents
1. [Overview](#1-overview)
2. [Interface](#2-interface)
3. [Config](#3-config)
4. [Buffer Management](#4-buffer-management)
5. [JIT Compiler](#5-jit-compiler)
6. [Extensions](#6-extensions)
7. [Hybrid-EP Kernels](#7-hybrid-ep-kernels)

---

## 1. Overview

### Architecture Diagram

![Hybrid-EP Workflow](Hybrid-EP-workflowsvg.svg)

### Code Structure
```
csrc/hybrid_ep/
├── hybrid_ep.*                    # Main HybridEPBuffer class
├── pybind_hybrid_ep.cu            # PyBind bindings
├── config.cuh                     # Config definitions
├── utils.cuh                      # Utility helpers and macros
├── allocator/                     # MNNVL/IPC memory allocator
├── backend/                       # Core dispatch/combine kernels
│   ├── hybrid_ep_backend.cuh      # Kernel implementations
│   ├── ibvcore.h                  # InfiniBand verbs definitions
│   └── topo_detection.cuh         # GPU topology detection
├── buffer/                        # Buffer coordinators
│   ├── intranode.*                # NVLCoordinator (intra-node communication)
│   └── internode.*                # RDMACoordinator (inter-node communication)
├── executor/                      # Kernel execution (dispatch/combine core)
├── extension/                     # Extensions (allgather, permute)
└── jit/                           # JIT kernel compiler
    
deep_ep/
├── hybrid_ep_buffer.py            # Python interface
└── buffer.py                      # Buffer management

tests/
├── test_hybrid_ep.py              # Functional tests
└── test_graphed_hybrid_ep.py      # CUDA Graph tests
```

---

## 2. Interface

---

### `__init__`

**Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `group` | `torch.distributed.ProcessGroup` | PyTorch distributed process group |
| `hidden_dim` | `int` | Hidden dimension of tokens |
| `max_num_of_tokens_per_rank` | `int` | Maximum tokens per rank, used for buffer allocation|
| `num_local_experts` | `int` | Number of experts on each rank |
| `use_fp8` | `bool` | Use FP8 quantization (default: False) |
| `num_sms_dispatch_api` | `int` | SMs for dispatch kernel |
| `num_sms_combine_api` | `int` | SMs for combine kernel |
| `load_cached_kernels` | `bool` | Load pre-compiled JIT kernels (default: False) |
| `use_shared_buffer` | `bool` | Share intra-node buffer between dispatch/combine (default: True) |
| `enable_custom_allgather` | `bool` | Use optimized intra-node allgather (default: False) |

---

### `dispatch` / `dispatch_with_permute`

Dispatch tokens to target experts. Use `dispatch_with_permute` for integrated permutation (see [6.2 Permutation](#62-permutation)).

> **Routing Input Modes** (choose one):
> - **Index-based**: `topk_idx` + `topk_weights` + `num_of_experts`
> - **Map-based**: `routing_map` + `probs`

**Common Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden` | `Tensor[N, D]` | Input token embeddings |
| `topk_idx` | `Tensor[N, K]` | Top-K expert indices per token |
| `topk_weights` | `Tensor[N, K]` | Top-K routing weights |
| `num_of_experts` | `int` | Total number of experts |
| `routing_map` | `Tensor[N, E]` | Boolean routing map |
| `probs` | `Tensor[N, E]` | Routing probabilities |
| `scaling_factor` | `Tensor` | FP8 scaling factor |
| `handle` | `tuple` | Cached metadata from previous call |

**Additional Inputs for `dispatch_with_permute`:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `num_of_experts_per_rank` | `int` | Experts per rank |
| `pad_multiple` | `int` | Pad output to multiple (for GEMM alignment) |
| `num_permuted_tokens` | `int` | Expected output size (for non_blocking) |
| `non_blocking` | `bool` | Skip sync, use GPU-side metadata (default: False) |

> **Non-blocking Mode:** When `non_blocking=True`, all stream synchronizations are skipped. The output buffer is allocated based on `num_permuted_tokens`. If the actual token count exceeds this size, excess tokens will be dropped and the `overflow_flag` in the handle (stored on device) will be set to `True`.

**Outputs:**
| Return | `dispatch` | `dispatch_with_permute` |
|--------|------------|-------------------------|
| `dispatched_token` | Tokens for local experts | Permuted tokens grouped by expert |
| `dispatched_probs` | Routing probabilities | Routing probabilities |
| `dispatched_scaling_factor` | FP8 scaling factors | FP8 scaling factors |
| `tokens_per_expert` | - | `Tensor[E]`: Token count per expert |
| `handle` | For `combine` | For `combine_with_unpermute` |

---

### `combine` / `combine_with_unpermute`

Combine tokens from experts back to original positions. Use corresponding method based on dispatch variant.

**Inputs:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden` | `Tensor` | Expert output embeddings |
| `probs` | `Tensor` | Routing probabilities for weighted sum |
| `handle` | `tuple` | Metadata from dispatch (required) |
| `pad_multiple` | `int` | Padding alignment (`combine_with_unpermute` only) |

**Outputs:**
| Return | Type | Description |
|--------|------|-------------|
| `combined_token` | `Tensor[N, D]` | Combined tokens in original order |
| `combined_probs` | `Tensor` | Aggregated probabilities |

---

### `empty_jit_cache`
Clear all cached JIT-compiled kernels from disk.
---

### Handle Structure

The `handle` returned by dispatch methods contains precomputed metadata for the corresponding combine call.

#### `dispatch` Handle

```python
handle = (
    sparse_to_dense_map,         # [0] Tensor
    rdma_to_attn_map,            # [1] Tensor
    attn_to_rdma_map,            # [2] Tensor
    num_dispatched_tokens_tensor,# [3] Tensor: Total dispatched token count
    local_expert_routing_map,    # [4] Tensor: Local expert routing information
    num_of_tokens,               # [5] int: Number of tokens per rank
    config,                      # [6] HybridEpConfigInstance: Runtime configuration
)
```

#### `dispatch_with_permute` Handle

```python
handle = (
    sparse_to_dense_map,         # [0] Tensor
    rdma_to_attn_map,            # [1] Tensor
    attn_to_rdma_map,            # [2] Tensor
    num_dispatched_tokens_tensor,# [3] Tensor: Total dispatched token count 
    local_expert_routing_map,    # [4] Tensor: Local expert routing information
    row_id_map,                  # [5] Tensor: Row permutation mapping for unpermute
    num_of_tokens_per_rank,      # [6] int: Number of tokens per rank
    config,                      # [7] HybridEpConfigInstance: Runtime configuration
    overflow_flag,               # [8] Tensor: Buffer overflow indicator
)
```

---

## 3. Config

Hybrid-EP uses two configuration structures defined in [`config.cuh`](../csrc/hybrid_ep/config.cuh):

- **`BufferConfig`**: A subset of parameters used solely for buffer size calculation, stored persistently in the buffer object.
- **`HybridEpConfigInstance`**: Contains all parameters needed for JIT-compiling and launching Hybrid-EP kernels. A new instance is created for each invocation.

Each run compares the new `HybridEpConfigInstance` against `BufferConfig` to detect whether existing buffers are sufficient. If not, a free-reallocate cycle is triggered (see [4. Buffer Management](#4-buffer-management)).

### Parameter Reference

#### Runtime Parameters

These parameters are typically derived from the model configuration:

| Parameter | Description |
|-----------|-------------|
| `hidden_dim` | Hidden size (must match model hidden dimension) |
| `max_num_of_tokens_per_rank` | Maximum sequence length for dispatch kernel input |
| `num_of_experts_per_rank` | Number of experts hosted by each rank |
| `num_of_nodes` | Number of NVLink domains (not OS nodes/containers) |
| `num_of_ranks_per_node` | Number of ranks within one NVLink domain |

#### Performance Tuning Parameters

These parameters are pre-tuned for optimal performance. Adjustments are generally not recommended, but can be made via environment variables for specific hardware configurations:

| Parameter | Env Variable | Description |
|-----------|--------------|-------------|
| `num_of_threads_per_block_preprocessing_api` | `NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API` | Thread-block width for preprocessing kernel |
| `num_of_blocks_preprocessing_api` | - | Grid size for preprocessing kernel |
| `num_of_stages_dispatch_api` | `NUM_OF_STAGES_DISPATCH_API` | Pipeline depth for dispatch. Larger improves occupancy but increases shared memory usage. Reduce if `hidden_dim` is very large |
| `num_of_blocks_dispatch_api` | - | Number of CTAs for dispatch; controls SM utilization |
| `num_of_stages_g2s_combine_api` | `NUM_OF_STAGES_G2S_COMBINE_API` | Pipeline depth for global-to-shared in combine. Same shared memory trade-off as dispatch |
| `num_of_stages_s2g_combine_api` | `NUM_OF_STAGES_S2G_COMBINE_API` | Pipeline depth for shared-to-global in combine |
| `num_of_blocks_combine_api` | - | Number of CTAs for combine kernels |

### Note on `max_num_of_tokens_per_rank`

During JIT compilation, `max_num_of_tokens_per_rank` serves as a template parameter for static resource allocation. At runtime, the actual `num_of_tokens_per_rank` is passed and must satisfy:

```
num_of_tokens_per_rank <= max_num_of_tokens_per_rank
```

Since `max_num_of_tokens_per_rank` also determines buffer allocation size, Hybrid-EP automatically updates this value on each run to ensure sufficient capacity

---

## 4. Buffer Management

TODO
1. shared buffer
2. buffer update

---

## 5. JIT Compiler

TODO
1. nvcc jit workflow 
2. jit cacahe

---

## 6. Extensions

### 6.1 Allgather

TODO

### 6.2 Permutation

TODO

---

## 7. Hybrid-EP Kernels

TODO

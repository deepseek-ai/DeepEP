# DeepEP

DeepEP (DeepEveryParallel) is a high-performance communication library for modern machine learning training and inference. The library currently focuses on expert parallelism (EP) — providing high-throughput and low-latency all-to-all GPU kernels (MoE dispatch and combine) with low-precision support including FP8 — while also offering experimental primitives for pipeline parallelism (PP), context parallelism (CP), and remote memory access (Engram), all designed for zero or minimal SM occupation. All kernels are compiled at runtime via a lightweight Just-In-Time (JIT) module, requiring no CUDA compilation during installation.

Despite its lightweight design, DeepEP's performance matches or exceeds hardware bandwidth limits across various configurations.

## News

- **V2 release**: A complete refactoring of Expert Parallelism — achieving extreme performance with several times fewer SM resources compared to V1, while supporting significantly larger scale-up and scale-out domains. V2 has also switched from the NVSHMEM backend to the more lightweight **NCCL Gin backend**.

### New features

- **Fully JIT** (Just-In-Time compilation)
- **NCCL Gin backend**
  - Header-only & lightweight
  - Able to reuse existing NCCL communicators
- **EPv2**
  - High-throughput and low-latency APIs unified into a single `ElasticBuffer` interface, with a new GEMM layout
  - Larger scale-up & scale-out domain support (up to EP2048)
  - Analytical SM & QP count calculation — no more auto-tuning needed
  - Both hybrid & direct modes remain supported
  - For V3-like legacy training, SM usage reduced from 24 to 4 - 6 while maintaining equivalent or better performance
- **0 SM Engram** (with RDMA)
- **0 SM PP** (with RDMA)
- **0 SM CP** (with Copy Engine)

### Notes

- Buffer size consumption is larger than V1
- 0 SM RDMA low-latency EP is no longer supported
- Engram, PP, and CP are experimental features

### Still on-going features

- **Elastic GPU & CPU buffers**: A contiguous virtual address space that maps to a hybrid of GPU and CPU physical memory under the hood, enabling fully automatic and transparent Engram or imbalanced EP
- Reducing intermediate buffer sizes by leveraging EP replay to handle load imbalance
- All-gather updates and reduce-scatter implementations for DP & TP

For the legacy V1 documentation (NVSHMEM-based), see [docs/legacy.md](docs/legacy.md).

## Performance

Following V3's configuration, we tested with 8K tokens per batch, 7168 hidden dimensions, top 8 experts, FP8 dispatching, and BF16 combining, and obtained the following results:

| Arch | NIC type | Topo | Dispatch Bottleneck Bandwidth | Combine Bottleneck Bandwidth | #SMs |
|--|--|--|--|--|--|
| SM90 | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 81 GB/s (RDMA) | 12 |
| SM90 | CX7 | EP 8 x 4 | 61 GB/s (RDMA) | 61 GB/s (RDMA) | 6 |
| SM100 | CX7 | EP 8 x 2 | 90 GB/s (RDMA) | 91 GB/s (RDMA) | 12 |
| SM100 | N/A | EP 8 | 726 GB/s (NVLink) | 740 GB/s (NVLink) | 64 (Max perf) |
| SM100 | N/A | EP 8 | 643 GB/s (NVLink) | 675 GB/s (NVLink) | 24 (Min #SM) |

Notes: the results are logical bandwidth. For example, under the `EP 8 x 2` case, 90 GB/s actually contains local rank traffic.

Comparing with V1, **V2 achieves up to 1.3x peak performance, while saving up to 4x SM count**.

We omit results for larger EP configurations for the time being, but encourage interested users to benchmark them directly. Based on our internal experience, we expect the kernel to continue saturating hardware bandwidth at scale.

For V1 performance data, see [docs/legacy.md](docs/legacy.md#performance).

## Quick start

### Requirements

- Hopper (SM90) GPUs, or other architectures with SM90 PTX ISA support
- Python 3.8 and above
- CUDA version
  - CUDA 12.3 and above for SM90 GPUs
- PyTorch 2.10 and above
- NCCL 2.30.4 and above
- NVLink for intranode communication
- RDMA network for internode communication

### Install NCCL dependency

We recommend using pip to install NCCL so that DeepEP can automatically locate it within the Python environment. You can install it using the following command:

```bash
pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
```

### Install NVSHMEM dependency

DeepEP also depends on NVSHMEM to provide support for legacy methods. Please refer to our [NVSHMEM Installation Guide](docs/nvshmem.md) for instructions.

### Development

```bash
# Build and make symbolic links for SO files
python setup.py build
# You may modify the specific SO names according to your own platform
ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.cpython-38-x86_64-linux-gnu.so

# Run test cases
# NOTES: you may modify the `init_dist` function in `tests/utils/envs.py`
# according to your own cluster settings, and launch into multiple nodes
python tests/elastic/test_ep.py
python tests/elastic/test_agrs.py
python tests/elastic/test_engram.py
python tests/elastic/test_pp.py
```

### Installation

```bash
python setup.py install
```

Then, import `deep_ep` in your Python project, and enjoy!

## Interfaces and examples

### Buffer initialization

In V2, all EP operations — high-throughput and low-latency — are unified under a single `ElasticBuffer` interface. The buffer can be initialized by specifying MoE settings directly, and the optimal SM and QP counts are calculated analytically.

```python
import torch
import torch.distributed as dist
from typing import Optional

from deep_ep import ElasticBuffer

# Communication buffer (will allocate at runtime)
_buffer: Optional[ElasticBuffer] = None

# Number of SMs to use for communication kernels (will be set at buffer creation)
_num_comm_sms: int = 0


def get_buffer(group: dist.ProcessGroup,
               num_max_tokens_per_rank: int,
               hidden: int,
               num_topk: int,
               num_experts: int,
               use_fp8_dispatch: bool = False) -> ElasticBuffer:
    """Initialize or retrieve the ElasticBuffer for EP communication."""
    global _buffer, _num_comm_sms

    # Check if we can reuse the existing buffer
    required_bytes = ElasticBuffer.get_buffer_size_hint(
        group, num_max_tokens_per_rank, hidden,
        num_topk=num_topk, use_fp8_dispatch=use_fp8_dispatch,
    )
    if _buffer is not None and _buffer.group == group and _buffer.num_bytes >= required_bytes:
        return _buffer

    # Allocate a new buffer with MoE settings
    # NOTES: V2 buffer size consumption is larger than V1
    _buffer = ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
    )

    # V2 analytically calculates the optimal SM count — no more auto-tuning needed
    # You may also specify `num_sms` manually in dispatch/combine calls to override
    _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)

    return _buffer
```

### Example use in model training or inference prefilling

V2 unifies the dispatch and combine APIs into a single `ElasticBuffer` interface. The example below shows how to use them for training (with backward passes) or inference prefilling.

```python
import torch
import torch.distributed as dist
from typing import Tuple, Union

from deep_ep import ElasticBuffer, EPHandle, EventOverlap


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int,
                     num_max_tokens_per_rank: int,
                     expert_alignment: int = 1) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
              torch.Tensor, torch.Tensor, EPHandle, EventOverlap]:
    """
    MoE dispatch: route tokens to the corresponding experts across all ranks.
    Supports both BF16 and FP8 (x as a tuple of [data, scale_factors]) inputs.
    """
    global _buffer, _num_comm_sms

    recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=expert_alignment,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    # `handle` contains routing metadata for the subsequent combine call
    # `handle.num_recv_tokens_per_expert_list` provides per-expert token counts for GEMM
    # Use `event.current_stream_wait()` to synchronize the compute stream before using results
    return recv_x, recv_topk_idx, recv_topk_weights, handle, event


def dispatch_backward(grad_recv_x: torch.Tensor,
                      grad_recv_topk_weights: torch.Tensor,
                      handle: EPHandle) -> Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    """The backward pass of MoE dispatch is actually a combine."""
    global _buffer, _num_comm_sms

    combined_grad_x, combined_grad_topk_weights, event = _buffer.combine(
        grad_recv_x,
        handle=handle,
        topk_weights=grad_recv_topk_weights,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_grad_x, combined_grad_topk_weights, event


def combine_forward(x: torch.Tensor,
                    handle: EPHandle) -> Tuple[torch.Tensor, EventOverlap]:
    """MoE combine: reduce expert outputs back to their original ranks."""
    global _buffer, _num_comm_sms

    combined_x, _, event = _buffer.combine(
        x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_x, event


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: EPHandle) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    """The backward pass of MoE combine is actually a dispatch."""
    global _buffer, _num_comm_sms

    grad_x, _, _, _, event = _buffer.dispatch(
        grad_combined_x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return grad_x, event
```

For communication-computation overlap, use the `EventOverlap` interface to manage dependencies between the communication stream and the compute stream:

```python
# After dispatch, overlap computation while communication is in-flight
recv_x, recv_topk_idx, recv_topk_weights, handle, event = dispatch_forward(...)

# ... do some independent computation here ...

# Wait for communication to finish before using results
event.current_stream_wait()

# Now safe to use recv_x, recv_topk_idx, recv_topk_weights
```

### Example use in inference decoding

For inference decoding, the same `ElasticBuffer` is used. The handle-caching pattern allows reusing routing metadata across iterations when the gating decisions remain unchanged, avoiding redundant CPU synchronization.

```python
import torch
from typing import Tuple, Optional, Union

from deep_ep import ElasticBuffer, EPHandle, EventOverlap


def decode_dispatch(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                    topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                    num_experts: int,
                    num_max_tokens_per_rank: int,
                    cached_handle: Optional[EPHandle] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
              torch.Tensor, torch.Tensor, EPHandle, EventOverlap]:
    """
    MoE dispatch for inference decoding.
    If `cached_handle` is provided, the layout is reused without CPU synchronization.
    """
    global _buffer, _num_comm_sms

    if cached_handle is not None:
        # Reuse cached handle: skip layout recomputation and CPU sync
        recv_x, _, _, handle, event = _buffer.dispatch(
            x,
            handle=cached_handle,
            num_sms=_num_comm_sms,
            async_with_compute_stream=True,
        )
        return recv_x, cached_handle.topk_idx, None, handle, event

    recv_x, recv_topk_idx, recv_topk_weights, handle, event = _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return recv_x, recv_topk_idx, recv_topk_weights, handle, event


def decode_combine(x: torch.Tensor,
                   handle: EPHandle) -> Tuple[torch.Tensor, EventOverlap]:
    """MoE combine for inference decoding."""
    global _buffer, _num_comm_sms

    combined_x, _, event = _buffer.combine(
        x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_x, event
```

### Environment variables

The library provides some environment variables, which may be useful:

- General
    - `EP_BUFFER_DEBUG`: `0` or `1`, print buffer initialization, SM approximation, and backend debugging information, `0` by default
    - `EP_SUPPRESS_NCCL_CHECK`: `0` or `1`, suppress NCCL version mismatch checking, `0` by default
    - `EP_AVOID_RECORD_STREAM`: `0` or `1`, avoid `record_stream` on output tensors, `0` by default
    - `EP_NUM_TOPK_IDX_BITS`: integer, override the number of bits for top-k index encoding, `0` (auto) by default
- Networking
    - `EP_NIC_NAME`: string, the default NIC name used to query NIC properties, `mlx5_0` by default
    - `EP_OVERRIDE_RDMA_SL`: integer, override the RDMA service level index for traffic isolation
    - `EP_DISABLE_GIN`: `0` or `1`, disable the NCCL Gin backend (fall back to non-Gin path), `0` by default
- JIT
    - `EP_JIT_DEBUG`: `0` or `1`, print JIT debugging information, `0` by default
    - `EP_JIT_CACHE_DIR`: string, cache directory for compiled kernels, `$HOME/.deep_ep` by default
    - `EP_JIT_NVCC_COMPILER`: string, NVCC compiler path; defaults to `torch.utils.cpp_extension.CUDA_HOME`
    - `EP_JIT_CPP_STANDARD`: integer, C++ standard version, `20` by default
    - `EP_JIT_PRINT_COMPILER_COMMAND`: `0` or `1`, print compilation commands, `0` by default
    - `EP_JIT_PTXAS_VERBOSE`: `0` or `1`, show detailed PTXAS output, `0` by default
    - `EP_JIT_PTXAS_CHECK`: `0` or `1`, assert no local memory usage in compiled kernels, `0` by default
    - `EP_JIT_WITH_LINEINFO`: `0` or `1`, embed source line info for profiling tools, `0` by default
    - `EP_JIT_DUMP_ASM`: `0` or `1`, dump both PTX and SASS, `0` by default
    - `EP_JIT_DUMP_PTX`: `0` or `1`, dump PTX output, `0` by default
    - `EP_JIT_DUMP_SASS`: `0` or `1`, dump SASS output, `0` by default
- Debug and profiling
    - `EP_GIN_GDAKI_DEBUG`: `0` or `1`, enable NCCL Gin GDAKI debugging output, `0` by default
    - `EP_USE_NVIDIA_TOOLS`: `0` or `1`, skip internal profiling when running under external NVIDIA tools, `0` by default
    - `EP_DISABLE_BARRIER_PROFILING`: `0` or `1`, disable barrier-based communication profiling in benchmarks, `0` by default
- Build
    - `EP_NCCL_ROOT_DIR`: string, path to the NCCL installation directory; auto-detected from the Python environment if not set
    - `EP_NVSHMEM_ROOT_DIR`: string, path to the NVSHMEM installation directory; auto-detected from the Python environment if not set
    - `TORCH_CUDA_ARCH_LIST`: string, list of target CUDA architectures, e.g. `"9.0"`
    - `DISABLE_SM90_FEATURES`: `0` or `1`, disable SM90 features for legacy methods, `0` by default
    - `DISABLE_AGGRESSIVE_PTX_INSTRS`: `0` or `1`, disable aggressive load/store instructions in legacy methods, `0` by default

Some environment variables are **persistent**: they are captured at build time and baked into the installed package as default values. At import time, these defaults are applied automatically unless overridden by current environment variables. The persistent variables are: `EP_JIT_CACHE_DIR`, `EP_JIT_PRINT_COMPILER_COMMAND`, `EP_NUM_TOPK_IDX_BITS`, `EP_NCCL_ROOT_DIR`.

For additional details, please refer to [the test code](tests/elastic/test_ep.py) or review the corresponding Python documentation.

## Network configurations

DeepEP is fully tested with InfiniBand networks. However, it is theoretically compatible with RDMA over Converged Ethernet (RoCE) as well.

### Traffic isolation

Traffic isolation is supported by InfiniBand through Virtual Lanes (VL).

To prevent interference between different types of traffic, we recommend segregating workloads across different virtual lanes as follows:

- expert-parallel workloads
- other workloads

For DeepEP V2, you can control the virtual lane assignment by setting the `sl_idx` argument or the `EP_OVERRIDE_RDMA_SL` environment variable.

### Adaptive routing

Adaptive routing is an advanced routing feature provided by InfiniBand switches that can evenly distribute traffic across multiple paths. Even though adaptive routing introduces additional latency, we still recommend enabling it under all network load conditions.

### Congestion control

Congestion control is disabled because it hurts maximum bandwidth. If congestion is unavoidable in some scenarios, we recommend assigning those workloads to low-priority virtual lanes.

### PCI atomic mode

If the hardware supports it, we recommend using the following command to set the NIC's `PCI_ATOMIC_MODE` to improve RDMA atomic operation performance:

```bash
sudo mlxconfig -y -d mlx5_$i set PCI_ATOMIC_MODE=4
```

## Experimental branches

- [Zero-copy](https://github.com/deepseek-ai/DeepEP/pull/453)
    - Removing the copy between PyTorch tensors and communication buffers, which reduces the SM usages significantly for normal kernels
    - This PR is authored by **Tencent Network Platform Department**
- [Eager](https://github.com/deepseek-ai/DeepEP/pull/437)
    - Using a low-latency protocol removes the extra RTT latency introduced by RDMA atomic OPs
- [Hybrid-EP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
    - A new backend implementation using TMA instructions for minimal SM usage and larger NVLink domain support
    - Fine-grained communication-computation overlap for single-batch scenarios
    - PCIe kernel support for non-NVLink environments
    - NVFP4 data type support
- [AntGroup-Opt](https://github.com/deepseek-ai/DeepEP/tree/antgroup-opt)
    - This optimization series is authored by **AntGroup Network Platform Department**
    - [Normal-SMFree](https://github.com/deepseek-ai/DeepEP/pull/347) Eliminating SM from RDMA path by decoupling comm-kernel execution from NIC token transfer, freeing SMs for compute
    - [LL-SBO](https://github.com/deepseek-ai/DeepEP/pull/483) Overlapping Down GEMM computation with Combine Send communication via signaling mechanism to reduce end-to-end latency
    - [LL-Layered](https://github.com/deepseek-ai/DeepEP/pull/500) Optimizing cross-node LL operator communication using rail-optimized forwarding and data merging to reduce latency
- [Mori-EP](https://github.com/deepseek-ai/DeepEP/tree/mori-ep)
    - ROCm/AMD GPU support powered by [MORI](https://github.com/ROCm/mori) backend (low-latency mode)

## Community forks

- [uccl/uccl-ep](https://github.com/uccl-project/uccl/tree/main/ep) - Enables running DeepEP on heterogeneous GPUs (e.g., Nvidia, AMD) and NICs (e.g., EFA, Broadcom, CX7)
- [Infrawaves/DeepEP_ibrc_dual-ports_multiQP](https://github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP) - Adds multi-QP solution and dual-port NIC support in IBRC transport
- [antgroup/DeepXTrace](https://github.com/antgroup/DeepXTrace) - A diagnostic analyzer for efficient and precise localization of slow ranks
- [ROCm/mori](https://github.com/ROCm/mori) - AMD's next-generation communication library for performance-critical AI workloads (e.g., Wide EP, KVCache transfer, Collectives)

## Acknowledgement

DeepEP V2 is built on top of the [NCCL](https://github.com/nvidia/nccl) Gin backend. Thanks to @sjeaugey, @pakmarkthub, @sb17v, @xiaofanl-nvidia, and the NCCL team for their support!

## License

This code repository is released under [the MIT License](LICENSE).

## Citation

```bibtex
@misc{deepep2025,
      title={DeepEP: an efficient expert-parallel communication library},
      author={Chenggang Zhao and Shangyan Zhou and Liyue Zhang and Chengqi Deng and Zhean Xu and Yuxuan Liu and Kuai Yu and Jiashi Li and Liang Zhao},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepEP}},
}
```

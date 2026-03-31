# 快速开始

这一页的目标很纯粹：用最短路径把你从“刚 clone 下来 DeepEP”带到“我已经明白 API 主骨架，知道真实集群里要配什么”。

## 1. 先过一遍硬件与软件门槛

结合仓库 `README.md` 与 `setup.py`，实际要求如下：

- Ampere（SM80）或 Hopper（SM90）级别 GPU；
- Python 3.8+；
- PyTorch 2.1+；
- SM80 至少 CUDA 11；SM90 建议 CUDA 12.3+；
- 节点内普通内核依赖 NVLink；
- 跨节点路径依赖 RDMA；
- 所有基于 RDMA 的特性还依赖 NVSHMEM。

如果你只想用单节点 NVLink 普通内核，理论上可以不装 NVSHMEM；但 internode 与 low-latency 功能都会被关闭。

## 2. 构建与安装

### 开发构建

```bash
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py build
ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.cpython-38-x86_64-linux-gnu.so
```

### 安装到环境中

```bash
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py install
```

### 关键环境变量

| 变量 | 含义 | 对应位置 |
| --- | --- | --- |
| `NVSHMEM_DIR` | NVSHMEM 安装目录 | `setup.py`、运行时初始化 |
| `DISABLE_SM90_FEATURES` | 禁用 Hopper 特性 | `setup.py`、`csrc/config.hpp` |
| `TORCH_CUDA_ARCH_LIST` | 目标架构列表，如 `9.0` | `setup.py` |
| `DISABLE_AGGRESSIVE_PTX_INSTRS` | 关闭激进 PTX 指令技巧 | `setup.py`、底层 kernel |
| `TOPK_IDX_BITS` | 选择 32 位或 64 位 expert index | `setup.py`、`csrc/config.hpp` |
| `NVSHMEM_IB_SL` | InfiniBand 虚拟 lane / service level | 集群部署时使用 |

## 3. 执行骨架

```mermaid
flowchart LR
    A[初始化分布式组] --> B[选择普通模式或低延迟模式]
    B --> C[根据 size hint 分配 buffer]
    C --> D[准备 top-k 路由张量]
    D --> E[Dispatch]
    E --> F[本地专家计算]
    F --> G[Combine]
```

如果你能把这张图吃透，DeepEP 的 API 主线就已经明白大半了。

## 4. 普通内核的最小调用骨架

```python
import torch
import torch.distributed as dist
from deep_ep import Buffer

Buffer.set_num_sms(24)  # 普通内核要求偶数

group = dist.group.WORLD
hidden_bytes = hidden * 2  # BF16 每元素 2 字节

dispatch_cfg = Buffer.get_dispatch_config(group.size())
combine_cfg = Buffer.get_combine_config(group.size())
num_nvl_bytes = max(
    dispatch_cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
    combine_cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
)
num_rdma_bytes = max(
    dispatch_cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
    combine_cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
)

buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, layout_event = \
    buffer.get_dispatch_layout(topk_idx, num_experts)

recv_x, recv_topk_idx, recv_topk_weights, local_expert_counts, handle, event = buffer.dispatch(
    x,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_tokens_per_rank=num_tokens_per_rank,
    num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
    is_token_in_rank=is_token_in_rank,
    num_tokens_per_expert=num_tokens_per_expert,
)

# ... 本地专家 GEMM ...

combined_x, combined_topk_weights, event = buffer.combine(
    expert_outputs,
    handle,
    topk_weights=recv_topk_weights,
)
```

### 用最朴素的话理解这些参数

- `topk_idx`：每个 token 想去哪些 expert。
- `get_dispatch_layout(...)`：把这个愿望单翻译成“发货计划单”。
- `dispatch(...)`：按计划把 token 发出去。
- `handle`：回程 combine 的“底单”，没有它就很难精确拼回去。

## 5. 低延迟模式的最小骨架

```python
from deep_ep import Buffer

num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
    num_max_dispatch_tokens_per_rank,
    hidden,
    group.size(),
    num_experts,
)

buffer = Buffer(
    group,
    0,
    num_rdma_bytes,
    low_latency_mode=True,
    num_qps_per_rank=num_experts // group.size(),
)

recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
    hidden_states,
    topk_idx,
    num_max_dispatch_tokens_per_rank,
    num_experts,
    return_recv_hook=True,
)

hook()  # 真正收 payload

combined_x, event, hook = buffer.low_latency_combine(
    expert_outputs,
    topk_idx,
    topk_weights,
    handle,
    return_recv_hook=True,
)

hook()
```

低延迟模式和普通模式的根本差异在于：

- 它吃更多的 RDMA buffer；
- 它优先优化 decode 延迟，而不是吞吐；
- 它可以把“发起网络请求”和“真正收取数据”拆开。

## 6. `EventOverlap` 到底是做什么的

`EventOverlap` 本质上就是一个更顺手的 CUDA event 封装，用来让计算流和通信流建立依赖，而不是全设备同步。

最常见的用法是：

- `previous_event` 告诉 DeepEP：通信开始前要先等一段计算；
- `async_finish=True` 让通信先在 comm stream 上飞，调用先返回；
- `current_stream_wait()` 则让你在真正需要结果时再等。

## 7. 集群排障最先看什么

在怪 DeepEP 之前，先查下面这几项：

- 本机 GPU 之间真的全是 NVLink 可见吗？
- 编译时目标架构是不是配对了当前 GPU？
- NVSHMEM 是否已正确安装与可见？
- process group 的 world size / rank 是否正确？
- 低延迟模式下，`num_qps_per_rank` 是否等于本地 expert 数？

`deep_ep/utils.py` 里的 `check_nvlink_connections(...)` 就是专门用来挡掉那些“看起来像多卡，实际上拓扑不满足”的情况。

## 8. 仓库里已有的测试入口

```bash
python tests/test_intranode.py
python tests/test_internode.py
python tests/test_low_latency.py
```

另外要注意 `tests/utils.py` 里的 `init_dist(...)` 很简化，作者也明说了：在你的真实集群里，通常要按自己的 launcher / 环境变量体系改写。

## 9. 下一步怎么读

- 想看系统全貌：去 [架构总览](architecture.md)
- 想看训练 / prefill：去 [普通内核路径](normal-kernels.md)
- 想看 decoding：去 [低延迟内核路径](low-latency.md)

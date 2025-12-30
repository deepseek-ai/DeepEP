# DeepEP Permute 流程分析

## 概述

DeepEP 是一个专为 Mixture-of-Experts (MoE) 模型和专家并行 (EP) 优化的通信库。在 MoE 模型中，"permute" 流程指的是将 tokens 根据路由结果（top-k 专家选择）重新分配到不同 rank 的过程，这是 **dispatch** 操作的核心。

## 整体架构

DeepEP 支持三种通信模式：

1. **Intranode（节点内）**: 使用 NVLink 进行高吞吐通信
2. **Internode（节点间）**: 使用 RDMA + NVLink 混合通信
3. **Low-latency（低延迟）**: 使用纯 RDMA 进行低延迟通信

## Permute 流程详解

### 阶段 1: Layout 计算（布局分析）

**文件**: `csrc/kernels/layout.cu`
**核心函数**: `get_dispatch_layout()`

这个阶段计算 tokens 如何分配到各个 rank 和 expert：

```cpp
// 主要输出
int* num_tokens_per_rank      // [num_ranks] - 每个 rank 接收的 token 数量
int* num_tokens_per_rdma_rank // [num_rdma_ranks] - 每个 RDMA rank 接收的 token 数量
int* num_tokens_per_expert    // [num_experts] - 每个 expert 处理的 token 数量
bool* is_token_in_rank        // [num_tokens, num_ranks] - token 是否发送到该 rank
```

**执行逻辑**:
1. 遍历所有 tokens 的 `topk_idx`（每个 token 选择的 top-k 专家）
2. 统计每个 expert 需要处理的 token 数量
3. 计算每个 rank 需要接收的 token 数量（因为 experts 分布在不同 ranks）
4. 标记每个 token 需要发送到哪些 ranks

**关键代码片段** (layout.cu:31-38):
```cpp
for (int i = thread_id; i < num_tokens; i += kNumThreads) {
    auto shifted_topk_idx = topk_idx + i * num_topk;
    for (int j = 0, expert_idx; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
            ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
    }
}
```

---

### 阶段 2: Dispatch 通知（Notify Dispatch）

**文件**: `csrc/kernels/intranode.cu`
**核心函数**: `notify_dispatch()`

这个阶段在各 rank 之间同步 token 分布信息：

**执行逻辑**:

1. **Barrier 同步**: 所有 ranks 等待对齐 (line 33)
   ```cpp
   barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);
   ```

2. **写入本地统计信息** (line 46-50):
   ```cpp
   per_rank_buffer[rank * kNumRanks + thread_id] = num_tokens_per_rank[thread_id];
   for (int i = 0; i < num_experts_per_rank; ++i)
       per_expert_buffer[rank * num_experts_per_rank + i] =
           num_tokens_per_expert[thread_id * num_experts_per_rank + i];
   ```

3. **再次 Barrier**: 等待所有 ranks 写入完成 (line 53)

4. **计算前缀和** (line 60-64):
   - 计算每个 rank 从其他 ranks 接收的 token 累计数量
   - 用于后续确定每个 token 在接收缓冲区的位置

5. **计算 Channel 前缀和** (line 92-110):
   - 将任务分配到多个 channels（并行通道）
   - 每个 channel 负责一部分 tokens 的传输

---

### 阶段 3: Token Dispatch（实际数据传输）

**文件**: `csrc/kernels/intranode.cu`
**核心函数**: `dispatch<kNumRanks, kNumThreads, kNumTMABytesPerWarp>()`

这是最核心的 permute 操作，实际完成 tokens 的重排和传输。

#### 3.1 发送端逻辑 (Sender)

**关键步骤**:

1. **初始化通道元数据** (line 306-310):
   ```cpp
   int value = responsible_channel > 0 ?
       channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1] : 0;
   st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1);
   ```

2. **遍历负责的 tokens** (line 320-398):
   - 每个 channel 负责一段连续的 tokens
   - 检查目标 rank 的接收队列是否有空间

3. **队列满检测** (line 324-337):
   ```cpp
   while (true) {
       int num_used_slots = cached_channel_tail_idx -
           ld_volatile_global(channel_head_idx.buffer());
       if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens)
           break;
       // 等待接收端消费数据
   }
   ```

4. **数据拷贝** (line 358-386):
   - **隐藏向量**: 将 token 的隐藏状态拷贝到接收端的循环缓冲区
     ```cpp
     auto shifted_channel_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
     auto shifted_x = x + token_idx * hidden_int4;
     UNROLLED_WARP_COPY(5, lane_id, hidden_int4,
         shifted_channel_x_buffers, shifted_x, __ldg, st_na_global);
     ```

   - **源索引**: 记录 token 的原始位置
     ```cpp
     channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);
     ```

   - **Top-K 信息**: 拷贝并转换专家索引（转为本地索引）
     ```cpp
     auto idx_value = __ldg(topk_idx + token_idx * num_topk + lane_id);
     idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end) ?
         idx_value - recv_expert_begin : -1;
     channel_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] = idx_value;
     ```

   - **量化参数**: 如果使用 FP8，拷贝量化 scales

5. **更新尾指针** (line 396-397):
   ```cpp
   st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
   ```

#### 3.2 接收端逻辑 (Receiver)

**关键步骤**:

1. **计算接收偏移** (line 410-411):
   ```cpp
   int rank_offset = responsible_rank > 0 ?
       rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank] : 0;
   ```

2. **等待通道元数据** (line 415-420):
   ```cpp
   while ((total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0);
   while ((num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0);
   ```

3. **循环接收数据** (line 434-518):
   - 检查发送端是否有新数据（通过 tail 指针）
   - 从循环缓冲区读取数据
   - 写入最终的接收缓冲区

4. **数据拷贝**:
   - **隐藏向量** (line 463-479):
     - SM90 使用 TMA (Tensor Memory Accelerator) 优化
     - 其他架构使用 warp 级拷贝

   - **源索引** (line 483-487)

   - **Top-K 信息** (line 490-498)

   - **量化参数** (line 501-507)

5. **更新头指针** (line 513-514):
   ```cpp
   st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx);
   ```

---

### 阶段 4: Combine（反向聚合）

**文件**: `csrc/kernels/intranode.cu`
**核心函数**: `combine()`

Combine 是 dispatch 的逆操作，将处理后的 tokens 发送回原始 rank：

1. 使用 dispatch 阶段保存的 `handle`（包含 `src_idx` 等元信息）
2. 根据 `src_idx` 确定每个 token 应该返回到哪个原始位置
3. 执行类似的发送-接收流程
4. 支持带权重的归约（使用 `topk_weights`）

---

## 关键优化技术

### 1. 循环缓冲区 (Ring Buffer)

使用循环缓冲区实现流水线传输：
- `head_idx`: 接收端已消费的位置
- `tail_idx`: 发送端已写入的位置
- 避免固定大小限制，支持动态流控

**代码位置**: intranode.cu:257-260

### 2. 多通道并行 (Multi-Channel)

将通信任务分配到多个 channels：
- 每个 channel 负责一部分 tokens
- channels 之间并行执行
- 提高带宽利用率

**代码位置**: intranode.cu:92-112, 230-233

### 3. Warp 级并行

使用 warp 级原语实现高效并行：
- `warp_reduce_sum`: warp 内归约
- `elect_one_sync`: 选举一个线程执行操作
- `__shfl_sync`: warp 内数据交换

**代码位置**: 分布在整个 dispatch 函数中

### 4. TMA 优化 (SM90)

在 Hopper 架构上使用 TMA 加速内存拷贝：
```cpp
tma_load_1d(tma_buffer, shifted_buffer_x_int4 + i * half_hidden_int4,
    tma_mbarrier, half_hidden_bytes);
mbarrier_arrive_and_expect_tx(tma_mbarrier, half_hidden_bytes);
mbarrier_wait(tma_mbarrier, tma_phase);
tma_store_1d(tma_buffer, shifted_recv_x_int4 + i * half_hidden_int4,
    half_hidden_bytes, false);
```

**代码位置**: intranode.cu:465-475

### 5. 内存序控制

使用不同的内存访问语义优化性能：
- `ld_volatile_global`: 易失性读取（同步）
- `st_relaxed_sys_global`: 松弛写入（高性能）
- `st_release_sys_global`: 释放语义（保证可见性）
- `ld_acquire_sys_global`: 获取语义（保证读取最新值）

**代码位置**: intranode.cu:308, 310, 397, 437

### 6. 前缀和优化

使用前缀和确定每个 token 在目标缓冲区的位置：
- 避免原子操作
- 支持批量分配
- 提高并行度

**代码位置**: intranode.cu:60-64

---

## 数据流示意

```
Input: topk_idx [num_tokens, num_topk]
       x [num_tokens, hidden]
       topk_weights [num_tokens, num_topk]

Step 1: Layout
├─> num_tokens_per_rank [num_ranks]
├─> num_tokens_per_expert [num_experts]
└─> is_token_in_rank [num_tokens, num_ranks]

Step 2: Notify Dispatch
├─> Barrier sync
├─> Exchange statistics
└─> Compute prefix sums

Step 3: Dispatch (Permute)
Sender (Even SM blocks):
├─> For each channel:
│   ├─> Check queue capacity
│   ├─> Copy x to ring buffer
│   ├─> Copy src_idx
│   ├─> Transform & copy topk_idx
│   └─> Update tail_idx

Receiver (Odd SM blocks):
├─> For each channel:
│   ├─> Wait for data (tail_idx)
│   ├─> Copy from ring buffer to final buffer
│   └─> Update head_idx

Output: recv_x [num_recv_tokens, hidden]
        recv_topk_idx [num_recv_tokens, num_topk]
        recv_topk_weights [num_recv_tokens, num_topk]
        recv_src_idx [num_recv_tokens]
```

---

## Python API 使用示例

```python
import torch
from deep_ep import Buffer

# 1. 计算 layout
num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
    buffer.get_dispatch_layout(topk_idx, num_experts)

# 2. Dispatch (执行 permute)
recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, _ = \
    buffer.dispatch(
        x=x,                           # [num_tokens, hidden]
        topk_idx=topk_idx,             # [num_tokens, num_topk]
        topk_weights=topk_weights,     # [num_tokens, num_topk]
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert
    )
# recv_x: [num_recv_tokens, hidden] - 重排后的 tokens

# 3. Expert 计算 (用户代码)
output = expert_forward(recv_x, recv_topk_idx)

# 4. Combine (逆 permute)
final_output, _, _ = buffer.combine(
    x=output,
    handle=handle,
    topk_weights=recv_topk_weights
)
```

---

## 性能特点

### Intranode (节点内)
- **带宽**: ~153 GB/s (dispatch), ~158 GB/s (combine)
- **通信方式**: NVLink
- **适用场景**: 8 ranks 以内的小规模并行

### Internode (节点间)
- **带宽**: ~43-58 GB/s (取决于 rank 数量)
- **通信方式**: RDMA + NVLink 混合
- **适用场景**: 16-64 ranks 的中大规模并行

### Low-latency (低延迟)
- **延迟**: 77-194 us (dispatch), 114-369 us (combine)
- **带宽**: ~39-127 GB/s
- **通信方式**: 纯 RDMA
- **适用场景**: 推理解码（延迟敏感）

---

## 总结

DeepEP 的 permute 流程（dispatch）是一个高度优化的 all-to-all 通信过程：

1. **三阶段设计**: Layout → Notify → Dispatch，清晰分离计算和通信
2. **流水线架构**: 使用循环缓冲区实现发送-接收流水线
3. **多级并行**: SM 级、Warp 级、线程级多层并行
4. **硬件优化**: 针对 NVLink、RDMA、TMA 等硬件特性深度优化
5. **内存优化**: 精细的内存序控制，最小化同步开销

这些设计使得 DeepEP 能够在大规模 MoE 模型中实现高效的 token 重排和专家并行通信。

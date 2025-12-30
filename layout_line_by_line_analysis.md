# DeepEP Layout.cu 逐行详解

## 文件概览

`layout.cu` 的核心功能是计算 MoE dispatch 的**布局信息**，即确定每个 token 应该发送到哪些 ranks 和 experts。这是 dispatch 操作的第一阶段。

---

## 头文件和命名空间 (Lines 1-7)

```cpp
1  #include "configs.cuh"      // 配置常量，如 NUM_MAX_NVL_PEERS
2  #include "exception.cuh"    // 异常处理宏，如 EP_DEVICE_ASSERT, EP_STATIC_ASSERT
3  #include "launch.cuh"       // Kernel 启动宏，如 SETUP_LAUNCH_CONFIG, LAUNCH_KERNEL
4
5  namespace deep_ep {
6
7  namespace layout {
```

**说明**：
- Line 1-3: 引入必要的头文件
- Line 5-7: 定义命名空间，避免符号冲突

---

## Kernel 函数签名 (Lines 9-18)

```cpp
9   template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
10  __global__ void get_dispatch_layout(const topk_idx_t* topk_idx,
11                                      int* num_tokens_per_rank,
12                                      int* num_tokens_per_rdma_rank,
13                                      int* num_tokens_per_expert,
14                                      bool* is_token_in_rank,
15                                      int num_tokens,
16                                      int num_topk,
17                                      int num_ranks,
18                                      int num_experts) {
```

**模板参数**：
- `kNumThreads`: 每个 block 的线程数（固定为 256）
- `kNumExpertsPerSM`: 每个 SM 负责统计的专家数量（固定为 4）
- `kNumRanksPerSM`: 每个 SM 负责统计的 rank 数量（固定为 8）

**输入参数**：
- `topk_idx`: `[num_tokens, num_topk]` - 每个 token 选择的 top-k 专家索引
- `num_tokens`: 总 token 数量
- `num_topk`: 每个 token 选择的专家数量（通常是 4 或 8）
- `num_ranks`: EP 组中的 rank 总数
- `num_experts`: 所有专家的总数

**输出参数**：
- `num_tokens_per_rank`: `[num_ranks]` - 每个 rank 需要接收的 token 数量
- `num_tokens_per_rdma_rank`: `[num_rdma_ranks]` - 每个 RDMA rank 需要接收的 token 数量（节点间通信用）
- `num_tokens_per_expert`: `[num_experts]` - 每个专家需要处理的 token 数量
- `is_token_in_rank`: `[num_tokens, num_ranks]` - 标记每个 token 是否发送到该 rank

---

## 线程和 Block 索引 (Lines 19-20)

```cpp
19  auto sm_id = static_cast<int>(blockIdx.x);
20  auto thread_id = static_cast<int>(threadIdx.x);
```

**说明**：
- `sm_id`: 当前 SM (block) 的 ID，范围 `[0, num_sms)`
- `thread_id`: block 内的线程 ID，范围 `[0, kNumThreads)` 即 `[0, 256)`

**设计思想**：
- 不同的 SM blocks 负责统计不同的 experts 或 ranks
- 同一个 block 内的线程协作完成该 block 负责的统计任务

---

## 第一部分：统计专家（Expert）信息 (Lines 22-52)

### Shared Memory 声明 (Line 23)

```cpp
23  __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
```

**说明**：
- **Shared memory** 数组，大小为 `256 × 4 = 1024` 个整数（4 KB）
- `num_tokens_per_expert_per_thread[thread_id][expert_local_id]` 表示：
  - 线程 `thread_id` 统计到的
  - 本地专家 `expert_local_id`（相对索引）接收的 token 数量

**为什么使用 shared memory？**
- 每个线程独立统计，避免原子操作
- 后续通过归约（reduction）汇总结果
- Shared memory 延迟低（~20 cycles），远快于 global memory（~400 cycles）

---

### 计算当前 SM 负责的专家范围 (Line 24)

```cpp
24  int expert_begin_idx = sm_id * kNumExpertsPerSM,
       expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
```

**说明**：
- 每个 SM 负责 `kNumExpertsPerSM = 4` 个连续的专家
- `expert_begin_idx`: 起始专家索引 = `sm_id × 4`
- `expert_end_idx`: 结束专家索引（不包含）= `min(起始 + 4, 总专家数)`

**示例**：
- 假设 `num_experts = 64`，需要 `64 / 4 = 16` 个 SMs
- SM 0 负责 experts [0, 4)
- SM 1 负责 experts [4, 8)
- ...
- SM 15 负责 experts [60, 64)

---

### 判断是否负责专家统计 (Line 25)

```cpp
25  if (expert_begin_idx < expert_end_idx) {
```

**说明**：
- 只有前 `ceil(num_experts / kNumExpertsPerSM)` 个 SMs 负责专家统计
- 其他 SMs 跳过这部分，执行后面的 rank 统计（Line 54 开始）

---

### 初始化 per-thread 计数器 (Lines 27-29)

```cpp
27  #pragma unroll
28  for (int i = 0; i < kNumExpertsPerSM; ++i)
29      num_tokens_per_expert_per_thread[thread_id][i] = 0;
```

**说明**：
- 将该线程负责的所有本地专家的计数器初始化为 0
- `#pragma unroll`: 编译器指令，循环展开（loop unrolling）
  - 因为 `kNumExpertsPerSM = 4` 是编译时常量，展开为 4 条赋值语句
  - 减少循环控制开销，提高性能

---

### 遍历 tokens 并统计（核心逻辑）(Lines 30-39)

```cpp
30  #pragma unroll
31  for (int i = thread_id; i < num_tokens; i += kNumThreads) {
32      auto shifted_topk_idx = topk_idx + i * num_topk;
33      #pragma unroll
34      for (int j = 0, expert_idx; j < num_topk; ++j) {
35          expert_idx = static_cast<int>(shifted_topk_idx[j]);
36          if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
37              ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
38      }
39  }
```

**逐行解析**：

**Line 31**: `for (int i = thread_id; i < num_tokens; i += kNumThreads)`
- **Grid-stride loop** 模式
- 每个线程从自己的 `thread_id` 开始，步长为 `kNumThreads = 256`
- 线程 0 处理 tokens: 0, 256, 512, ...
- 线程 1 处理 tokens: 1, 257, 513, ...
- 保证所有 tokens 都被覆盖，且负载均衡

**Line 32**: `auto shifted_topk_idx = topk_idx + i * num_topk;`
- 计算第 `i` 个 token 的 topk_idx 数组的起始地址
- `topk_idx` 是一维数组，逻辑上是 `[num_tokens, num_topk]`
- `shifted_topk_idx` 指向 `topk_idx[i, :]` 的首元素

**Line 34**: `for (int j = 0, expert_idx; j < num_topk; ++j)`
- 遍历第 `i` 个 token 选择的所有 top-k 专家
- `j` 范围 `[0, num_topk)`，通常 `num_topk = 4 or 8`

**Line 35**: `expert_idx = static_cast<int>(shifted_topk_idx[j]);`
- 读取第 `j` 个专家的全局索引
- `topk_idx_t` 通常是 `int64_t`，转为 `int`

**Line 36-37**: 条件判断和计数
```cpp
if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
    ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
```
- **Line 36**: 判断该专家是否由当前 SM 负责
  - `expert_idx` 是全局索引
  - 只有在 `[expert_begin_idx, expert_end_idx)` 范围内才统计

- **Line 37**: 增加计数
  - `expert_idx - expert_begin_idx`: 转换为本地索引 `[0, kNumExpertsPerSM)`
  - 该线程负责的该专家的计数器 +1

**示例**：
```
假设 num_tokens = 1000, num_topk = 4, num_experts = 64
SM 0 负责 experts [0, 4)

线程 0 处理：
  - token 0: topk_idx = [2, 15, 30, 45]
    → expert 2 在范围内，num_tokens_per_expert_per_thread[0][2] += 1
  - token 256: topk_idx = [1, 3, 20, 50]
    → expert 1, 3 在范围内，对应计数器各 +1
  ...
```

---

### Block 内同步 (Line 40)

```cpp
40  __syncthreads();
```

**说明**：
- **Block 内所有线程同步**
- 确保所有线程都完成了 per-thread 统计
- 之后才能安全地读取其他线程的统计结果

---

### 静态断言检查 (Line 43)

```cpp
43  EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
```

**说明**：
- 编译时检查：`kNumExpertsPerSM (4) <= kNumThreads (256)`
- 保证后续归约时每个专家至少有一个线程负责

---

### 归约：汇总所有线程的统计结果 (Lines 44-50)

```cpp
44  if (expert_begin_idx + thread_id < expert_end_idx) {
45      int sum = 0;
46      #pragma unroll
47      for (int i = 0; i < kNumThreads; ++i)
48          sum += num_tokens_per_expert_per_thread[i][thread_id];
49      num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
50  }
```

**逐行解析**：

**Line 44**: `if (expert_begin_idx + thread_id < expert_end_idx)`
- 只有部分线程参与归约
- 线程 `t` 负责归约本地专家 `t`（如果存在）
- 示例：如果 SM 负责 4 个专家，只有线程 0-3 参与

**Line 47-48**: 归约循环
```cpp
for (int i = 0; i < kNumThreads; ++i)
    sum += num_tokens_per_expert_per_thread[i][thread_id];
```
- 线程 `t` 遍历所有 256 个线程的统计结果
- 累加 `num_tokens_per_expert_per_thread[i][t]`
- 得到本地专家 `t` 的总 token 数量

**Line 49**: 写入全局内存
```cpp
num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
```
- `expert_begin_idx + thread_id`: 转换为全局专家索引
- 将结果写入输出数组

**并行示意**：
```
SM 0 负责 experts [0, 4):
  - 线程 0 归约 expert 0: sum(num_tokens_per_expert_per_thread[:][0])
  - 线程 1 归约 expert 1: sum(num_tokens_per_expert_per_thread[:][1])
  - 线程 2 归约 expert 2: sum(num_tokens_per_expert_per_thread[:][2])
  - 线程 3 归约 expert 3: sum(num_tokens_per_expert_per_thread[:][3])
  - 线程 4-255 空闲
```

---

### 提前返回 (Line 51)

```cpp
51  return;
```

**说明**：
- 负责专家统计的 SMs 完成任务后直接返回
- 不执行后面的 rank 统计部分

---

## 第二部分：统计 Rank 信息 (Lines 54-120)

### RDMA 检查 (Lines 54-55)

```cpp
54  if (num_tokens_per_rdma_rank != nullptr)
55      EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS);
```

**说明**：
- 如果需要统计 RDMA rank（节点间通信），检查配置合法性
- `NUM_MAX_NVL_PEERS`: 每个节点的 GPU 数量（通常是 8）
- 节点间通信要求：
  - `num_ranks` 是 `NUM_MAX_NVL_PEERS` 的倍数
  - `num_ranks > NUM_MAX_NVL_PEERS`（至少 2 个节点）

---

### 计算 RDMA Rank 数量 (Line 58)

```cpp
58  constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
```

**说明**：
- 每个 SM 负责 `kNumRanksPerSM = 8` 个 ranks
- 对应 `8 / 8 = 1` 个 RDMA rank（一个节点）

---

### Shared Memory 声明 (Lines 59-60)

```cpp
59  __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
60  __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];
```

**说明**：
- `num_tokens_per_rank_per_thread`: `256 × 8 = 2048` 个整数（8 KB）
- `num_tokens_per_rdma_rank_per_thread`: `256 × 1 = 256` 个整数（1 KB）
- 同样使用 per-thread 统计 + 归约的模式

---

### 计算当前 SM 负责的 Rank 范围 (Lines 61-63)

```cpp
61  auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
62  int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM,
       rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
63  int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_NVL_PEERS,
       rdma_rank_end_idx = rank_end_idx / NUM_MAX_NVL_PEERS;
```

**逐行解析**：

**Line 61**: `sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;`
- 计算专家统计用了多少个 SMs
- 向上取整：`ceil(num_experts / kNumExpertsPerSM)`
- 示例：64 个专家需要 `ceil(64/4) = 16` 个 SMs

**Line 62**: 计算 rank 范围
- `sm_id - sm_begin`: 当前 SM 在 rank 统计中的相对索引
- `rank_begin_idx`: 起始 rank 索引 = `(sm_id - 16) × 8`
- `rank_end_idx`: 结束 rank 索引（不包含）

**示例**：
```
假设 num_experts = 64, num_ranks = 32
sm_begin = 16（前 16 个 SMs 负责专家统计）

SM 16: rank_begin_idx = (16-16)×8 = 0,  rank_end_idx = 8  → ranks [0, 8)
SM 17: rank_begin_idx = (17-16)×8 = 8,  rank_end_idx = 16 → ranks [8, 16)
SM 18: rank_begin_idx = (18-16)×8 = 16, rank_end_idx = 24 → ranks [16, 24)
SM 19: rank_begin_idx = (19-16)×8 = 24, rank_end_idx = 32 → ranks [24, 32)
```

**Line 63**: 计算 RDMA rank 范围
- `rdma_rank_begin_idx = rank_begin_idx / 8`: 起始节点 ID
- `rdma_rank_end_idx = rank_end_idx / 8`: 结束节点 ID

---

### 判断是否负责 rank 统计 (Line 64)

```cpp
64  if (rank_begin_idx < rank_end_idx) {
```

**说明**：
- 只有分配到 rank 统计任务的 SMs 执行
- 多余的 SMs 直接结束（Line 120）

---

### 计算专家分布信息 (Lines 65-67)

```cpp
65  const auto num_expert_per_rank = num_experts / num_ranks;
66  auto expert_begin = rank_begin_idx * num_expert_per_rank;
67  auto expert_end = rank_end_idx * num_expert_per_rank;
```

**说明**：
- `num_expert_per_rank`: 每个 rank 负责的专家数量（假设均匀分布）
- `expert_begin`: 当前 SMs 负责的 ranks 所拥有的起始专家索引
- `expert_end`: 结束专家索引

**示例**：
```
假设 num_experts = 64, num_ranks = 32
每个 rank 负责 64/32 = 2 个专家

SM 16 负责 ranks [0, 8):
  - expert_begin = 0 × 2 = 0
  - expert_end = 8 × 2 = 16
  - 即专家 [0, 16) 分布在 ranks [0, 8)
```

---

### 初始化计数器 (Lines 70-75)

```cpp
70  #pragma unroll
71  for (int i = 0; i < kNumRanksPerSM; ++i)
72      num_tokens_per_rank_per_thread[thread_id][i] = 0;
73  #pragma unroll
74  for (int i = 0; i < kNumRDMARanksPerSM; ++i)
75      num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
```

**说明**：
- 初始化 rank 和 RDMA rank 的 per-thread 计数器为 0

---

### 遍历 tokens 并统计（核心逻辑）(Lines 76-100)

```cpp
76  #pragma unroll
77  for (int i = thread_id; i < num_tokens; i += kNumThreads) {
78      auto shifted_topk_idx = topk_idx + i * num_topk;
79      int is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
80      #pragma unroll
81      for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
82          expert_idx = static_cast<int>(shifted_topk_idx[j]);
83          if (expert_begin <= expert_idx and expert_idx < expert_end) {
84              // Count single rank
85              rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
86              is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
87          }
88      }
```

**逐行解析**：

**Line 77**: Grid-stride loop，同专家统计部分

**Line 78**: 获取第 `i` 个 token 的 topk_idx

**Line 79**: 局部数组，标记该 token 是否发送到各 rank
```cpp
int is_in_rank[kNumRanksPerSM] = {0};         // 8 个元素
int is_in_rdma_rank[kNumRDMARanksPerSM] = {0}; // 1 个元素
```
- 这些是**寄存器数组**（小且固定大小）
- `is_in_rank[r] > 0` 表示该 token 至少有一个专家在 rank `r`

**Line 81-88**: 遍历 token 的 top-k 专家

**Line 82**: 读取专家全局索引

**Line 83**: 判断该专家是否属于当前 SMs 负责的 ranks
- `expert_begin <= expert_idx < expert_end`
- 只统计在负责范围内的专家

**Line 85**: 计算专家所属的 rank
```cpp
rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
```
- `expert_idx / num_expert_per_rank`: 专家所属的全局 rank ID
- `- rank_begin_idx`: 转换为本地 rank ID `[0, kNumRanksPerSM)`

**示例**：
```
假设 num_expert_per_rank = 2, rank_begin_idx = 0
expert 0 → rank 0/2 = 0 → rank_idx = 0 - 0 = 0
expert 1 → rank 1/2 = 0 → rank_idx = 0 - 0 = 0
expert 2 → rank 2/2 = 1 → rank_idx = 1 - 0 = 1
expert 3 → rank 3/2 = 1 → rank_idx = 1 - 0 = 1
```

**Line 86**: 标记该 token 需要发送到该 rank
```cpp
is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
```
- `is_in_rank[rank_idx]++`: 记录该 token 在该 rank 的专家数量
  - 注意：这里是累加，不是布尔值
  - 但后续会转为布尔值（Line 93: `> 0`）

- `is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++`:
  - `rank_idx / 8`: 计算 rank 所属的节点
  - 标记该 token 需要发送到该节点

---

### 写入 is_token_in_rank 并累加计数 (Lines 90-95)

```cpp
90  auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
91  #pragma unroll
92  for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
93      shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
94      num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
95  }
```

**Line 90**: 计算 `is_token_in_rank[i, :]` 的起始地址

**Line 92-95**: 遍历本地 ranks

**Line 93**: 写入全局内存
```cpp
shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
```
- `is_token_in_rank[i, j + rank_begin_idx]` 表示 token `i` 是否发送到 rank `j + rank_begin_idx`
- `is_in_rank[j] > 0`: 转换为布尔值
  - 只要该 token 在该 rank 有**至少一个**专家，就标记为 `true`

**Line 94**: 累加 per-thread 计数
```cpp
num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
```
- 如果该 token 需要发送到 rank `j`，计数器 +1
- `(is_in_rank[j] > 0)` 返回 0 或 1（C++ 中 `bool` 转 `int`）

---

### 累加 RDMA rank 计数 (Lines 97-99)

```cpp
97  #pragma unroll
98  for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
99      num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
```

**说明**：
- 类似 Line 94，但针对 RDMA ranks（节点级别）
- 记录需要发送到每个节点的 token 数量

---

### Block 内同步 (Line 101)

```cpp
101 __syncthreads();
```

**说明**：
- 确保所有线程完成 per-thread 统计
- 准备进行归约

---

### 归约：汇总 rank 统计 (Lines 104-111)

```cpp
104 EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
105 if (rank_begin_idx + thread_id < rank_end_idx) {
106     int sum = 0;
107     #pragma unroll
108     for (int i = 0; i < kNumThreads; ++i)
109         sum += num_tokens_per_rank_per_thread[i][thread_id];
110     num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
111 }
```

**说明**：
- 与专家归约（Lines 44-50）类似
- 线程 `t` 负责归约本地 rank `t` 的统计结果
- 累加所有线程的 `num_tokens_per_rank_per_thread[:][t]`
- 写入 `num_tokens_per_rank[rank_begin_idx + t]`

---

### 归约：汇总 RDMA rank 统计 (Lines 113-119)

```cpp
113 if (num_tokens_per_rdma_rank != nullptr and rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
114     int sum = 0;
115     #pragma unroll
116     for (int i = 0; i < kNumThreads; ++i)
117         sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
118     num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
119 }
```

**说明**：
- 如果需要 RDMA rank 统计（`!= nullptr`），执行归约
- 写入 `num_tokens_per_rdma_rank` 数组

---

## Host 函数（Kernel 启动）(Lines 123-149)

### 函数签名 (Lines 123-132)

```cpp
123 void get_dispatch_layout(const topk_idx_t* topk_idx,
124                          int* num_tokens_per_rank,
125                          int* num_tokens_per_rdma_rank,
126                          int* num_tokens_per_expert,
127                          bool* is_token_in_rank,
128                          int num_tokens,
129                          int num_topk,
130                          int num_ranks,
131                          int num_experts,
132                          cudaStream_t stream) {
```

**说明**：
- Host 端函数，用于启动 kernel
- 参数与 kernel 函数一致（除了模板参数）

---

### 配置参数 (Line 133)

```cpp
133 constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;
```

**说明**：
- 固定的性能调优参数
- `kNumThreads = 256`: 每个 block 256 个线程（典型配置）
- `kNumExpertsPerSM = 4`: 每个 SM 统计 4 个专家
- `kNumRanksPerSM = 8`: 每个 SM 统计 8 个 ranks

---

### 计算 SM 数量 (Line 134)

```cpp
134 int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) +
                (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
```

**说明**：
- 总 SMs = 专家统计 SMs + Rank 统计 SMs
- `(num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM`:
  - 向上取整：`ceil(num_experts / 4)`
  - 专家统计需要的 SMs

- `(num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM`:
  - 向上取整：`ceil(num_ranks / 8)`
  - Rank 统计需要的 SMs

**示例**：
```
num_experts = 64, num_ranks = 32
专家 SMs: ceil(64/4) = 16
Rank SMs: ceil(32/8) = 4
总 SMs: 16 + 4 = 20
```

---

### 静态断言 (Line 135)

```cpp
135 EP_STATIC_ASSERT(kNumRanksPerSM % NUM_MAX_NVL_PEERS == 0, "Invalid number of ranks per SM");
```

**说明**：
- 编译时检查：`kNumRanksPerSM (8) % NUM_MAX_NVL_PEERS (8) == 0`
- 保证 RDMA rank 计算的正确性

---

### 设置启动配置 (Line 137)

```cpp
137 SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
```

**说明**：
- 宏定义，设置 kernel 启动参数：
  - Grid size: `num_sms` 个 blocks
  - Block size: `kNumThreads = 256` 个线程
  - CUDA stream: `stream`

---

### 启动 Kernel (Lines 138-148)

```cpp
138 LAUNCH_KERNEL(&cfg,
139               (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),
140               topk_idx,
141               num_tokens_per_rank,
142               num_tokens_per_rdma_rank,
143               num_tokens_per_expert,
144               is_token_in_rank,
145               num_tokens,
146               num_topk,
147               num_ranks,
148               num_experts);
```

**说明**：
- 使用宏启动 kernel（封装了错误检查等）
- Line 139: 指定模板参数实例化 kernel
- Lines 140-148: 传递参数

---

## 算法总结

### 整体流程

1. **SM 分工**：
   - 前 N 个 SMs 负责统计 experts
   - 后 M 个 SMs 负责统计 ranks

2. **Per-thread 统计**：
   - 每个线程独立统计自己负责的 tokens
   - 使用 shared memory 存储中间结果

3. **归约汇总**：
   - Block 内同步
   - 部分线程负责归约所有线程的统计结果
   - 写入全局内存

### 性能优化技术

1. **Grid-stride loop**：负载均衡，支持任意数量的 tokens
2. **Shared memory**：避免原子操作，降低延迟
3. **Loop unrolling** (`#pragma unroll`)：减少循环开销
4. **Per-thread + 归约**：并行化 + 最终汇总，平衡并行度和同步开销
5. **分段处理**：专家和 rank 分别由不同 SMs 处理，避免资源浪费

### 时间复杂度

- **Per-thread 统计**：`O(num_tokens / kNumThreads × num_topk)` = `O(num_tokens × num_topk / 256)`
- **归约**：`O(kNumThreads)` = `O(256)`
- **总体**：`O(num_tokens × num_topk / 256)`，高度并行

### 空间复杂度

- **Shared memory（专家统计）**：`256 × 4 × 4 bytes = 4 KB`
- **Shared memory（rank 统计）**：`256 × (8 + 1) × 4 bytes = 9 KB`
- **总 shared memory**：约 13 KB（远小于 48-164 KB 的硬件限制）

---

## 设计亮点

1. **灵活的两阶段设计**：
   - 专家统计和 rank 统计分离
   - 可以根据实际规模动态分配 SMs

2. **避免原子操作**：
   - 使用 per-thread 计数 + 归约
   - 原子操作成本高（100+ cycles），归约只需 shared memory 读取（20 cycles）

3. **高并行度**：
   - 多 SMs 并行处理不同专家/ranks
   - 单 SM 内多线程并行处理 tokens

4. **内存访问优化**：
   - Coalesced access：线程以 stride-256 访问 topk_idx
   - Shared memory 重用：先专家后 rank，复用同一块内存

5. **编译时优化**：
   - 模板参数 + `constexpr` + `#pragma unroll`
   - 编译器生成高度优化的代码

---

## 典型执行示例

```
输入：
  num_tokens = 4096
  num_topk = 8
  num_experts = 64
  num_ranks = 32

Grid 配置：
  专家 SMs: ceil(64/4) = 16 SMs
  Rank SMs: ceil(32/8) = 4 SMs
  总共: 20 SMs × 256 threads = 5120 threads

执行：
  SM 0-15: 统计 experts [0,64)
  SM 16-19: 统计 ranks [0,32)

每个线程处理：
  tokens: thread_id, thread_id+256, thread_id+512, ...
  约 4096 / 256 = 16 个 tokens
  每个 token 检查 8 个 top-k 专家

输出：
  num_tokens_per_expert: [64] - 每个专家接收的 token 数
  num_tokens_per_rank: [32] - 每个 rank 接收的 token 数
  is_token_in_rank: [4096, 32] - 布尔矩阵，约 16 MB
```

---

这就是 `layout.cu` 的完整逐行解析！这个 kernel 是 DeepEP dispatch 流程的基石，后续的数据传输都依赖于它计算的布局信息。

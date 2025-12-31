# NVSHMEM 深度指南：背景知识与 API 实战

## 目录

1. [NVSHMEM 简介](#nvshmem-简介)
2. [核心概念](#核心概念)
3. [对称内存详解](#对称内存详解)
4. [API 使用示例](#api-使用示例)
5. [性能对比：NVSHMEM vs MPI](#性能对比nvshmem-vs-mpi)
6. [实战案例](#实战案例)

---

## NVSHMEM 简介

### 什么是 NVSHMEM？

**NVSHMEM** (NVIDIA Shared Memory) 是 NVIDIA 开发的基于 OpenSHMEM 标准的并行编程接口，专为 NVIDIA GPU 集群设计，提供高效且可扩展的 GPU 间通信能力。

**最新版本**：NVSHMEM 3.4.5（截至 2025 年）

### 核心特性

1. **PGAS 模型** (Partitioned Global Address Space)
   - 在多个 GPU 的内存中创建全局地址空间
   - 可通过细粒度操作访问远程 GPU 内存

2. **GPU 直接发起通信**
   - 无需 CPU 参与，GPU kernel 直接发起数据传输
   - 消除 CPU-GPU 同步开销

3. **异步通信**
   - 与 MPI 的阻塞式 send/recv 不同
   - 使用异步、单边通信原语

4. **易用的对称内存分配**
   - 提供简单接口分配跨 GPU 对称分布的内存

### NVSHMEM vs 传统 MPI

| 特性 | NVSHMEM | CUDA-aware MPI |
|------|---------|----------------|
| **通信发起者** | GPU kernel | CPU |
| **同步开销** | 无 CPU-GPU 同步 | 需要 CPU-GPU 同步 |
| **通信模式** | 单边（Put/Get） | 双边（Send/Recv） |
| **细粒度通信** | 支持（thread 级） | 不支持 |
| **编程模型** | PGAS | 消息传递 |

---

## 核心概念

### 1. PE (Processing Element)

在 NVSHMEM 中，每个 GPU 称为一个 **PE**。

```cpp
int my_pe = nvshmem_my_pe();        // 获取当前 PE 的 ID
int n_pes = nvshmem_n_pes();        // 获取总 PE 数量
```

**示例**：
```
4 个 GPU 集群：
  PE 0: GPU 0
  PE 1: GPU 1
  PE 2: GPU 2
  PE 3: GPU 3
```

### 2. 对称对象 (Symmetric Objects)

**对称对象**是在所有 PE 的对称堆（symmetric heap）上分配的内存，具有以下特性：

- 在所有 PE 上的**虚拟地址相同**
- 可以被任何 PE 的 GPU kernel 直接访问
- 使用 PE ID + 对称地址访问远程内存

### 3. 全局地址空间

```
PE 0 内存           PE 1 内存           PE 2 内存
┌──────────┐       ┌──────────┐       ┌──────────┐
│ 私有内存 │       │ 私有内存 │       │ 私有内存 │
├──────────┤       ├──────────┤       ├──────────┤
│ 对称堆   │◄─────►│ 对称堆   │◄─────►│ 对称堆   │
│ 0x7000.. │       │ 0x7000.. │       │ 0x7000.. │
│          │       │          │       │          │
│ data[0]  │       │ data[1]  │       │ data[2]  │
└──────────┘       └──────────┘       └──────────┘
     ▲                  ▲                  ▲
     │                  │                  │
     └──────────────────┴──────────────────┘
          可跨 PE 直接访问（相同虚拟地址）
```

---

## 对称内存详解

### 对称堆分配策略

NVSHMEM 支持两种对称堆分配策略：

#### 1. 动态分配（默认，使用 CUDA VMM）

```bash
# 启用动态分配（默认）
export NVSHMEM_DISABLE_CUDA_VMM=0
```

**优势**：
- 按需分配，灵活高效
- 使用 CUDA Virtual Memory Management (VMM) API
- 无需预先指定堆大小

#### 2. 静态分配

```bash
# 禁用 CUDA VMM，使用静态分配
export NVSHMEM_DISABLE_CUDA_VMM=1

# 指定对称堆大小（例如 4 GB）
export NVSHMEM_SYMMETRIC_SIZE=4294967296
```

**优势**：
- 预分配，启动时固定
- 适用于不支持 VMM 的旧架构

---

### 对称内存分配 API

#### nvshmem_malloc - 对称堆分配

```cpp
void* nvshmem_malloc(size_t size);
```

**特性**：
- **集体操作**（Collective）：所有 PE 必须同时调用
- 返回对称地址，在所有 PE 上虚拟地址相同
- 从对称堆分配（vs `malloc` 从私有堆分配）

**示例 1：基本分配**

```cpp
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void init_kernel(int* data, int my_pe) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        data[0] = my_pe * 100;  // 每个 PE 写入不同的值
    }
}

int main() {
    // 初始化 NVSHMEM
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // 在对称堆上分配 1024 个整数（所有 PE 必须调用）
    int* symmetric_data = (int*) nvshmem_malloc(1024 * sizeof(int));

    // 初始化数据
    init_kernel<<<1, 256>>>(symmetric_data, my_pe);
    cudaDeviceSynchronize();

    // 访问远程 PE 的数据（见后续示例）

    // 释放对称内存（集体操作）
    nvshmem_free(symmetric_data);

    nvshmem_finalize();
    return 0;
}
```

**关键点**：
- `nvshmem_malloc` 是**集体操作**，所有 PE 必须传递**相同的 size**
- 返回的指针在所有 PE 上虚拟地址相同
- 必须使用 `nvshmem_free` 释放（也是集体操作）

---

#### nvshmem_align - 对齐分配

```cpp
void* nvshmem_align(size_t alignment, size_t size);
```

**用途**：分配对齐的对称内存（例如 128 字节对齐）

**示例**：

```cpp
// 分配 128 字节对齐的 4096 字节对称内存
void* aligned_data = nvshmem_align(128, 4096);
```

---

#### nvshmem_free - 释放对称内存

```cpp
void nvshmem_free(void* ptr);
```

**注意**：
- **集体操作**，所有 PE 必须调用
- 只能释放由 `nvshmem_malloc` 或 `nvshmem_align` 分配的内存

---

### 静态对称数据（全局变量）

除了动态分配，还可以声明静态对称对象：

```cpp
// 声明对称全局变量
__device__ int symmetric_counter;

__global__ void increment_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        symmetric_counter++;
    }
}
```

**编译要求**：
```bash
nvcc -rdc=true -lcuda -lnvshmem -o app app.cu
```

- `-rdc=true`：启用可重定位设备代码（Relocatable Device Code）
- 允许 NVSHMEM 识别设备全局变量为对称对象

---

## API 使用示例

### 1. 初始化与终止

```cpp
#include <nvshmem.h>
#include <nvshmemx.h>

int main() {
    // 初始化 NVSHMEM
    nvshmem_init();

    int my_pe = nvshmem_my_pe();    // 获取当前 PE ID
    int n_pes = nvshmem_n_pes();    // 获取总 PE 数量

    printf("PE %d of %d\n", my_pe, n_pes);

    // ... 应用逻辑 ...

    // 终止 NVSHMEM
    nvshmem_finalize();
    return 0;
}
```

---

### 2. 远程内存访问（RMA）

#### nvshmem_put - 写入远程内存

```cpp
void nvshmem_TYPE_put(TYPE* dest, const TYPE* source, size_t nelems, int pe);
```

**功能**：将本地数据拷贝到远程 PE 的对称内存

**示例：单向数据传输**

```cpp
__global__ void put_kernel(int* symmetric_data, int my_pe, int n_pes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0 && my_pe == 0) {
        // PE 0 将数据写入 PE 1 的对称内存
        int local_data[4] = {100, 200, 300, 400};
        nvshmem_int_put(symmetric_data, local_data, 4, 1);
        // dest=PE 1 的 symmetric_data, source=local_data, nelems=4, pe=1
    }
}
```

**变体**：
- `nvshmem_int_p(dest, value, pe)`：写入单个值
- `nvshmem_int_put_nbi(...)`：非阻塞版本

---

#### nvshmem_get - 从远程读取

```cpp
void nvshmem_TYPE_get(TYPE* dest, const TYPE* source, size_t nelems, int pe);
```

**功能**：从远程 PE 的对称内存读取数据到本地

**示例：读取远程数据**

```cpp
__global__ void get_kernel(int* symmetric_data, int* local_buffer, int my_pe) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0 && my_pe == 1) {
        // PE 1 从 PE 0 读取数据
        nvshmem_int_get(local_buffer, symmetric_data, 4, 0);
        // dest=local_buffer, source=PE 0 的 symmetric_data, nelems=4, pe=0
    }
}
```

**变体**：
- `nvshmem_int_g(source, pe)`：读取单个值并返回
- `nvshmem_int_get_nbi(...)`：非阻塞版本

---

#### Block 级优化 API

NVSHMEM 提供 block 级 API，利用整个 thread block 并行拷贝：

```cpp
void nvshmemx_TYPE_put_block(TYPE* dest, const TYPE* source, size_t nelems, int pe);
```

**示例：Block 级 Put**

```cpp
__global__ void put_block_kernel(float* symmetric_data, float* local_data, int my_pe) {
    // 所有线程参与拷贝（如果目标 GPU 支持 P2P）
    if (my_pe == 0) {
        nvshmemx_float_put_block(symmetric_data, local_data, 1024, 1);
        // NVSHMEM 运行时会利用 block 内所有线程并行拷贝
    }
}
```

**优势**：
- 如果目标 GPU 通过 P2P 连接，运行时会利用所有线程并发拷贝
- 显著提高大数据传输的带宽

---

### 3. 同步操作

#### nvshmem_barrier_all - 全局屏障

```cpp
void nvshmem_barrier_all(void);
```

**功能**：所有 PE 同步，类似 MPI_Barrier

**示例：确保数据传输完成**

```cpp
__global__ void sync_example(int* data, int my_pe, int n_pes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        // PE 0 写入数据到所有其他 PE
        if (my_pe == 0) {
            for (int i = 1; i < n_pes; i++) {
                nvshmem_int_p(data, 42, i);
            }
        }

        // 所有 PE 等待数据传输完成
        nvshmem_barrier_all();

        // 现在所有 PE 都可以安全读取 data
        printf("PE %d: data = %d\n", my_pe, *data);
    }
}
```

---

#### nvshmem_quiet - 等待所有 RMA 完成

```cpp
void nvshmem_quiet(void);
```

**功能**：等待当前 PE 发起的所有 RMA 操作完成

**示例**：

```cpp
__global__ void quiet_example(int* data, int my_pe) {
    if (threadIdx.x == 0) {
        // 发起多个 Put 操作
        nvshmem_int_p(data, 100, (my_pe + 1) % 4);
        nvshmem_int_p(data + 1, 200, (my_pe + 2) % 4);

        // 等待所有 Put 完成
        nvshmem_quiet();

        // 现在可以安全地修改本地数据
    }
}
```

---

### 4. 集体通信（Collective Operations）

#### nvshmem_barrier - Team 屏障

```cpp
void nvshmem_barrier(nvshmem_team_t team);
```

**功能**：指定 team 内的 PE 同步

**示例**：

```cpp
nvshmem_barrier(NVSHMEM_TEAM_WORLD);  // 等价于 nvshmem_barrier_all()
```

---

#### nvshmem_broadcast - 广播

```cpp
void nvshmem_TYPE_broadcast(nvshmem_team_t team, TYPE* dest, const TYPE* source,
                            size_t nelems, int PE_root);
```

**功能**：将 root PE 的数据广播到 team 内所有 PE

**示例：PE 0 广播数据**

```cpp
__global__ void broadcast_kernel(int* data, int my_pe) {
    if (threadIdx.x == 0) {
        // PE 0 准备数据
        if (my_pe == 0) {
            data[0] = 999;
        }

        // PE 0 广播到所有 PE
        nvshmem_int_broadcast(NVSHMEM_TEAM_WORLD, data, data, 1, 0);
        // team, dest, source, nelems, PE_root

        // 现在所有 PE 的 data[0] == 999
    }
}
```

---

#### nvshmem_alltoall - 全交换

```cpp
void nvshmem_TYPE_alltoall(nvshmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems);
```

**功能**：每个 PE 向所有其他 PE 发送不同的数据块

**示例：4 PE 全交换**

```cpp
__global__ void alltoall_kernel(int* send_buf, int* recv_buf, int my_pe, int n_pes) {
    if (threadIdx.x == 0) {
        // 准备发送数据
        for (int i = 0; i < n_pes; i++) {
            send_buf[i] = my_pe * 100 + i;
        }

        // 全交换（每个 PE 发送 1 个元素到每个 PE）
        nvshmem_int_alltoall(NVSHMEM_TEAM_WORLD, recv_buf, send_buf, 1);

        // 结果：recv_buf[i] = i * 100 + my_pe
        for (int i = 0; i < n_pes; i++) {
            printf("PE %d recv from PE %d: %d\n", my_pe, i, recv_buf[i]);
        }
    }
}
```

**输出**（4 PEs）：
```
PE 0: [0, 100, 200, 300]
PE 1: [1, 101, 201, 301]
PE 2: [2, 102, 202, 302]
PE 3: [3, 103, 203, 303]
```

---

#### nvshmem_fcollect - 全收集

```cpp
void nvshmem_TYPE_fcollect(nvshmem_team_t team, TYPE* dest, const TYPE* source, size_t nelems);
```

**功能**：收集所有 PE 的数据到每个 PE（拼接）

**示例**：

```cpp
__global__ void fcollect_kernel(int* local_data, int* collected_data, int my_pe) {
    if (threadIdx.x == 0) {
        local_data[0] = my_pe * 10;

        // 收集所有 PE 的数据
        nvshmem_int_fcollect(NVSHMEM_TEAM_WORLD, collected_data, local_data, 1);

        // 每个 PE 的 collected_data: [0, 10, 20, 30, ...]
    }
}
```

---

#### Reduction 操作

```cpp
void nvshmem_TYPE_OPERATION_reduce(nvshmem_team_t team, TYPE* dest, const TYPE* source, size_t nreduce);
```

**支持的操作**：`sum`, `prod`, `min`, `max`, `and`, `or`, `xor`

**示例：求和归约**

```cpp
__global__ void reduce_kernel(int* local_val, int* result, int my_pe) {
    if (threadIdx.x == 0) {
        *local_val = my_pe + 1;  // PE i 的值为 i+1

        // 所有 PE 求和，结果在所有 PE 上
        nvshmem_int_sum_reduce(NVSHMEM_TEAM_WORLD, result, local_val, 1);

        // 4 PEs: result = 1 + 2 + 3 + 4 = 10
        printf("PE %d: sum = %d\n", my_pe, *result);
    }
}
```

---

### 5. Block 级集体操作

使用 `nvshmemx_*_block` API 可以让整个 thread block 参与集体操作：

```cpp
__global__ void block_collective_kernel(int* data, int my_pe) {
    // 所有线程参与 fcollect
    nvshmemx_int_fcollect_block(NVSHMEM_TEAM_WORLD, data, data, 256);
    // 比单线程版本快得多
}
```

---

## 性能对比：NVSHMEM vs MPI

### 实测性能数据

#### GROMACS 分子动力学模拟

来源：[Redesigning GROMACS Halo Exchange](https://arxiv.org/html/2509.21527v1)

| 系统规模 | GPU 数 | NVSHMEM (ns/day) | MPI (ns/day) | 提升 |
|---------|--------|------------------|--------------|------|
| 45k atoms | 4 | **1649** | 1126 | **+46%** |
| 180k atoms | 4 | **1103** | 1058 | **+4%** |
| 180k atoms | 8 | **1249** | 973 | **+28%** |

**结论**：
- 小系统：NVSHMEM 优势显著（**46%**）
- 大系统，少 GPU：MPI 略优（1-3%）
- 扩展性：NVSHMEM 在 8 GPU 上优势增大（**28%**）

---

#### Kokkos Conjugate Gradient Solver

来源：LLNL Sierra 超级计算机测试

- NVSHMEM 实现**显著优于** CUDA-aware MPI
- 代码量大幅减少
- GPU 直接发起通信，消除 CPU-GPU 同步瓶颈

---

### 性能优势分析

#### NVSHMEM 优势场景

1. **通信密集型**
   - 频繁的小消息通信
   - Halo exchange（边界交换）
   - 细粒度数据依赖

2. **高扩展性**
   - 多 GPU/节点场景
   - 强扩展性（固定问题规模，增加 GPU）

3. **GPU 主导计算**
   - 无需 CPU 参与通信
   - 减少 CPU-GPU 数据移动

#### MPI 优势场景

1. **计算密集型**
   - 大规模计算，少量通信
   - 单 GPU 处理大问题

2. **遗留代码**
   - 已有 MPI 代码库
   - 移植成本考虑

---

### 关键技术差异

| 特性 | NVSHMEM | CUDA-aware MPI |
|------|---------|----------------|
| **通信发起** | GPU kernel 直接发起 | CPU 发起，需要 CPU-GPU 同步 |
| **细粒度通信** | 支持 thread 级 Put/Get | 不支持（只能 block/grid 级） |
| **单边通信** | 天然支持（Put/Get） | 需要 MPI_Put/Get（较少支持） |
| **同步开销** | 无 CPU-GPU 同步 | 有显著同步开销 |
| **编程复杂度** | PGAS 模型，直观 | 消息传递，较复杂 |

---

## 实战案例

### 案例 1：分布式向量求和

**目标**：每个 PE 计算本地向量和，然后全局归约

```cpp
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>

#define N 1024

__global__ void vector_sum_kernel(float* local_vec, float* local_sum,
                                   float* global_sum, int my_pe) {
    __shared__ float shared_sum;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 每个线程计算部分和
    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        thread_sum += local_vec[i];
    }

    // Block 内归约
    atomicAdd(&shared_sum, thread_sum);
    __syncthreads();

    // 第一个线程保存本地和
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *local_sum = shared_sum;

        // 全局求和归约
        nvshmem_float_sum_reduce(NVSHMEM_TEAM_WORLD, global_sum, local_sum, 1);
    }
}

int main() {
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // 分配对称内存
    float* local_vec = (float*) nvshmem_malloc(N * sizeof(float));
    float* local_sum = (float*) nvshmem_malloc(sizeof(float));
    float* global_sum = (float*) nvshmem_malloc(sizeof(float));

    // 初始化向量（每个 PE 的值不同）
    cudaMemset(local_vec, 0, N * sizeof(float));
    float init_val = (float)(my_pe + 1);
    cudaMemcpy(local_vec, &init_val, sizeof(float), cudaMemcpyHostToDevice);

    // 执行求和
    vector_sum_kernel<<<4, 256>>>(local_vec, local_sum, global_sum, my_pe);
    cudaDeviceSynchronize();

    // 读取结果
    float result;
    cudaMemcpy(&result, global_sum, sizeof(float), cudaMemcpyDeviceToHost);

    if (my_pe == 0) {
        printf("Global sum: %f\n", result);
    }

    nvshmem_free(local_vec);
    nvshmem_free(local_sum);
    nvshmem_free(global_sum);

    nvshmem_finalize();
    return 0;
}
```

**编译**：
```bash
nvcc -rdc=true -I${NVSHMEM_HOME}/include -L${NVSHMEM_HOME}/lib \
     -lnvshmem -lcuda -o vector_sum vector_sum.cu
```

**运行**（4 GPUs）：
```bash
mpirun -np 4 ./vector_sum
```

---

### 案例 2：环形通信（Ring Communication）

**目标**：每个 PE 向下一个 PE 发送数据，形成环

```cpp
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void ring_kernel(int* data, int my_pe, int n_pes) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        // 初始化本地数据
        *data = my_pe * 100;

        // 向下一个 PE 发送（环形拓扑）
        int next_pe = (my_pe + 1) % n_pes;
        nvshmem_int_p(data, *data, next_pe);

        // 等待所有传输完成
        nvshmem_quiet();

        // 屏障同步
        nvshmem_barrier_all();

        // 现在 data 包含前一个 PE 的数据
        printf("PE %d received: %d (from PE %d)\n",
               my_pe, *data, (my_pe - 1 + n_pes) % n_pes);
    }
}

int main() {
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    int* data = (int*) nvshmem_malloc(sizeof(int));

    ring_kernel<<<1, 256>>>(data, my_pe, n_pes);
    cudaDeviceSynchronize();

    nvshmem_free(data);
    nvshmem_finalize();
    return 0;
}
```

**输出**（4 PEs）：
```
PE 0 received: 300 (from PE 3)
PE 1 received: 0 (from PE 0)
PE 2 received: 100 (from PE 1)
PE 3 received: 200 (from PE 2)
```

---

### 案例 3：分布式矩阵乘法（简化版）

```cpp
#include <nvshmem.h>
#include <nvshmemx.h>

#define M 1024  // 矩阵行数
#define K 1024  // 共享维度
#define N 1024  // 矩阵列数

__global__ void matmul_kernel(float* A, float* B_symmetric, float* C,
                              int my_pe, int n_pes) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // 访问本地 A 矩阵
        for (int k = 0; k < K; k++) {
            // 计算 B 矩阵分布在哪个 PE
            int b_pe = (col * K + k) / (K * N / n_pes);
            int b_offset = (col * K + k) % (K * N / n_pes);

            // 从远程 PE 读取 B 元素
            float b_val = nvshmem_float_g(B_symmetric + b_offset, b_pe);

            sum += A[row * K + k] * b_val;
        }

        C[row * N + col] = sum;
    }
}

int main() {
    nvshmem_init();

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // 分配内存
    float* A;
    cudaMalloc(&A, M * K * sizeof(float));  // 本地私有

    // B 矩阵分布在多个 PE 上（对称内存）
    int b_local_size = (K * N) / n_pes;
    float* B_symmetric = (float*) nvshmem_malloc(b_local_size * sizeof(float));

    float* C;
    cudaMalloc(&C, M * N * sizeof(float));

    // 初始化矩阵（省略）

    // 执行分布式矩阵乘法
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<grid, block>>>(A, B_symmetric, C, my_pe, n_pes);

    cudaDeviceSynchronize();

    // 清理
    cudaFree(A);
    nvshmem_free(B_symmetric);
    cudaFree(C);

    nvshmem_finalize();
    return 0;
}
```

---

### 案例 4：DeepEP 中的 NVSHMEM 使用

DeepEP 使用 NVSHMEM 实现低延迟的 MoE dispatch/combine：

```cpp
// 来自 DeepEP 的简化示例

// 1. 初始化 NVSHMEM
nvshmemx_init_attr_t attr;
attr.mpi_comm = &mpi_comm;
nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

// 2. 分配对称缓冲区
void* rdma_buffer = nvshmem_malloc(num_rdma_bytes);

// 3. GPU kernel 直接使用 NVSHMEM Put
__global__ void low_latency_dispatch_kernel(...) {
    // 计算目标 PE
    int target_pe = get_target_pe(expert_id);

    // 直接从 GPU 发起 RDMA 写入
    nvshmem_putmem_nbi(remote_addr, local_data, size, target_pe);
}

// 4. 非阻塞等待
nvshmem_quiet();  // 等待所有 Put 完成
```

**优势**：
- GPU kernel 直接发起 RDMA，无 CPU 参与
- 支持细粒度、高并发的通信
- 延迟降低到 **77-194 微秒**（vs MPI 的毫秒级）

---

## 环境变量配置

### 常用环境变量

```bash
# 禁用 CUDA VMM（使用静态对称堆）
export NVSHMEM_DISABLE_CUDA_VMM=1

# 设置静态对称堆大小（4 GB）
export NVSHMEM_SYMMETRIC_SIZE=4294967296

# 启用 InfiniBand IBGDA（GPU Direct Async）
export NVSHMEM_IB_ENABLE_IBGDA=1

# 设置每个 PE 的 RC QP 数量
export NVSHMEM_IBGDA_NUM_RC_PER_PE=24

# 设置 QP 深度
export NVSHMEM_QP_DEPTH=1024

# 禁用 P2P（强制使用 IB）
export NVSHMEM_DISABLE_P2P=0

# 禁用 NVLink SHARP
export NVSHMEM_DISABLE_NVLS=1

# 设置虚拟 lane（流量隔离）
export NVSHMEM_IB_SL=0

# 禁用多节点 NVLink
export NVSHMEM_DISABLE_MNNVL=1

# 设置最大 teams 数量
export NVSHMEM_MAX_TEAMS=7
```

---

## 编译与运行

### 编译命令

```bash
# 基本编译
nvcc -rdc=true -I${NVSHMEM_HOME}/include \
     -L${NVSHMEM_HOME}/lib -lnvshmem -lcuda \
     -o myapp myapp.cu

# 使用 MPI 启动（推荐）
nvcc -rdc=true -I${NVSHMEM_HOME}/include \
     -I${MPI_HOME}/include \
     -L${NVSHMEM_HOME}/lib -L${MPI_HOME}/lib \
     -lnvshmem -lmpi -lcuda \
     -o myapp myapp.cu
```

### 运行方式

#### 方式 1：使用 mpirun（推荐）

```bash
mpirun -np 4 ./myapp
```

#### 方式 2：使用 NVSHMEM 启动器

```bash
nvshmrun -np 4 ./myapp
```

---

## 最佳实践

### 1. 内存分配

- ✅ 使用 `nvshmem_malloc` 分配对称内存
- ✅ 确保所有 PE 传递相同的 `size`
- ❌ 不要混用 `cudaMalloc` 和 `nvshmem_malloc` 用于 RMA

### 2. 同步

- ✅ 使用 `nvshmem_quiet()` 等待自己的 Put/Get 完成
- ✅ 使用 `nvshmem_barrier_all()` 全局同步
- ❌ 不要假设 Put/Get 立即完成

### 3. 性能优化

- ✅ 使用 `nvshmemx_*_block` API 利用整个 block
- ✅ 使用非阻塞操作 (`*_nbi`) 隐藏延迟
- ✅ 批量传输大数据块，减少调用次数
- ❌ 避免频繁的小消息传输

### 4. 调试

```bash
# 启用详细日志
export NVSHMEM_DEBUG=TRACE

# 检查 NVSHMEM 配置
export NVSHMEM_INFO=1
```

---

## 参考资源

### 官方文档
- [NVSHMEM Developer Page](https://developer.nvidia.com/nvshmem)
- [NVSHMEM 3.4.5 Documentation](https://docs.nvidia.com/nvshmem/api/index.html)
- [NVSHMEM API Reference](https://docs.nvidia.com/nvshmem/api/api.html)
- [NVSHMEM Examples](https://docs.nvidia.com/nvshmem/api/examples.html)

### 学术论文
- [Redesigning GROMACS Halo Exchange (2025)](https://arxiv.org/html/2509.21527v1)
- [Dynamic Symmetric Heap Allocation in NVSHMEM](https://link.springer.com/chapter/10.1007/978-3-031-04888-3_12)
- [Evaluating One-sided Communication on CPUs and GPUs](https://dl.acm.org/doi/fullHtml/10.1145/3624062.3624182)

### GitHub 资源
- [NVIDIA/nvshmem](https://github.com/NVIDIA/NVSHMEM)
- [NVSHMEM Releases](https://github.com/NVIDIA/NVSHMEM/releases)

### 相关技术
- [PyTorch Symmetric Memory](https://docs.pytorch.org/docs/stable/symmetric_memory.html)
- [CUDA Virtual Memory Management (VMM)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-memory-management)

---

## 总结

NVSHMEM 提供了强大的 GPU 间通信能力：

✅ **核心优势**：
- GPU 直接发起通信，无 CPU 参与
- 对称内存简化分布式编程
- 细粒度、高并发通信
- 显著优于 MPI（通信密集型场景）

✅ **适用场景**：
- MoE 模型通信（如 DeepEP）
- 分布式深度学习
- 科学计算（分子动力学、CFD）
- 高性能图处理

✅ **关键 API**：
- `nvshmem_malloc` - 对称内存分配
- `nvshmem_put/get` - 远程内存访问
- `nvshmem_barrier_all` - 全局同步
- `nvshmem_*_reduce` - 集体归约

掌握 NVSHMEM，可以充分发挥多 GPU 集群的通信性能！🚀

---

**Sources:**
- [NVSHMEM Developer Page](https://developer.nvidia.com/nvshmem)
- [NVSHMEM 3.4.5 Documentation](https://docs.nvidia.com/nvshmem/api/introduction.html)
- [NVSHMEM Memory Management](https://docs.nvidia.com/nvshmem/api/gen/api/memory.html)
- [NVSHMEM Collective Operations](https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html)
- [Redesigning GROMACS with NVSHMEM](https://arxiv.org/html/2509.21527v1)
- [Dynamic Symmetric Heap Allocation](https://link.springer.com/chapter/10.1007/978-3-031-04888-3_12)
- [Evaluating One-sided Communication Performance](https://dl.acm.org/doi/fullHtml/10.1145/3624062.3624182)
- [PyTorch Symmetric Memory](https://docs.pytorch.org/docs/stable/symmetric_memory.html)

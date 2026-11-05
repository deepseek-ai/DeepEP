# DeepEP

[ [English](README.md) | 简体中文 ]

DeepEP (DeepEveryParallel) 是一款专为机器学习训练与推理设计的高性能通信库。它为混合专家模型（MoE）的分发（dispatch）与合并（combine）提供了高吞吐、低延迟的专家并行（EP）All-to-All GPU 算子，支持 FP8 分发，以及针对冗余专家的 NVLink 权重与梯度交换。此外，它还提供了针对流水线并行（PP）、上下文并行（CP）、数据并行（DP）以及远程内存访问（Engram）的实验性原语。通信算子在运行时通过 [DeepJIT](https://github.com/deepseek-ai/DeepJIT) 进行即时编译，配套扩展在安装过程中构建。

## 最新动态 (News)

- **昇腾（Ascend）版本发布**
  - 在华为昇腾 950 NPU 上提供相同的 API 与完整性能表现
  - 详情请查阅 [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend)

- **V2.5 版本发布**：
  - 将 `ElasticBuffer` 拆分为 `EPBuffer`、`EngramBuffer`、`PPBuffer` 和 `BucketBuffer`，共享 `BufferBase` 生命周期
  - 新增 `BufferAllocator`，用于在构建缓冲区前规划对称张量分配
  - 通过 `BucketBuffer` 新增批量 All-Gather、Reduce-Scatter 与 All-Reduce 操作，为普通 PyTorch 张量提供会话（session）支持
  - 新增 `EPBuffer.lb_prefetch_weights` 与 `EPBuffer.lb_reduce_grads`，用于动态冗余专家：在专家计算前通过 NVLink 预取专家权重与量化比例因子，随后在反向传播期间将冗余专家的 FP32 梯度累加至原始专家中。这些原语支持了由 [MoonEP](https://github.com/MoonshotAI/MoonEP) 与 [UltraEP](https://github.com/Dots-Infra/UltraEP) 探索的专家复制方案；API 与集成要求详见[专家负载均衡](#专家负载均衡)
  - 支持延迟 EP 后处理（deferred EP epilogues）、缓存展开布局以及专家间的零填充（zero padding）
  - 支持在 GPU 或 CPU 上存储多层 Engram，每层配有一个等待钩子（wait hook）
  - 彻底移除 V1 版本，包括其 API、NVSHMEM 后端及遗留文档。NVSHMEM 不再作为依赖项

- **V2 版本发布**：对专家并行进行了全面重构，支持更大规模的纵向扩展（scale-up）与横向扩展（scale-out）域，并引入了轻量级的 **NCCL Gin 后端**。

### 新特性

- **JIT 编译通信算子**：基于 DeepJIT 驱动
- **NCCL Gin 后端**
  - 轻量级设备端通信 API
  - 能够复用现有的 NCCL 通信器（communicators）
- **EPv2**
  - 将高吞吐与低延迟 API 统一整合为单一的 `EPBuffer` 接口，并为分组专家 GEMM 提供了展开布局（expanded layout）
  - 支持更大规模的 Scale-Up 与 Scale-Out 域
  - 解析式 SM 与 QP 数量计算 —— 无需再进行自动调优（auto-tuning）
  - 混合模式（hybrid）与直接模式（direct）均继续受支持
- **Engram**（实验性，基于 RDMA）
- **PP**（流水线并行，实验性，基于 RDMA）
- **Bucket 集合通信**（实验性）：用于 CP 和 DP（All-Gather、Reduce-Scatter 以及 All-Reduce）

### 注意事项

- EP 分发（dispatch）与合并（combine）需要 GPU SM 参与；不支持零 SM（zero-SM）RDMA EP
- Bucket、Engram 以及 PP 均为实验性功能
- Engram 需要提供 `ncclGinOptFlagsWarpGet` 支持的 NCCL 构建版本

## 快速上手 (Quick start)

### 环境要求

- Linux
- Python 3.10 及以上
- NVIDIA Hopper 或更新架构的 GPU
- 用于 DeepJIT 编译的 CUDA Toolkit 13.1 及以上版本，并支持目标 GPU
- 支持 `std::format` 的 C++20 编译器与标准库
- 支持 CUDA 的 PyTorch 2.10 及以上版本
- NCCL 2.32.3 及以上版本
- 节点内通信需支持 NVLink
- 节点间通信需支持 RDMA 网络

安装过程会针对 CUDA 与 NCCL 库构建主机端 C++ 扩展。GPU 算子由 DeepJIT 在运行时针对当前设备编译，因此安装过程不需要可见的 GPU 或 `TORCH_CUDA_ARCH_LIST`。请确保运行时 CUDA Toolkit 和主机端编译器可用。自动带宽检测使用 `nvidia-smi` 和 `ibstat`；`BucketBuffer` 当前要求 NVLink 和 RDMA 带宽均可被检测，即使通信组仅使用其中一种传输方式。

### 安装 NCCL 依赖

安装与您的 CUDA 环境匹配的 NCCL 软件包，以便 DeepEP 可以定位其头文件与库文件：

```bash
# CUDA 13.x
python -m pip install "nvidia-nccl-cu13>=2.32.3" --no-deps
# 对于 CUDA 12.x，请改用 nvidia-nccl-cu12
```

若使用自定义 NCCL 安装，请将 `EP_NCCL_ROOT_DIR` 设置为包含 `include/` 和 `lib/` 的目录。PyTorch 和 DeepEP 必须加载相同的 NCCL 动态库。使用早于 2.31 的 NCCL 头文件构建时，还要求编译期与运行期的 NCCL 版本完全一致。

### 安装步骤

```bash
# 初始化 DeepJIT 子模块
git submodule update --init --recursive

# 构建 wheel 包并安装到当前 Python 环境
bash install.sh
```

随后即可在 Python 项目中导入 `deep_ep`。

### 开发与测试

```bash
# 构建扩展并链接到源码树中
bash develop.sh

# 运行测试用例
python tests/ep/test_ep.py
python tests/bucket/test_all_gather.py
python tests/bucket/test_reduce_scatter.py
python tests/bucket/test_all_reduce.py
python tests/buffer/test_allocation.py
python tests/ep/test_prefetch_weights.py
python tests/ep/test_reduce_grads.py
python tests/engram/test_engram.py
python tests/pp/test_pp.py
```

测试脚本需要 NumPy，并在本地启动 GPU 工作进程。对于多节点测试，使用共享的 `MASTER_ADDR` 和 `MASTER_PORT` 在每个节点上启动相同脚本，并将 `WORLD_SIZE` 设置为节点数，`RANK` 设置为节点索引。这些是 [deep_ep/utils/envs.py](deep_ep/utils/envs.py) 中 `init_dist` 的约定惯例；如有需要可针对您的集群进行调整。除非已设置，否则测试默认使用 `NCCL_IB_SL=1` 和 `EP_OVERRIDE_RDMA_SL=1`。权重预取与梯度规约测试使用单个 NVLink 域；PP 测试需要纯 RDMA 组。

## 接口与使用示例 (Interfaces and examples)

### 缓冲区初始化 (Buffer initialization)

高吞吐与低延迟 EP 操作共享统一的 `EPBuffer` 接口。直接使用 MoE 配置初始化缓冲区；SM 和 QP 数量通过解析式估算，并可在每次调用时进行覆盖。

为每个 EP 组创建并复用缓冲区。所有 rank 必须对 `num_max_tokens_per_rank` 达成一致；选择一个涵盖预期训练、Prefill 和 Decoding 批次的通用容量。下面的示例管理单个 EP 组。在替换其缓冲区前请完成所有未决操作。

```python
import torch.distributed as dist
from typing import Optional

from deep_ep import EPBuffer

# 通信缓冲区（将在运行时分配）
_buffer: Optional[EPBuffer] = None

# 用于通信算子的 SM 数量（将在创建缓冲区时设置）
_num_comm_sms: int = 0


def get_buffer(group: dist.ProcessGroup,
               num_max_tokens_per_rank: int,
               hidden: int,
               num_topk: int,
               num_experts: int,
               use_fp8_dispatch: bool = False) -> EPBuffer:
    """初始化或获取用于 EP 通信的 EPBuffer。"""
    global _buffer, _num_comm_sms

    # 检查是否可以复用已有缓冲区
    required_bytes = EPBuffer.get_buffer_size_hint(
        group, num_max_tokens_per_rank, hidden,
        num_topk=num_topk, use_fp8_dispatch=use_fp8_dispatch,
    )
    if _buffer is not None and _buffer.group == group and _buffer.num_bytes >= required_bytes:
        _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)
        return _buffer

    # 使用 MoE 配置分配新缓冲区
    _buffer = EPBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=use_fp8_dispatch,
    )

    # 根据网络拓扑与通信量估算 SM 数量
    # 也可以在 dispatch/combine 调用中手动指定 `num_sms` 进行覆盖
    _num_comm_sms = _buffer.get_theoretical_num_sms(num_experts, num_topk)

    return _buffer
```

### 模型训练与推理中的使用示例

训练、推理 Prefill 以及推理 Decoding 均使用相同的 `EPBuffer` 分发（dispatch）和合并（combine）API。该示例针对分组专家 GEMM 使用展开布局，并将前向后处理推迟至框架等待其结果时执行。将 `expert_alignment` 设置为分组 GEMM 的 token 对齐量；对于 DeepGEMM，请使用 `deep_gemm.get_mk_alignment_for_contiguous_layout()`。

`do_cpu_sync=True` 可获取精确的输出大小和 CPU 侧专家计数，常用于训练与 Prefill。在 Decoding 等使用 GPU 侧接收计数时将其设为 `False`：输出将分配为配置的容量，且专家 GEMM 必须使用 `handle.psum_num_recv_tokens_per_expert` 来确定有效的专家范围。新的路由决策需要全新的 dispatch；仅凭 decoding 并不能使旧的路由句柄可复用。

```python
import torch
from typing import Optional, Tuple, Union

from deep_ep import EPBuffer, EPHandle, EventHandle, EventOverlap


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int,
                     num_max_tokens_per_rank: int,
                     expert_alignment: int = 1,
                     do_cpu_sync: bool = True,
                     previous_event: Optional[EventHandle] = None) -> EventOverlap:
    """
    MoE 分发：将 token 路由到所有 rank 上的相应专家。
    支持 BF16 与 FP8 输入（对于 FP8，x 为包含 [data, scale_factors] 的元组）。
    在返回的 event 上等待即可获取展开张量与路由句柄。
    """
    global _buffer, _num_comm_sms

    return _buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=expert_alignment,
        num_sms=_num_comm_sms,
        previous_event=previous_event,
        async_with_compute_stream=True,
        allocate_on_comm_stream=previous_event is not None,
        do_cpu_sync=do_cpu_sync,
        do_expand=True,
        do_zero_padding=True,
        use_tma_aligned_col_major_sf=True,
        defer_epilogue=True,
    )


def dispatch_backward(grad_recv_x: torch.Tensor,
                      grad_recv_topk_weights: torch.Tensor,
                      handle: EPHandle,
                      bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    """MoE 分发的反向传播实际上是一个合并（combine）操作。"""
    global _buffer, _num_comm_sms

    combined_grad_x, combined_grad_topk_weights, event = _buffer.combine(
        grad_recv_x,
        handle=handle,
        bias=bias,
        topk_weights=grad_recv_topk_weights,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
    )

    return combined_grad_x, combined_grad_topk_weights, event


def combine_forward(x: torch.Tensor,
                    handle: EPHandle,
                    bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                    previous_event: Optional[EventHandle] = None) -> EventOverlap:
    """MoE 合并：将各专家的输出规约还原回其原始 rank。"""
    global _buffer, _num_comm_sms

    return _buffer.combine(
        x,
        handle=handle,
        bias=bias,
        num_sms=_num_comm_sms,
        previous_event=previous_event,
        async_with_compute_stream=True,
        allocate_on_comm_stream=previous_event is not None,
        defer_epilogue=True,
    )


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: EPHandle) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    """MoE 合并的反向传播实际上是一个分发（dispatch）操作。"""
    global _buffer, _num_comm_sms

    grad_x, _, _, _, event = _buffer.dispatch(
        grad_combined_x,
        handle=handle,
        num_sms=_num_comm_sms,
        async_with_compute_stream=True,
        do_expand=True,
        do_zero_padding=True,
        use_tma_aligned_col_major_sf=True,
    )

    return grad_x, event
```

为了实现通信与计算重叠（Overlap），先发起通信，然后在专家计算需要其结果的关键点进行等待：

```python
# Decoding 阶段使用 do_cpu_sync=False，配合 GPU 侧专家计数
event = dispatch_forward(...)

# ... 此处可执行一些独立的计算任务 ...

# 执行延迟分发后处理并获取其结果
recv_x, _, recv_topk_weights, handle = event.current_stream_wait()

# ... 执行专家 GEMM 并应用 recv_topk_weights 生成 expert_output ...
event = combine_forward(expert_output, handle)

# ... 此处可执行一些独立的计算任务 ...
combined_x, _ = event.current_stream_wait()
```

展开后的 `recv_topk_weights` 为一维张量，每个专家行对应一个权重值。专家计算在 combine 之前应用这些路由权重；combine 则对传入的专家输出进行累加。Combine 的输入必须为 BF16。`bias` 可以在 combine 后处理期间加上共享专家（shared-expert）输出或残差。

训练过程会保存前向传播的 `handle`，以供 `combine_backward` 和 `dispatch_backward` 使用。上述反向辅助函数会返回张量与 event；在使用张量之前必须在对应 event 上等待。缓存的 dispatch 会直接复用已保存的展开布局，无需再次进行接收计数的 CPU 同步。在句柄的所有调用完成之前，请保持原始 `topk_idx` 不变。

### 延迟后处理 (Deferred epilogues)

Dispatch 与 Combine 支持将 `defer_epilogue=True` 与 `async_with_compute_stream=True` 搭配使用。在此模式下，调用将直接返回一个 `EventOverlap`。调用 `.wait()` 会在当前流上执行延迟的后处理，对于 dispatch 返回 `(recv_x, recv_topk_idx, recv_topk_weights, handle)`，对于 combine 返回 `(combined_x, combined_topk_weights)`。

返回的结果由首次 wait 产生，因此应妥善保留供后续计算使用。如果没有其他重叠计算任务，请立即调用 `.wait()`。`do_zero_padding=True` 会清除专家之间的对齐间隙；在非 CPU 同步模式下，它并不会将未使用的输出容量转化为有效数据。

如果在通信调用前已有独立的计算排入队列，请在入队该计算前通过 `_buffer.capture()` 捕获输入就绪事件，并作为 `previous_event` 传入。在此情况下，根据 EP 要求，前向辅助函数会设置 `allocate_on_comm_stream=True`。延迟后处理依然在 `.wait()` 时于当前流上执行；这也允许在此处消费由中间计算产生的 combine `bias`。

### Bucket 集合通信原语（实验性）

`BucketBuffer` 为 CP 和 DP 工作负载提供批量的 `all_gather`、`reduce_scatter` 以及 `all_reduce` 操作。在常规用法中，输入必须驻留在 bucket 自身的存储空间中。在创建缓冲区前使用 `BufferAllocator` 规划张量。当构建缓冲区时，规划中的 meta 张量将成为缓冲区内的 CUDA 视图（view）。

```python
import torch

from deep_ep import BufferAllocator, BucketBuffer

# group 为已初始化的 CP 或 DP 进程组
allocation_plan = BufferAllocator()
x = allocation_plan.allocate((group.size(), 1024), torch.float32)
buffer = BucketBuffer(group, allocation_plan)

# 将每个 rank 的本地分片收集（gather）到已注册的张量中
x[group.rank()].fill_(group.rank())
gathered_x = buffer.all_gather(x[group.rank()]).wait()

# Reduce-scatter 与 all-reduce 使用缓冲区内的 FP32 张量
x.fill_(1)
shard = buffer.reduce_scatter(x).wait()
x.fill_(1)
reduced_x = buffer.all_reduce(x).wait()
```

所有 rank 必须使用相同的分配规划。每个集合通信原语亦支持接收张量列表，`.wait()` 将返回相应的张量或张量列表。返回结果为缓冲区的视图；All-Gather 返回展平的视图。Reduce-Scatter 和 All-Reduce 默认接收 FP32 输入并累加分量；可以使用 `scale` 对结果进行缩放。Reduce-Scatter 还支持 `comm_precision="bf16"`，在保持 FP32 输入与输出张量的同时进行较低精度的通信。

若要使用在 bucket 外部署的张量，请将集合通信调用包装在 `buffer.session()` 中。Session 会负责管理临时 bucket 存储以及必要的中转暂存拷贝：

```python
with buffer.session():
    reduced_x = buffer.all_reduce(external_x).wait()
    # 在复用 session 存储前消费或拷贝 reduced_x
```

纯 NVLink 的 All-Gather 路径亦支持外部源，并显式传入 bucket 内的 `dsts`；对于常规的 bucket 外部输入，请使用 session。

这三种集合通信原语均支持 NVLink、RDMA 以及混合域。NVLink 规约需要 NCCL multimem 支持。All-Gather 由复制引擎驱动（`num_sms=0`）；Reduce-Scatter 与 All-Reduce 使用 GPU SM。示例请参见 [Bucket 测试](tests/bucket) 与 [分配测试](tests/buffer/test_allocation.py)，包括向多个通信组注册的使用方式。

### 专家负载均衡 (Expert load balancing)

动态专家复制（Dynamic expert replication）将高负载专家的计算分摊到其他额外的 GPU 上。[MoonEP](https://github.com/MoonshotAI/MoonEP) 探索了通过动态冗余专家、权重预取和梯度规约实现的在线规划。[UltraEP](https://github.com/Dots-Infra/UltraEP)（[论文](https://arxiv.org/abs/2606.04101)）根据门控（post-gating）后的实时负载规划复制与 token 重路由，并将专家复制限制在 NVLink 域内。这些工作启发了 `EPBuffer` 提供的冗余专家通信原语。

- `lb_prefetch_weights(redundant_expert_weights, expert_weights, redundancy_mapping)`：在专家计算前，将原始专家权重推送到指定的冗余槽位。它支持单个张量或相匹配的非空张量序列，因此权重与量化比例因子可一同传输而无需数据类型转换。
- `lb_reduce_grads(redundant_expert_grads, expert_grads, redundancy_mapping)`：通过 NVLink 读取冗余专家的 FP32 梯度，并就地累加到原始专家的梯度中。使用与前向传播相同的映射，并在规约前将 `expert_grads` 初始化为本地分量或零。

调用方负责提供复制规划、将 token 重路由到选定的专家实例，并管理专家 GEMM。两项交换操作均为各自 NVLink 域内的集合通信：该域内的每个 rank 都必须调用它们，包括未分配冗余槽位的 rank。它们在通信流上异步运行并返回 `EventOverlap`；在消费其结果之前请调用 `.wait()`。`previous_event` 可以指定输入就绪的时机，`num_sms=0` 则采用解析式 SM 估算。

使用 `lb_allocation_plan_or_num_bytes` 预留冗余权重与梯度存储空间，可传入 `BufferAllocator` 规划或对齐的字节数。该负载均衡（LB）区域独立于 EP 分发/合并缓冲区。冗余张量必须驻留其中；原始专家权重与梯度则可使用常规的 CUDA 分配。在 NVLink 域内的每个 rank 上，规划必须具备相同的形状与分配顺序。

`redundancy_mapping` 必须是形状为 `[num_nvlink_ranks, num_redundant_experts]` 的连续 CUDA int32 张量，且在该域内每个 rank 上的内容必须完全相同。条目 `[r, c]` 将某个专家分配给域本地 rank `r` 上的冗余槽位 `c`，未使用的槽位则设为 `-1`。专家 ID 格式为 `owner_rank * num_local_experts + local_expert_idx`（使用域本地 rank 索引）。副本应分配给对等节点；rank 自身的专家不得出现在其自身的冗余槽位中。未使用的槽位将被自动跳过。

所有权重与梯度张量必须为连续的 CUDA 张量。权重张量的首维为 `num_local_experts`，冗余张量的首维为 `num_redundant_experts`；两者的后续维度形状可以不同，只要每对匹配项中每个专家的字节数相同且为 `deep_ep.get_num_tma_alignment()`（32 字节）的倍数即可。域内的所有 rank 必须具有相同的 `num_local_experts`。梯度规约接收二维 FP32 张量；需将每个专家的参数展平成行，并保持相同的字节对齐。

```python
import torch

from deep_ep import BufferAllocator, EPBuffer

# 在每个 rank 上使用相同的规划；示例中每个专家行为 1024 个元素
lb_plan = BufferAllocator()
redundant_weights = lb_plan.allocate((num_redundant_experts, 1024), torch.bfloat16)
redundant_grads = lb_plan.allocate((num_redundant_experts, 1024), torch.float32)
buffer = EPBuffer(
    group,
    num_max_tokens_per_rank=num_max_tokens_per_rank,
    hidden=hidden,
    num_topk=num_topk,
    lb_allocation_plan_or_num_bytes=lb_plan,
)

# expert_weights: 连续 CUDA BF16 [num_local_experts, 1024]
# redundancy_mapping: 由调用方的复制规划器提供
weights_ready = buffer.lb_prefetch_weights(
    redundant_weights, expert_weights, redundancy_mapping,
)
# ... 独立的计算可与此数据交换重叠 ...
weights_ready.wait()
# ... 使用预取的权重执行专家计算 ...

# 在反向传播写入 redundant_grads 与本地 expert_grads 后：
grads_ready = buffer.lb_reduce_grads(
    redundant_grads, expert_grads, redundancy_mapping,
)
# ... 独立的反向计算可与此规约重叠 ...
grads_ready.wait()
# expert_grads 现在包含来自所有冗余实例的累加贡献
```

在相应的数据交换完成之前，请保持映射与张量存储空间有效。如果冗余存储空间在不同网络层或 microbatch 间复用，需在反向计算前恢复所需的权重，并在复用其输入前完成梯度规约。规约操作不会自动清除冗余梯度；在下一次累加前请就地覆盖或清零。完整分配与正确性验证示例请参见[权重预取测试](tests/ep/test_prefetch_weights.py)和[梯度规约测试](tests/ep/test_reduce_grads.py)。

### Engram（实验性）

`EngramBuffer` 支持 GPU/CPU 存储以及通过 RDMA 进行的多层数据拉取。使用 `get_storage_size_hint` 与 `get_theoretical_config` 确定缓冲区及其 QP 大小，然后通过 `set_config` 和 `write` 填充各层数据表。

`fetch(indices)` 接收形状为 `[num_layers, num_tokens, num_entries_per_token]` 的 int32 张量，并为每一层返回一个完成钩子。在使用该层拉取的数据之前，请调用对应的钩子。BF16、FP8 以及 CPU 存储示例请参阅 [Engram 测试](tests/engram/test_engram.py)。

### 流水线并行（实验性）

`PPBuffer` 在纯 RDMA 流水线组中的相邻 rank 之间提供 `send(x, dst_rank_idx)` 和 `recv(x, src_rank_idx)`。在构建时设置 `num_max_tensor_bytes` 和 `num_max_inflight_tensors` 以预留发送/接收槽位。输入与输出必须为连续的 CUDA 张量，且指针与大小均需满足 32 字节对齐。用法详见 [PP 测试](tests/pp/test_pp.py)。

### 环境变量配置

请在导入 `deep_ep` 和创建缓冲区之前设置运行时环境变量。除另有说明外，布尔标志均使用 `0`/`1`。以下默认值假定未打包编译期默认值。

**运行时与网络配置**

| 变量名 | 默认值 | 作用说明 |
| --- | --- | --- |
| `EP_BUFFER_DEBUG` | 未设置 | 设为 `1` 可打印初始化、拓扑、缓冲区大小以及 SM 估算等诊断信息。保持未设置以禁用所有诊断；Python 侧检查亦将字符串 `"0"` 视为启用。 |
| `EP_SUPPRESS_NCCL_CHECK` | `0` | 跳过导入时对重复 NCCL 库以及与所选安装二进制一致性的检查。不会绕过 C++ 设备通信器的兼容性检查。 |
| `EP_AVOID_RECORD_STREAM` | `0` | 对于 EP 和 bucket 操作，在完成事件中保留张量，而非调用 `record_stream`。请保持该事件存活直到通信完成。 |
| `EP_REUSE_NCCL_COMM` | `1` | 当 PyTorch 进程组后端公开 `_comm_ptr` 时复用其 NCCL 通信器；否则创建由 DeepEP 管理的通信器。 |
| `EP_DEFAULT_RDMA_SL` | 未设置 | 当缓冲区未提供 `sl_idx` 时，设置 Gin 流量类别/服务等级（SL）。若两者均未设置，则使用 NCCL 的默认值。 |
| `EP_OVERRIDE_RDMA_SL` | 未设置 | 同时覆盖 `EP_DEFAULT_RDMA_SL` 与缓冲区的 `sl_idx`。 |
| `EP_DISABLE_GIN` | `0` | 仅使用 NVLink 时跳过 Gin 初始化。RDMA 操作必须使用 Gin；该标志不会选择其他 RDMA 后端。 |
| `EP_NUM_MAX_LOCAL_RANKS` | `16` | 仅限 Engram：在混合模式下估算用于 `NCCL_WIN_STRIDE` 大小计算的注册存储量。这仅为尺寸估算，而非 rank 数量限制。 |

**JIT 编译配置**

DeepJIT 会优先读取 `EP_JIT_*`，然后将对应的 `DJ_JIT_*` 变量作为全局后备选项。请在首次算子编译前完成配置。显式指定的 `EP_JIT_*` 值（包括打包的默认值）优先级高于 `DJ_JIT_*`。

| 变量名 | 默认值 | 作用说明 |
| --- | --- | --- |
| `EP_JIT_DEBUG` | `0` | 启用编译器命令与算子加载诊断、PTXAS 输出、源行信息以及 PTX/SASS 转储。 |
| `EP_JIT_CACHE_DIR` | `$HOME/.dj` | 缓存根目录或以冒号分隔的根目录列表。按顺序检索所有根目录，并将新编译的产物写入首个目录中。 |
| `EP_JIT_NVCC_COMPILER` | 检测到的 toolkit 的 `bin/nvcc` | 覆盖 NVCC 可执行文件路径；但仍需能够发现有效的 CUDA Toolkit 根目录。 |
| `EP_JIT_CPP_STANDARD` | `20` | 传递给 NVCC 的 C++ 标准版本。DeepEP 要求 C++20 或更新版本。 |
| `EP_JIT_PRINT_COMPILER_COMMAND` | `0` | 打印编译器与反汇编器命令。 |
| `EP_JIT_PRINT_LOAD_TIME` | `0` | 打印算子二进制加载耗时。 |
| `EP_JIT_PTXAS_VERBOSE` | `0` | 启用并打印详细的 PTXAS 编译输出。 |
| `EP_JIT_CHECK_NO_SPILLS` | `0` | 拒绝包含寄存器溢出（register spills）的编译算子。 |
| `EP_JIT_CHECK_NO_LOCAL_MEMORY` | `0` | 拒绝使用局部内存（local memory）的编译算子。 |
| `EP_JIT_WITH_LINEINFO` | `0` | 嵌入用于性能分析的源代码行信息。 |
| `EP_JIT_DUMP_ASM` | `0` | 在缓存未命中时同时生成 PTX 和 SASS 构件。 |
| `EP_JIT_DUMP_PTX` | `0` | 在缓存未命中时生成 PTX 构件。 |
| `EP_JIT_DUMP_SASS` | `0` | 在缓存未命中时生成 SASS 构件；需要工具包中的 `cuobjdump`。 |
| `EP_GIN_GDAKI_DEBUG` | `0` | 编译 JIT 算子时启用 NCCL Gin GDAKI 设备级调试。 |

DeepJIT 依次通过 `CUDA_HOME`、`CUDA_PATH`、`PATH` 中的 `nvcc` 以及 `/usr/local/cuda` 发现 CUDA Toolkit。

**构建与依赖发现配置**

| 变量名 | 默认值 | 作用说明 |
| --- | --- | --- |
| `EP_NCCL_ROOT_DIR` | 自动检测 | 包含 `include/` 和 `lib/` 的 NCCL 安装路径，在构建期与导入期使用。优先级高于 `NCCL_DIR`，其后为 NVIDIA Python 包发现机制。 |
| `EP_NUM_TOPK_IDX_BITS` | `64` | 构建期的 top-k 索引位宽（`32` 或 `64`）。请针对路由索引使用导出的 `deep_ep.topk_idx_t`。更改此值需要重新构建扩展。 |

当在软件包构建过程中设置时，这些变量将存储为导入期默认值：`EP_JIT_CACHE_DIR`、`EP_JIT_PRINT_COMPILER_COMMAND`、`EP_JIT_CPP_STANDARD`、`EP_NUM_TOPK_IDX_BITS`、`EP_NCCL_ROOT_DIR`、`EP_DEFAULT_RDMA_SL` 以及 `EP_OVERRIDE_RDMA_SL`。导入时已存在的环境变量具有更高优先级。在运行时覆盖 `EP_NUM_TOPK_IDX_BITS` 不会改变已编译的索引类型。

**测试性能分析配置**

这些标志仅影响 [deep_ep/utils/testing.py](deep_ep/utils/testing.py) 中的 `bench_kineto`，不影响生产通信：

| 变量名 | 默认值 | 作用说明 |
| --- | --- | --- |
| `EP_USE_NVIDIA_TOOLS` | `0` | 当使用 Nsight 或 Compute Sanitizer 时跳过内部分析器。启用此项时报告的内部计时为占位符。 |
| `EP_DISABLE_BARRIER_PROFILING` | `0` | 禁用在每次分析迭代前插入的屏障与延迟。 |

## 网络配置 (Network configurations)

DeepEP 已经过 InfiniBand 网络的充分测试。但在理论上它亦兼容 RoCE（RDMA over Converged Ethernet）网络。

### 流量隔离

InfiniBand 通过虚拟通道（Virtual Lanes / VL）支持流量隔离。

为防止不同类型的流量相互干扰，我们建议按如下方式将工作负载隔离到不同的虚拟通道中：

- 专家并行（EP）工作负载
- 其他工作负载

通过缓冲区的 `sl_idx` 参数选择 RDMA 服务等级（SL）。优先级顺序为：`EP_OVERRIDE_RDMA_SL` > `sl_idx` > `EP_DEFAULT_RDMA_SL` > NCCL 默认值。网络的 SL 到 VL 映射将决定承载该流量的具体虚拟通道。

### 自适应路由

自适应路由（Adaptive Routing）是由 InfiniBand 交换机提供的高级路由功能，能够跨多条路径均匀分布流量。尽管自适应路由会引入额外的延迟，但我们仍建议在所有网络负载条件下均启用该功能。

### 拥塞控制

对于追求最大带宽的工作负载，我们建议在验证网络配置后禁用拥塞控制。如果拥塞不可避免，请将这些工作负载放置在较低优先级的虚拟通道上。DeepEP 本身不会在网络交换架构上配置拥塞控制。

### PCI 原子操作模式

若硬件支持，我们建议使用以下命令配置网卡的 `PCI_ATOMIC_MODE`，以提升 RDMA 原子操作性能：

```bash
# 将 mlx5_0 替换为目标网卡设备
sudo mlxconfig -y -d mlx5_0 set PCI_ATOMIC_MODE=4
```

## 实验性分支 (Experimental branches)

以下链接描述了独立实现与前沿研究分支。其特性与依赖项仅适用于对应分支；在将其与当前 API 集成前请查阅具体分支说明：

- [Zero-copy](https://github.com/deepseek-ai/DeepEP/pull/453)
    - 消除 PyTorch 张量与通信缓冲区之间的拷贝开销，显著降低常规算子的 SM 占用
    - 本 PR 由**腾讯网络平台部（Tencent Network Platform Department）**贡献
- [Eager](https://github.com/deepseek-ai/DeepEP/pull/437)
    - 采用低延迟协议，消除 RDMA 原子操作引入的额外 RTT 延迟
- [Hybrid-EP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
    - 采用 TMA 指令的新后端实现，以实现极低的 SM 占用和更大规模的 NVLink 域支持
    - 针对单批次（single-batch）场景的细粒度通信-计算重叠
    - 针对非 NVLink 环境的 PCIe 算子支持
    - NVFP4 数据类型支持
- [AntGroup-Opt](https://github.com/deepseek-ai/DeepEP/tree/antgroup-opt)
    - 该优化系列由**蚂蚁集团网络平台部（AntGroup Network Platform Department）**贡献
    - [Normal-SMFree](https://github.com/deepseek-ai/DeepEP/pull/347)：通过将通信算子执行与网卡 token 传输解耦，从 RDMA 路径中消除 SM 占用，释放 SM 用于计算
    - [LL-SBO](https://github.com/deepseek-ai/DeepEP/pull/483)：通过信号机制将 Down GEMM 计算与 Combine Send 通信重叠，降低端到端延迟
    - [LL-Layered](https://github.com/deepseek-ai/DeepEP/pull/500)：利用导轨优化转发（rail-optimized forwarding）与数据合并优化跨节点 LL 算子通信，降低通信延迟
- [Mori-EP](https://github.com/deepseek-ai/DeepEP/tree/mori-ep)
    - 由 [MORI](https://github.com/ROCm/mori) 后端驱动的 ROCm/AMD GPU 支持（低延迟模式）
- [nvDev](https://github.com/deepseek-ai/DeepEP/tree/nvDev)
    - 基于 V2 的分支，集成了最新 CUDA 特性（如 Compute Fabric Transport / CFT），为小 token 场景带来更优延迟表现

## 社区分支与衍生项目 (Community forks)

- [uccl/uccl-ep](https://github.com/uccl-project/uccl/tree/main/ep) —— 支持在异构 GPU（如 NVIDIA、AMD）和网卡（如 EFA、Broadcom、CX7）上运行 DeepEP
- [Infrawaves/DeepEP_ibrc_dual-ports_multiQP](https://github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP) —— 在 IBRC 传输中添加多 QP 解决方案与双端口网卡支持
- [antgroup/DeepXTrace](https://github.com/antgroup/DeepXTrace) —— 用于高效、精准定位慢节点（slow ranks）的诊断分析工具
- [ROCm/mori](https://github.com/ROCm/mori) —— AMD 针对关键性能 AI 工作负载（如 Wide EP、KVCache 传输、集合通信）的下一代通信库

## 致谢 (Acknowledgement)

DeepEP 构建于 [NCCL](https://github.com/nvidia/nccl) Gin 后端之上。感谢 [@sjeaugey](https://github.com/sjeaugey)、[@pakmarkthub](https://github.com/pakmarkthub)、[@sb17v](https://github.com/sb17v)、[@xiaofanl-nvidia](https://github.com/xiaofanl-nvidia) 以及 NCCL 团队的大力支持！

我们亦对 [MoonEP](https://github.com/MoonshotAI/MoonEP) 和 [UltraEP](https://github.com/Dots-Infra/UltraEP) 在动态专家复制以及支撑专家负载均衡的权重/梯度交换方面的工作表示诚挚感谢。

## 许可证 (License)

本代码仓库依据 [MIT 许可证](LICENSE) 发布。

## 引用 (Citation)

```bibtex
@misc{deepep2025,
      title={DeepEP: an efficient expert-parallel communication library},
      author={Chenggang Zhao and Shangyan Zhou and Liyue Zhang and Chengqi Deng and Zhean Xu and Yuxuan Liu and Kuai Yu and Jiashi Li and Liang Zhao},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepEP}},
}
```

---

*声明：本中文文档由社区志愿者维护，仅供参考。如有出入，请以[英文原版](README.md)为准。*

# 数学与直觉

这一页的任务不是炫公式，而是把 DeepEP 里的那些“看起来像天书”的符号，翻译成你能一眼看懂的现实意义。

## 1. 每个通信系统都必须先回答两个问题

在 DeepEP 真正搬运任何一个字节之前，它必须先回答：

1. **每个 token 到底要去哪里？**
2. **每条路到底要预留多少 buffer？**

你看到的大部分复杂代码，本质上都只是这两个问题的高性能实现。

## 2. Dispatch layout 的本质：集合问题

设：

- `T`：token 数；
- `K`：top-k 数；
- `E`：expert 总数；
- `R`：rank 总数；
- `E_local = E / R`：每个 rank 持有的 expert 数。

如果 token `t` 的第 `j` 个选择 expert 是 `e_{t, j}`，那么它所属的 rank 就是：

$$
\operatorname{rank}(e_{t,j}) = \left\lfloor \frac{e_{t,j}}{E_{local}} \right\rfloor.
$$

随后，DeepEP 会构造 token `t` 的目标 rank 集合：

$$
D_t = \{\operatorname{rank}(e_{t,j}) \mid e_{t,j} \neq -1\}.
$$

注意这个词是 **集合**。意思是：如果一个 token 选中了两个 expert，而这两个 expert 恰好都在同一个 rank 上，那么这个 rank 只算一次。

接着定义指标量：

$$
I_{t,r} = \mathbf{1}[r \in D_t].
$$

那么 layout kernel 实际上在算：

$$
N_r = \sum_{t=0}^{T-1} I_{t,r}
$$

这就是发往 rank `r` 的 token 数。

而每个 expert 被选中的次数则是：

$$
M_e = \sum_{t=0}^{T-1} \sum_{j=0}^{K-1} \mathbf{1}[e_{t,j} = e].
$$

### 菜市场级别类比

想象有 4 个大妈去菜市场，4 个摊位分别卖不同菜。

- 某位大妈可能想买番茄和土豆；
- 如果这两样菜刚好都在同一个摊位卖，那她只走到这个摊位一次；
- `I_{t,r}` 的意思就是：“第 `t` 位大妈需不需要去第 `r` 个摊位”；
- `N_r` 的意思就是：“第 `r` 个摊位最后要接待多少位顾客”。

layout kernel 干的就是这个事，只不过它在 GPU 上用极高并发算出来。

## 3. 代入一个极简数字例子

假设：

- 8 个 expert；
- 4 个 rank；
- 每个 rank 2 个 expert；
- top-k = 2。

门控结果如下：

| Token | 选中 expert | 目标 rank 集合 |
| --- | --- | --- |
| `t0` | `[1, 6]` | `{0, 3}` |
| `t1` | `[0, 2]` | `{0, 1}` |
| `t2` | `[3, 7]` | `{1, 3}` |
| `t3` | `[4, -1]` | `{2}` |

于是指标矩阵就是：

| Token \ Rank | 0 | 1 | 2 | 3 |
| --- | ---: | ---: | ---: | ---: |
| `t0` | 1 | 0 | 0 | 1 |
| `t1` | 1 | 1 | 0 | 0 |
| `t2` | 0 | 1 | 0 | 1 |
| `t3` | 0 | 0 | 1 | 0 |

因此：

$$
(N_0, N_1, N_2, N_3) = (2, 2, 1, 2).
$$

这串数字，正是 `get_dispatch_layout(...)` 返回的核心内容之一。

## 4. 前缀和根本没那么玄，它就是“排座位”

计数告诉你：某辆车上有多少乘客。前缀和告诉你：**每个乘客应该坐哪一排**。

如果一个目标 rank 从 4 个 sender 那里分别收到计数 `[1, 2, 0, 1]`，它的前缀和就是：

$$
(1, 3, 3, 4).
$$

含义就是：

- sender 0 占区间 `[0, 1)`；
- sender 1 占区间 `[1, 3)`；
- sender 2 占区间 `[3, 3)`；
- sender 3 占区间 `[3, 4)`。

DeepEP 里的各种 prefix matrix，本质就是把这个“排座位”逻辑扩展到了 rank、channel、RDMA 多层级上。

```mermaid
flowchart LR
    Count[各 sender 计数] --> Prefix[前缀和]
    Prefix --> Slots[确定 buffer 写入区间]
    Slots --> Transport[读写双方无冲突对齐]
```

## 5. Combine 的数学本质：带权归并

设 `h_{t,j}` 表示 token `t` 在第 `j` 个 expert 上计算出的结果，`w_{t,j}` 表示对应门控权重。

那么理想的 combine 输出就是：

$$
y_t = \sum_{j=0}^{K-1} w_{t,j} h_{t,j}.
$$

如果还传入了 bias `b_t^{(0)}` 和 `b_t^{(1)}`，最终输出就变成：

$$
\hat{y}_t = y_t + b_t^{(0)} + b_t^{(1)}.
$$

所以 combine 不是“把东西搬回来”这么简单，它实际上同时完成了：

- 恢复原 token 语义；
- 对多个 expert 输出做 reduce；
- 视情况再叠加 bias。

## 6. FP8 scale 用最接地气的话解释

DeepEP 常按 128 个 hidden channel 为一组处理数据。

对某一组，先定义：

$$
\alpha = \max_i |x_i|.
$$

然后 scale 近似设成：

$$
s = \frac{\alpha}{448}.
$$

再把数值量化成近似的 FP8 表示：

$$
q_i \approx \frac{x_i}{s}, \qquad x_i \approx q_i s.
$$

### 一组小学算术级例子

假设某个 128 通道块的最大绝对值是 `4.48`，那么：

$$
s = \frac{4.48}{448} = 0.01.
$$

于是：

- `1.12` 约等于 `112 × 0.01`；
- `-2.24` 约等于 `-224 × 0.01`；
- `4.48` 约等于 `448 × 0.01`。

这就是为什么 low-latency API 往往返回两个 tensor：

- 一个是 FP8 payload；
- 一个是每 128 通道对应的 scale。

## 7. 为什么 low-latency buffer 会涨得这么快

`csrc/kernels/configs.cuh` 里的公式如果直接看，会让人头疼；但拆成“payload + metadata”两部分就不神秘了。

对于普通内核，NVLink buffer 粗略可以写成：

$$
B_{NVL} \approx C \cdot R_{NVL} \cdot (\text{metadata} + T_{recv} \cdot \text{hidden bytes} + T_{recv} \cdot \text{auxiliary fields}),
$$

其中 `C = num_sms / 2`，代表 channel 数。

对于 low-latency mode，一条 dispatch message 大致大小是：

$$
\text{bytes}_{dispatch} = 16 + \max(2H, H + 4H/128),
$$

这里：

- `16` 是 `int4` 控制头；
- `2H` 是 BF16 载荷；
- `H + 4H/128` 是 FP8 载荷加 scale。

而 combine message 大小大致是：

$$
\text{bytes}_{combine} = 2H + 4H/128.
$$

### 用 `H = 7168` 代进去

- `num_scales = 7168 / 128 = 56`
- BF16 payload = `2H = 14336` bytes
- FP8 payload + scales = `7168 + 56 * 4 = 7392` bytes
- dispatch message = `16 + max(14336, 7392) = 14352` bytes
- combine message = `14336 + 224 = 14560` bytes

这就是 README 为什么会提醒 low-latency 模式很吃内存：它本来就是按“艰难情况”静态预留的。

## 8. SourceMeta 位域其实没那么可怕

`csrc/kernels/internode.cu` 里的 `SourceMeta` 会存两类信息：

- 源 RDMA rank；
- 一个 8 bit 掩码，表示源节点内哪些 NVLink peer 跟这个 token 有关。

比如 bit mask 是 `10110010`，意思就是那几个对应位上的本地 NVLink 位置被选中了。它不是玄学，只是“谁关心这个 token”被压缩存成了位图。

## 9. 最后只记两条直觉就够了

如果你看完这一页只想记住两句话，那就记住：

1. **layout 本质上是集合 + 前缀和问题。**
2. **通信本质上是把 payload 按真实硬件拓扑填进队列。**

一旦接受这两个直觉，DeepEP 里大部分公式与 buffer 结构都会突然变得顺眼很多。

# Math and Mental Models

This page translates DeepEP's formulas into everyday intuition. The goal is not to impress you with symbols. The goal is to make the symbols feel obvious.

## 1. The two questions every transport must answer

Before DeepEP can move a single byte, it must answer:

1. **Where should each token go?**
2. **How much buffer space should each route get?**

Everything else in the code is a highly optimized answer to those two questions.

## 2. Dispatch layout as a set problem

Let:

- `T` be the number of tokens,
- `K` be the top-k count,
- `E` be the number of experts,
- `R` be the number of ranks,
- `E_local = E / R` be the number of experts per rank.

If token `t` selects expert `e_{t, j}`, then the owner rank is:

$$
\operatorname{rank}(e_{t,j}) = \left\lfloor \frac{e_{t,j}}{E_{local}} \right\rfloor.
$$

DeepEP then forms the **destination set** of token `t`:

$$
D_t = \{\operatorname{rank}(e_{t,j}) \mid e_{t,j} \neq -1\}.
$$

Notice the word **set**. If a token picks two experts on the same rank, that rank appears only once.

Now define the indicator:

$$
I_{t,r} = \mathbf{1}[r \in D_t].
$$

Then the layout kernel computes:

$$
N_r = \sum_{t=0}^{T-1} I_{t,r}
$$

for the number of tokens sent to rank `r`, and

$$
M_e = \sum_{t=0}^{T-1} \sum_{j=0}^{K-1} \mathbf{1}[e_{t,j} = e]
$$

for the number of times expert `e` is selected.

### Super-simple market analogy

Imagine 4 shoppers and 4 vegetable stalls run by 4 vendors.

- A shopper may want tomatoes and potatoes.
- If both vegetables are sold by the same vendor, the shopper only walks to that vendor once.
- `I_{t,r}` just means “shopper `t` needs vendor `r`”.
- `N_r` means “how many shoppers will line up at vendor `r`”.

That is all the layout kernel is doing, just at GPU speed.

## 3. Worked example with real numbers

Suppose:

- 8 experts,
- 4 ranks,
- 2 experts per rank,
- top-k = 2.

Let the expert selections be:

| Token | Experts | Destination set |
| --- | --- | --- |
| `t0` | `[1, 6]` | `{0, 3}` |
| `t1` | `[0, 2]` | `{0, 1}` |
| `t2` | `[3, 7]` | `{1, 3}` |
| `t3` | `[4, -1]` | `{2}` |

Then the indicator matrix is:

| Token \ Rank | 0 | 1 | 2 | 3 |
| --- | ---: | ---: | ---: | ---: |
| `t0` | 1 | 0 | 0 | 1 |
| `t1` | 1 | 1 | 0 | 0 |
| `t2` | 0 | 1 | 0 | 1 |
| `t3` | 0 | 0 | 1 | 0 |

So:

$$
(N_0, N_1, N_2, N_3) = (2, 2, 1, 2).
$$

That vector is exactly the kind of tensor returned by `get_dispatch_layout(...)`.

## 4. Prefix sums are just seat numbers

Counts tell you **how many** passengers board each bus. Prefix sums tell you **where each passenger should sit**.

If one destination rank receives counts `[1, 2, 0, 1]` from four senders, the prefix sum is:

$$
(1, 3, 3, 4).
$$

That means:

- sender 0 occupies slots `[0, 1)`,
- sender 1 occupies slots `[1, 3)`,
- sender 2 occupies slots `[3, 3)`,
- sender 3 occupies slots `[3, 4)`.

DeepEP's prefix matrices are just that idea generalized across ranks, channels, and RDMA levels.

```mermaid
flowchart LR
    Count[Per-sender counts] --> Prefix[Prefix sums]
    Prefix --> Slots[Deterministic buffer slots]
    Slots --> Transport[Writers and readers agree without conflict]
```

## 5. Combine is weighted reduction

Let `h_{t,j}` be the output produced for token `t` by its `j`-th selected expert, and let `w_{t,j}` be the corresponding gate weight.

Then the ideal combine result is:

$$
y_t = \sum_{j=0}^{K-1} w_{t,j} h_{t,j}.
$$

If DeepEP is also asked to add optional bias tensors `b_t^{(0)}` and `b_t^{(1)}`, the final value becomes:

$$
\hat{y}_t = y_t + b_t^{(0)} + b_t^{(1)}.
$$

So the transport graph is doing more than a gather; it is returning values to the original token order **and** applying the MoE reduction semantics.

## 6. FP8 scaling in plain language

DeepEP often handles payloads in blocks of 128 hidden channels.

For one such block, define:

$$
\alpha = \max_i |x_i|.
$$

Then the scale is approximately:

$$
s = \frac{\alpha}{448}.
$$

The FP8 value is then a clipped, quantized version of:

$$
q_i \approx \frac{x_i}{s}, \qquad x_i \approx q_i s.
$$

### Tiny numeric example

Suppose a 128-channel block has its largest magnitude equal to `4.48`.

Then:

$$
s = \frac{4.48}{448} = 0.01.
$$

So values like:

- `1.12` become about `112`,
- `-2.24` become about `-224`,
- `4.48` become about `448`.

Later, DeepEP multiplies the FP8 payload by the stored scale to recover a BF16 approximation.

That is why the low-latency API may return **two** tensors: payload and scales.

## 7. Why low-latency buffer sizes grow so fast

The formulas in `csrc/kernels/configs.cuh` are easier to understand if you separate them into payload and metadata.

For normal kernels, the NVLink-side buffer is roughly:

$$
B_{NVL} \approx C \cdot R_{NVL} \cdot (\text{metadata} + T_{recv} \cdot \text{hidden bytes} + T_{recv} \cdot \text{auxiliary fields}),
$$

where `C = num_sms / 2` is the number of channels.

For low-latency mode, one dispatch message is sized as:

$$
\text{bytes}_{dispatch} = 16 + \max(2H, H + 4H/128),
$$

where:

- `16` is the `int4` control header,
- `2H` is the BF16 path,
- `H + 4H/128` is the FP8 payload plus scales.

And one combine message is sized as:

$$
\text{bytes}_{combine} = 2H + 4H/128.
$$

### Example with `H = 7168`

- `num_scales = 7168 / 128 = 56`
- BF16 payload = `2H = 14336` bytes
- FP8 payload + scales = `7168 + 56 * 4 = 7392` bytes
- dispatch message = `16 + max(14336, 7392) = 14352` bytes
- combine message = `14336 + 224 = 14560` bytes

This is why the README warns that low-latency mode is memory-hungry: the layout is deliberately provisioned for the hard case.

## 8. Source metadata as a bitfield

`csrc/kernels/internode.cu` defines a `SourceMeta` structure that stores:

- the source RDMA rank,
- an 8-bit mask telling which NVLink peers inside that source node own the token.

If the bit mask is `10110010`, it means those specific local NVLink positions were selected. The exact bit positions are not magic. They are just a compact way to carry “who inside the source node cares about this token?”

## 9. The two guiding intuitions

If you remember only two things, remember these:

1. **Layout is a set-and-prefix-sum problem.**
2. **Transport is a queue-filling problem over real hardware fabrics.**

The symbols in the code become much less scary once you recognize those two patterns.

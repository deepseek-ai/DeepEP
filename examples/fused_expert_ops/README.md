# Fused BF16 expert epilogues

This optional Triton example prepares expert gradients and combine inputs between DeepEP collectives and the expert GEMMs. Run these helpers inside the model's existing forward and backward implementation. The CUDA tensor arguments use contiguous storage and FP32 route weights.

```bash
python examples/fused_expert_ops/test_fused_expert_ops.py
```

The checks require CUDA PyTorch and Triton. The DeepEP extension is needed when connecting the example to the dispatch and combine calls below.

## Compact receive

For standard compact dispatch, `recv_topk_idx` and `recv_topk_weights` have shape `[R, K]`, where `R` is the number of local receive rows and `K` is the router top-k. This example uses `do_cpu_sync=True` and `expert_alignment=1` to obtain exact unpadded receive sizes. Build the row maps alongside the expert layout used by the GEMMs.

```python
import torch

from fused_expert_ops import (
    pack_expert_input_grads,
    pack_expert_outputs,
    prepare_expert_grads,
)

num_tokens, topk = recv_topk_idx.shape
flat_experts = recv_topk_idx.flatten()
active = (flat_experts >= 0).nonzero().flatten()
route_slots = active[flat_experts[active].argsort(stable=True)]
expert_to_token = route_slots // topk
expert_weights = recv_topk_weights.flatten().index_select(0, route_slots)
token_to_expert = torch.full_like(recv_topk_idx, -1, dtype=torch.long)
expert_rows = torch.arange(route_slots.numel(), device=route_slots.device)
token_to_expert.flatten().index_copy_(0, route_slots, expert_rows)
```

`expert_to_token` maps expert rows back to compact receive rows. `token_to_expert` lists the expert rows to sum for each compact token, with `-1` for inactive lanes. Its lane order defines the FP32 sum order. Keep these maps in the same expert order as the grouped GEMMs and reuse them while the routing layout stays valid.

```python
# Forward. expert_y is the BF16 down-projection output in expert order.
packed_y = pack_expert_outputs(expert_y, expert_weights, token_to_expert=token_to_expert)
y, _, _ = buffer.combine(packed_y, handle)

# Backward. Replay the compact layout from forward.
dy_recv, _, _, _, _ = buffer.dispatch(dy, handle=handle)
d_expert_y, d_route = prepare_expert_grads(dy_recv, expert_y, expert_weights, expert_to_token=expert_to_token)
d_route_slots = torch.zeros_like(recv_topk_weights)
d_route_slots.flatten().index_copy_(0, route_slots, d_route)

# Run expert backward GEMMs with d_expert_y. gate_dx and up_dx are their BF16 input gradients.
packed_dx = pack_expert_input_grads(gate_dx, up_dx, token_to_expert=token_to_expert)
dx, d_weights, _ = buffer.combine(packed_dx, handle, topk_weights=d_route_slots)
```

Each compact pack reads the required expert rows, reduces local contributions in FP32, and writes the BF16 result once. `prepare_expert_grads` reads the upstream gradient once for both weighting and the route-weight gradient dot product. The FP32 dot reduction can differ from PyTorch's reduction order in its last bits.

## Expanded receive

For native `do_expand=True`, the outputs of cached dispatch and the expert route weights already follow expert row order. Pass the expert-ordered tensors directly and leave the maps at their defaults. The forward and backward pack helpers return one BF16 row per expert row, and `d_route` is already the one-dimensional route gradient for expanded combine. Keep `allow_multiple_reduction=True` for expanded combine with route gradients.

The direct path assumes every supplied row is valid. For an aligned expanded layout with padding, use an identity row map with `-1` in padded rows for gradient preparation. Pass the same map with shape `[M, 1]` to the pack helpers to zero those rows in the expanded combine inputs. A sync-free capacity buffer needs its valid ranges from the dispatch handle.

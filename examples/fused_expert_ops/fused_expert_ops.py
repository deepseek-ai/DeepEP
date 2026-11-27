"""Optional Triton epilogues for a BF16 MoE step using DeepEP."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_expert_grads(
    dy, expert_y, weights, expert_to_token, d_expert_y, d_weights,
    expert_rows, token_rows,
    HIDDEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, DIRECT: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_H)
    valid_row = row < expert_rows
    if DIRECT:
        token = row.to(tl.int64)
        active = valid_row
    else:
        token = tl.load(expert_to_token + row, mask=valid_row, other=-1).to(tl.int64)
        active = valid_row & (token >= 0) & (token < token_rows)
    valid_col = col[None, :] < HIDDEN
    expert_offset = row.to(tl.int64)[:, None] * HIDDEN + col[None, :]
    token_offset = token[:, None] * HIDDEN + col[None, :]
    active_element = active[:, None] & valid_col
    grad = tl.load(dy + token_offset, mask=active_element, other=0).to(tl.float32)
    scale = tl.load(weights + row, mask=active, other=0)
    tl.store(d_expert_y + expert_offset, grad * scale[:, None], mask=valid_row[:, None] & valid_col)
    y = tl.load(expert_y + expert_offset, mask=active_element, other=0).to(tl.float32)
    tl.store(d_weights + row, tl.sum(grad * y, axis=1), mask=valid_row)


@triton.jit
def _pack_expert_rows(
    first, second, token_to_expert, out, token_rows, expert_rows,
    HIDDEN: tl.constexpr, ROUTES: tl.constexpr, WEIGHTED: tl.constexpr, DIRECT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    valid_token = token < token_rows
    valid_col = col[None, :] < HIDDEN
    acc = tl.full((BLOCK_M, BLOCK_K), 0, tl.float32)
    for lane in tl.static_range(ROUTES):
        if DIRECT:
            row = token.to(tl.int64)
            active = valid_token
        else:
            map_offset = token.to(tl.int64) * ROUTES + lane
            row = tl.load(token_to_expert + map_offset, mask=valid_token, other=-1).to(tl.int64)
            active = valid_token & (row >= 0) & (row < expert_rows)
        offset = row[:, None] * HIDDEN + col[None, :]
        mask = active[:, None] & valid_col
        a = tl.load(first + offset, mask=mask, other=0).to(tl.float32)
        if WEIGHTED:
            scale = tl.load(second + row, mask=active, other=0)
            acc += a * scale[:, None]
        else:
            b = tl.load(second + offset, mask=mask, other=0).to(tl.float32)
            acc += a + b
    dst = token.to(tl.int64)[:, None] * HIDDEN + col[None, :]
    tl.store(out + dst, acc, mask=valid_token[:, None] & valid_col)


def _check_tensor(tensor, name, device, dtypes, shape):
    if tensor.device != device or tensor.dtype not in dtypes or not tensor.is_contiguous() or tensor.shape != shape:
        raise ValueError(f'{name} must be contiguous on {device}, with shape {tuple(shape)} and dtype in {dtypes}')


def _matrix_shape(tensor, name):
    if tensor.ndim != 2 or tensor.shape[1] == 0 or not tensor.is_cuda:
        raise ValueError(f'{name} must be a CUDA matrix with a positive hidden dimension')
    _check_tensor(tensor, name, tensor.device, (torch.bfloat16,), tensor.shape)
    return tensor.shape


def prepare_expert_grads(dy, expert_y, weights, *, expert_to_token=None):
    """Return BF16 expert-row dY and FP32 per-route weight gradients.

    Pass a compact token row for each expert row, using -1 for padding.
    Without a map, dy is already in expert order and all rows are valid.
    Expert outputs are the unweighted down-projection values.
    """
    expert_rows, hidden = _matrix_shape(expert_y, 'expert_y')
    token_rows, dy_hidden = _matrix_shape(dy, 'dy')
    if dy.device != expert_y.device or dy_hidden != hidden:
        raise ValueError('dy and expert_y must have the same device and hidden dimension')
    _check_tensor(weights, 'weights', expert_y.device, (torch.float32,), (expert_rows,))
    if expert_to_token is None:
        if token_rows != expert_rows:
            raise ValueError('direct dY must have one row per expert output')
    else:
        _check_tensor(expert_to_token, 'expert_to_token', expert_y.device,
                      (torch.int32, torch.int64), (expert_rows,))
    if not token_rows:
        return torch.zeros_like(expert_y), torch.zeros_like(weights)
    d_expert_y = torch.empty_like(expert_y)
    d_weights = torch.empty_like(weights)
    if expert_rows:
        block_m = 4
        with torch.cuda.device(expert_y.device):
            _prepare_expert_grads[(triton.cdiv(expert_rows, block_m),)](
                dy, expert_y, weights, expert_to_token, d_expert_y, d_weights, expert_rows, token_rows,
                hidden, block_m, triton.next_power_of_2(hidden), expert_to_token is None,
                num_warps=8, enable_fp_fusion=False,
            )
    return d_expert_y, d_weights


def _pack(first, second, token_to_expert, weighted):
    expert_rows, hidden = _matrix_shape(first, 'expert rows')
    second_shape = (expert_rows,) if weighted else first.shape
    _check_tensor(second, 'weights' if weighted else 'up_dx', first.device,
                  (torch.float32,) if weighted else (torch.bfloat16,), second_shape)
    if token_to_expert is None:
        token_rows, routes = expert_rows, 1
    else:
        if token_to_expert.ndim != 2:
            raise ValueError('token_to_expert must have shape [token rows, route slots]')
        token_rows, routes = token_to_expert.shape
        _check_tensor(token_to_expert, 'token_to_expert', first.device, (torch.int32, torch.int64),
                      (token_rows, routes))
    if not expert_rows or not routes:
        return first.new_zeros((token_rows, hidden))
    out = first.new_empty((token_rows, hidden))
    if token_rows:
        block_m, block_k = 16, 256
        with torch.cuda.device(first.device):
            _pack_expert_rows[(triton.cdiv(token_rows, block_m), triton.cdiv(hidden, block_k))](
                first, second, token_to_expert, out, token_rows, expert_rows,
                hidden, routes, weighted, token_to_expert is None, block_m, block_k,
                num_warps=4, enable_fp_fusion=False,
            )
    return out


def pack_expert_outputs(expert_y, weights, *, token_to_expert=None):
    """Multiply by FP32 route weights and produce a BF16 combine input.

    A compact reverse map lists expert rows in their desired sum order, with
    -1 for missing routes. With no map, return weighted expert rows for native
    expanded combine.
    """
    return _pack(expert_y, weights, token_to_expert, weighted=True)


def pack_expert_input_grads(gate_dx, up_dx, *, token_to_expert=None):
    """Add gate and up input gradients and produce a BF16 combine input.

    Pass a compact reverse map to reduce all local contributions per token.
    With no map, return one input-gradient row per expert row for expanded combine.
    """
    return _pack(gate_dx, up_dx, token_to_expert, weighted=False)

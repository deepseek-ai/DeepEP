import torch

from fused_expert_ops import pack_expert_input_grads, pack_expert_outputs, prepare_expert_grads


def check_grads(dy, y, weights, row_map=None):
    got_dy, got_scores = prepare_expert_grads(dy, y, weights, expert_to_token=row_map)
    if row_map is None:
        active = torch.ones(y.size(0), device=y.device, dtype=torch.bool)
        value = dy.float()
    else:
        active = row_map >= 0
        value = dy.index_select(0, row_map.clamp_min(0)).float()
    grad = torch.zeros_like(y)
    score = torch.zeros_like(weights)
    grad[active] = (value[active] * weights[active, None]).to(y.dtype)
    score[active] = (value[active] * y[active].float()).sum(1)
    torch.testing.assert_close(got_dy, grad, rtol=0, atol=0)
    torch.testing.assert_close(got_scores, score, rtol=1e-5, atol=3e-5)


def compact_case(hidden, index_dtype):
    # Includes token/row zero, empty tokens, gaps between lanes, and a poisoned unused expert row.
    lanes = torch.tensor([[0, -1, 7, 9], [-1, -1, -1, -1], [1, 5, -1, -1], [-1, 4, 3, 6], [8, 2, 11, 10]],
                         device='cuda', dtype=index_dtype)
    valid = lanes >= 0
    owners = torch.full((13,), -1, device='cuda', dtype=index_dtype)
    token_ids = torch.arange(lanes.size(0), device='cuda', dtype=index_dtype)[:, None].expand_as(lanes)
    owners[lanes[valid].long()] = token_ids[valid]
    dy = torch.randn((lanes.size(0), hidden), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((owners.numel(), hidden), device='cuda', dtype=torch.bfloat16)
    gate_dx, up_dx = torch.randn_like(y), torch.randn_like(y)
    weights = torch.randn(owners.numel(), device='cuda', dtype=torch.float32)
    weights[3] = 0
    weights[-1] = float('nan')
    for tensor in (y, gate_dx, up_dx):
        tensor[-1].fill_(float('nan'))
    check_grads(dy, y, weights, owners)

    expected_y = torch.zeros_like(dy, dtype=torch.float32)
    expected_dx = torch.zeros_like(expected_y)
    for lane in lanes.unbind(1):
        active = lane >= 0
        rows = lane[active].long()
        expected_y[active] += y[rows].float() * weights[rows, None]
        expected_dx[active] += gate_dx[rows].float() + up_dx[rows].float()
    actual_y = pack_expert_outputs(y, weights, token_to_expert=lanes)
    actual_dx = pack_expert_input_grads(gate_dx, up_dx, token_to_expert=lanes)
    torch.testing.assert_close(actual_y, expected_y.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(actual_dx, expected_dx.bfloat16(), rtol=0, atol=0)


def expanded_case():
    y = torch.randn((19, 257), device='cuda', dtype=torch.bfloat16)
    gate_dx, up_dx, dy = (torch.randn_like(y) for _ in range(3))
    weights = torch.randn(y.size(0), device='cuda', dtype=torch.float32)
    check_grads(dy, y, weights)
    torch.testing.assert_close(pack_expert_outputs(y, weights), (y.float() * weights[:, None]).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(pack_expert_input_grads(gate_dx, up_dx), (gate_dx.float() + up_dx.float()).bfloat16(),
                               rtol=0, atol=0)
    rows = torch.arange(y.size(0), device='cuda')
    rows[2::3] = -1
    padded = rows < 0
    for tensor in (y, gate_dx, up_dx, dy, weights):
        tensor[padded] = float('nan')
    check_grads(dy, y, weights, rows)
    dense_y = torch.zeros_like(y)
    dense_dx = torch.zeros_like(y)
    dense_y[~padded] = (y[~padded].float() * weights[~padded, None]).bfloat16()
    dense_dx[~padded] = (gate_dx[~padded].float() + up_dx[~padded].float()).bfloat16()
    torch.testing.assert_close(pack_expert_outputs(y, weights, token_to_expert=rows[:, None]), dense_y, rtol=0, atol=0)
    torch.testing.assert_close(pack_expert_input_grads(gate_dx, up_dx, token_to_expert=rows[:, None]), dense_dx,
                               rtol=0, atol=0)


def empty_cases():
    y = torch.empty((0, 35), device='cuda', dtype=torch.bfloat16)
    weights = torch.empty(0, device='cuda', dtype=torch.float32)
    lanes = torch.full((4, 3), -1, device='cuda', dtype=torch.int32)
    y_empty = pack_expert_outputs(y, weights, token_to_expert=lanes)
    assert y_empty.shape == (4, 35) and not y_empty.count_nonzero().item()
    dx_empty = pack_expert_input_grads(y, y, token_to_expert=lanes)
    torch.testing.assert_close(dx_empty, y_empty, rtol=0, atol=0)
    assert pack_expert_outputs(y, weights).shape == (0, 35)
    assert pack_expert_outputs(y_empty, torch.ones(4, device='cuda'), token_to_expert=lanes[:, :0].contiguous()).eq(0).all()
    owners = torch.full((4,), -1, device='cuda', dtype=torch.int64)
    d_y, d_score = prepare_expert_grads(y, y_empty, torch.ones(4, device='cuda'), expert_to_token=owners)
    assert d_y.shape == (4, 35) and d_score.shape == (4,) and d_y.eq(0).all() and d_score.eq(0).all()
    assert prepare_expert_grads(y, y, weights)[0].shape == (0, 35)


if __name__ == '__main__':
    torch.manual_seed(769)
    compact_case(37, torch.int32)
    compact_case(3072, torch.int64)
    expanded_case()
    empty_cases()
    print('fused expert epilogue checks passed')

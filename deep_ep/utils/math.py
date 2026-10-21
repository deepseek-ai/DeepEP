import torch
from typing import Tuple


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def safe_div(a, b) -> float:
    try:
        return a / b
    except ZeroDivisionError as e:
        if a == 0:
            return 0
        else:
            raise


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


@torch.compile(dynamic=True)
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    aligned_n = align(n, 128)
    x_padded = torch.nn.functional.pad(x, (0, aligned_n - n), mode='constant', value=0)
    x_padded_view = x_padded.view(m, -1, 128)
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_padded_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, aligned_n)[:, :n].contiguous(), (x_amax / 448.0).view(m, -1)


@torch.compile(dynamic=True)
def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor) -> torch.Tensor:
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)

    assert x_fp8.dim() == 2
    m, n = x_fp8.shape
    aligned_n = align(n, 128)
    x_fp8_padded = torch.nn.functional.pad(x_fp8, (0, aligned_n - n), mode='constant', value=0)
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32_padded = x_fp8_padded.to(torch.float32).view(x_fp8.shape[0], -1, 128)
    x_scales = x_scales.view(x_fp8.shape[0], -1, 1)
    return (x_fp32_padded * x_scales).view(x_fp8_padded.shape).to(torch.bfloat16)[:, :n].contiguous()


def inplace_unique(x: torch.Tensor, num_slots: int) -> None:
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def create_grouped_scores(scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


def hash_tensor(t: torch.Tensor) -> int:
    # `view(torch.int)` needs a contiguous tensor whose elements are 4 bytes wide,
    # so it raises for non-contiguous tensors and for dtypes such as `bool`,
    # `int64` or `float16`. Normalize with a byte view to stay dtype-agnostic.
    t = t.contiguous()
    if t.element_size() == 4:
        return t.view(torch.int).sum().item()
    return t.view(torch.uint8).sum().item()


def hash_tensors(*tensors) -> int:
    value = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            value ^= hash_tensors(*t)
        elif t is not None and isinstance(t, torch.Tensor):
            value ^= hash_tensor(t)
    return value


def count_bytes(*tensors) -> int:
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total

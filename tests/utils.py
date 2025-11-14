# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from typing import Optional, Tuple, Union

BLOCK_SIZE = 16
def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        'backend': 'nccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in sig.parameters:
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def align_up(x, y):
    return (x + y - 1) // y * y


def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2
    m, n = x.shape
    aligned_n = align_up(n, 128)
    x_padded = torch.nn.functional.pad(x, (0, aligned_n - n), mode='constant', value=0)
    x_padded_view = x_padded.view(m, -1, 128)
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_padded_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, aligned_n)[:, :n].contiguous(), (x_amax / 448.0).view(m, -1)

    
def cast_fp8_to_bf16(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)

    assert x_fp8.dim() == 2
    m, n = x_fp8.shape
    aligned_n = align_up(n, 128)
    x_fp8_padded = torch.nn.functional.pad(x_fp8, (0, aligned_n - n), mode='constant', value=0)
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32_padded = x_fp8_padded.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32_padded * x_scales).view(x_fp8_padded.shape).to(torch.bfloat16)[:,:n].contiguous()

def get_global_token_idxs(recv_count: torch.Tensor, recv_src_info: torch.Tensor, recv_layout_range: torch.Tensor, num_local_experts: int, num_ranks: int, num_tokens: int):
    rank = dist.get_rank()
    int_mask = (2 ** 32) - 1
    begin_idx = torch.zeros((num_local_experts, num_ranks), dtype=torch.int, device='cuda')
    count = torch.zeros((num_local_experts, num_ranks), dtype=torch.int, device='cuda')
    global_token_idxs = torch.ones((num_local_experts, num_ranks * num_tokens), dtype=torch.int, device='cuda') * -1
    for local_expert in range(num_local_experts):
        num_valid_tokens = recv_count[local_expert].item()
        for src_rank in range(num_ranks):
            begin_idx_local, count_local = (recv_layout_range[local_expert][src_rank] >> 32).item(), (recv_layout_range[local_expert][src_rank] & int_mask).item()
            begin_idx[local_expert, src_rank], count[local_expert, src_rank] = begin_idx_local, count_local
            for recv_idx in range(begin_idx_local, begin_idx_local + count_local):
                global_token_idxs[local_expert, recv_idx] = recv_src_info[local_expert, recv_idx] + src_rank * num_tokens
    return global_token_idxs


def get_pair_token_idx(global_token_idxs_test: torch.Tensor, global_token_idxs_ref: torch.Tensor, local_expert: int, token_idx: int):
    global_token_idxs_temp = global_token_idxs_test[local_expert, token_idx]    
    idx_arr = torch.nonzero(global_token_idxs_ref[local_expert, :] == global_token_idxs_temp, as_tuple=False)
    assert idx_arr.numel() == 1, f'idx_arr.numel(): {idx_arr.numel()}'
    return idx_arr.item(), global_token_idxs_temp


def recover_swizzled_scales(scale, m, n):
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // BLOCK_SIZE
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    # Recover the swizzled scaling factor to linear layout
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_n)).to(torch.float32)
    return result[:m, :scale_n]

def recover_experts_swizzled_scales(scale, l, m, n):
    recovered_tensor = torch.empty((l, m, n//16), dtype=torch.float32, device=scale.device)
    for i in range(l):
        recovered_tensor[i] = recover_swizzled_scales(scale[i], m, n)
    return recovered_tensor
def quantize_bfloat16_to_nvfp4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 16 == 0
    m, n = x.shape
    x_global_amax = x.abs().float().amax(dim=1).view(m, -1).clamp(1e-4)
    sf_scales = (6.0 * 448.0) / x_global_amax
    x_global_quantized = (x * sf_scales).view(m, n)
    x_view = x_global_quantized.view(m, -1, 16)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    x_quantized = (x_view * (6.0 / x_amax.unsqueeze(2))).view(m, n)
    x_nvfp4_packed = pack_8xnvfp4_to_int32(x_quantized)
    x_scales = (x_amax / 6.0).view(m, -1)
    return x_nvfp4_packed, x_scales.to(torch.float8_e4m3fn).view(dtype=torch.int32), sf_scales

def pack_8xnvfp4_to_int32(x: torch.Tensor) -> torch.Tensor:
    num_tokens, hidden = x.shape
    NVFP4_TABLE = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=x.device)
    NVFP4_INTREPL = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device=x.device)
    sign = (x < 0).to(torch.int32)
    diff = (x.abs().unsqueeze(-1) - NVFP4_TABLE.view(1, 1, 8)).abs()
    idx = diff.argmin(dim=-1)
    repl = NVFP4_INTREPL[idx]
    int32_elm = ((sign << 3) + repl)
    int32_elm = int32_elm.reshape(num_tokens, hidden // 8, 8)
    shift = int32_elm << torch.tensor([28, 24, 20, 16, 12, 8, 4, 0], dtype=torch.int32, device=x.device)
    uint32_shift = shift.view(dtype=torch.uint32)
    final = uint32_shift.sum(dim=-1).to(dtype=torch.uint32).view(dtype=torch.int32)
    return final

def int32_to_8floats_lookup(tensor: torch.Tensor, table: torch.Tensor, msb_first: bool = True) -> torch.Tensor:
    """
    Decomposes each int32 in the input tensor into 8 4-bit values,
    and converts them into float values using a lookup table.

    Args:
        tensor: (int32 Tensor) Tensor of any shape, e.g., [B, N]
        table: (float Tensor) A 1D lookup table of length 16 that maps all 4-bit values to floats
        msb_first: (bool) Whether the most significant 4 bytes should be put at the first element in the result list

    Returns:
        float32 Tensor: Merges the last two dimensions, so shape is [..., n*M], where n is the number of int32 and 8 per int32.
    """
    assert tensor.dtype == torch.int32, "Input must be of int32 type"
    assert table.numel() == 16 and table.ndim == 1, "Lookup table must be 1D with length 16"

    result = []
    for i in range(8):
        if msb_first:
            shift = (7 - i) * 4
        else:
            shift = i * 4
        idx = ((tensor >> shift) & 0xF).long()  # Extract 4-bit index [0, 15]
        val = table[idx].unsqueeze(-1)  # Lookup and preserve dimensions
        result.append(val)

    out = torch.cat(result, dim=-1)  # Output shape: [..., 8]
    # Merge the last two dimensions if shape is [..., M, 8]
    out = out.reshape(*out.shape[:-2], -1) if out.ndim > 2 else out
    return out


def uint8_to_2floats_lookup(tensor: torch.Tensor, table: torch.Tensor, msb_first: bool = True) -> torch.Tensor:
    """
    Decomposes each uint8 in the input tensor into 2 4-bit values,
    and converts them into float values using a lookup table.

    Args:
        tensor: (uint8 Tensor) Tensor of any shape, e.g., [B, M]
        table: (float Tensor) A 1D lookup table of length 16 that maps all 4-bit values to floats
        msb_first: (bool) Whether the most significant 4 bytes should be put at the first element in the result list

    Returns:
        float32 Tensor: Merges the last two dimensions, so shape is [..., n*M], where n is 2, 
        which isthe number of 4-bit values per uint8.
    """
    assert tensor.dtype == torch.uint8, "Input must be of uint8 type"

    result = []
    for i in range(2):
        if msb_first:
            shift = (1 - i) * 4
        else:
            shift = i * 4
        idx = ((tensor >> shift) & 0xF).long()  # Extract 4-bit index [0, 15]
        val = table[idx].unsqueeze(-1)  # Lookup and preserve dimensions
        result.append(val)

    out = torch.cat(result, dim=-1)  # Output shape: [..., 2]
    # Merge the last two dimension
    out = out.reshape(*out.shape[:-2], -1) if out.ndim > 1 else out
    return out


def cast_nvfp4_to_bf16(x_nvfp4: torch.Tensor, x_scales: torch.Tensor, x_global_scale: float, use_ue8m0_for_sf: bool = False):
    NVFP4_TABLE = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1.0, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=x_nvfp4.device)  
    assert x_nvfp4.dtype == torch.uint8, "Input must be of int8 type, but got " + str(x_nvfp4.dtype)
    assert x_scales.ndim == 6, "Input scales must be of 6 dimensions"
    assert x_scales.shape[0] == 32 and x_scales.shape[1] == 4 and x_scales.shape[3] == 4, "Input scales shape must be [32, 4, rm, 4, rk, l]"
    _, _, rm, _, rk, l = x_scales.shape
    assert x_nvfp4.ndim == 3, "Input nvfp4 must be of 3 dimensions"
    assert x_nvfp4.shape[2] == l, "Input nvfp4 shape must be [m, k//2, l], but got " + str(x_nvfp4.shape)
    x_nvfp4 = x_nvfp4.permute(2, 0, 1)
    
    if use_ue8m0_for_sf:
        assert x_scales.dtype == torch.int8, "Input scales must be of int8 type if use_ue8m0_for_sf is True"
        x_scales = x_scales.view(dtype=torch.int8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    else:
        assert x_scales.dtype == torch.float8_e4m3fn, "Input scales must be of float8_e4m3fn type if use_ue8m0_for_sf is False"
        x_scales = x_scales.to(torch.float32)
    x_scales = x_scales * (1 / x_global_scale)
    
    x_fp32 = uint8_to_2floats_lookup(x_nvfp4, table=NVFP4_TABLE, msb_first=False).to(torch.float32)
    x_scales_view = x_scales.permute(5, 2, 4, 0, 1, 3).view(l, rm, -1)
    x_scales_view_recover = torch.empty((l, rm*128, rk*4), dtype=torch.float32, device=x_scales.device)
    for i in range(l):
        x_scales_view_recover[i] = recover_swizzled_scales(x_scales_view[i], rm*128, rk*64)
    x_fp32_dequantized = x_fp32 * x_scales_view_recover.repeat_interleave(16, dim=-1)[:, :x_nvfp4.shape[1], :]

    return x_fp32_dequantized.contiguous().to(torch.bfloat16)


def per_token_cast_back(x: torch.Tensor, x_scales: torch.Tensor, x_global_scale: torch.Tensor = None, use_ue8m0_for_sf: bool = False, src_data_format: str = 'fp8'):
    if src_data_format == 'fp8':
        return cast_fp8_to_bf16(x, x_scales)
    elif src_data_format == 'nvfp4':
        return cast_nvfp4_to_bf16(x, x_scales, x_global_scale, use_ue8m0_for_sf)
    else:
        raise ValueError(f"Unsupported src_data_format: {src_data_format}")

def dequantize_nvfp4_back_to_bfloat16(x_nvfp4: torch.Tensor, x_scales: torch.Tensor, x_sf_scale: torch.Tensor, use_ue8m0_for_nvfp4_sf: bool = False):
    NVFP4_TABLE = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1.0, -1.5, -2, -3, -4, -6], dtype=torch.float32, device=x_nvfp4.device)   
    if use_ue8m0_for_nvfp4_sf:
        x_scales = x_scales.view(dtype=torch.int8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    else:
        x_scales = x_scales.view(dtype=torch.float8_e4m3fn).to(torch.float32)
    x_sf_scale = 1 / x_sf_scale
    x_scales = x_scales * x_sf_scale
    
    x_fp32 = int32_to_8floats_lookup(x_nvfp4, NVFP4_TABLE)
    
    x_fp32 = x_fp32.view(*x_fp32.shape[:-1], -1, 16)
    x_scales = x_scales.view(*x_scales.shape[:-1], -1, 1)
    x_fp32 = x_fp32 * x_scales
    x_fp32 = x_fp32.view(*x_nvfp4.shape[:-1], -1).to(torch.bfloat16)

    return x_fp32

def inplace_unique(x: torch.Tensor, num_slots: int):
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


def create_grouped_scores(scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int):
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


def bench(fn, num_warmups: int = 50, num_tests: int = 50, post_fn=None):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times)


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(fn, kernel_names: Union[str, tuple], num_tests: int = 30, suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None, barrier_comm_profiling: bool = False,
                 num_kernels_per_period: int = 1):
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))
                for _ in range(num_tests):
                    fn()
                torch.cuda.synchronize()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = prof.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) == 1, f'Errors of the kernel {name} in the profiling table'

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Return average kernel durations
    units = {'ms': 1e3, 'us': 1e6}
    kernel_durations = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_durations.append(float(time_str.replace(unit, '')) / scale)
                        break
                break

    # Expand the kernels by periods
    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            prof.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())

        for i, kernel_name in enumerate(kernel_names):
            events = [event for event in profile_data['traceEvents'] if f'::{kernel_name}' in event['name']]
            events = sorted(events, key=lambda event: event['ts'])
            durations = [event['dur'] / 1e6 for event in events]
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations[i] = [sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                               for j in range(num_kernels_per_period)]

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]


def hash_tensor(t: torch.Tensor):
    return t.view(torch.int).sum().item()


def permute(
    tokens,
    routing_map,
    scaling_factor: torch.Tensor = None,
    probs: torch.Tensor = None,
    num_out_tokens: int = None,
):
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool().T.contiguous()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = (
        torch.arange(num_tokens, device=routing_map.device)
        .unsqueeze(0)
        .expand(num_experts, -1)
    )

    sorted_indices = token_indices.masked_select(routing_map)

    if probs is not None:
        permuted_probs = probs.T.contiguous().masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)
    permuted_scaling_factor = scaling_factor.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, permuted_scaling_factor


class TorchRef:
    def __init__(
        self,
        ep_group: torch.distributed.ProcessGroup,
        num_of_experts: int,
        num_of_ranks_per_node: int,
    ):
        self.ep_group = ep_group
        self.group_rank = torch.distributed.get_rank(self.ep_group)
        self.group_size = torch.distributed.get_world_size(self.ep_group)
        self.local_rank = self.group_rank % num_of_ranks_per_node
        self.num_of_experts = num_of_experts
        self.num_local_experts = num_of_experts // self.group_size

    def _local_expert_range(self):
        start = self.group_rank * self.num_local_experts
        end = start + self.num_local_experts  # [start, end)
        return start, end

    def _local_expert_range_per_node(self):
        start = self.local_rank * self.num_local_experts
        end = start + self.num_local_experts  # [start, end)
        return start, end

    def _select_local_tokens(
        self,
        global_hidden: torch.Tensor,
        global_probs: torch.Tensor,
        global_scaling_factor: torch.Tensor | None,
        global_routing_map: torch.Tensor,
    ):
        start, end = self._local_expert_range()
        row_mask = global_routing_map[:, start:end].any(dim=1)

        dispatched_hidden = global_hidden[row_mask]
        dispatched_probs = (
            global_probs[row_mask, start:end] if global_probs is not None else None
        )
        dispatched_scaling_factor = (
            global_scaling_factor[row_mask]
            if global_scaling_factor is not None
            else None
        )

        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
        )

    def dispatch(
        self,
        hidden: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor = None,
        scaling_factor: torch.Tensor = None,
        local_expert_routing_map: torch.Tensor = None,
        out_token_num: int = None,
        enable_permute: bool = False,
        pad_multiple: int = 0,
    ):
        seq_len, hidden_dim = hidden.shape
        # Cache sizes for combine
        self._last_seq_len = seq_len
        self._last_hidden_dim = hidden_dim
        # gather the routing map
        global_routing_map = torch.empty(
            seq_len * self.group_size,
            self.num_of_experts,
            device=hidden.device,
            dtype=torch.bool,
        )
        torch.distributed.all_gather_into_tensor(
            global_routing_map, routing_map, self.ep_group
        )

        # dispatch the hidden tensor
        global_hidden = torch.empty(
            seq_len * self.group_size,
            hidden_dim,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        torch.distributed.all_gather_into_tensor(global_hidden, hidden, self.ep_group)

        # dispatch the probs tensor
        if probs is not None:
            global_probs = torch.empty(
                seq_len * self.group_size,
                self.num_of_experts,
                device=probs.device,
                dtype=probs.dtype,
            )
            torch.distributed.all_gather_into_tensor(global_probs, probs, self.ep_group)
        else:
            global_probs = None

        # dispatch the scaling factor tensor
        if scaling_factor is not None:
            global_scaling_factor = torch.empty(
                seq_len * self.group_size,
                hidden_dim // 128,
                device=scaling_factor.device,
                dtype=scaling_factor.dtype,
            )
            torch.distributed.all_gather_into_tensor(
                global_scaling_factor, scaling_factor, self.ep_group
            )
        else:
            global_scaling_factor = None

        (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
        ) = self._select_local_tokens(
            global_hidden=global_hidden,
            global_probs=global_probs,
            global_scaling_factor=global_scaling_factor,
            global_routing_map=global_routing_map,
        )

        if enable_permute:
            dispatched_hidden, dispatched_probs, dispatched_scaling_factor = permute(
                dispatched_hidden,
                local_expert_routing_map,
                dispatched_scaling_factor,
                dispatched_probs,
                out_token_num,
            )

            if pad_multiple > 0:
                token_per_expert = local_expert_routing_map.sum(dim=0)
                padd_m_split = [
                    (i + pad_multiple - 1) // pad_multiple * pad_multiple
                    for i in token_per_expert
                ]

                device = dispatched_hidden.device
                dtype = dispatched_hidden.dtype
                hidden_dim = dispatched_hidden.shape[1]

                padded_hidden_list = []
                padded_probs_list = [] if dispatched_probs is not None else None
                padded_scaling_list = (
                    [] if dispatched_scaling_factor is not None else None
                )

                start_idx = 0
                for expert_idx, (actual_tokens, padded_tokens) in enumerate(
                    zip(token_per_expert.tolist(), padd_m_split)
                ):
                    end_idx = start_idx + int(actual_tokens)
                    padding_size = padded_tokens - int(actual_tokens)

                    # pad token
                    expert_hidden = dispatched_hidden[start_idx:end_idx]
                    pad_hidden = torch.zeros(
                        padding_size, hidden_dim, device=device, dtype=dtype
                    )
                    expert_hidden = torch.cat([expert_hidden, pad_hidden], dim=0)
                    padded_hidden_list.append(expert_hidden)

                    # pad probs
                    if dispatched_probs is not None:
                        expert_probs = dispatched_probs[start_idx:end_idx]
                        pad_probs = torch.zeros(
                            padding_size, device=device, dtype=dispatched_probs.dtype
                        )
                        expert_probs = torch.cat([expert_probs, pad_probs], dim=0)
                        padded_probs_list.append(expert_probs)

                    # pad scaling factor
                    if dispatched_scaling_factor is not None:
                        expert_scaling = dispatched_scaling_factor[start_idx:end_idx]
                        pad_scaling = torch.zeros(
                            padding_size,
                            hidden_dim // 128,
                            device=device,
                            dtype=dispatched_scaling_factor.dtype,
                        )
                        expert_scaling = torch.cat([expert_scaling, pad_scaling], dim=0)
                        padded_scaling_list.append(expert_scaling)

                    start_idx = end_idx

                dispatched_hidden = torch.cat(padded_hidden_list, dim=0)
                if dispatched_scaling_factor is not None:
                    dispatched_scaling_factor = torch.cat(padded_scaling_list, dim=0)
                if dispatched_probs is not None:
                    dispatched_probs = torch.cat(padded_probs_list, dim=0)

        return (
            dispatched_hidden,
            dispatched_probs,
            dispatched_scaling_factor,
        )

def count_rdma_send_from_routing_map(routing_map: torch.Tensor, local_node_id: int, num_of_nodes: int):
    num_of_tokens, num_of_experts = routing_map.shape
    num_of_experts_per_node = num_of_experts // num_of_nodes

    # 1. reshape the routing map 
    routing_map = routing_map.reshape(num_of_tokens, num_of_nodes, num_of_experts_per_node)
    # # 2. Set the local node map to zeros
    # To align with the implementation in the DeepEP, we do not set the local node map to zeros.
    # routing_map[:, local_node_id, :] = 0
    # 3. reduce_max for unique routed tokens for each node 
    rdma_routing_map = routing_map.max(dim=-1).values 

    return rdma_routing_map.sum().item()


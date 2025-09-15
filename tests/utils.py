import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from typing import Optional, Union

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


def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

    
def cast_fp8_to_bf16(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)

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

def int32_to_8floats_lookup(tensor: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """
    Decomposes each int32 in the input tensor into 8 4-bit values,
    and converts them into float values using a lookup table.

    Args:
        tensor: (int32 Tensor) Tensor of any shape, e.g., [B, N]
        table: (float Tensor) A 1D lookup table of length 16 that maps all 4-bit values to floats

    Returns:
        float32 Tensor: Merges the last two dimensions, so shape is [..., n*M], where n is the number of int32 and 8 per int32.
    """
    assert tensor.dtype == torch.int32, "Input must be of int32 type"
    assert table.numel() == 16 and table.ndim == 1, "Lookup table must be 1D with length 16"

    result = []
    for i in range(8):
        shift = i * 4
        idx = ((tensor >> shift) & 0xF).long()  # Extract 4-bit index [0, 15]
        val = table[idx].unsqueeze(-1)  # Lookup and preserve dimensions
        result.append(val)

    out = torch.cat(result, dim=-1)  # Output shape: [..., 8]
    # Merge the last two dimensions if shape is [..., M, 8]
    out = out.reshape(*out.shape[:-2], -1) if out.ndim > 2 else out
    return out


def uint8_to_2floats_lookup(tensor: torch.Tensor) -> torch.Tensor:
    """
    Decomposes each uint8 in the input tensor into 2 4-bit values,
    and converts them into float values using a lookup table.

    Args:
        tensor: (uint8 Tensor) Tensor of any shape, e.g., [B, N]

    Returns:
        float32 Tensor: Merges the last two dimensions, so shape is [..., n*M], where n is the number of uint8 and 2 per uint8.
    """
    NVFP4_TABLE = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1.0, -1.5, -2, -3, -4, -6], dtype=torch.float32, device='cuda')
    assert tensor.dtype == torch.uint8, "Input must be of uint8 type"

    result = []
    for i in range(2):
        shift = i * 4
        idx = ((tensor >> shift) & 0xF).long()  # Extract 4-bit index [0, 15]
        val = NVFP4_TABLE[idx].unsqueeze(-1)  # Lookup and preserve dimensions
        result.append(val)

    out = torch.cat(result, dim=-1)  # Output shape: [..., 2]
    # Merge the last two dimension
    out = out.reshape(*out.shape[:-2], -1) if out.ndim > 1 else out
    return out


def cast_nvfp4_to_bf16(x_nvfp4: torch.Tensor, x_scales: torch.Tensor, x_global_scale: float, use_ue8m0_for_sf: bool = False):
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
    
    x_fp32 = uint8_to_2floats_lookup(x_nvfp4).to(torch.float32)
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

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
    # avoid NaN
    if x.numel() == 0 and y.numel() == 0:
        return 0.0
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()

def get_global_token_indices(distribution_type, num_experts, num_tokens, num_ranks, num_topk, 
                             imbalance_factor, simulation_seed=0):
    """
    Generates and returns the global top-k indices for all tokens across all ranks,
    matching the target imbalance factor.
    """
    if imbalance_factor <= 1.0:
        # use uniform
        print(f"Distribution: uniform, Target ratio: {imbalance_factor}", flush=True)
        expert_probs = torch.ones(num_experts, dtype=torch.float32, device='cuda')
        expert_probs /= expert_probs.sum()
        
        total_tokens = num_tokens * num_ranks
        token_probs_expanded = expert_probs.unsqueeze(0).expand(total_tokens, -1)
        
        torch.manual_seed(simulation_seed)
        global_topk_idx = torch.multinomial(token_probs_expanded, num_samples=num_topk, replacement=False)
        return global_topk_idx
    else:
        # Imbalanced case: find params and get indices directly
        params, global_topk_idx = find_distribution_parameters_for_target_ratio(
            distribution_type=distribution_type,
            target_imbalance_ratio=imbalance_factor,
            num_experts=num_experts,
            num_tokens_per_rank=num_tokens,
            num_ranks=num_ranks,
            num_topk=num_topk,
            simulation_seed=simulation_seed
        )
        
        print(f'Distribution: {distribution_type}, Target ratio: {imbalance_factor}, '
              f'Found parameters: {params}', flush=True)
        return global_topk_idx.contiguous()
    
def _simulate_global_sampling(
    distribution_type: str, 
    params_dict: dict, 
    num_experts: int, 
    total_tokens: int,
    num_ranks: int,
    num_topk: int,
    simulation_seed: int = 0,
    permutation: torch.Tensor = None
):
    """
    Simulates global sampling and returns both the resulting imbalance ratio and the generated indices.
    """
    num_local_experts = num_experts // num_ranks
    expert_probs = _generate_expert_probs(
        distribution_type, params_dict, num_experts, permutation=permutation
    )
    token_probs_expanded = expert_probs.unsqueeze(0).expand(total_tokens, -1)
    
    torch.manual_seed(simulation_seed)
    global_topk_idx = torch.multinomial(
        token_probs_expanded, num_samples=num_topk, replacement=False
    )
    
    rank_counts = torch.zeros(num_ranks, dtype=torch.int64, device='cuda')
    valid_indices = global_topk_idx.flatten()
    valid_indices = valid_indices[valid_indices >= 0]
    
    for rank in range(num_ranks):
        start_expert = rank * num_local_experts
        end_expert = (rank + 1) * num_local_experts
        mask = (valid_indices >= start_expert) & (valid_indices < end_expert)
        rank_counts[rank] = mask.sum().item()
    
    max_count = rank_counts.max().item()
    avg_count = rank_counts.float().mean().item()
    
    imbalance_ratio = max_count / avg_count if avg_count > 0 else 0.0
    
    return imbalance_ratio, global_topk_idx
def find_distribution_parameters_for_target_ratio(
    distribution_type: str,
    target_imbalance_ratio: float,
    num_experts: int,
    num_tokens_per_rank: int,
    num_ranks: int,
    num_topk: int,
    max_iterations=20,
    tolerance=0.02,
    simulation_seed=0
):
    """
    Finds parameters and returns them along with the final sampled indices.
    Returns:
        tuple: (dict of parameters, torch.Tensor of global_topk_idx)
    """
    total_tokens = num_tokens_per_rank * num_ranks
    torch.manual_seed(simulation_seed)
    permutation = torch.randperm(num_experts, device='cuda')
    
    def simulate_ratio_only(params_dict):
        ratio, _ = _simulate_global_sampling(
            distribution_type, params_dict, num_experts, total_tokens, 
            num_ranks, num_topk, simulation_seed, permutation=permutation
        )
        return ratio
    
    search_config = _get_search_config(distribution_type, target_imbalance_ratio)
    final_params = _binary_search_parameters(
        simulate_ratio_only, 
        search_config, 
        target_imbalance_ratio,
        max_iterations,
        tolerance
    )
    _, final_global_topk_idx = _simulate_global_sampling(
        distribution_type, final_params, num_experts, total_tokens,
        num_ranks, num_topk, simulation_seed, permutation=permutation 
    )
    return final_params, final_global_topk_idx
def _generate_expert_probs(distribution_type: str, params: dict, num_experts: int, permutation: torch.Tensor = None):
    """Generate expert probabilities, with optional shuffling."""
    if distribution_type == 'powerlaw':
        alpha = params['alpha']
        ranks = torch.arange(1, num_experts + 1, device='cuda', dtype=torch.float32)
        popularity_values = ranks ** (-alpha if alpha > 0 else 0)
        
    elif distribution_type == 'lognormal':
        sigma = params['sigma']
        log_normal_dist = torch.distributions.LogNormal(loc=0.0, scale=sigma)
        popularity_values = log_normal_dist.sample((num_experts,)).to('cuda')
        popularity_values, _ = torch.sort(popularity_values, descending=True)
        popularity_values.clamp_(min=1e-9)
        
    elif distribution_type == 'gamma':
        shape = params['shape']
        gamma_dist = torch.distributions.Gamma(concentration=shape, rate=2.0)
        popularity_values = gamma_dist.sample((num_experts,)).to('cuda')
        popularity_values, _ = torch.sort(popularity_values, descending=True)
        popularity_values.clamp_(min=1e-9)
        
    else:
        raise ValueError(f"Unsupported distribution: {distribution_type}")
    
    if permutation is not None:
        shuffled_values = torch.zeros_like(popularity_values)
        shuffled_values.scatter_(0, permutation, popularity_values)
        popularity_values = shuffled_values
    
    return popularity_values / popularity_values.sum()
def _get_search_config(distribution_type: str, target_ratio: float):
    """Get search bounds and parameter names for each distribution"""
    
    if distribution_type == 'powerlaw':
        return {
            'param_name': 'alpha',
            'low': 0.0,
            'high': 5.0
        }
        
    elif distribution_type == 'lognormal':
        return {
            'param_name': 'sigma', 
            'low': 0.1,
            'high': 5.0
        }
        
    elif distribution_type == 'gamma':
        return {
            'param_name': 'shape',
            'low': 0.001,  # Shape must be > 0
            'high': 2.0,
            'inverse_relationship': True # reverse binary search
        }
        
    else:
        raise ValueError(f"No search config for distribution: {distribution_type}")
def _binary_search_parameters(simulate_fn, search_config, target_ratio, max_iterations, tolerance):
    """Generic binary search for distribution parameters"""
    
    param_name = search_config['param_name']
    param_low = search_config['low']
    param_high = search_config['high']
    inverse_relationship = search_config.get('inverse_relationship', False) 
    
    for i in range(max_iterations):
        param_mid = (param_low + param_high) / 2
        actual_ratio = simulate_fn({param_name: param_mid})
        
        if abs(actual_ratio - target_ratio) / target_ratio < tolerance:
            return {param_name: param_mid}
        
        condition_for_increasing_param = actual_ratio < target_ratio
        if inverse_relationship:
            condition_for_increasing_param = not condition_for_increasing_param
        
        if condition_for_increasing_param:
            param_low = param_mid
        else:
            param_high = param_mid
    
    final_param = (param_low + param_high) / 2
    return {param_name: final_param}

def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


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
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
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

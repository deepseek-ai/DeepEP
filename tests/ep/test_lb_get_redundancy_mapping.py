"""Check LB prepare route identity, copy capacity, stable order and token-storage isolation."""
import argparse

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.testing import bench, generate_lognormal_topk


# Every `INVALID_SLOT_PERIOD`-th token carries an invalid first slot: the outputs must pass negative
# entries through untouched, not only route the valid ones
INVALID_SLOT_PERIOD = 7


def make_topk(num_ranks: int, num_experts: int, num_tokens: int, num_topk: int, seed: int) -> torch.Tensor:
    """Draw rank-specific routes from shared lognormal expert popularities, with masked slots."""
    topk_idx = torch.stack([generate_lognormal_topk(num_tokens, num_experts, num_topk, rank_idx, seed)
                        for rank_idx in range(num_ranks)]).to(deep_ep.topk_idx_t).cuda()
    topk_idx[:, ::INVALID_SLOT_PERIOD, 0] = -1
    return topk_idx


def histogram(values: torch.Tensor, size: int) -> torch.Tensor:
    """A `[size]` int64 count of the nonnegative ids in `values`."""
    return torch.zeros(size, dtype=torch.int64, device=values.device).index_add_(
        0, values.to(torch.int64), torch.ones_like(values, dtype=torch.int64))


def check(topk_idx: torch.Tensor, remapped_topk_idx: torch.Tensor, redundancy_mapping: torch.Tensor,
          num_experts: int, num_nvl_ranks: int, token_alignment: int, group: dist.ProcessGroup) -> None:
    """Check route identity, capacity, load conservation and stable destination order."""
    num_ranks = topk_idx.shape[0]
    num_rdma_ranks = num_ranks // num_nvl_ranks
    num_main_experts_per_rank = num_experts // num_ranks
    num_domain_experts = num_nvl_ranks * num_main_experts_per_rank
    assert redundancy_mapping.ndim == 2 and redundancy_mapping.shape[0] == num_nvl_ranks
    assert redundancy_mapping.dtype == torch.int32 and redundancy_mapping.device == topk_idx.device and redundancy_mapping.is_contiguous()
    num_redundant_experts_per_rank = redundancy_mapping.shape[1]
    assert 0 <= num_redundant_experts_per_rank <= num_main_experts_per_rank
    assert remapped_topk_idx.shape == topk_idx.shape[1:] and remapped_topk_idx.dtype == topk_idx.dtype
    assert bool(((redundancy_mapping >= -1) & (redundancy_mapping < num_domain_experts)).all()), 'invalid mapping expert'
    for nvl_rank_idx, row in enumerate(redundancy_mapping.cpu().tolist()):
        domain_expert_indices = [expert_idx for expert_idx in row if expert_idx >= 0]
        assert len(domain_expert_indices) == len(set(domain_expert_indices)), 'duplicate redundant expert'
        assert all(expert_idx // num_main_experts_per_rank != nvl_rank_idx for expert_idx in domain_expert_indices), 'redundant home expert'
    redundancy_mappings = [torch.empty_like(redundancy_mapping) for _ in range(num_ranks)]
    if num_redundant_experts_per_rank:
        dist.all_gather(redundancy_mappings, redundancy_mapping, group=group)
        for domain_idx in range(num_rdma_ranks):
            domain_slice = redundancy_mappings[domain_idx * num_nvl_ranks:(domain_idx + 1) * num_nvl_ranks]
            assert all(torch.equal(other, domain_slice[0]) for other in domain_slice[1:]), 'NVLink domain ranks disagree on the mapping'

    # Gather routes and preserve every negative input verbatim
    all_remapped_topk_idx = torch.empty(topk_idx.shape, dtype=remapped_topk_idx.dtype, device=topk_idx.device)
    if topk_idx.numel():
        dist.all_gather_into_tensor(all_remapped_topk_idx.view(-1), remapped_topk_idx.contiguous().view(-1), group=group)
    is_valid_route = topk_idx >= 0
    assert torch.equal(all_remapped_topk_idx[~is_valid_route], topk_idx[~is_valid_route]), 'invalid routes must be preserved'
    num_physical_experts_per_rank = num_main_experts_per_rank + num_redundant_experts_per_rank
    physical_expert_idx = all_remapped_topk_idx[is_valid_route].to(torch.int64)
    assert bool(((physical_expert_idx >= 0) & (physical_expert_idx < num_ranks * num_physical_experts_per_rank)).all()), 'invalid physical expert'
    dst_rank_idx = physical_expert_idx // num_physical_experts_per_rank
    dst_domain_idx = dst_rank_idx // num_nvl_ranks
    dst_nvl_rank_idx = dst_rank_idx % num_nvl_ranks
    local_physical_expert_idx = physical_expert_idx % num_physical_experts_per_rank
    logical_expert_idx = dst_rank_idx * num_main_experts_per_rank + local_physical_expert_idx
    is_redundant_expert = local_physical_expert_idx >= num_main_experts_per_rank
    if bool(is_redundant_expert.any()):
        domain_mappings = torch.stack([redundancy_mappings[domain_idx * num_nvl_ranks] for domain_idx in range(num_rdma_ranks)])
        domain_expert_idx = domain_mappings[
            dst_domain_idx[is_redundant_expert],
            dst_nvl_rank_idx[is_redundant_expert],
            local_physical_expert_idx[is_redundant_expert] - num_main_experts_per_rank,
        ].to(torch.int64)
        assert bool((domain_expert_idx >= 0).all()), 'route landed on an unassigned redundant slot'
        logical_expert_idx[is_redundant_expert] = dst_domain_idx[is_redundant_expert] * num_domain_experts + domain_expert_idx
    assert torch.equal(logical_expert_idx, topk_idx[is_valid_route].to(torch.int64)), 'routes must keep their expert'

    # Splitting an expert must conserve routes and add no aligned load
    num_tokens_per_expert = histogram(topk_idx[is_valid_route], num_experts)
    num_tokens_per_rank_expert = histogram(dst_rank_idx * num_experts + logical_expert_idx, num_ranks * num_experts).view(num_ranks, num_experts)
    assert torch.equal(num_tokens_per_rank_expert.sum(0), num_tokens_per_expert), 'expert counts changed'
    loads_per_rank_expert = (num_tokens_per_rank_expert + token_alignment - 1) // token_alignment
    assert torch.equal(loads_per_rank_expert.sum(0), (num_tokens_per_expert + token_alignment - 1) // token_alignment), 'a split expert must not add a load'
    has_partial_load = (num_tokens_per_rank_expert % token_alignment != 0) & (num_tokens_per_rank_expert > 0)
    assert int(has_partial_load.sum(0).max()) <= 1, 'an expert may keep its partial load on one destination only'

    # Exact balance holds within each NVLink domain at full copy capacity; zero slots preserve home placement
    loads_per_domain_rank = loads_per_rank_expert.sum(1).view(num_rdma_ranks, num_nvl_ranks)
    main_loads_per_domain_rank = ((num_tokens_per_expert + token_alignment - 1) // token_alignment).view(
        num_rdma_ranks, num_nvl_ranks, num_main_experts_per_rank).sum(2)
    total_loads_per_domain = main_loads_per_domain_rank.sum(1, keepdim=True)
    target_loads_per_domain_rank = total_loads_per_domain // num_nvl_ranks + (
        torch.arange(num_nvl_ranks, device=topk_idx.device) < total_loads_per_domain % num_nvl_ranks)
    if num_redundant_experts_per_rank == num_main_experts_per_rank:
        assert torch.equal(loads_per_domain_rank, target_loads_per_domain_rank), \
            f'aligned loads are unbalanced: {loads_per_domain_rank.tolist()}'
    elif num_redundant_experts_per_rank == 0:
        assert torch.equal(loads_per_domain_rank, main_loads_per_domain_rank), 'zero slots must preserve home placement'

    # Destinations are nondecreasing in each original expert's route sequence
    input_expert_idx = topk_idx[is_valid_route].to(torch.int64)
    for expert_idx in range(num_experts):
        dst_rank_indices = dst_rank_idx[input_expert_idx == expert_idx]
        assert bool((dst_rank_indices[1:] >= dst_rank_indices[:-1]).all()), f'expert {expert_idx} destinations are out of stable order'


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_experts = args.num_experts
    assert num_experts % num_ranks == 0 and 0 < args.num_topk <= num_experts
    num_main_experts_per_rank = num_experts // num_ranks
    num_benchmark_redundant_experts_per_rank = args.num_redundant_experts_per_rank
    assert num_benchmark_redundant_experts_per_rank is None or 0 <= num_benchmark_redundant_experts_per_rank <= num_main_experts_per_rank
    redundant_expert_counts_per_rank = (
        sorted({0, 1, num_main_experts_per_rank - 1, num_main_experts_per_rank})
        if num_benchmark_redundant_experts_per_rank is None else [num_benchmark_redundant_experts_per_rank]
    )
    num_benchmark_redundant_experts_per_rank = (
        num_main_experts_per_rank
        if num_benchmark_redundant_experts_per_rank is None else num_benchmark_redundant_experts_per_rank
    )
    buffer = deep_ep.EPBuffer(group, num_bytes=deep_ep.get_num_allocation_alignment(), explicitly_destroy=True)
    num_rdma_ranks, num_nvl_ranks = buffer.num_rdma_ranks, buffer.num_nvlink_ranks

    def prepare(input_topk_idx: torch.Tensor, token_alignment: int, num_case_redundant_experts_per_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        remapped_topk_idx, redundancy_mapping = buffer.lb_get_redundancy_mapping(input_topk_idx, num_experts, num_case_redundant_experts_per_rank, token_alignment, args.num_sms)
        return remapped_topk_idx, redundancy_mapping

    def verify(all_topk_idx: torch.Tensor, token_alignment: int, num_case_redundant_experts_per_rank: int) -> None:
        original_topk_idx = all_topk_idx[rank_idx].clone()
        remapped_topk_idx, redundancy_mapping = prepare(all_topk_idx[rank_idx], token_alignment, num_case_redundant_experts_per_rank)
        assert redundancy_mapping.shape == (num_nvl_ranks, num_case_redundant_experts_per_rank)
        check(all_topk_idx, remapped_topk_idx, redundancy_mapping, num_experts, num_nvl_ranks, token_alignment, group)
        repeated_remapped_topk_idx, repeated_redundancy_mapping = prepare(all_topk_idx[rank_idx], token_alignment, num_case_redundant_experts_per_rank)
        assert torch.equal(repeated_remapped_topk_idx, remapped_topk_idx) and torch.equal(repeated_redundancy_mapping, redundancy_mapping), 'prepare is not repeatable'
        assert torch.equal(all_topk_idx[rank_idx], original_topk_idx), 'prepare modified its input'

    dist_print(f'Config: ranks={num_rdma_ranks}x{num_nvl_ranks}, experts={num_experts}, tokens={args.num_tokens}, '
               f'top-k={args.num_topk}, token_alignments={args.token_alignments}, copy_counts={redundant_expert_counts_per_rank}', once_in_node=True)
    topk_idx = make_topk(num_ranks, num_experts, args.num_tokens, args.num_topk, args.seed)

    # Q is required; positional and keyword calls must use the same order.
    try:
        buffer.lb_get_redundancy_mapping(topk_idx[rank_idx], num_experts)
    except TypeError:
        pass
    else:
        raise AssertionError('lb_get_redundancy_mapping accepted a missing redundant slot count')
    full_remapped_topk_idx, full_redundancy_mapping = prepare(topk_idx[rank_idx], 1, num_main_experts_per_rank)
    keyword_remapped_topk_idx, keyword_redundancy_mapping = buffer.lb_get_redundancy_mapping(
        topk_idx[rank_idx], num_experts, num_redundant_experts_per_rank=num_main_experts_per_rank,
        token_alignment=1, num_sms=args.num_sms)
    assert torch.equal(full_remapped_topk_idx, keyword_remapped_topk_idx) and torch.equal(full_redundancy_mapping, keyword_redundancy_mapping)

    for token_alignment in args.token_alignments:
        for num_case_redundant_experts_per_rank in redundant_expert_counts_per_rank:
            verify(topk_idx, token_alignment, num_case_redundant_experts_per_rank)
            dist_print(f'PASS {token_alignment=}, {num_case_redundant_experts_per_rank=}', once_in_node=True)

    # A reduced SM count still has to cover every route, so it must reproduce the full-SM result
    for token_alignment in args.token_alignments:
        full_remapped, full_mapping = buffer.lb_get_redundancy_mapping(
            topk_idx[rank_idx], num_experts, num_main_experts_per_rank, token_alignment, 0)
        for num_sms in (1, 8):
            sm_remapped, sm_mapping = buffer.lb_get_redundancy_mapping(
                topk_idx[rank_idx], num_experts, num_main_experts_per_rank, token_alignment, num_sms)
            assert torch.equal(sm_remapped, full_remapped), f'{num_sms=} dropped routes'
            assert torch.equal(sm_mapping, full_mapping), f'{num_sms=} changed the mapping'
        dist_print(f'PASS reduced SM sweep {token_alignment=}', once_in_node=True)

    # Empty input, fully masked ranks, tails, and the extremes of the index dtype
    index_info = torch.iinfo(deep_ep.topk_idx_t)
    negative_topk_indices = [-1, -7, index_info.min, index_info.min + 1]
    masked_topk_idx = torch.tensor(negative_topk_indices, dtype=deep_ep.topk_idx_t,
                                   device=topk_idx.device).view(1, -1, 1).repeat(num_ranks, 1, 1)
    mixed_topk_idx = torch.cat((masked_topk_idx, torch.zeros((num_ranks, 1, 1), dtype=deep_ep.topk_idx_t,
                                                             device=topk_idx.device)), dim=1)
    uniform_topk_idx = torch.arange(num_experts, dtype=deep_ep.topk_idx_t,
                                    device=topk_idx.device).view(1, -1, 1).repeat(num_ranks, 1, 1)
    for all_topk_idx in (topk_idx[:, :0], masked_topk_idx, mixed_topk_idx, uniform_topk_idx):
        for num_case_redundant_experts_per_rank in redundant_expert_counts_per_rank:
            verify(all_topk_idx, 1, num_case_redundant_experts_per_rank)
            verify(all_topk_idx, 128, num_case_redundant_experts_per_rank)

    # Prepare scratch must not overlap token storage, before or after resize
    for resize in (False, True):
        if resize:
            buffer.resize(buffer.num_bytes, 0)
        token_storage = buffer.runtime.lb_storage.as_strided((buffer.num_bytes,), (1,), storage_offset=0)
        token_storage.fill_(0xa5)
        input_topk_idx = token_storage[:num_experts * deep_ep.topk_idx_t.itemsize].view(deep_ep.topk_idx_t).view(-1, 1)
        input_topk_idx.copy_(uniform_topk_idx[rank_idx])
        token_storage_before = token_storage.clone()
        remapped_topk_idx, redundancy_mapping = prepare(input_topk_idx, 1, num_main_experts_per_rank)
        check(uniform_topk_idx, remapped_topk_idx, redundancy_mapping, num_experts, num_nvl_ranks, 1, group)
        assert torch.equal(token_storage, token_storage_before), 'prepare scratch overlaps token storage'
        del input_topk_idx, token_storage, token_storage_before
    dist_print('Boundary and storage-isolation checks passed', once_in_node=True)

    num_routes = num_ranks * args.num_tokens * args.num_topk
    dist_print('Performance:', once_in_node=True)
    for token_alignment in args.token_alignments:
        mean, minimum, _ = bench(lambda: prepare(topk_idx[rank_idx], token_alignment, num_benchmark_redundant_experts_per_rank))
        dist_print(f' > token_alignment={token_alignment:3d}, {num_routes} routes: '
                   f'{minimum * 1e6:7.1f} us min, {mean * 1e6:7.1f} us mean, '
                   f'{num_routes / minimum / 1e6:6.1f} M routes/s', once_in_node=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test LB prepare')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--num-tokens', type=int, default=8192, help='Tokens per rank')
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--token-alignments', type=int, nargs='+', default=[128, 1])
    parser.add_argument('--num-sms', type=int, default=0, help='0 uses all SMs')
    parser.add_argument('--num-redundant-experts-per-rank', type=int, default=None,
                        help='Explicit slot count; default sweeps capacity boundaries and benchmarks full capacity')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

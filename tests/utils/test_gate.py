import torch

from deep_ep.utils.math import ceil_div
from deep_ep.utils.gate import get_unbalanced_scores


def test_unbalanced_scores():
    print('Testing gate score generation (Output with num_tokens = 4096, num_experts = 512):')
    for num_tokens in [1, 4096]:
        for num_experts_per_rank in [1, 4, 8, 16]:
            for num_ranks in [2, 4, 8, 16, 64, 72]:
                num_experts = num_experts_per_rank * num_ranks
                for num_topk in [1, 2, 4, 6, 8, 9]:
                    if num_topk > num_experts:
                        continue
                    for ratio in [1.0, 2.0, 4.0]:
                        for precise, synthetic in [(1, 0), (0, 0), (0, 1)]:
                            total_rank_count = torch.zeros(num_ranks, device='cuda')

                            # This is the requirement from precise generation algorithm
                            lower_bound_per_token = max(1, ceil_div(num_topk, num_experts_per_rank))
                            upper_bound_per_token = min(min(num_topk, num_ranks), int((num_ranks - 1) / ratio) + 1)
                            if lower_bound_per_token > upper_bound_per_token:
                                continue

                            # Repeat for each rank
                            for rank_idx in range(num_ranks):
                                scores = get_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, ratio, precise, synthetic)
                                _topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
                                assert torch.isfinite(_topk_weights).all()
                                assert (topk_idx.sort(dim=-1).values.diff(dim=-1) != 0).all()
                                topk_idx = topk_idx // num_experts_per_rank
                                row_indices = torch.arange(num_tokens).unsqueeze(1).expand(num_tokens, num_topk).flatten()
                                topk_idx = topk_idx.flatten()
                                rank_count = torch.zeros((num_tokens, num_ranks), device='cuda')
                                rank_count[row_indices, topk_idx] = 1
                                rank_count = rank_count.sum(dim=0)
                                total_rank_count += rank_count

                            # Calculate the actual ratio and inequality
                            max_count = total_rank_count.max().item()
                            other_mean = (total_rank_count.sum().item() - max_count) / (num_ranks - 1) if num_ranks > 1 else 1
                            practical_ratio = max_count / max(other_mean, 1)
                            inequality = total_rank_count[1:].max().item() / max(total_rank_count[1:].min().item(), 1)
                            total_sent_tokens = int(total_rank_count.sum().item())
                            if num_tokens > 1000:
                                if num_ranks in [8, 64] and num_experts_per_rank == 8:
                                    print(f' > {precise=}, {synthetic=}, {num_ranks=:2d}, {num_topk=}, expected_ratio={ratio} | '
                                        f'ratio={practical_ratio:6.3f}, {inequality=:6.3f}, {total_sent_tokens=:7d}')

                                # Only check the ratio and inequality in precise mode
                                if precise:
                                    assert abs(practical_ratio - ratio) / ratio < 0.1 and inequality < 1.02, \
                                            f'Failed to generate unbalanced scores with following config: \n' \
                                            f'{precise=}, {num_tokens=}, {num_experts=:3d}, {num_ranks=:2d}, {num_topk=}, expected_ratio={ratio} | ' \
                                            f'ratio={practical_ratio:6.3f}, {inequality=:6.3f}, {total_sent_tokens=:7d}'
    print()


if __name__ == '__main__':
    test_unbalanced_scores()

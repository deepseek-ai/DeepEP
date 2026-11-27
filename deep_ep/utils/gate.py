import torch


# Precise mode
def _generate_topk_idx(rank_count: torch.Tensor, num_tokens: int, num_experts: int, num_ranks: int, num_topk: int) -> torch.Tensor:
    """
    Map rank count to expert indices
    """
    assert torch.equal(torch.sum(rank_count, dim=1), torch.ones(num_tokens, dtype=torch.int, device='cuda') * num_topk)
    assert (num_tokens, num_ranks) == rank_count.shape
    num_experts_per_rank = num_experts // num_ranks

    # Generate base value
    base_vals = torch.arange(num_experts, device='cuda').view(1, num_ranks, num_experts_per_rank).expand(num_tokens, num_ranks, num_experts_per_rank)

    # Randomize the ordering within each row
    rand_vals = torch.rand(num_tokens, num_ranks, num_experts_per_rank, device='cuda')
    perm_indices = torch.argsort(rand_vals, dim=-1)
    permuted = torch.gather(base_vals, 2, perm_indices)

    # Create the mask
    k_idx = torch.arange(num_experts_per_rank, device='cuda').view(1, 1, num_experts_per_rank).expand(num_tokens, num_ranks, num_experts_per_rank)
    rank_count_expanded = rank_count.unsqueeze(2).expand(num_tokens, num_ranks, num_experts_per_rank)
    mask = k_idx < rank_count_expanded

    # Get the final indices by masking and reshaping
    selected = permuted[mask]  # (num_tokens * num_topk,)
    topk_idx = selected.view(num_tokens, num_topk)

    return topk_idx


def _generate_rank_count(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float) -> torch.Tensor:
    """
    Generate rank count tensor for a given number of tokens, experts, ranks, and top-k.

    This function generates a tensor of shape `(num_tokens, num_ranks)` where each element `[i, j]` represents
    the number of topk experts that token `i` have on rank `j`. The distribution is such that
    one special rank gets `ratio` times more traffic than the others.
    """
    num_experts_per_rank = num_experts // num_ranks
    num_normal_ranks = num_ranks - 1

    assert ratio >= 1.0, 'ratio must be no less than 1.0'

    # Generate rank count of each token from random distribution
    random_scores = torch.rand(num_tokens, num_experts, device='cuda')
    topk_weights_, topk_indices = torch.topk(random_scores, num_topk, dim=1, largest=True, sorted=False)
    topk_indices //= num_experts_per_rank
    sorted_topk_indices = torch.sort(topk_indices, dim=1)[0]
    topk_indices_diff_mask = sorted_topk_indices[:, 1:] != sorted_topk_indices[:, :-1]
    a = topk_indices_diff_mask.sum(dim=1) + 1

    # Upper bound for this generating algorithm
    upper_bound_per_token = int(num_normal_ranks / ratio) + 1

    # Clamp the value in range [1, upper_bound_per_token] for each token
    a = torch.clamp(a, None, upper_bound_per_token)

    # Consider the special rank
    sum_a = torch.sum(a).item()
    normal_token_count = int(sum_a / (num_normal_ranks + ratio))
    special_token_count = sum_a - normal_token_count * num_normal_ranks
    special_token_count = min(special_token_count, int(normal_token_count * ratio) + 1)

    # Tokens that the special rank must be in topk
    must_mask = (a == num_ranks)
    must_count = int(must_mask.sum().item())
    special_token_count = max(must_count, special_token_count)
    assert must_count <= special_token_count, 'Too many tokens with full rank assignment'

    # Tokens that the special rank can optionally be in topk
    optional_token_indices = torch.where(must_mask == 0)[0]
    optional_token_indices = optional_token_indices[torch.randperm(num_tokens - must_count, device='cuda')][:special_token_count - must_count]
    must_token_indices = torch.where(must_mask != 0)[0]
    special_token_row_index = torch.cat(([must_token_indices, optional_token_indices]))

    # Generate permutations for normal ranks
    rank_perm = (torch.randperm(num_normal_ranks, device='cuda') + 1).repeat(num_tokens * num_topk // num_normal_ranks + 1)

    # Compute cumulative sum of a to get starting indices in b for each row
    a_cumsum = torch.cumsum(torch.cat((torch.tensor([0], device='cuda'), a)), dim=0)
    row_starts = a_cumsum[:-1]  # Starting indices for each row in b, shape (n,)

    # Insert special rank index into the permutation for special tokens
    rank_perm_with_special_rank = torch.zeros(num_tokens * num_topk, dtype=torch.long, device='cuda')  # (n * k,)
    special_token_mask = torch.zeros(num_tokens * num_topk, dtype=torch.bool, device='cuda')
    special_token_flattened_row_index = row_starts[special_token_row_index]
    special_token_mask[special_token_flattened_row_index] = 1
    all_indices = torch.arange(num_tokens * num_topk, device='cuda')
    non_special_indices = all_indices[special_token_mask != True]
    rank_perm_with_special_rank[non_special_indices] = rank_perm[:len(non_special_indices)]

    # Create column index grids
    col_idx = torch.arange(num_topk, device='cuda').view(1, num_topk)  # (1, num_topk)

    # Compute modulo indices: col_idx % a[i] for each row
    # torch.max is used to avoid zeros in case a[i] = 0 (which happens when the only topk rank is the special rank)
    mod_idx = col_idx % a.view(num_tokens, 1)  # (n, num_topk)

    # Compute indices in b: row_start + (col % a[i])
    b_indices = row_starts.view(num_tokens, 1) + mod_idx  # (n, k)

    # Gather values from b using computed indices
    result = rank_perm_with_special_rank[b_indices]

    # Shuffle rows randomly to avoid any pattern
    shuffle_indices = torch.randperm(num_tokens, device='cuda')
    result = result[shuffle_indices]  # Shuffle rows

    # Create rank count tensor
    rank_count = torch.zeros((num_tokens, num_ranks), dtype=torch.int32, device='cuda')
    rank_count.scatter_add_(dim=1, index=result, src=torch.ones_like(result, dtype=torch.int32))
    return rank_count


def get_precise_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float):
    """
    Generate precise unbalanced scores for testing.

    Note that this function generates a distribution with precise unbalanced distribution,
    which **differs from real distribution**.
    """
    # Generate num topk experts for each rank
    rank_count = _generate_rank_count(num_tokens, num_experts, num_ranks, num_topk, ratio)

    # Generate scores in a low distribution
    threshold = 0.9
    scores = torch.empty((num_tokens, num_experts), dtype=torch.float32, device='cuda')
    scores.uniform_(to=threshold)

    # Generate topk indices and change their scores to a high distribution
    topk_idx = _generate_topk_idx(rank_count, num_tokens, num_experts, num_ranks, num_topk)
    topk_scores = torch.empty((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    topk_scores.uniform_(threshold + 1e-6, 1.0)
    row_idx = torch.arange(num_tokens).unsqueeze(1).expand(num_tokens, num_topk)
    scores[row_idx, topk_idx] = topk_scores
    return scores


# Random mode
def _get_scores_by_factor(num_tokens: int, num_experts: int, num_ranks: int, factor: float) -> torch.Tensor:
    num_experts_per_rank = num_experts // num_ranks
    scores = torch.empty((num_tokens, num_experts), dtype=torch.float32, device='cuda')
    scores[:, :num_experts_per_rank].uniform_(to=factor)
    scores[:, num_experts_per_rank:].uniform_(to=1)
    return scores


def _map_unbalanced_ratio_to_factor(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float) -> float:
    num_iterations = 20
    factor_l, factor_r = 1.0, 100.0

    num_experts_per_rank = num_experts // num_ranks
    for _i in range(num_iterations):
        factor_mid = (factor_l + factor_r) / 2
        scores = _get_scores_by_factor(num_tokens, num_experts, num_ranks, factor_mid)
        _, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        rank_idx = topk_idx // num_experts_per_rank
        one_hot = torch.nn.functional.one_hot(rank_idx, num_ranks)
        counts = one_hot.any(dim=1).to(torch.float).sum(dim=0)
        if counts[0].item() > counts[1:].mean().item() * ratio:
            factor_r = factor_mid
        else:
            factor_l = factor_mid
    return factor_l


def get_random_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float):
    """Generate unbalanced scores with a given ratio.
    """
    factor = 1.0
    if ratio != 1.0:
        factor = _map_unbalanced_ratio_to_factor(num_tokens, num_experts, num_ranks, num_topk, ratio)
    return _get_scores_by_factor(num_tokens, num_experts, num_ranks, factor)


# Synthetic mode
def _count_tokens_per_rank(topk_idx: torch.Tensor, num_experts_per_rank: int, num_ranks: int) -> torch.Tensor:
    counts = torch.zeros((topk_idx.size(0), num_ranks), dtype=torch.int, device=topk_idx.device)
    counts.scatter_(1, topk_idx // num_experts_per_rank, 1)
    return counts.sum(dim=0)


def _generate_synthetic_logits(num_tokens: int, num_experts: int) -> torch.Tensor:
    # Routing priors (obtained from real scores)
    # These are independent hyperparameters that can be tuned to match the real score distribution
    SOURCE_VARIANCE = 0.18
    BLOCK_VARIANCE = 0.20
    TOKEN_VARIANCE = 0.62
    BLOCK_SIZE = 512
    NEIGHBOR_MIX = 0.37

    # Construct scores from source preferences, block preferences, and adjacent-token noise correlation
    source_bias = torch.randn((num_experts,), dtype=torch.float32, device='cuda')
    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_bias = torch.randn((num_blocks, num_experts), dtype=torch.float32, device='cuda')
    white = torch.randn((num_blocks * BLOCK_SIZE + 1, num_experts), dtype=torch.float32, device='cuda')
    scores = torch.add(white[1:], white[:-1], alpha=NEIGHBOR_MIX)
    scores.mul_((TOKEN_VARIANCE / (1 + NEIGHBOR_MIX ** 2)) ** 0.5)
    scores.add_(source_bias, alpha=SOURCE_VARIANCE ** 0.5)
    scores.view(num_blocks, BLOCK_SIZE, num_experts).add_(block_bias[:, None, :], alpha=BLOCK_VARIANCE ** 0.5)
    return scores[:num_tokens]


def _calibrate_synthetic_rank_bias(scores: torch.Tensor, num_ranks: int, num_topk: int,
                                    ratio: float) -> torch.Tensor:
    # Load calibration hyperparameters
    # These are independent hyperparameters that control the calibration process
    MAX_ITERS = 64
    TOLERANCE = 0.05
    LEARNING_RATE = 0.4
    DECAY_AFTER = 8
    MAX_BIAS_STEP = 1.0
    COUNT_SMOOTHING = 1.0
    LOG_EPSILON = 1e-12

    num_tokens, num_experts = scores.shape
    num_local_experts = num_experts // num_ranks
    topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False).indices
    counts = _count_tokens_per_rank(topk_idx, num_local_experts, num_ranks).float()

    # Keep background preferences, with mean 1 and maximum at most (1 + ratio) / 2
    preference = counts[1:] + COUNT_SMOOTHING
    preference /= preference.mean()
    mix = ((ratio - 1) / (2 * (preference.max() - 1).clamp_min(LOG_EPSILON))).clamp(max=1)
    target = torch.ones(num_ranks, dtype=torch.float32, device=scores.device)
    target[1:] += mix * (preference - 1)
    target[0] = ratio

    # Fix input logits, adjust bias, and retain the best one from the bounded iteration
    bias = torch.zeros_like(target)
    best_bias, best_error = bias.clone(), float('inf')
    for iteration in range(MAX_ITERS):
        if iteration:
            logits = (scores.view(num_tokens, num_ranks, num_local_experts) + bias[None, :, None]).view(num_tokens, num_experts)
            topk_idx = torch.topk(logits, num_topk, dim=-1, sorted=False).indices
            counts = _count_tokens_per_rank(topk_idx, num_local_experts, num_ranks).float()
        # Check the whole target distribution, relative to the mean background load
        observed = counts / counts[1:].mean().clamp_min(LOG_EPSILON)
        error = (observed - target).abs().max().item() / ratio
        if error < best_error:
            best_error, best_bias = error, bias.clone()
        if error <= TOLERANCE:
            break
        # One optimization step towards the target
        observed = (counts + COUNT_SMOOTHING) / (counts[1:].mean() + COUNT_SMOOTHING)
        step = LEARNING_RATE * min(1., DECAY_AFTER / (iteration + 1))
        bias.add_((step * (target.clamp_min(LOG_EPSILON).log() - observed.log())).clamp(-MAX_BIAS_STEP, MAX_BIAS_STEP))
        bias.sub_(bias.mean())

    return best_bias


def get_synthetic_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int,
                                    ratio: float) -> torch.Tensor:
    """Generate synthetic unbalanced scores for testing.

    It uses statistical priors from actual routing data to create a synthetic distribution.
    Due to its approximate nature, the generated distribution may **differ from real distribution**.
    If the requested ratio is unachievable, the function will return the best scores found.
    """
    assert num_tokens >= 0 and 1 <= num_topk <= num_experts
    assert num_ranks > 0 and num_experts % num_ranks == 0
    assert ratio >= 1.0

    scores = _generate_synthetic_logits(num_tokens, num_experts)
    if num_tokens and num_ranks > 1:
        bias = _calibrate_synthetic_rank_bias(scores, num_ranks, num_topk, ratio)
        scores.view(num_tokens, num_ranks, -1).add_(bias[None, :, None])
    return torch.nn.functional.softplus(scores)


def get_unbalanced_scores(num_tokens: int, num_experts: int, num_ranks: int, num_topk: int, ratio: float,
                          precise: bool = False, synthetic: bool = False):
    if synthetic:
        return get_synthetic_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, ratio)
    elif precise:
        return get_precise_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, ratio)
    else:
        return get_random_unbalanced_scores(num_tokens, num_experts, num_ranks, num_topk, ratio)

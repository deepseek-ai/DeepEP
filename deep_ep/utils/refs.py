import math
import torch
import torch.distributed as dist
from typing import Optional, Tuple, Union

from .envs import get_global_seed
from .math import ceil_div


def dispatch(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
             topk_idx: torch.Tensor, topk_weights: Optional[torch.Tensor],
             num_max_tokens_per_rank: int, num_experts: int):
    """
    The reference implementation of dispatching tokens to experts across multiple ranks.

    Not expanded. Sorted by rank and then by token within each rank (i.e. sorted by `src_token_global_idx`).

    Arguments:
    - `x`: Input tokens, `[num_tokens, hidden]` or (`[num_tokens, hidden], [num_tokens, hidden_sf]`)
    - `topk_idx`: Top-k expert indices for each token, `[num_tokens, num_topk]`
    - `topk_weights`: Top-k weights for each token, can be None, `[num_tokens, num_topk]`
    - `num_max_tokens_per_rank`: Maximum number of tokens per rank, must >= actual number of tokens per rank and aligned with DeepEP's `num_max_tokens_per_rank` since we're going to calculate `src_token_global_idx = src_rank_idx*num_max_tokens_per_rank + src_token_local_idx`
    - `num_experts`: Total number of experts across all ranks

    Returns:
    - `recv_x`, `recv_topk_idx`, and `recv_topk_weights`: Received tokens, top-k indices, and top-k weights after dispatching. Out of range `recv_topk_idx` (i.e. that expert is not on the current rank) are set to -1. `recv_topk_weights` is `None` when `topk_weights` is not provided.
    - `recv_src_token_idx`: Received `src_token_global_idx` for each received token
    - `num_recv_tokens_per_rank`: Number of received tokens from each rank, `[num_ranks]`
    """
    # TODO: support forwarding
    rank_idx = dist.get_rank()
    num_ranks = dist.get_world_size()

    assert num_experts % num_ranks == 0
    num_experts_per_rank = num_experts // num_ranks

    # Unpack SF
    use_fp8 = isinstance(x, tuple)
    x, sf = x if use_fp8 else (x, None)

    # TODO: use SF bytes instead of hardcoded recipe
    num_tokens, hidden = x.size()
    num_tokens_, num_topk = topk_idx.size()
    assert num_tokens == num_tokens_
    if sf is not None:
        num_tokens__, hidden_sf = sf.size()
        assert num_tokens == num_tokens__
        assert hidden_sf == ceil_div(hidden, 128)
    if topk_weights is not None:
        num_tokens__, num_topk_ = topk_weights.size()
        assert num_tokens == num_tokens__
        assert num_topk == num_topk_

    # Prepare per-peer send buffers
    send_x_list = []
    send_sf_list = []
    send_topk_idx_list = []
    send_topk_weights_list = []
    send_src_token_idx_list = []
    num_send_tokens_per_rank = torch.zeros((num_ranks, ), dtype=torch.int, device=x.device)
    for dst_rank_idx in range(num_ranks):
        expert_start_idx = dst_rank_idx * num_experts_per_rank
        expert_end_idx = expert_start_idx + num_experts_per_rank

        # Get the indices of tokens
        mask_to_send = ((expert_start_idx <= topk_idx) & (topk_idx < expert_end_idx)).any(dim=1)
        indices_to_send = mask_to_send.nonzero(as_tuple=True)[0]
        num_send_tokens_per_rank[dst_rank_idx] = indices_to_send.numel()

        # Select the data for tokens
        x_to_send = x[indices_to_send]
        sf_to_send = sf[indices_to_send] if use_fp8 else None
        topk_idx_to_send = topk_idx[indices_to_send]
        topk_weights_to_send = topk_weights[indices_to_send] if topk_weights is not None else None
        masked_topk_idx = torch.where((expert_start_idx <= topk_idx_to_send) & (topk_idx_to_send < expert_end_idx),
                                      topk_idx_to_send, torch.full_like(topk_idx_to_send, -1))

        send_x_list.append(x_to_send)
        send_sf_list.append(sf_to_send)
        send_topk_idx_list.append(masked_topk_idx)
        send_topk_weights_list.append(topk_weights_to_send)
        send_src_token_idx_list.append(indices_to_send)

    send_x = torch.cat(send_x_list, dim=0)
    send_sf = torch.cat(send_sf_list, dim=0) if use_fp8 else None
    send_topk_idx = torch.cat(send_topk_idx_list, dim=0)
    send_topk_weights = torch.cat(send_topk_weights_list, dim=0) if topk_weights is not None else None
    send_src_token_idx = torch.cat(send_src_token_idx_list, dim=0).to(torch.int)
    send_src_token_idx += rank_idx * num_max_tokens_per_rank

    # Exchange size
    num_recv_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device=x.device)
    dist.all_to_all_single(num_recv_tokens_per_rank, num_send_tokens_per_rank)
    num_recv_tokens = int(num_recv_tokens_per_rank.sum().item())

    # Exchange main data
    num_send_tokens_per_rank = num_send_tokens_per_rank.tolist()
    num_recv_tokens_per_rank = num_recv_tokens_per_rank.tolist()
    recv_x = torch.empty((num_recv_tokens, hidden), dtype=x.dtype, device=x.device)
    recv_sf = torch.empty((num_recv_tokens, hidden_sf), dtype=sf.dtype, device=x.device) if use_fp8 else None
    recv_topk_idx = torch.empty((num_recv_tokens, num_topk), dtype=topk_idx.dtype, device=x.device)
    recv_topk_weights = torch.empty((num_recv_tokens, num_topk), dtype=topk_weights.dtype, device=x.device) \
        if topk_weights is not None else None
    recv_src_token_idx = torch.empty((num_recv_tokens, ), dtype=torch.int, device=x.device)
    dist.all_to_all_single(recv_x, send_x, num_recv_tokens_per_rank, num_send_tokens_per_rank)
    if use_fp8:
        dist.all_to_all_single(recv_sf, send_sf, num_recv_tokens_per_rank, num_send_tokens_per_rank)
    dist.all_to_all_single(recv_topk_idx, send_topk_idx, num_recv_tokens_per_rank, num_send_tokens_per_rank)
    if topk_weights is not None:
        dist.all_to_all_single(recv_topk_weights, send_topk_weights, num_recv_tokens_per_rank, num_send_tokens_per_rank)
    dist.all_to_all_single(recv_src_token_idx, send_src_token_idx, num_recv_tokens_per_rank, num_send_tokens_per_rank)

    # Mask top-k indices
    expert_start_idx = rank_idx * num_experts_per_rank
    expert_end_idx = expert_start_idx + num_experts_per_rank
    mask = (expert_start_idx <= recv_topk_idx) & (recv_topk_idx < expert_end_idx)
    recv_topk_idx = recv_topk_idx - expert_start_idx
    recv_topk_idx.masked_fill_(~mask, -1)

    # Pack SF
    recv_x = (recv_x, recv_sf) if use_fp8 else recv_x

    return (recv_x, recv_topk_idx, recv_topk_weights,
            recv_src_token_idx, torch.tensor(num_recv_tokens_per_rank, dtype=torch.int))


def generate_pre_combine_data(src_token_global_idx: torch.Tensor,
                              num_max_tokens_per_rank: int, num_topk: int, hidden: int) -> torch.Tensor:
    """
    Generate data needed for combine from `src_token_global_idx`.
    Recall that `src_token_global_idx = src_rank_idx * num_max_tokens_per_rank + src_token_local_idx`.
    The generated data (denoted as `y`) of the i-th token has a shape of [num_topk, hidden], with

    `y[j, k] = sin((token_seeds * P % max_seed + 1) / max_seed * (k + 1) + sin(seed))`

    where `P=131071` is a large prime, `token_seeds` is calculated via `token_seeds = src_token_global_idx[i] * num_topk + j`,
    and `max_seed = num_ranks * num_max_tokens_per_rank * num_topk`.

    Arguments:
    - `src_token_global_idx`: Source token global indices, `[num_tokens]`

    Returns:
    - Generated data, `[num_tokens, num_topk, hidden]`
    """
    num_ranks = dist.get_world_size()
    token_seeds = (src_token_global_idx.unsqueeze(1) * num_topk +
                   torch.arange(num_topk, device=src_token_global_idx.device).unsqueeze(0))  # [num_tokens, num_topk]
    max_seed = num_ranks * num_max_tokens_per_rank * num_topk
    result = torch.sin(
        (((token_seeds * 131071 % max_seed).float() + 1) / max_seed).unsqueeze(-1) *
        torch.arange(1, hidden + 1, device=src_token_global_idx.device, dtype=torch.float32).broadcast_to(1, 1, hidden) +
        math.sin(float(get_global_seed()))
    )
    return result.to(torch.bfloat16)


def ordered_accumulate(data: torch.Tensor, initial_value: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Accumulate `data` in order along the num_topk dimension.

    Arguments:
    - `data`: Data to be accumulated, `[num_tokens, num_topk, hidden]`
    - `initial_value`: Initial value for accumulation, `[num_tokens, hidden]`

    Returns:
    - Result, `[num_tokens, hidden]`
    """
    num_topk = data.shape[1]
    if initial_value is None:
        result = torch.zeros((data.shape[0], data.shape[2]), dtype=torch.float32, device=data.device)
    else:
        result = initial_value.clone()
    for i in range(num_topk):
        result += data[:, i, :].float()
    return result.to(data.dtype)


def combine(y: torch.Tensor, topk_idx: torch.Tensor,
            num_scaleout_ranks: int, num_scaleup_ranks: int, num_experts: int,
            bias: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
            reduce_in_local: bool, reduce_in_scaleup: bool) -> torch.Tensor:
    """
    The reference implementation of (possibly multi-level reduction) combining tokens.

    Arguments:
    - `y`: Input tokens to be combined, `[num_tokens, num_topk, hidden]`
    - `topk_idx`: `[num_tokens, num_topk]`
    - `reduce_in_local` and `reduce_in_scaleup`: Whether to do reduction within rank and within scale-up group.
       - `(True, True)` -> Hybrid combine
       - `(True, False)` -> Non-hybrid combine
       - `(False, False)` -> Equivalent to `allow_multiple_reduction` is False
       Pay attention that `reduce_in_scaleup` = `True` or `False` is NOT equivalent even if `num_scaleout_ranks == 1` due to `bias` handling.

    Returns:
    - Combined result, `[num_tokens, hidden]`
    """
    num_ranks = num_scaleout_ranks * num_scaleup_ranks
    num_tokens, hidden = y.shape[0], y.shape[2]
    num_topk = y.shape[1]
    assert not (not reduce_in_local and reduce_in_scaleup), 'Invalid reduction configuration'

    def grouped_reduce(data_to_reduce: torch.Tensor, group_id: torch.Tensor) -> torch.Tensor:
        """
        Perform in-place grouped reduction on `data_to_reduce` according to `group_id`.
        The summation within each group are performed in strict order along the `num_topk` dimension.
        The result for each group is stored at the rightmost token of that group, and other tokens are set to zero.

        Arguments:
        - `data_to_reduce`: Data to be reduced, `[num_tokens, num_topk, hidden]`
        - `group_id`: group IDs for each token, `[num_tokens, num_topk]`
        """
        # Shuffle to make tokens with the same group_id contiguous
        group_id, src_indices = torch.sort(group_id, dim=-1, stable=True)
        # transformed_src_indices[i, j] = i * num_topk + src_indices[i, j]
        transformed_src_indices = (
            (src_indices + torch.arange(0, num_tokens, device=y.device).unsqueeze(-1) * num_topk).flatten())
        data_to_reduce = data_to_reduce.view(-1, hidden)[transformed_src_indices].view(num_tokens, num_topk, hidden)
        # Perform segmented reduce within each group
        cur_accum_buf = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=y.device)
        for i in range(num_topk):
            is_segment_break = torch.full((num_tokens, ), True, dtype=torch.bool, device=y.device) \
                if i == num_topk - 1 else group_id[:, i] != group_id[:, i + 1]
            cur_accum_buf += data_to_reduce[:, i, :].float()
            # For one token, if `is_segment_break` is True,
            # save the accumulated value and clear the buffer, otherwise, clear `data_to_reduce[:, i, :]`
            segment_break_token_indices = torch.where(is_segment_break)[0]
            data_to_reduce[segment_break_token_indices, i] = cur_accum_buf[segment_break_token_indices].to(data_to_reduce.dtype)
            cur_accum_buf[segment_break_token_indices] = 0.0
            non_segment_break_token_indices = torch.where(~is_segment_break)[0]
            data_to_reduce[non_segment_break_token_indices, i] = 0.0
        # Unshuffle
        # noinspection PyShadowingNames
        result = torch.empty_like(data_to_reduce)
        result.view(-1, hidden)[transformed_src_indices] = data_to_reduce.view(-1, hidden)
        return result.view(num_tokens, num_topk, hidden)
    
    num_experts_per_rank = num_experts // num_ranks
    if reduce_in_local:
        y = grouped_reduce(y, topk_idx // num_experts_per_rank)
    if reduce_in_scaleup:
        y = grouped_reduce(y, topk_idx // (num_experts_per_rank * num_scaleup_ranks))
    bias_sum = bias[0].float() + bias[1].float() if isinstance(bias, tuple) else bias.float() if bias is not None else None
    result = ordered_accumulate(y, bias_sum)
    return result

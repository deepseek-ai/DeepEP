#pragma once

#include <cooperative_groups.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

// Pre-computes, for every `(token, top-k)` entry, the destination slot the token will occupy in
// the receiver's buffer, making the placement a pure function of `topk_idx`.
//
// The emitted value is `row_base_idx * kNumMaxTokensPerRank + row-local slot`, where the
// row-local slot is this rank's running count of deduplicated entries sent to that destination
// rank, and `row_base_idx` is the sender's scale-up rank index for the direct dispatch, or its
// scale-out rank index for the hybrid one (whose scale-up buffer segment is selected by the
// forwarding rank). A row is private to one source rank, which can never send more than
// `kNumMaxTokensPerRank` deduplicated entries to a single destination, so rows never overflow.
template <int kNumSMs, int kNumWarps,
          int kNumRanks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumThreads = kNumWarps * 32>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_deterministic_prologue_impl(
    topk_idx_t* topk_idx,
    int* rank_count_buffer,
    int* dst_buffer_slot_idx,
    const int num_tokens,
    const int row_base_idx
) {
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto global_warp_idx = sm_idx * kNumWarps + warp_idx;

    // Token region the current warp is responsible for
    const auto num_tokens_per_warp = math::ceil_div(num_tokens, kNumSMs * kNumWarps);
    const auto start_token_idx = global_warp_idx * num_tokens_per_warp;
    const auto end_token_idx = min(start_token_idx + num_tokens_per_warp, num_tokens);

    // Group configs
    // NOTES: Group refers to the tokens that each warp handles concurrently
    constexpr int kNumTokensPerGroup = 32 / kNumTopk;
    const auto token_idx_offset = lane_idx / kNumTopk;
    const unsigned token_mask = ((1u << kNumTopk) - 1) << (token_idx_offset * kNumTopk);
    EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k");

    // Shared memory for reduction
    // NOTES: Each warp owns separate shared memory region for separate sum.
    extern __shared__ int8_t smem[];
    const auto rank_count_global_psum = math::advance_ptr<int>(smem, 0);
    const auto rank_count_warp_sum = math::advance_ptr<int>(rank_count_global_psum, (kNumRanks + warp_idx * kNumRanks) * sizeof(int));
    const auto rank_count_warp_psum = math::advance_ptr<int>(rank_count_warp_sum, kNumWarps * kNumRanks * sizeof(int));

    // Initialize to zero before reduce
    for (int i = thread_idx; i < kNumRanks * (1 + 2 * kNumWarps); i += kNumThreads)
        reinterpret_cast<int*>(smem)[i] = 0;
    __syncthreads();

    // Util functions
    const auto map_expert_to_rank_idx = [&](const int& expert_idx) {
        return expert_idx >= 0 ? expert_idx / kNumExpertsPerRank : -1;
    };
    const auto is_unique = [&](const int& rank_idx) {
        return ((ptx::match(rank_idx) & token_mask) >> lane_idx) == 1;
    };
    const auto count_ones_before = [&](const unsigned& mask, const int& bit_idx) {
        return __popc(mask & ((1u << bit_idx) - 1));
    };
    const auto get_other_rank_count_warp_sum = [&](const int& other_warp_idx) {
        // NOTES: pass negative num_bytes to advance pointer
        return math::advance_ptr<int>(rank_count_warp_sum, (other_warp_idx - warp_idx) * kNumRanks * sizeof(int));
    };

    // Each warp scan the tokens separately
    for (int i = start_token_idx; i < end_token_idx; i += kNumTokensPerGroup) {
        const auto token_idx = i + token_idx_offset;
        const auto is_active_thread = lane_idx < kNumTopk * kNumTokensPerGroup and token_idx < end_token_idx;
        const int expert_idx = is_active_thread ? static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
        const auto rank_idx = map_expert_to_rank_idx(expert_idx);

        // Avoid duplicate messages to a single rank
        const auto deduped_rank_idx = is_unique(rank_idx) ? rank_idx : -1;
        const auto rank_idx_mask = ptx::match(deduped_rank_idx);

        // Let the one with the largest lane index send the count
        if ((rank_idx_mask >> lane_idx) == 1 and deduped_rank_idx >= 0)
            rank_count_warp_sum[deduped_rank_idx] += __popc(rank_idx_mask);
    }
    __syncthreads();

    // Get block sum and store to global
    for (int rank_idx = thread_idx; rank_idx < kNumRanks; rank_idx += kNumThreads) {
        int rank_count_block_sum = 0;
        for (int i = 0; i < kNumWarps; i++)
            rank_count_block_sum += get_other_rank_count_warp_sum(i)[rank_idx];
        rank_count_buffer[sm_idx * kNumRanks + rank_idx] = rank_count_block_sum;
    }
    cooperative_groups::this_grid().sync();

    // Get the prefix sum before the current SM
    for (int rank_idx = lane_idx; rank_idx < kNumRanks; rank_idx += 32) {
        int rank_count = 0;
        for (int i = warp_idx; i < sm_idx; i += kNumWarps)
            rank_count += rank_count_buffer[i * kNumRanks + rank_idx];
        atomicAdd_block(rank_count_global_psum + rank_idx, rank_count);
    }
    __syncthreads();

    // Get each warp's prefix sum
    for (int rank_idx = lane_idx; rank_idx < kNumRanks; rank_idx += 32) {
        int rank_count = rank_count_global_psum[rank_idx];
        for (int i = 0; i < warp_idx; i++)
            rank_count += get_other_rank_count_warp_sum(i)[rank_idx];
        rank_count_warp_psum[rank_idx] = rank_count;
    }
    __syncwarp();

    // Each warp scan the tokens separately
    for (int i = start_token_idx; i < end_token_idx; i += kNumTokensPerGroup) {
        const auto token_idx = i + token_idx_offset;
        const auto is_active_thread = lane_idx < kNumTopk * kNumTokensPerGroup and token_idx < end_token_idx;
        const auto expert_idx = is_active_thread ? static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
        const auto rank_idx = map_expert_to_rank_idx(expert_idx);

        // Avoid duplicate messages to a single rank
        const auto deduped_rank_idx = is_unique(rank_idx) ? rank_idx : -1;
        const auto rank_idx_mask = ptx::match(deduped_rank_idx);

        // Store to target buffer
        const auto stored_dst_slot_idx = deduped_rank_idx >= 0 ?
                                         rank_count_warp_psum[deduped_rank_idx] + count_ones_before(rank_idx_mask, lane_idx) : -1;
        const auto value = stored_dst_slot_idx >= 0 ?
                           row_base_idx * kNumMaxTokensPerRank + stored_dst_slot_idx : -1;
        if (is_active_thread)
            dst_buffer_slot_idx[i * kNumTopk + lane_idx] = value;

        // Let the one with the largest lane index send the count
        if ((rank_idx_mask >> lane_idx) == 1 and deduped_rank_idx >= 0)
            rank_count_warp_psum[deduped_rank_idx] += __popc(rank_idx_mask);
        __syncwarp();
    }
}

}  // namespace deep_ep::elastic

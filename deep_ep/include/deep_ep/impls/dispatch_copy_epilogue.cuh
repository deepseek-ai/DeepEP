#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

template <bool kDoExpand, bool kCachedMode, bool kDoZeroPadding,
          // NOTES: this channel concept only applies for scale-out ranks
          int kNumSMs, int kNumChannels, int kNumWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk, int kExpertAlignment,
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          int kNumThreads = kNumWarps * 32,
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          bool kDoCreateLinkedList = (kNumScaleoutRanks > 1 and not kCachedMode)>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_copy_epilogue_impl(void* buffer, void* workspace,
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            int* psum_num_recv_tokens_per_expert,
                            void* recv_x, sf_pack_t* recv_sf,
                            topk_idx_t* recv_topk_idx, float* recv_topk_weights,
                            int* recv_src_metadata,
                            int* channel_linked_list,
                            int* num_unaligned_recv_tokens_per_expert,
                            int num_recv_tokens,
                            const int recv_sf_token_stride, const int recv_sf_hidden_stride,
                            const int scaleout_rank_idx, const int scaleup_rank_idx) {
    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;

    // For top-k index transformations
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    const auto rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;
    const auto expert_start_idx = kNumExpertsPerRank * rank_idx, expert_end_idx = kNumExpertsPerRank * (rank_idx + 1);

    // Buffer layouts
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(warp_idx).get_token_buffer(0);
    const auto scaleup_buffer = layout::BufferLayout<false>(token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);

    // Init TMA
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // Will block until the main dispatch kernel has finished and all data are visible
    // NOTES: PDL is used, please do not use `__ldg`
    cudaGridDependencySynchronize();

    // For no CPU sync case, the number of received tokens should be read from the GPU tensor
    if (num_recv_tokens == kNumMaxTokensPerRank * kNumRanks)
        num_recv_tokens = psum_num_recv_tokens_per_scaleup_rank[kNumScaleupRanks - 1];

    // Current rank indices should be maintained
    int current_rank_idx = -1, stored_psum_num_recv_tokens;
    int current_rank_start = 0, current_rank_end = 0;
    #pragma unroll
    for (int i = global_warp_idx; i < num_recv_tokens; i += kNumWarps * kNumSMs) {
        // Calculate token index in the buffer
        while (i >= current_rank_end) {
            current_rank_idx += 1;
            EP_DEVICE_ASSERT(current_rank_idx < kNumScaleupRanks);
            const auto stored_lane_idx = current_rank_idx % 32;
            if (stored_lane_idx == 0 and current_rank_idx + lane_idx < kNumScaleupRanks)
                stored_psum_num_recv_tokens = psum_num_recv_tokens_per_scaleup_rank[current_rank_idx + lane_idx];
            current_rank_start = current_rank_end;
            current_rank_end = ptx::exchange(stored_psum_num_recv_tokens, stored_lane_idx);
        }
        const auto buffer_token = scaleup_buffer.get_rank_buffer(current_rank_idx).get_token_buffer(i - current_rank_start);

        // Wait buffer releases
        ptx::tma_store_wait();
        __syncwarp();

        // Issue TMA loads
        // Including all stuffs: data, SF, top-k metadata
        if (ptx::elect_one_sync()) {
            ptx::tma_load_1d(tma_buffer.get_base_ptr(), buffer_token.get_base_ptr(),
                             mbarrier_ptr, tma_buffer.get_num_bytes<false>());
            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, tma_buffer.get_num_bytes<false>());
        }
        __syncwarp();

        // Load target expert indices separately to tolerate TMA load latency
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        int dst_expert_idx = -1;
        if (lane_idx < kNumTopk)
            dst_expert_idx = buffer_token.get_topk_idx_ptr()[lane_idx];
        __syncwarp();

        // Validate target expert indices and store for non-expand mode
        const auto in_range = expert_start_idx <= dst_expert_idx and dst_expert_idx < expert_end_idx;
        const auto master_src_topk_idx = ptx::get_master_lane_idx(ptx::gather(in_range));
        dst_expert_idx = in_range ? dst_expert_idx - expert_start_idx : -1;
        EP_DEVICE_ASSERT(ptx::deduplicate(dst_expert_idx, lane_idx) or dst_expert_idx == -1);
        if (not kDoExpand and lane_idx < kNumTopk)
            recv_topk_idx[i * kNumTopk + lane_idx] = static_cast<topk_idx_t>(dst_expert_idx);
        __syncwarp();

        // Calculate target indices in the tensor
        constexpr int kMetadataStride = 2 + kNumTopk;
        int dst_tensor_idx = -1;
        if (not kDoExpand and ptx::elect_one_sync()) {
            dst_tensor_idx = i;
        } else if (kDoExpand and kCachedMode and lane_idx < kNumTopk) {
            // Cached expand mode: read pre-computed dst_tensor_idx from metadata
            dst_tensor_idx = recv_src_metadata[i * kMetadataStride + 2 + lane_idx];
        } else if (kDoExpand and not kCachedMode and dst_expert_idx >= 0) {
            dst_tensor_idx = atomicAdd(psum_num_recv_tokens_per_expert + dst_expert_idx, 1);
        }
        __syncwarp();

        // Wait for TMA arrival
        if (ptx::elect_one_sync())
            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
        __syncwarp();

        // Maintain linked list
        if constexpr (kDoCreateLinkedList) {
            if (ptx::elect_one_sync())
                channel_linked_list[tma_buffer.get_linked_list_idx_ptr()[master_src_topk_idx]] = i;
            __syncwarp();
        }

        // Issue TMA stores for data
        if (kDoExpand ? (dst_tensor_idx >= 0) : ptx::elect_one_sync()) {
            ptx::tma_store_1d(math::advance_ptr(recv_x, static_cast<int64_t>(dst_tensor_idx) * kNumHiddenBytes),
                              tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
            ptx::tma_store_commit();
        }
        __syncwarp();

        // Store SF
        if constexpr (kNumSFPacks > 0) {
            constexpr auto kNumFullIters = kNumSFPacks / 32;
            const bool do_last_iter = (kNumSFPacks % 32 != 0) and (kNumFullIters * 32 + lane_idx < kNumSFPacks);
            EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");

            // Load into registers
            const auto smem_src_ptr = tma_buffer.get_sf_ptr();
            sf_pack_t reg_src[kNumFullIters + 1];
            #pragma unroll
            for (int k = 0; k < kNumFullIters; ++ k)
                reg_src[k] = smem_src_ptr[k * 32 + lane_idx];
            if (do_last_iter)
                reg_src[kNumFullIters] = smem_src_ptr[kNumFullIters * 32 + lane_idx];

            // Prepare strides
            const auto recv_sf_token_stride_i64 = static_cast<int64_t>(recv_sf_token_stride);
            const auto recv_sf_hidden_stride_i64 = static_cast<int64_t>(recv_sf_hidden_stride);

            // Iterate through all valid indices and store into output buffer
            auto mask = kDoExpand ? ptx::gather(dst_tensor_idx >= 0) : 1;
            while (mask) {
                const int valid_lane_idx = __ffs(mask) - 1;
                const auto gmem_dst = math::advance_ptr<sf_pack_t>(recv_sf,
                    ptx::exchange(dst_tensor_idx, valid_lane_idx) * (recv_sf_token_stride_i64 * sizeof(sf_pack_t)));
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k)
                    gmem_dst[(k * 32 + lane_idx) * recv_sf_hidden_stride_i64] = reg_src[k];
                if (do_last_iter)
                    gmem_dst[(kNumFullIters * 32 + lane_idx) * recv_sf_hidden_stride_i64] = reg_src[kNumFullIters];
                mask ^= 1 << valid_lane_idx;
            }
        }

        // Store the top-k weights
        if (kDoExpand and recv_topk_weights != nullptr and dst_tensor_idx >= 0) {
            recv_topk_weights[dst_tensor_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];
        } else if (not kDoExpand and recv_topk_weights != nullptr and lane_idx < kNumTopk) {
            // For backward, weights are optional
            recv_topk_weights[i * kNumTopk + lane_idx] = tma_buffer.get_topk_weights_ptr()[lane_idx];
        }
        __syncwarp();

        // Write source token index (skip in cached mode as metadata is reused)
        // And:
        //   - Non-hybrid mode: the source scaleup peer rank index and master top-k lane index
        //   - Hybrid mode: the slot index and master top-k lane index
        if constexpr (not kCachedMode) {
            if (ptx::elect_one_sync()) {
                recv_src_metadata[i * kMetadataStride + 0] = *tma_buffer.get_src_token_global_idx_ptr();
                if constexpr (kNumScaleoutRanks == 1) {
                    recv_src_metadata[i * kMetadataStride + 1] = current_rank_idx * kNumTopk + master_src_topk_idx;
                } else {
                    recv_src_metadata[i * kMetadataStride + 1] = (i - current_rank_start) * kNumTopk + master_src_topk_idx;
                }
            }
            __syncwarp();

            // Write reduction source indices
            if (kDoExpand and lane_idx < kNumTopk)
                recv_src_metadata[i * kMetadataStride + 2 + lane_idx] = dst_tensor_idx;
            __syncwarp();
        }
    }

    // Maintain linked list's ending
    // Or you can understand it as writing the tail at once
    if constexpr (kDoCreateLinkedList) {
        constexpr int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32);
        const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);
        for (int i = global_warp_idx; i < kNumChannels; i += kNumSMs * kNumWarps) {
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                if (const auto k = j * 32 + lane_idx; j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks) {
                    channel_linked_list[
                        *workspace_layout.get_channel_scaleup_tail_ptr(i, k)
                    ] = -1;

                    // Clean for combine usages
                    *workspace_layout.get_channel_scaleup_tail_ptr(i, k) = 0;
                }
            }
            __syncwarp();
        }
    }

    // Zero padding: clear the alignment gaps between experts in the expanded output
    if constexpr (kDoZeroPadding and kDoExpand) {
        // Wait for last TMA store from main loop to complete
        ptx::tma_store_wait();
        __syncwarp();

        // Zero out the TMA buffer's hidden region in smem
        ptx::st_bulk<kNumHiddenBytes>(tma_buffer.get_hidden_ptr());

        // Read all expert unaligned counts in parallel
        constexpr int kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32);
        int num_experts_per_lane[kNumExpertsPerLane];
        #pragma unroll
        for (int i = 0; i < kNumExpertsPerLane; ++ i) {
            const int expert_idx = i * 32 + lane_idx;
            num_experts_per_lane[i] = expert_idx < kNumExpertsPerRank ?
                num_unaligned_recv_tokens_per_expert[expert_idx] : 0;
        }

        // Single while loop over all padding tokens
        int wave_idx = 0, wave_pad_psum = 0, wave_tensor_psum = 0, pad_idx = global_warp_idx;
        while (true) {
            // Jump into the right position
            int dst_tensor_idx;
            while (wave_idx < kNumExpertsPerLane) {
                // Assign current wave expert token count
                int wave_num_experts_per_lane;
                #pragma unroll
                for (int i = 0; i < kNumExpertsPerLane; ++ i)
                    wave_num_experts_per_lane = i == wave_idx ? num_experts_per_lane[i] : wave_num_experts_per_lane;
                int wave_num_pads_per_lane = 0;
                if (wave_idx * 32 + lane_idx < kNumExpertsPerRank) {
                    wave_num_pads_per_lane = kExpertAlignment - (wave_num_experts_per_lane % kExpertAlignment);
                    wave_num_pads_per_lane = wave_num_pads_per_lane == kExpertAlignment ? 0 : wave_num_pads_per_lane;
                }
                int wave_num_pads = ptx::reduce_add(wave_num_pads_per_lane);

                // Check whether hit the current wave
                if (pad_idx < wave_pad_psum + wave_num_pads) {
                    const int local_pad_idx = pad_idx - wave_pad_psum;
                    const int pad_psum = ptx::warp_inclusive_sum(wave_num_pads_per_lane, lane_idx);
                    const int owner_lane_idx = __ffs(__ballot_sync(0xffffffff, pad_psum > local_pad_idx)) - 1;
                    dst_tensor_idx = wave_tensor_psum + ptx::exchange(
                        ptx::warp_exclusive_sum(wave_num_experts_per_lane + wave_num_pads_per_lane, lane_idx) +
                        wave_num_experts_per_lane + local_pad_idx - (pad_psum - wave_num_pads_per_lane),
                        owner_lane_idx);
                    break;
                }

                // Move to the next wave
                wave_idx ++;
                wave_pad_psum += wave_num_pads;
                wave_tensor_psum += ptx::reduce_add(wave_num_experts_per_lane + wave_num_pads_per_lane);
            }
            if (wave_idx >= kNumExpertsPerLane)
                break;

            // Zero data via TMA store
            if (ptx::elect_one_sync()) {
                ptx::tma_store_1d(math::advance_ptr(recv_x, static_cast<int64_t>(dst_tensor_idx) * kNumHiddenBytes),
                                  tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
                ptx::tma_store_commit();
            }
            __syncwarp();

            // Zero weights
            if (recv_topk_weights != nullptr and ptx::elect_one_sync())
                recv_topk_weights[dst_tensor_idx] = 0.0f;
            __syncwarp();

            // Zero SF
            if constexpr (kNumSFPacks > 0) {
                const auto recv_sf_token_stride_i64 = static_cast<int64_t>(recv_sf_token_stride);
                const auto recv_sf_hidden_stride_i64 = static_cast<int64_t>(recv_sf_hidden_stride);
                constexpr sf_pack_t zero_sf_pack = {0};
                const auto gmem_dst = math::advance_ptr<sf_pack_t>(recv_sf,
                    dst_tensor_idx * (recv_sf_token_stride_i64 * sizeof(sf_pack_t)));
                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k)
                    gmem_dst[(k * 32 + lane_idx) * recv_sf_hidden_stride_i64] = zero_sf_pack;
                if constexpr (kNumSFPacks % 32 != 0) {
                    if (kNumFullIters * 32 + lane_idx < kNumSFPacks)
                        gmem_dst[(kNumFullIters * 32 + lane_idx) * recv_sf_hidden_stride_i64] = zero_sf_pack;
                }
            }
            __syncwarp();

            // Try to fill the next one
            pad_idx += kNumSMs * kNumWarps;
        }
    }
}

}  // namespace deep_ep::elastic

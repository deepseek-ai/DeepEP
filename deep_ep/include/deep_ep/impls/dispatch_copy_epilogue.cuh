#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

template <bool kDoExpand, bool kCachedMode, bool kDoScatterToNvS,
          // NOTES: this channel concept only applies for scale-out ranks
          int kNumSMs, int kNumChannels, int kNumWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
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
                            float* recv_aux_weights,   // per-row router-weight gradient scalar output
                            int* recv_src_metadata,
                            int* channel_linked_list,
                            int num_recv_tokens,
                            const int recv_sf_token_stride, const int recv_sf_hidden_stride,
                            const int scaleout_rank_idx, const int scaleup_rank_idx,
                            // Fused combine-backward scatter. When kDoScatterToNvS, additionally scatter
                            // each received row to out_nvs[scatter_src_metadata[i, 2+k]] (the
                            // forward-recorded NvS destinations). Both null/unused otherwise.
                            const int* scatter_src_metadata, void* out_nvs) {
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
        int dst_tensor_idx = -1;
        if constexpr (not kDoExpand) {
            if (ptx::elect_one_sync())
                dst_tensor_idx = i;
        } else if constexpr (kCachedMode) {
            // Cached+expand replay. The fresh expand dispatch that built this handle recorded, per
            // (recv-token i, k-slot lane), the atomically-assigned expanded row into
            // recv_src_metadata[:, 2+lane] (see the write near the end of this loop). The expanded row
            // order within an expert is otherwise race-dependent and not reproducible across dispatch
            // calls, so a cached replay must reuse that recorded map (passed in via the handle's
            // recv_src_metadata) instead of re-running the racy atomicAdd, guaranteeing the replay is
            // row-for-row aligned with the forward. `-1` rows (unselected k-slots) stay -1 and are
            // skipped by the `dst_tensor_idx >= 0` store guards below.
            if (lane_idx < kNumTopk)
                dst_tensor_idx = recv_src_metadata[i * (2 + kNumTopk) + 2 + lane_idx];
        } else if (kDoExpand and dst_expert_idx >= 0) {
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

        // Fused combine-backward scatter (non-expand path). Each valid lane k additionally writes the same
        // received hidden row to out_nvs[scatter_src_metadata[i, 2+k]] -- the NvS destinations recorded by
        // the forward expand dispatch (deterministic, no atomicAdd). This is bit-identical to the Python
        // gather path out[meta[i,2+k]] = d_base[i]: same bytes, same destinations, no reduction. Mirrors the
        // expand store above (per-lane TMA store from the same shared source to a per-lane destination).
        if constexpr (kDoScatterToNvS) {
            constexpr int kScatterMetadataStride = 2 + kNumTopk;
            if (lane_idx < kNumTopk) {
                const int nvs_row = scatter_src_metadata[i * kScatterMetadataStride + 2 + lane_idx];
                if (nvs_row >= 0) {
                    ptx::tma_store_1d(math::advance_ptr(out_nvs, static_cast<int64_t>(nvs_row) * kNumHiddenBytes),
                                      tma_buffer.get_hidden_ptr(), kNumHiddenBytes);
                    ptx::tma_store_commit();
                }
            }
            __syncwarp();
        }

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
        // Aux scalar in parallel with recv_topk_weights, expand mode only (the per-row [NvS] scalar)
        if (kDoExpand and recv_aux_weights != nullptr and dst_tensor_idx >= 0) {
            recv_aux_weights[dst_tensor_idx] = tma_buffer.get_aux_weights_ptr()[lane_idx];
        }
        __syncwarp();

        // Write source token index
        // And:
        //   - Non-hybrid mode: the source scaleup peer rank index and master top-k lane index
        //   - Hybrid mode: the slot index and master top-k lane index
        constexpr int kMetadataStride = 2 + kNumTopk;
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
}

}  // namespace deep_ep::elastic

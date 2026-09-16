#pragma once

// MIT License
//
// Copyright (c) 2025 DeepSeek
// Changes and additions copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/impls/combine_utils.cuh>
#include <deep_ep/impls/proxy_ring.cuh>

namespace deep_ep::elastic {

template <bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs,
          int kNumScaleupWarps, int kNumForwardWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),
          int kNumScaleupUpdateInterval = 3,
          int kBatchSize = 12,
          int kProxyRingDepth = kProxyRingDepthDefault,
          int kNumChannelsPerSM = kNumForwardWarps,
          int kNumChannels = kNumChannelsPerSM * kNumSMs,
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          int kNumDataWarps = kNumScaleupWarps + kNumForwardWarps,
          int kProxyWarpIdx = kNumDataWarps,
          int kNumWarps = kNumDataWarps + 1,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          bool kUseScaleoutRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),
          bool kUseScaleupRankLayout = use_rank_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>(),
          int kNumTokensInScaleoutLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleoutRanks, kNumTopk>(),
          int kNumTokensInScaleupLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumScaleupRanks, kNumTopk>()>
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_unordered_combine_impl(nv_bfloat16* x,
                    float* topk_weights,
                    int* src_metadata,
                    int* psum_num_recv_tokens_per_scaleup_rank,
                    int* token_metadata_at_forward,
                    int* channel_linked_list,
                    const int* token_map_at_dispatch,
                    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                    void* buffer, void* workspace,
                    const int scaleout_rank_idx, const int scaleup_rank_idx,
                    int num_reduced_tokens, const int num_combined_tokens) {
    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    constexpr bool kDoExpandedSend = not kAllowMultipleReduction and kUseExpandedLayout;

    // Combine vector type selection
    using combine_vec_t = typename CombineVecTraits<kNumHiddenBytes>::vec_t;
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);

    // Workspaces
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);

    // We should assign the real number of received tokens if without CPU sync
    if (num_reduced_tokens == kNumMaxTokensPerRank * kNumRanks)
        num_reduced_tokens = __ldg(psum_num_recv_tokens_per_scaleup_rank + kNumScaleupRanks - 1);

    // Token layouts
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);

    // TMA buffers — one per DATA warp only (scale-up + forward).
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto tma_buffer_layout = layout::BufferLayout<true>(token_layout, kNumDataWarps, 1, smem);
    // The proxy warp never touches `tma_buffer`, clamp its index to 0.
    const auto tma_buffer = tma_buffer_layout
        .get_rank_buffer(warp_idx < kNumDataWarps ? warp_idx : 0).get_token_buffer(0);

    // Proxy warp hand-off rings — placed in `smem[]` right after the TMA buffers.
    const auto proxy_ring_layout = ProxyRingLayout(
        kNumForwardWarps, kProxyRingDepth, tma_buffer_layout.get_buffer_end_ptr());
    for (int f = thread_idx; f < kNumForwardWarps; f += kNumThreads) {
        *proxy_ring_layout.get_head(f) = 0;
        *proxy_ring_layout.get_tail(f) = 0;
        *proxy_ring_layout.get_done(f) = 0;
    }
    __syncthreads();

    // All the buffer layouts
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, kNumTokensInScaleupLayout, kNumScaleoutRanks * kNumMaxTokensPerRank,
        buffer);
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleoutRanks,
        kNumChannels * (kNumMaxTokensPerChannel * (kAllowMultipleReduction ? 1 : kNumTopk)),
        scaleup_buffer.get_buffer_end_ptr());
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleoutRanks,
        kNumChannels * (kNumMaxTokensPerChannel * (kAllowMultipleReduction ? 1 : kNumTopk)),
        scaleout_recv_buffer.get_buffer_end_ptr());

    // Init TMA for scale-up and forward warps
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // NCCL Gin handle
    // For data warp, each warp is a channel, for proxy warp each lane is a channel.
    const auto gin_channel_in_sm = warp_idx < kNumDataWarps ?
        (warp_idx % kNumChannelsPerSM) :
        (lane_idx < kNumForwardWarps ? lane_idx : 0);
    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm_idx, gin_channel_in_sm);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // Global parallel barriers for scale-out subteam and scale-up subteam
    // NOTES: this barrier needs a grid sync, as there are channel scale-up tail cleaning before
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridCombineTag0, false, true, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);

    // Adjust register count at certain cases
    // TODO: support more cases, or try to make channel count more aligned
    const bool kAdjustRegisters = (kNumChannelsPerSM == 4 or kNumChannelsPerSM == 8) and not kUseExpandedLayout;
    constexpr int kNumRegistersForScaleupWarps = 40;
    constexpr int kNumRegistersForForwardWarps = 256 - kNumRegistersForScaleupWarps;

    // Different warp roles
    if (warp_idx < kNumScaleupWarps) {
        const auto channel_idx = sm_idx * kNumChannelsPerSM + warp_idx;

        // Adjust registers
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_dealloc<kNumRegistersForScaleupWarps>();

        // Shift into the right buffer if using rank layout
        if constexpr (kUseScaleupRankLayout)
            scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Expanding send mode must not be backward
        if constexpr (kDoExpandedSend)
            EP_DEVICE_ASSERT(topk_weights == nullptr);

        // Tail issuer
        // `st.release.sys` is pretty slow, so do it by an interval
        int update_counter = 0;
        int stored_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        int stored_old_num_tokens_sent[kNumScaleupRanksPerLane] = {};
        const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
        const auto update_tails = [&](const bool& finish = false) {
            ++ update_counter;
            if (finish or update_counter == kNumScaleupUpdateInterval) {
                // Wait all TMA stores to finish
                ptx::tma_store_wait();
                __syncwarp();

                // Issue
                #pragma unroll
                for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                    if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {
                        // NOTES: save some traffic with `stored_old_num_tokens_sent`
                        // Also, we cannot rewrite a finished slot, if the peer is going to clean it
                        if (stored_num_tokens_sent[i] != stored_old_num_tokens_sent[i])
                            ptx::st_release_sys(gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j), stored_num_tokens_sent[i]);
                        stored_old_num_tokens_sent[i] = stored_num_tokens_sent[i];
                    }
                }
                update_counter = 0;
            }
            __syncwarp();
        };

        // Shape of `channel_linked_list`: `[kNumChannels, kNumMaxTokensPerChannel + 1, kNumScaleupRanks]`
        // Iterate until all scale-up peers finish
        int dst_scaleup_rank_idx = channel_idx;
        int stored_ll_idx[kNumScaleupRanksPerLane] = {}, stored_token_idx[kNumScaleupRanksPerLane] = {};
        #pragma unroll
        for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
            stored_token_idx[i] = -1;
        while (true) {
            // Load token indices in the list
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                const auto j = i * 32 + lane_idx;
                stored_token_idx[i] = i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks ?
                    __ldg(channel_linked_list +
                          channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumScaleupRanks +
                          stored_ll_idx[i] * kNumScaleupRanks + j) : -1;
            }
            __syncwarp();

            // Check whether all ranks are finished
            bool exited = true;
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                exited &= ptx::all(stored_token_idx[i] < 0);
            if (exited)
                break;

            // Process tokens for all ranks together using bitmask to skip inactive ranks
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up ranks for 64-bit mask");
            using mask_t = std::conditional_t<(kNumScaleupRanks <= 32), uint32_t, uint64_t>;
            mask_t wip_mask = 0;
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                wip_mask |= static_cast<mask_t>(ptx::gather(stored_token_idx[j] >= 0)) << (j * 32);
            while (wip_mask) {
                // Find next active rank after `dst_scaleup_rank_idx` (round-robin)
                const auto start = (dst_scaleup_rank_idx + 1) % kNumScaleupRanks;
                const auto hi_mask = (wip_mask >> start) << start;
                dst_scaleup_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);
                wip_mask ^= static_cast<mask_t>(1) << dst_scaleup_rank_idx;

                // Exchange token index from the owning lane using static partition iteration
                int token_idx = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto src_lane_idx = dst_scaleup_rank_idx - j * 32;
                    token_idx = src_lane_idx == lane_idx ? stored_token_idx[j] : token_idx;
                }
                token_idx = ptx::exchange(token_idx, dst_scaleup_rank_idx % 32);

                // Get source metadata and decide the destination buffer
                constexpr int kMetadataStride = 2 + kNumTopk;
                const auto src_global_token_idx = __ldg(src_metadata + token_idx * kMetadataStride + 0);
                const auto src_token_idx = src_global_token_idx % kNumMaxTokensPerRank;
                const auto src_scaleout_rank_idx = src_global_token_idx / (kNumMaxTokensPerRank * kNumScaleupRanks);
                auto token_buffer = [&]() {
                    if constexpr (kUseScaleupRankLayout) {
                        const auto src_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) / kNumTopk;
                        return scaleup_buffer.get_token_buffer(src_slot_idx);
                    } else {
                        const auto master_topk_idx = __ldg(src_metadata + token_idx * kMetadataStride + 1) % kNumTopk;
                        return scaleup_buffer
                            .get_rank_buffer(master_topk_idx)
                            .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                    }
                }();
                token_buffer.set_base_ptr(gin.get_sym_ptr<ncclTeamTagLsa>(token_buffer.get_base_ptr(), dst_scaleup_rank_idx));

                // Some checks
                EP_STATIC_ASSERT(kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");

                // Read source indices for expand mode
                int stored_topk_slot_idx = -1;
                if constexpr (kUseExpandedLayout) {
                    if (lane_idx < kNumTopk)
                        stored_topk_slot_idx = __ldg(src_metadata + token_idx * kMetadataStride + (2 + lane_idx));
                    __syncwarp();
                }

                // 3 cases:
                //  - no-expand, expand + no-reduce
                //  - expand + reduce
                //  - expand + send all
                auto reduce_valid_mask = ptx::gather(stored_topk_slot_idx >= 0);
                auto no_local_reduce = not kUseExpandedLayout or (kAllowMultipleReduction and __popc(reduce_valid_mask) == 1);
                if (no_local_reduce) {
                    int token_idx_in_tensor = token_idx;
                    if constexpr (kUseExpandedLayout)
                        token_idx_in_tensor = ptx::exchange(stored_topk_slot_idx, ptx::get_master_lane_idx(reduce_valid_mask));

                    // Directly load
                    if (ptx::elect_one_sync()) {
                        const auto load_ptr =
                            math::advance_ptr(x, static_cast<int64_t>(token_idx_in_tensor) * kNumHiddenBytes);
                        ptx::tma_store_wait();
                        ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                    }
                    __syncwarp();
                } else if constexpr (kAllowMultipleReduction) {
                    // Do local reduction
                    // Sort valid top-k indices to front
                    int topk_slot_idx[kNumTopk];
                    compute_topk_slots(
                        topk_slot_idx, reduce_valid_mask,
                        [=](const int& idx) {
                            return ptx::exchange(stored_topk_slot_idx, idx);
                        }
                    );

                    // Reduce into shared memory
                    constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
                    combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumRanks)>(
                        lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                        /* Get source base */ [=](const int& slot_idx) {
                            return math::advance_ptr<combine_vec_t>(
                                x, slot_idx * static_cast<int64_t>(kNumHiddenBytes));
                        },
                        /* Wait buffer release */ [=]() {
                            ptx::tma_store_wait();
                            __syncwarp();
                        }
                    );
                    ptx::tma_store_fence();
                    __syncwarp();
                } else {
                    // No local reduction, send all data (expanded send)
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        int topk_slot_idx = ptx::exchange(stored_topk_slot_idx, k);
                        if (topk_slot_idx < 0)
                            continue;

                        if (ptx::elect_one_sync()) {
                            // Load
                            const auto load_ptr = math::advance_ptr(x, static_cast<int64_t>(kDoExpandedSend ? topk_slot_idx : token_idx) * kNumHiddenBytes);
                            ptx::tma_store_wait();
                            ptx::tma_load_1d(tma_buffer.get_base_ptr(), load_ptr, mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                            // NOTES: We don't need to care about `topk_weights` since we are in expand mode

                            // Store
                            const auto dst_token_buffer = scaleup_buffer
                                .get_rank_buffer(k)
                                .get_token_buffer(src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx);
                            ptx::tma_store_1d(
                                gin.get_sym_ptr<ncclTeamTagLsa>(dst_token_buffer.get_base_ptr(), dst_scaleup_rank_idx),
                                tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                            ptx::tma_store_commit();
                        }
                        __syncwarp();
                    }
                }

                // Write top-k weights (expanded send handled inside the loop above)
                if (not kDoExpandedSend and topk_weights != nullptr and lane_idx < kNumTopk) {
                    float value = 0;
                    if constexpr (kUseExpandedLayout) {
                        if (stored_topk_slot_idx >= 0)
                            value = __ldg(topk_weights + stored_topk_slot_idx);
                    } else {
                        value = __ldg(topk_weights + (token_idx * kNumTopk + lane_idx));
                    }
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = value;
                    ptx::tma_store_fence();
                }
                __syncwarp();

                // Issue TMA stores into remote scale-up buffer
                // NOTES: `kDoExpandedSend` mode has already issued
                if (not kDoExpandedSend and ptx::elect_one_sync()) {
                    // Wait TMA arrival (only for non-reduced cases)
                    if (no_local_reduce) {
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                    }

                    // Issue stores
                    ptx::tma_store_1d(
                        token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                        token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    stored_num_tokens_sent[j] += (j * 32 + lane_idx) == dst_scaleup_rank_idx;
                __syncwarp();
            }

            // Update the tails together
            // NOTES: TMA wait is inside
            update_tails();

            // Move linked list
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i)
                stored_ll_idx[i] += (stored_token_idx[i] >= 0);
        }

        // Update for the unissued ones
        update_tails(true);
    } else if (warp_idx < kNumDataWarps) {
        const auto forward_warp_idx = warp_idx - kNumScaleupWarps;
        const auto channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;

        // Indexed-signal id owned by this channel within its GIN context. Uses the
        // companion helper to `get_qp_mode` (called at kernel entry) so the id is
        // unique among all channels sharing this warp's QP, regardless of which of
        // `get_qp_mode`'s branches picked the mapping.
        const auto signal_id = static_cast<ncclGinSignal_t>(
            comm::get_qp_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm_idx, forward_warp_idx));

        // Adjust registers
        if constexpr (kAdjustRegisters)
            ptx::warpgroup_reg_alloc<kNumRegistersForForwardWarps>();

        constexpr int kNumSlotsPerChannel = kNumMaxTokensPerChannel * (kAllowMultipleReduction ? 1 : kNumTopk);
        scaleout_send_buffer = scaleout_send_buffer.get_channel_buffer<kNumSlotsPerChannel>(channel_idx);
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumSlotsPerChannel>(channel_idx);

        int slot_of_per_channel[kNumScaleoutRanks] = {};

        // Shape of `token_metadata_at_forward`: `[kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel + 1, kNumForwardMetadataDims]`
        constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);

        // Per Channel/ScaleOutRank batch metadata
        int batch_count[kNumScaleoutRanks] = {};
        int batch_start_slot[kNumScaleoutRanks];

        ProxyPutDesc* const my_ring = proxy_ring_layout.get_ring(forward_warp_idx);
        unsigned& my_head = *proxy_ring_layout.get_head(forward_warp_idx);
        const unsigned* const my_tail_ptr = proxy_ring_layout.get_tail(forward_warp_idx);

        // Hand a completed batch to the proxy instead of issuing the put here. Callers
        // must be under `elect_one_sync`. Backpressure: spin while ring is full.
        const auto issue_batched_rdma = [&](const int& dst) {
            if (batch_count[dst] == 0)
                return;
            const auto send_ptr = scaleout_send_buffer
                .get_rank_buffer(dst)
                .get_token_buffer(batch_start_slot[dst])
                .get_base_ptr();
            const auto recv_ptr = scaleout_recv_buffer
                .get_rank_buffer(scaleout_rank_idx)
                .get_token_buffer(batch_start_slot[dst])
                .get_base_ptr();

            // Wait for a free ring slot.
            while (my_head - ptx::ld_acquire_cta(my_tail_ptr) >= static_cast<unsigned>(kProxyRingDepth)) {}

            const unsigned slot = my_head % static_cast<unsigned>(kProxyRingDepth);
            my_ring[slot].send_ptr = send_ptr;
            my_ring[slot].recv_ptr = recv_ptr;
            my_ring[slot].num_bytes = batch_count[dst] * static_cast<int>(token_layout.get_num_bytes<false>());
            my_ring[slot].dst = dst;

            __threadfence();
            ptx::st_release_cta(&my_head, my_head + 1);

            batch_count[dst] = 0;
        };

        // Update per Channel/ScaleOutRank batch metadata and GIN PUT when threshold met
        const auto record_slot_and_maybe_flush = [&](const int& dst, const int& slot) {
            if (batch_count[dst] == 0)
                batch_start_slot[dst] = slot;
            batch_count[dst] += 1;
            if (batch_count[dst] == kBatchSize)
                issue_batched_rdma(dst);
        };

        int last_src_scaleout_rank_idx = -1;
        int last_slot_written = -1;
        const auto flush_last_tma_and_record_batch = [&]() {
            if (last_src_scaleout_rank_idx >= 0 and ptx::elect_one_sync()) {
                ptx::tma_store_wait();

                 // Issue only if not local rank
                if (last_src_scaleout_rank_idx != scaleout_rank_idx)
                    record_slot_and_maybe_flush(last_src_scaleout_rank_idx, last_slot_written);
            }
            __syncwarp();
        };

        // Replay the dispatch
        int stored_num_tokens_recv[kNumScaleupRanksPerLane] = {}, stored_cached_scaleup_tail[kNumScaleupRanksPerLane] = {};
        for (int i = 0; ; ++ i) {
            const auto src_token_global_idx = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims);
            const auto src_rank_idx = src_token_global_idx / kNumMaxTokensPerRank;
            const auto src_scaleout_rank_idx = src_rank_idx / kNumScaleupRanks;
            const auto src_token_idx = src_token_global_idx % kNumMaxTokensPerRank;
            auto stored_src_scaleup_rank_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + lane_idx) : -1;
            auto stored_src_slot_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + kNumTopk + lane_idx) : -1;
            if (src_token_global_idx < 0)
                break;

            // Scaleup rank mask
            EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Too many scale-up peers");
            using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;
            const auto scaleup_mask = ptx::reduce_or(
                stored_src_scaleup_rank_idx >= 0 ?
                (mask_t(1) << stored_src_scaleup_rank_idx) : mask_t(0));
            bool stored_is_scaleup_rank_needed[kNumScaleupRanksPerLane];
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_is_scaleup_rank_needed[j] = (scaleup_mask >> (j * 32 + lane_idx)) & 1;

            // Wait all tails to arrive
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                bool arrived = true;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    arrived &= not stored_is_scaleup_rank_needed[j] or stored_num_tokens_recv[j] < stored_cached_scaleup_tail[j];
                if (ptx::all(arrived))
                    return true;

                // Reload cached
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto k = j * 32 + lane_idx;
                    stored_cached_scaleup_tail[j] = j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks ?
                        ptx::ld_acquire_sys(workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k)) : -1;
                }

                // Timeout
                if (is_last_check) {
                    #pragma unroll
                    for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                        printf("DeepEP combine (scale-up wait) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "channel: %d, lane: %d, recv: %d, tail: %d (wait=%d)\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                               channel_idx, j * 32 + lane_idx,
                               stored_num_tokens_recv[j],
                               stored_cached_scaleup_tail[j],
                               stored_is_scaleup_rank_needed[j]);
                    }
                }
                return false;
            });

            // Increase received count
            #pragma unroll
            for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                stored_num_tokens_recv[j] += static_cast<int>(stored_is_scaleup_rank_needed[j]);
            
            if constexpr (not kAllowMultipleReduction) {
                // Cases where multiple reduction is disabled. We need to forward all data from scaleup peers to scaleout peers
                // TODO: Let scale-up warps directly put data into `send_buffer`?
                const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
                auto topk_valid_mask = kUseExpandedLayout ?
                    ptx::gather(stored_src_scaleup_rank_idx >= 0) :
                    ptx::gather(ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);  // Deduplicate w.r.t. scaleup rank index if expanded mode is disabled
                const int num_valid_topk = __popc(topk_valid_mask);
                if (ptx::elect_one_sync()) {
                    int packed_offset_in_entry = 0;
                    #pragma unroll
                    for (int k = 0; k < kNumTopk; ++ k) {
                        if ((topk_valid_mask & (1u << k)) == 0u)
                            continue;

                        // Issue TMA load, and wait
                        ptx::tma_load_1d(
                            tma_buffer.get_base_ptr(), scaleup_buffer.get_rank_buffer(k).get_token_buffer(src_slot_idx).get_base_ptr(),
                            mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);

                        const int slot = slot_of_per_channel[src_scaleout_rank_idx] + packed_offset_in_entry;
                        const auto recv_buffer_ptr = scaleout_recv_buffer
                            .get_rank_buffer(scaleout_rank_idx)
                            .get_token_buffer(slot)
                            .get_base_ptr();
                        const auto send_buffer_ptr = src_scaleout_rank_idx == scaleout_rank_idx ?
                            recv_buffer_ptr :
                            scaleout_send_buffer
                                .get_rank_buffer(src_scaleout_rank_idx)
                                .get_token_buffer(slot)
                                .get_base_ptr();
                        ++ packed_offset_in_entry;
                        ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), token_layout.get_num_bytes<false>());
                        ptx::tma_store_commit();
                        ptx::tma_store_wait();

                        topk_valid_mask ^= 1u << k;
                        if (src_scaleout_rank_idx != scaleout_rank_idx)
                            record_slot_and_maybe_flush(src_scaleout_rank_idx, slot);
                    }
                }
                __syncwarp();
                slot_of_per_channel[src_scaleout_rank_idx] += num_valid_topk;
            } else {
                // NOTES: we must do deduplicate and only add once from one rank
                auto reduce_valid_mask = ptx::gather(
                    ptx::deduplicate(stored_src_scaleup_rank_idx, lane_idx) and stored_src_scaleup_rank_idx >= 0);

                // Calculate the source buffer index
                int stored_src_buffer_idx = 0;
                if constexpr (kUseScaleupRankLayout) {
                    stored_src_buffer_idx =
                        stored_src_scaleup_rank_idx * scaleup_buffer.num_max_tokens_per_rank + stored_src_slot_idx;
                } else {
                    const auto src_slot_idx = src_scaleout_rank_idx * kNumMaxTokensPerRank + src_token_idx;
                    stored_src_buffer_idx = stored_src_slot_idx == -1 ? -1 :
                        lane_idx * scaleup_buffer.num_max_tokens_per_rank + src_slot_idx;
                }
                
                // Preprocess top-k indices
                int topk_slot_idx[kNumTokensInScaleupLayout];
                compute_topk_slots(
                    topk_slot_idx, reduce_valid_mask,
                    [=](const int& idx) {
                        return ptx::exchange(stored_src_buffer_idx, idx);
                    }
                );

                // Do reduce
                constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, kAdjustRegisters ? 8 : 4>();
                combine_reduce<kHiddenVec, kUnrollFactor, math::constexpr_ceil_div(kNumTopk, kNumScaleoutRanks)>(
                    lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()),
                    /* Get source base */ [=](const int& slot_idx) {
                        return static_cast<combine_vec_t*>(scaleup_buffer.get_token_buffer(slot_idx, true).get_base_ptr());
                    },
                    /* Wait buffer release */ [=]() {
                        flush_last_tma_and_record_batch();
                    }
                );

                // Merge topk weights
                // NOTES: the slot indices must follow the master lane
                stored_src_buffer_idx = ptx::exchange(
                    stored_src_buffer_idx, ptx::get_master_lane_idx(ptx::match(stored_src_scaleup_rank_idx)));
                if (stored_src_scaleup_rank_idx >= 0) {
                    tma_buffer.get_topk_weights_ptr()[lane_idx] =
                        scaleup_buffer.get_token_buffer(stored_src_buffer_idx, true)
                                    .get_topk_weights_ptr()[lane_idx];
                }
                ptx::tma_store_fence();
                __syncwarp(); // Necessary to let the leader lane see the writes

                // Assign send and receive buffers
                // NOTES: as we only have 1 destination, we will use "send" as "recv" for local transfer
                const int recv_slot = slot_of_per_channel[src_scaleout_rank_idx];
                const auto recv_token_buffer = scaleout_recv_buffer
                    .get_rank_buffer(scaleout_rank_idx)
                    .get_token_buffer(recv_slot);
                const auto send_token_buffer = src_scaleout_rank_idx == scaleout_rank_idx ?
                    recv_token_buffer :
                    scaleout_send_buffer
                    .get_rank_buffer(src_scaleout_rank_idx)
                    .get_token_buffer(slot_of_per_channel[src_scaleout_rank_idx]);
                slot_of_per_channel[src_scaleout_rank_idx] += 1;

                // Write into scale-out send buffer or local rank recv buffer bypass
                if (ptx::elect_one_sync()) {
                    ptx::tma_store_1d(send_token_buffer.get_base_ptr(), tma_buffer.get_base_ptr(),
                                    token_layout.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                __syncwarp();

                // Record RDMA info to issue later
                last_src_scaleout_rank_idx = src_scaleout_rank_idx;
                last_slot_written = recv_slot;
            }
        }

        // Issue the last TMA and record operation for last RDMA
        if constexpr (kAllowMultipleReduction)
            flush_last_tma_and_record_batch();

        // Issue the last RDMA per channel/scaleout-warp
        if (ptx::elect_one_sync()) {
            #pragma unroll
            for (int dst = 0; dst < kNumScaleoutRanks; ++ dst)
                issue_batched_rdma(dst);

            ptx::st_release_cta(proxy_ring_layout.get_done(forward_warp_idx), 1);
        }
        __syncwarp();

        // Clean scaleup tails
        #pragma unroll
        for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
            const auto k = j * 32 + lane_idx;
            if (j < (kNumScaleupRanksPerLane - 1) or k < kNumScaleupRanks)
                *workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, k) = 0;
        }
        __syncwarp();

        // Update, wait and clean
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid ranks");

        // Derive the expected inbound put count for this channel by counting my own
        // outbound routing decisions during dispatch. Count "distinct valid non-local packed
        // entries per T" — same rule works for reduce and expand modes because of how
        // dispatch fills the map.
        //
        // Concrete example — token T=0 on channel C, topK=4, my scaleout_rank_idx=3,
        // routing (k=0 → scaleout 1, k=1 → scaleout 0, k=2 → scaleout 0, k=3 → scaleout 3):
        //
        //   Reduce mode (map = [ pack(1,0,C), pack(0,0,C), pack(0,0,C), pack(3,0,C) ]):
        //     k=0: valid, rank=1 (remote), first sight of pack(1,0,C)         → count
        //     k=1: valid, rank=0 (remote), first sight of pack(0,0,C)         → count
        //     k=2: valid, rank=0 (remote), pack(0,0,C) already seen (dup)     → skip
        //     k=3: valid, rank=3 (== self, local-bypass, no put fires)        → skip
        //   Expected puts for T=0 = 2. Peer R=0 sends one reduced partial covering k=1+k=2.
        //
        //   Expand mode (map = [ pack(1,0,C), pack(0,0,C), pack(0,1,C), pack(3,0,C) ]):
        //     k=0: valid, rank=1 (remote), unique packed value  → count
        //     k=1: valid, rank=0 (remote), unique packed value  → count
        //     k=2: valid, rank=0 (remote), unique packed value  → count
        //     k=3: valid, rank=3 (self)                          → skip
        //   Expected puts for T=0 = 3. Peer R=0 sends two separate partials (k=1, k=2)
        //   at distinct slots because each valid k has its own packed slot in expand mode.
        int local_expected_count_per_rank[kNumScaleoutRanks] = {};
        {
            #pragma unroll
            for (int t_in_channel = lane_idx; t_in_channel < kNumMaxTokensPerChannel; t_in_channel += 32) {
                const int token_idx = channel_idx + t_in_channel * kNumChannels;
                if (token_idx >= num_combined_tokens)
                    continue;
                // Load this token's map entries once, then dedup+count within the topK block.
                int packed_entries[kNumTopk];
                #pragma unroll
                for (int k = 0; k < kNumTopk; ++ k)
                    packed_entries[k] = __ldg(token_map_at_dispatch + token_idx * kNumTopk + k);
                #pragma unroll
                for (int k = 0; k < kNumTopk; ++ k) {
                    const int p = packed_entries[k];
                    if (p < 0)
                        continue;
                    int rank, slot, channel;
                    unpack_combine_recv_addr(p, rank, slot, channel);
                    // Sanity: token T dispatched by us with `T % kNumChannels == channel_idx`
                    // must have its map entry's channel field equal to channel_idx.
                    EP_DEVICE_ASSERT(channel == channel_idx);
                    if (rank == scaleout_rank_idx)
                        continue;   // local-bypass, no RDMA put
                    bool is_duplicate = false;
                    #pragma unroll
                    for (int prior_k = 0; prior_k < k; ++ prior_k)
                        is_duplicate |= (packed_entries[prior_k] == p);
                    if (not is_duplicate)
                        local_expected_count_per_rank[rank] += 1;
                }
            }
        }

        // Each lane in the Forward warp (= channel) accumulated the expected tokens per scale-out
        // rank in its own slice of the map. All-reduce across lanes so every lane sees
        // the full per-rank total. Then divide by the batch size, that's the aggregation
        // granularity we expect on the wire.
        int num_expected_arrivals = 0;
        #pragma unroll
        for (int r = 0; r < kNumScaleoutRanks; ++ r) {
            const int rank_total = ptx::reduce_add(local_expected_count_per_rank[r]);
            num_expected_arrivals += math::ceil_div(rank_total, kBatchSize);
        }
        __syncwarp();
        // Wait for the per-channel indexed signal to accumulate `num_expected_arrivals`
        // increments from the remote senders. Bump the shadow by the expected delta to
        // get the target count, then poll the actual signal until it catches up. only
        // one lane polls. We poll with a timeout (instead of the blocking
        // `waitSignalMeetShadow`) so a stuck peer surfaces a diagnostic rather than
        // hanging.
        if (ptx::elect_one_sync()) {
            const auto shadow_ptr = gin.gin.getSignalShadowPtr(signal_id);
            const auto target = (*shadow_ptr += static_cast<uint64_t>(num_expected_arrivals));
            comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                const auto signal = gin.gin.readSignal(signal_id, 64, cuda::memory_order_acquire);
                if (signal >= target)
                    return true;

                if (is_last_check) {
                    printf("DeepEP combine (scale-out wait all) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                           "channel: %d, signal: %lu, target: %lu\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                           channel_idx, signal, target);
                }
                return false;
            });
        }
        __syncwarp();
    } else {
        // Proxy warp loop shape: arch-selected at JIT compile time.
        //   SM100+ (B200) -> sequential single-lane sweeper.
        //   otherwise (H200) -> parallel multi-lane (each lane owns its ring).
        //
        // B200 forward warps run faster so batches arrive at the proxy unevenly,
        // driving high lane divergence in the parallel design. H200 forward warps
        // are slower, so batches arrive more evenly and most lanes have work
        // together -> low divergence.
        //
        // Divergence cost per put step: sum(Ln for n in 0..kNumForwardWarps-1)
        // (all lanes' PC time) + lane-context-switch overhead. On B200 that beats
        // the serial sweep; on H200 the parallel throughput gain dominates.
        #if __CUDA_ARCH__ >= 1000
        constexpr bool kSingleLaneSweeper = true;
        #else
        constexpr bool kSingleLaneSweeper = false;
        #endif

        if constexpr (kSingleLaneSweeper) {
            // Sequential design: one lane sweeps all rings, issues puts serially.
            // Owns per-ring GIN contexts (built inline so each put routes to its
            // channel's QP). Best when per-put cost is high enough that warp-lockstep
            // parallelism doesn't pay off (B200 SM100).
            if (ptx::elect_one_sync()) {
                int num_forward_warps_done_cnt = 0;
                unsigned tail[kNumForwardWarps] = {};
                bool ring_done[kNumForwardWarps] = {};
                ncclGinSignal_t ring_signal_id[kNumForwardWarps];

                alignas(handle::NCCLGin) unsigned char gin_ctx_storage[kNumForwardWarps * sizeof(handle::NCCLGin)];
                auto* const gin_ctx = reinterpret_cast<handle::NCCLGin*>(gin_ctx_storage);
                #pragma unroll
                for (int forward_warp_idx = 0; forward_warp_idx < kNumForwardWarps; ++ forward_warp_idx) {
                    ring_signal_id[forward_warp_idx] = static_cast<ncclGinSignal_t>(
                        comm::get_qp_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm_idx, forward_warp_idx));
                    const auto [qp, mode] =
                        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm_idx, forward_warp_idx);
                    new (&gin_ctx[forward_warp_idx]) handle::NCCLGin(nccl_dev_comm, nccl_window, qp, mode);
                }

                while (num_forward_warps_done_cnt < kNumForwardWarps) {
                    int gin_put_ops = 0;

                    #pragma unroll
                    for (int forward_warp_idx = 0; forward_warp_idx < kNumForwardWarps; ++ forward_warp_idx) {
                        if (ring_done[forward_warp_idx])
                            continue;

                        const unsigned head = ptx::ld_acquire_cta(proxy_ring_layout.get_head(forward_warp_idx));

                        if (tail[forward_warp_idx] == head) {
                            if (ptx::ld_acquire_cta(proxy_ring_layout.get_done(forward_warp_idx)) == 1 and
                                tail[forward_warp_idx] == ptx::ld_acquire_cta(proxy_ring_layout.get_head(forward_warp_idx))) {
                                ring_done[forward_warp_idx] = true;
                                ++ num_forward_warps_done_cnt;
                            }
                            continue;
                        }

                        const auto& desc = proxy_ring_layout.get_ring(forward_warp_idx)[
                            tail[forward_warp_idx] % static_cast<unsigned>(kProxyRingDepth)];
                        gin_ctx[forward_warp_idx].put<ncclTeamTagRail>(
                            desc.recv_ptr, desc.send_ptr,
                            desc.num_bytes,
                            desc.dst,
                            0, /*flags=0*/
                            ncclGin_SignalAdd{ring_signal_id[forward_warp_idx], static_cast<uint64_t>(1)}
                        );

                        ++ tail[forward_warp_idx];
                        ptx::st_release_cta(proxy_ring_layout.get_tail(forward_warp_idx), tail[forward_warp_idx]);
                        ++ gin_put_ops;
                    }

                    if (gin_put_ops == 0)
                        __nanosleep(1000);
                }
            }
            __syncwarp();
        } else {
            // Parallel design: each lane represents one forward warp within this SM
            // and drains its own ring. Warp-lockstep parallelism across up to
            // kNumForwardWarps lanes — best when per-put cost is small (H200 SM90).
            if (lane_idx < kNumForwardWarps) {
                const auto channel_idx = sm_idx * kNumChannelsPerSM + lane_idx;
                const auto signal_id = static_cast<ncclGinSignal_t>(
                    comm::get_qp_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm_idx, lane_idx));

                ProxyPutDesc* const ring = proxy_ring_layout.get_ring(lane_idx);
                unsigned tail = 0;

                while (true) {
                    const unsigned head = ptx::ld_acquire_cta(proxy_ring_layout.get_head(lane_idx));
                    if (tail == head) {
                        if (ptx::ld_acquire_cta(proxy_ring_layout.get_done(lane_idx)) != 0) {
                            // Re-check head to avoid missing a descriptor published just
                            // before `done` became visible.
                            if (tail == ptx::ld_acquire_cta(proxy_ring_layout.get_head(lane_idx)))
                                break;
                        } else if (tail != 0 and (tail % static_cast<unsigned>(kNumForwardWarps)) == 0) {
                            __nanosleep(1000); // release context for data warps
                        }
                        continue;
                    }

                    const auto& desc = ring[tail % static_cast<unsigned>(kProxyRingDepth)];
                    gin.put<ncclTeamTagRail>(
                        desc.recv_ptr, desc.send_ptr,
                        desc.num_bytes,
                        desc.dst,
                        0, /*flags=0*/
                        ncclGin_SignalAdd{signal_id, static_cast<uint64_t>(1)}
                    );

                    ptx::st_release_cta(proxy_ring_layout.get_tail(lane_idx), tail + 1);
                    ++ tail;
                }
            }
        }
    }

    // No barrier at epilogue
}

}  // namespace deep_ep::elastic

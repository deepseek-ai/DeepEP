#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

template <bool kIsScaleupNVLink,
          bool kDoCPUSync,
          bool kReuseSlotIndices,
          int kNumSMs,
          int kNumNotifyWarps, int kNumDispatchWarps,
          int kNumRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk, int kExpertAlignment,
          int kNumQPs, int64_t kNumTimeoutCycles,
          int kNumNotifyThreads = kNumNotifyWarps * 32,
          int kNumDispatchThreads = kNumDispatchWarps * 32,
          int kNumThreads = kNumNotifyThreads + kNumDispatchThreads,
          typename team_t = std::conditional_t<kIsScaleupNVLink, ncclTeamTagLsa, ncclTeamTagWorld>>
__global__ void __launch_bounds__(kNumThreads, 1)
dispatch_impl(
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,
    topk_idx_t* copied_topk_idx,
    int* cumulative_local_expert_recv_stats,
    int* psum_num_recv_tokens_per_scaleup_rank,
    int* psum_num_recv_tokens_per_expert,
    int* num_unaligned_recv_tokens_per_expert,
    int* dst_buffer_slot_idx,
    const int num_tokens,
    const int sf_token_stride, const int sf_hidden_stride,
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window, void* buffer,
    void* workspace, void* mapped_host_workspace,
    const int rank_idx
) {
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    EP_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();

    // Workspaces
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, kNumExperts);
    const auto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace, 1, kNumRanks, kNumExperts);

    // The kernel uses a fixed space of dynamic shared memory (no static shared memory)
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");

    // Named barrier indices
    constexpr int kNotifyBarrierIndex = 1;

    // Gin handle
    // We treat each warp as a "channel"
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumDispatchWarps, (kNumNotifyWarps > 0)>(
        sm_idx, warp_idx - kNumNotifyWarps, warp_idx < kNumNotifyWarps);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // Barrier without TMA store flush, without prologue grid sync
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag0, false, false, true>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // Different warp roles
    if (warp_idx < kNumNotifyWarps) {
        // Assign shared memory
        constexpr int kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);

        // Clean initial counts
        // NOTES: if you want to change the order of different warp roles, please take care of the `thread_idx`
        int *rank_count = rank_expert_count, *expert_count = rank_expert_count + kNumRanks;
        #pragma unroll
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Atomic add on shared memory
        EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes");
        const auto global_warp_idx = warp_idx * kNumSMs + sm_idx;
        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
            // Expert choice can not be redundant
            // NOTES: no assertions here as they are expensive
            const auto dst_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            if (dst_expert_idx >= 0)
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            // Rank choice should do deduplication here
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
                atomicAdd_block(rank_count + dst_rank_idx, 1);
        }
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Do full-grid reduction
        #pragma unroll
        for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        // Do the remaining work by SM 0
        if (sm_idx == 0) {
            // Reduce all SM's count
            // Wait all SMs' arrival
            #pragma unroll
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>(true, [=](const bool& is_last_check) {
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);
                    if ((status >> 32) == kNumSMs) {
                        // Write into shared memory
                        // Write into send buffer if with RDMA
                        const auto encoded =
                            math::encode_decode_positive(static_cast<int>(status & 0xffffffffll));
                        rank_expert_count[i] = encoded;
                        if constexpr (not kIsScaleupNVLink)
                            workspace_layout.get_scaleup_rank_expert_count_ptr<true>()[i] = encoded;

                        // Clean for the next usage
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    if (is_last_check) {
                        printf("DeepEP notify (GPU reduction) timeout, rank: %d/%d, "
                               "thread: %d, status: %d | %d, expected: %d\n",
                               rank_idx, kNumRanks, thread_idx,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumSMs);
                    }
                    return false;
                });
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // TODO: for further optimization, we can fuse rank and expert counters
            // Issue scaleup rank count writes to peers
            for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                // Rank counters
                const auto dst_rank_counter =
                    workspace_layout.get_scaleup_rank_count_ptr<false>() + rank_idx;
                gin.put_value<team_t>(dst_rank_counter, static_cast<int64_t>(rank_count[i]), i,
                                      ncclGinOptFlagsAggregateRequests);
            }
            __syncwarp();

            // Issue scaleup expert count writes to peers
            if constexpr (kIsScaleupNVLink) {
                // NVLink per-element copy
                // We don't use TMA as the dtype of shared memory and global is different
                for (int i = thread_idx; i < kNumExperts; i += kNumNotifyThreads) {
                    const auto idx = kNumExpertsPerRank * rank_idx + (i % kNumExpertsPerRank);
                    gin.put_value<team_t>(
                        workspace_layout.get_scaleup_expert_count_ptr<false>() + idx,
                        static_cast<int64_t>(expert_count[i]), i / kNumExpertsPerRank);
                }
            } else {
                // RDMA bulk copy
                for (int i = thread_idx; i < kNumRanks; i += kNumNotifyThreads) {
                    const auto src_ptr = workspace_layout.get_scaleup_expert_count_ptr<true>() + kNumExpertsPerRank * i;
                    const auto dst_ptr = workspace_layout.get_scaleup_expert_count_ptr<false>() + kNumExpertsPerRank * rank_idx;
                    gin.put<team_t>(dst_ptr, src_ptr, kNumExpertsPerRank * sizeof(int64_t), i);
                }
            }

            // This is necessary, as the waited results will rewrite the shared memory
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Wait for rank and expert count
            const auto start_clock = clock64();
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    // NOTES: the global memory type has 64 bits
                    const auto count = static_cast<int>(
                        ptx::ld_volatile<int64_t>(workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + i));
                    const auto decoded = math::encode_decode_positive(count);
                    if (math::is_decoded_positive_ready(decoded)) {
                        workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] = 0;
                        rank_expert_count[i] = decoded;
                        return true;
                    }

                    if (is_last_check)
                        printf("DeepEP notify timeout, rank: %d, thread: %d, count: %d\n", rank_idx, i, decoded);
                    return false;
                }, start_clock);
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Reduce expert count and add stats
            for (int i = thread_idx; i < kNumExpertsPerRank; i += kNumNotifyThreads) {
                int sum = 0;
                #pragma unroll
                for (int j = 0; j < kNumRanks; ++ j)
                    sum += expert_count[j * kNumExpertsPerRank + i];

                // Write unaligned count before aligning
                if (num_unaligned_recv_tokens_per_expert != nullptr)
                    num_unaligned_recv_tokens_per_expert[i] = sum;

                expert_count[i] = math::align(sum, kExpertAlignment);

                // Update statistics counters
                if (cumulative_local_expert_recv_stats != nullptr)
                    atomicAdd(cumulative_local_expert_recv_stats + i, sum);
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Write host workspace
            if constexpr (kDoCPUSync) {
                for (int i = thread_idx; i < kNumRanks + kNumExpertsPerRank; i += kNumNotifyThreads) {
                    host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i] =
                        math::encode_decode_positive(rank_expert_count[i]);
                }
                __syncwarp();
            }

            // Do prefix sum by the warps
            // NOTES: we may have fast implementation with `cub::BlockScan`, but it is too heavy to use
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
                int psum = 0;
                #pragma unroll
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {
                    const auto idx = i * 32 + lane_idx;
                    const auto mem_idx = idx - is_exclusive;
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);

                    // Store into global memory
                    if (idx < n + is_exclusive)
                        out[idx] = sum;

                    // Update `psum` by using the last lane's value
                    psum = ptx::exchange(sum, 31);
                }
            };
            if (warp_idx == 0) {
                // Inclusive prefix sum
                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumRanks, 0);
            } else if (warp_idx == 1) {
                // Exclusive prefix sum for later expanding
                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }
        }
    } else {
        const int dispatch_warp_idx = warp_idx - kNumNotifyWarps;

        // Buffer layouts
        const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
        constexpr int kNumTmaBuffers = kIsScaleupNVLink ? 2 : 1;
        const auto warp_tma_buffers = layout::BufferLayout<true>(token_layout, kNumDispatchWarps, kNumTmaBuffers,
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(dispatch_warp_idx);
        auto recv_buffer = layout::BufferLayout<false>(token_layout, kNumRanks, kNumMaxTokensPerRank, buffer);
        auto send_buffer = layout::BufferLayout<false>(token_layout, 1, kNumMaxTokensPerRank, recv_buffer.get_buffer_end_ptr());
        recv_buffer = recv_buffer.get_rank_buffer(rank_idx);

        // Init TMA
        ptx::arrival_phase phase_0 = 0, phase_1 = 0;
        #pragma unroll
        for (int i = 0; i < kNumTmaBuffers; ++ i) {
            const auto tma_buffer = warp_tma_buffers.get_token_buffer(i);
            if (ptx::elect_one_sync())
                ptx::mbarrier_init_with_fence(tma_buffer.get_mbarrier_ptr(), 1);
        }
        __syncwarp();

        // Iterate all tokens
        const auto token_start = dispatch_warp_idx * kNumSMs + sm_idx;
        const auto token_stride = kNumDispatchWarps * kNumSMs;
        for (int token_idx = token_start, token_iter = 0;
             token_idx < num_tokens; token_idx += token_stride, ++ token_iter) {
            const auto token_i64_idx = static_cast<int64_t>(token_idx);
            const int tma_buffer_idx = token_iter & (kNumTmaBuffers - 1);
            const auto tma_buffer = warp_tma_buffers.get_token_buffer(tma_buffer_idx);
            const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
            auto& phase = tma_buffer_idx == 0 ? phase_0 : phase_1;

            // Wait TMA store arrivals
            if constexpr (kIsScaleupNVLink)
                ptx::tma_store_wait<1>();
            else
                ptx::tma_store_wait();
            __syncwarp();

            // Issue data TMA
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            __syncwarp();

            // Issue SF TMA or cp.async
            if constexpr (kNumSFPacks > 0) {
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k) {
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                __syncwarp();
            }

            // Load top-k indices and weights
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
            int stored_dst_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
                const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
                stored_dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                if (copied_topk_idx != nullptr)
                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
            }
            __syncwarp();

            // Add source metadata (rank index and token index)
            // Please ensure no TMA buffer shared memory writes after this part
            if (ptx::elect_one_sync())
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
            ptx::tma_store_fence();
            __syncwarp();

            // Deduplicate ranks and assign slots
            int stored_dst_slot_idx = -1;
            if constexpr (kReuseSlotIndices) {
                if (lane_idx < kNumTopk)
                    stored_dst_slot_idx = __ldg(dst_buffer_slot_idx + token_idx * kNumTopk + lane_idx);
                stored_dst_slot_idx = stored_dst_slot_idx >= 0 ?
                    (stored_dst_slot_idx - rank_idx * kNumMaxTokensPerRank) : -1;
            } else {
                if (ptx::deduplicate(stored_dst_rank_idx, lane_idx) and stored_dst_rank_idx >= 0)
                    stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_rank_idx, 1);
                if (lane_idx < kNumTopk) {
                    const auto value = stored_dst_slot_idx >= 0 ?
                        rank_idx * kNumMaxTokensPerRank + stored_dst_slot_idx : -1;
                    dst_buffer_slot_idx[token_idx * kNumTopk + lane_idx] = value;
                }
            }
            __syncwarp();

            // Wait TMA load arrival
            // NOTES: this arrive must be after the `ptx::cp_async_mbarrier_arrive`
            if (ptx::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
            }
            __syncwarp();

            // TMA store to send buffer
            auto send_buffer_ptr = send_buffer.get_token_buffer(token_idx).get_base_ptr();
            if constexpr (not kIsScaleupNVLink) {
                if (ptx::elect_one_sync())
                    ptx::tma_store_1d(send_buffer_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                ptx::tma_store_commit();
                __syncwarp();
            }

            // Issue TMA NVLink stores
            EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid top-k selection");
            const auto dst_ptr = stored_dst_slot_idx >= 0 ?
                gin.get_sym_ptr<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(), stored_dst_rank_idx) :
                nullptr;
            if (dst_ptr != nullptr)
                ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            ptx::tma_store_commit();
            __syncwarp();

            // Issue RDMA put
            if constexpr (not kIsScaleupNVLink) {
                // Wait the send buffer store to arrive
                ptx::tma_store_wait<1>();
                __syncwarp();

                // NOTES: we should skip the NVLink accessible ranks
                if (stored_dst_slot_idx >= 0 and dst_ptr == nullptr) {
                    gin.put<team_t>(recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                    send_buffer_ptr, tma_buffer.get_num_bytes<false>(), stored_dst_rank_idx);
                }
                __syncwarp();
            }
        }
    }

    // Barrier to ensure data arrival
    comm::gpu_barrier<kIsScaleupNVLink, 1, kNumRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kDispatchTag1, true, true, false>(
        gin, workspace_layout, 0, rank_idx, sm_idx, thread_idx);

    // Trigger the copy epilogue kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // Clean atomic counters
    EP_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    if (not kReuseSlotIndices and sm_idx == 0 and thread_idx < kNumRanks)
        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;
}

}  // namespace deep_ep::elastic

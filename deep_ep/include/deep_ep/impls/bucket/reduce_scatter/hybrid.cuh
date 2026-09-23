#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/utils.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

namespace deep_ep::bucket {


template <int kNumRDMARanks, int kNumNVLRanks,
          int kNumSMs,
          int kNumMultimemWarps, int kNumMultimemWarpsPerGroup, int kNumIssueWarps, int kNumReduceWarps,
          int kNumSmemBytesPerReduceWarp,
          int kNumQPs,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx,
          int kNumThreads = (kNumMultimemWarps + kNumIssueWarps + kNumReduceWarps) * 32,
          int kNumMultimemGroups = kNumMultimemWarps / kNumMultimemWarpsPerGroup,
          int kNumMultimemThreadsPerGroup = kNumMultimemWarpsPerGroup * 32>
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_reduce_scatter(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                      void* workspace_ptr, void* buffer, void* buffer_multimem,
                      const float scale,
                      const int num_buckets,
                      const __grid_constant__ comm::BucketList buckets) {
    // Check slots
    // Chunk indices fit in int for every supported workspace size.
    constexpr int kNumChunkBytes = 32 * 1024;
    constexpr int kNumIssueChunksPerBatch = 2;
    constexpr int kNumSlotsPerRank = layout::ChunkStorage::kNumBytes / kNumChunkBytes / kNumRDMARanks;
    EP_STATIC_ASSERT(kNumSlotsPerRank > 0, "Insufficient chunk storage for hybrid reduce-scatter");
    EP_STATIC_ASSERT(kNumMultimemWarps % kNumMultimemWarpsPerGroup == 0,
                     "Multimem warps must be evenly divided across groups");
    EP_STATIC_ASSERT(kNumIssueWarps == kNumMultimemGroups,
                     "Each multimem group requires one issue warp");
    EP_STATIC_ASSERT(kNumMultimemWarps % 4 == 0 and (kNumIssueWarps + kNumReduceWarps) % 4 == 0,
                     "Register roles must align to warp groups");
    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Insufficient issue lanes for RDMA peers");
    EP_STATIC_ASSERT(kNumSmemBytesPerReduceWarp > 0 and
                     kNumSmemBytesPerReduceWarp % kNumTMAAlignmentBytes == 0,
                     "Invalid shared-memory size per reduce warp");
    EP_STATIC_ASSERT(kNumRDMARanks * kNumSlotsPerRank <= layout::ReduceScatterSignals::kNumTotalSlots,
                     "Reduce-scatter slots exceed the signal workspace");

    // Registers
    constexpr int kNumMultimemRegisters = 192;
    constexpr int kNumNonMultimemRegisters = 96;
    EP_STATIC_ASSERT(32 * (kNumMultimemWarps * kNumMultimemRegisters +
                          (kNumIssueWarps + kNumReduceWarps) * kNumNonMultimemRegisters) <= 65536 - 2048,
                     "Too many registers");

    // Barriers
    constexpr int kMultimemBarrierBaseIdx = 1;
    EP_STATIC_ASSERT(kMultimemBarrierBaseIdx + kNumMultimemGroups - 1 < 16, "Too many multimem groups for named barriers");

    // Indices
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    const auto rdma_rank_idx = ncclTeamRail(nccl_dev_comm).rank;
    const auto nvl_rank_idx = ncclTeamLsa(nccl_dev_comm).rank;
    const auto global_thread_idx = sm_idx * kNumThreads + thread_idx;

    // Init Gin
    constexpr int kNumChannelsPerSM = kNumMultimemGroups + kNumReduceWarps;
    constexpr int kNumUsedQPs = std::min(kNumQPs, kNumSMs * kNumChannelsPerSM);
    const int channel_idx = warp_idx < kNumMultimemWarps ?
        warp_idx / kNumMultimemWarpsPerGroup : warp_idx - kNumMultimemWarps;
    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM>(sm_idx, channel_idx);
    const comm::NCCLGin gin(nccl_dev_comm, window, qp_idx, sharing_mode);

    // Shared memory and mbarriers
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    __shared__ __align__(8) uint64_t mbarriers[kNumReduceWarps];
    if (thread_idx < kNumReduceWarps)
        ptx::mbarrier_init_with_fence(reinterpret_cast<ptx::mbarrier*>(mbarriers + thread_idx), 1);
    __syncthreads();

    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    auto& bucket_signals = workspace.signals[kContextIdx];
    auto& barrier_signals = bucket_signals.barrier;
    auto& signals = bucket_signals.reduce_scatter;

    // Every launch starts a fresh receive-credit ring.
    for (int i = global_thread_idx; i < kNumRDMARanks * kNumSlotsPerRank; i += kNumSMs * kNumThreads) {
        const int peer_rank_idx = i / kNumSlotsPerRank;
        const int slot_idx = i % kNumSlotsPerRank;
        *signals.get_head_ptr<kNumSlotsPerRank>(peer_rank_idx, slot_idx) = 0;
        *signals.get_tail_ptr<kNumSlotsPerRank>(peer_rank_idx, slot_idx) = 0;
    }
    comm::gpu_barrier<true, kNumRDMARanks, kNumNVLRanks,
                      kNumSMs, kNumThreads, kNumQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag,
                      false, true, true>(
        gin, barrier_signals, rdma_rank_idx, nvl_rank_idx,
        sm_idx, thread_idx);

    // Different warp roles
    if (warp_idx < kNumMultimemWarps) {
        ptx::warpgroup_reg_realloc<kNumMultimemRegisters, kNumThreads>();

        // Multimem reduce
        constexpr int kNumGlobalMultimemGroups = kNumSMs * kNumMultimemGroups;
        const int multimem_group_idx = warp_idx / kNumMultimemWarpsPerGroup;
        const int multimem_thread_idx = warp_idx % kNumMultimemWarpsPerGroup * 32 + lane_idx;

        // NOTES: must visit the same chunk sequence as the paired issue warp
        auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumGlobalMultimemGroups>(num_buckets, buckets, sm_idx * kNumMultimemGroups + multimem_group_idx);
        while (iterator.next()) {
            // NVLink domain reduction
            #pragma unroll 1
            for (int peer_begin = 0; peer_begin < kNumRDMARanks; peer_begin += kNumIssueChunksPerBatch) {
                const int num_batch_peers = min(kNumIssueChunksPerBatch, kNumRDMARanks - peer_begin);
                #pragma unroll 1
                for (int i = 0; i < num_batch_peers; ++ i) {
                    const auto dst_rdma_rank_idx = (rdma_rank_idx + peer_begin + i) % kNumRDMARanks;
                    const auto dst_rank_idx = dst_rdma_rank_idx * kNumNVLRanks + nvl_rank_idx;
                    const auto src_offset = iterator.bucket.offset + dst_rank_idx * iterator.bucket.num_bytes + iterator.chunk_offset_in_bucket;
                    const auto src_multimem_ptr = math::advance_ptr<const float4>(buffer_multimem, src_offset);
                    const auto src_ptr = math::advance_ptr<float4>(buffer, src_offset);
                    utils::iterate_ld_st<kNumMultimemWarpsPerGroup, 32, 16>(
                        multimem_thread_idx,
                        iterator.num_chunk_bytes / static_cast<int>(sizeof(float4)),
                        src_multimem_ptr, src_ptr,
                        [&](const float4* ptr, const int& lhs = 1, const int& rhs = 0) {
                            const auto v = ptx::multimem_ld_reduce_add_f32_with_gt_pred(ptr, lhs, rhs);
                            return kWithScale ? ptx::fmul4(v, scale) : v;
                        },
                        [](float4* ptr, const float4& value, const int& lhs = 1, const int& rhs = 0) { ptx::st_with_gt_pred(ptr, value, lhs, rhs); });
                }

                // Notify issue warps
                ptx::named_barrier_unaligned<kNumMultimemThreadsPerGroup + 32>(kMultimemBarrierBaseIdx + multimem_group_idx);
            }
        }
    } else {
        ptx::warpgroup_reg_realloc<kNumNonMultimemRegisters, kNumThreads>();

        if (warp_idx < kNumMultimemWarps + kNumIssueWarps) {
            // RDMA issue
            constexpr int kNumGlobalMultimemGroups = kNumSMs * kNumMultimemGroups;
            const auto multimem_group_idx = warp_idx - kNumMultimemWarps;
            const auto chunks = layout::ChunkView2D<kNumRDMARanks, kNumSlotsPerRank, kNumChunkBytes>{workspace.chunk_storage};

            // NOTES: must visit the same chunk sequence as the paired multimem group
            auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumGlobalMultimemGroups>(num_buckets, buckets, sm_idx * kNumMultimemGroups + multimem_group_idx);
            while (iterator.next()) {
                const int slot_idx = iterator.chunk_idx % kNumSlotsPerRank;
                const int head_target = iterator.chunk_idx / kNumSlotsPerRank;
                #pragma unroll 1
                for (int peer_begin = 0; peer_begin < kNumRDMARanks; peer_begin += kNumIssueChunksPerBatch) {
                    const int num_batch_peers = min(kNumIssueChunksPerBatch, kNumRDMARanks - peer_begin);
                    const int dst_rdma_rank_idx = (rdma_rank_idx + peer_begin + lane_idx) % kNumRDMARanks;
                    const int dst_rank_idx = dst_rdma_rank_idx * kNumNVLRanks + nvl_rank_idx;
                    const auto src_ptr = math::advance_ptr<float4>(
                        buffer, iterator.bucket.offset + dst_rank_idx * iterator.bucket.num_bytes + iterator.chunk_offset_in_bucket);
                    const auto head_ptr = signals.get_head_ptr<kNumSlotsPerRank>(dst_rdma_rank_idx, slot_idx);
                    const auto tail_ptr = signals.get_tail_ptr<kNumSlotsPerRank>(rdma_rank_idx, slot_idx);

                    // Wait until the previous generation has been consumed
                    // NOTES: this only depends on the remote consumers, so it overlaps with the local reduction
                    comm::timeout_while<kNumTimeoutCycles>(lane_idx < num_batch_peers, [=](const bool& is_last_check) {
                        const auto head = ptx::ld_volatile<int64_t>(head_ptr);
                        if (head >= head_target)
                            return true;

                        if (is_last_check) {
                            printf("DeepEP hybrid reduce-scatter send timeout: RDMA rank %d, NVL rank %d, "
                                   "peer %d, chunk %d, head target %d, head %lld\n",
                                   rdma_rank_idx, nvl_rank_idx, dst_rdma_rank_idx,
                                   iterator.chunk_idx, head_target, static_cast<long long>(head));
                        }
                        return false;
                    });

                    // Wait reduction finish
                    ptx::named_barrier_unaligned<kNumMultimemThreadsPerGroup + 32>(kMultimemBarrierBaseIdx + multimem_group_idx);

                    // Issue RDMA
                    if (lane_idx < num_batch_peers and dst_rdma_rank_idx == rdma_rank_idx) {
                        // The local chunk is already in place
                        ptx::red_add_rel_gpu(tail_ptr, 1);
                    } else if (lane_idx < num_batch_peers) {
                        // Push the reduced chunk and publish its arrival
                        gin.put<ncclTeamTagRail>(
                            chunks.get_chunk_ptr(rdma_rank_idx, slot_idx),
                            src_ptr, iterator.num_chunk_bytes, dst_rdma_rank_idx,
                            ncclGinOptFlagsDefault,
                            ncclGin_StrongVASignalAdd {
                                .signalWindow = window,
                                .signalOffset = gin.get_sym_offset(tail_ptr),
                                .value = 1,
                            });
                    }
                    __syncwarp();
                }
            }
        } else {
            // Reduce warps: accumulate remote chunks into the local output
            // Only 1 lane is involved
            if (ptx::elect_one_sync()) {
                constexpr int kNumGlobalReduceWarps = kNumSMs * kNumReduceWarps;

                const int rank_idx = rdma_rank_idx * kNumNVLRanks + nvl_rank_idx;
                const int reduce_warp_idx = warp_idx - kNumMultimemWarps - kNumIssueWarps;
                const auto mbarrier_ptr = reinterpret_cast<ptx::mbarrier*>(mbarriers + reduce_warp_idx);
                const auto smem_ptr = smem + reduce_warp_idx * kNumSmemBytesPerReduceWarp;
                const auto chunks = layout::ChunkView2D<kNumRDMARanks, kNumSlotsPerRank, kNumChunkBytes>{workspace.chunk_storage};
                ptx::arrival_phase phase = 0;

                auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumGlobalReduceWarps>(num_buckets, buckets, sm_idx * kNumReduceWarps + reduce_warp_idx);
                while (iterator.next()) {
                    const auto slot_idx = iterator.chunk_idx % kNumSlotsPerRank;
                    const auto tail_target = iterator.chunk_idx / kNumSlotsPerRank + 1;
                    const auto head_ptr = signals.get_head_ptr<kNumSlotsPerRank>(rdma_rank_idx, slot_idx);
                    const auto dst_ptr = math::advance_ptr<float>(
                        buffer, iterator.bucket.offset + rank_idx * iterator.bucket.num_bytes + iterator.chunk_offset_in_bucket);

                    #pragma unroll 1
                    for (int i = 0; i < kNumRDMARanks; ++ i) {
                        const int src_rdma_rank_idx = (rdma_rank_idx - i + kNumRDMARanks) % kNumRDMARanks;
                        const auto tail_ptr = signals.get_tail_ptr<kNumSlotsPerRank>(src_rdma_rank_idx, slot_idx);

                        // Wait arrival
                        comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                            const auto tail = ptx::ld_acquire_sys(tail_ptr);
                            if (tail >= tail_target)
                                return true;
                            if (is_last_check) {
                                printf("DeepEP hybrid reduce-scatter receive timeout: RDMA rank %d, NVL rank %d, "
                                       "peer %d, chunk %d, tail target %d, tail %lld\n",
                                       rdma_rank_idx, nvl_rank_idx, src_rdma_rank_idx,
                                       iterator.chunk_idx, tail_target, static_cast<long long>(tail));
                            }
                            return false;
                        });

                        // No local reduction
                        if (i == 0) {
                            // Release the local generation
                            ptx::red_add_rel_gpu(head_ptr, 1);
                            continue;
                        }

                        // Reduce the remote chunk through shared memory into the output
                        // NOTES: TMA atomic add can ensure deterministic if no other SMs involved
                        const auto recv_buffer_ptr = chunks.get_chunk_ptr(src_rdma_rank_idx, slot_idx);
                        #pragma unroll 1
                        for (int offset = 0; offset < iterator.num_chunk_bytes; offset += kNumSmemBytesPerReduceWarp) {
                            const int num_bytes = min(kNumSmemBytesPerReduceWarp, iterator.num_chunk_bytes - offset);
                            ptx::tma_store_wait();
                            ptx::tma_load_1d(
                                smem_ptr,
                                math::advance_ptr<const void>(recv_buffer_ptr, offset),
                                mbarrier_ptr, num_bytes);
                            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, num_bytes);
                            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                            ptx::tma_store_reduce_add_f32(math::advance_ptr<float>(dst_ptr, offset), smem_ptr, num_bytes);
                            ptx::tma_store_commit();
                        }

                        // Release the remote ring slot only if a later chunk will reuse it.
                        if (iterator.chunk_idx < iterator.num_total_chunks - kNumSlotsPerRank) {
                            gin.signal<ncclTeamTagRail>(
                                src_rdma_rank_idx,
                                ncclGin_StrongVASignalAdd {
                                    .signalWindow = window,
                                    .signalOffset = gin.get_sym_offset(head_ptr),
                                    .value = 1,
                                });
                        }
                    }
                }
            }
        }
    }

    // Barrier to ensure data arrival
    comm::gpu_barrier<true, kNumRDMARanks, kNumNVLRanks,
                      kNumSMs, kNumThreads, kNumUsedQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag,
                      true, true, false>(
        gin, barrier_signals, rdma_rank_idx, nvl_rank_idx,
        sm_idx, thread_idx);
}

}  // namespace deep_ep::bucket

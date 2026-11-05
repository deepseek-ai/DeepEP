#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/utils.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

namespace deep_ep::bucket {

template <int kNumRanks, int kNumSMs,
          int kNumIssueWarps, int kNumReduceWarps,
          int kNumSmemBytesPerReduceWarp,
          int kNumQPs,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx,
          int kNumWarps = kNumIssueWarps + kNumReduceWarps,
          int kNumThreads = kNumWarps * 32,
          int kNumGlobalIssueWarps = kNumSMs * kNumIssueWarps,
          int kNumGlobalReduceWarps = kNumSMs * kNumReduceWarps,
          int kNumGlobalThreads = kNumSMs * kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
rdma_all_reduce(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                void* workspace_ptr, void* buffer,
                const float scale,
                const int num_buckets,
                const __grid_constant__ comm::BucketList buckets) {
    // Check slots
    // Chunk indices fit in int for every supported workspace size.
    EP_STATIC_ASSERT(kNumRanks > 1, "RDMA all-reduce requires multiple RDMA ranks");
    constexpr int kNumChunkBytes = (kNumRanks <= 64 ? 64 : 32) * 1024;
    constexpr int kNumSlotsPerRank = layout::ChunkStorage::kNumBytes / kNumChunkBytes / kNumRanks;
    EP_STATIC_ASSERT(kNumSlotsPerRank > 0, "Insufficient chunk storage for RDMA all-reduce");
    EP_STATIC_ASSERT(kNumSmemBytesPerReduceWarp > 0 and
                     kNumSmemBytesPerReduceWarp % kNumTMAAlignmentBytes == 0,
                     "Invalid shared-memory size per reduce warp");
    EP_STATIC_ASSERT(2 * kNumRanks * kNumSlotsPerRank <= layout::AllReduceSignals::kNumTotalSlots,
                     "All-reduce slots exceed the signal workspace");

    // Issue and reduce warps must form complete warp groups
    EP_STATIC_ASSERT(kNumIssueWarps % 4 == 0 and kNumReduceWarps % 4 == 0,
                     "Issue and reduce warps must align to warp groups");

    // Indices
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    const auto rank_idx = ncclTeamRail(nccl_dev_comm).rank;
    const auto global_thread_idx = sm_idx * kNumThreads + thread_idx;

    // Init Gin
    constexpr int kNumChannelsPerSM = kNumIssueWarps + kNumReduceWarps;
    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM>(sm_idx, warp_idx);
    const comm::NCCLGin gin(nccl_dev_comm, window, qp_idx, sharing_mode);

    // Construct workspace layouts
    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    auto& bucket_signals = workspace.signals[kContextIdx];
    auto& barrier_signals = bucket_signals.barrier;
    auto& signals = bucket_signals.all_reduce;
    const auto rs_head = signals.counters;
    const auto rs_tail = rs_head + kNumRanks * kNumSlotsPerRank;

    // Shared memory and mbarriers for TMA reduction
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    __shared__ __align__(8) uint64_t mbarriers[kNumReduceWarps];

    // Every launch starts with fresh signal counters
    EP_STATIC_ASSERT(sizeof(layout::AllReduceSignals) % sizeof(int4) == 0 and
                     alignof(layout::AllReduceSignals) % alignof(int4) == 0,
                     "Signals must be a multiple of int4");
    constexpr auto kNumSignalVecs = static_cast<int>(sizeof(layout::AllReduceSignals) / sizeof(int4));
    const auto signal_vecs = reinterpret_cast<int4*>(&signals);
    #pragma unroll
    for (int i = global_thread_idx; i < kNumSignalVecs; i += kNumGlobalThreads)
        signal_vecs[i] = make_int4(0, 0, 0, 0);
    if (thread_idx < kNumReduceWarps)
        ptx::mbarrier_init_with_fence(reinterpret_cast<ptx::mbarrier*>(mbarriers + thread_idx), 1);

    // Barrier to make the cleared signals visible and ensure data arrival
    comm::gpu_barrier<true, kNumRanks, 1,
                      kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);

    const auto chunks = layout::ChunkView2D<kNumRanks, kNumSlotsPerRank, kNumChunkBytes>{workspace.chunk_storage};

    if (warp_idx < kNumIssueWarps) {
        // Each issue warp sends its chunks through a single elected lane and GIN channel
        const int global_issue_warp_idx = sm_idx * kNumIssueWarps + warp_idx;
        auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumGlobalIssueWarps>(
            num_buckets, buckets, global_issue_warp_idx);
        if (ptx::elect_one_sync()) {
            while (iterator.next()) {
                const int dst_rank_idx = iterator.chunk_idx % kNumRanks;
                if (dst_rank_idx == rank_idx)
                    continue;

                const int chunk_idx_in_shard = iterator.chunk_idx / kNumRanks;
                const int slot_idx = chunk_idx_in_shard % kNumSlotsPerRank;
                const int64_t rs_head_target = chunk_idx_in_shard / kNumSlotsPerRank;
                const auto rs_head_ptr = rs_head + dst_rank_idx * kNumSlotsPerRank + slot_idx;
                const auto rs_tail_ptr = rs_tail + rank_idx * kNumSlotsPerRank + slot_idx;

                // Wait release
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    // NOTES: no need to acquire
                    const auto head_val = ptx::ld_volatile<int64_t>(rs_head_ptr);
                    if (head_val >= rs_head_target)
                        return true;
                    if (is_last_check) {
                        printf("DeepEP RDMA all-reduce send timeout: rank %d, peer %d, "
                               "chunk %d, head target %lld, head %lld\n",
                               rank_idx, dst_rank_idx, iterator.chunk_idx,
                               static_cast<long long>(rs_head_target), static_cast<long long>(head_val));
                    }
                    return false;
                });

                const auto src_ptr = math::advance_ptr(
                    buffer, iterator.bucket.offset + iterator.chunk_offset_in_bucket);
                gin.put<ncclTeamTagRail>(
                    chunks.get_chunk_ptr(rank_idx, slot_idx),
                    src_ptr, iterator.num_chunk_bytes, dst_rank_idx, 0,
                    ncclGin_StrongVASignalAdd {
                        .signalWindow = window,
                        .signalOffset = gin.get_sym_offset(rs_tail_ptr),
                        .value = 1,
                    });
            }
        }
    } else {
        // Reduce warps consume their owned chunks in the deterministic cyclic rank order
        const int reduce_warp_idx = warp_idx - kNumIssueWarps;
        const int global_reduce_warp_idx = sm_idx * kNumReduceWarps + reduce_warp_idx;
        const auto mbarrier_ptr = reinterpret_cast<ptx::mbarrier*>(mbarriers + reduce_warp_idx);
        const auto smem_ptr = smem + reduce_warp_idx * kNumSmemBytesPerReduceWarp;
        ptx::arrival_phase phase = 0;
        auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumRanks * kNumGlobalReduceWarps>(
            num_buckets, buckets, rank_idx + kNumRanks * global_reduce_warp_idx);

        while (iterator.next()) {
            const int chunk_idx_in_shard = iterator.chunk_idx / kNumRanks;
            const int slot_idx = chunk_idx_in_shard % kNumSlotsPerRank;
            const int64_t rs_tail_target = chunk_idx_in_shard / kNumSlotsPerRank + 1;
            const auto rs_head_ptr = rs_head + rank_idx * kNumSlotsPerRank + slot_idx;
            const auto out_ptr = math::advance_ptr<float>(
                buffer, iterator.bucket.offset + iterator.chunk_offset_in_bucket);

            // Without scaling, all peers use TMA reduction
            // With scaling, leave the last peer for a warp-wide register reduction
            constexpr int kNumTMAPeers = kWithScale ? kNumRanks - 2 : kNumRanks - 1;
            #pragma unroll 1
            for (int i = 1; i <= kNumTMAPeers; ++ i) {
                const int src_rank_idx = (rank_idx + i) % kNumRanks;
                const auto rs_tail_ptr = rs_tail + src_rank_idx * kNumSlotsPerRank + slot_idx;

                if (ptx::elect_one_sync()) {
                    comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                        const auto tail_val = ptx::ld_acquire_sys(rs_tail_ptr);
                        if (tail_val >= rs_tail_target)
                            return true;
                        if (is_last_check) {
                            printf("DeepEP RDMA all-reduce receive timeout: rank %d, peer %d, "
                                   "chunk %d, tail target %lld, tail %lld\n",
                                   rank_idx, src_rank_idx, iterator.chunk_idx,
                                   static_cast<long long>(rs_tail_target), static_cast<long long>(tail_val));
                        }
                        return false;
                    });

                    const auto recv_ptr = chunks.get_chunk_ptr(src_rank_idx, slot_idx);
                    #pragma unroll 1
                    for (int offset = 0; offset < iterator.num_chunk_bytes; offset += kNumSmemBytesPerReduceWarp) {
                        const int num_bytes = min(kNumSmemBytesPerReduceWarp, iterator.num_chunk_bytes - offset);
                        ptx::tma_store_wait();
                        ptx::tma_load_1d(
                            smem_ptr, math::advance_ptr<const void>(recv_ptr, offset),
                            mbarrier_ptr, num_bytes);
                        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, num_bytes);
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                        ptx::tma_store_reduce_add_f32(
                            math::advance_ptr<float>(out_ptr, offset), smem_ptr, num_bytes);
                        ptx::tma_store_commit();
                    }
                    ptx::tma_store_wait();
                    gin.signal<ncclTeamTagRail>(
                        src_rank_idx,
                        ncclGin_StrongVASignalAdd {
                            .signalWindow = window,
                            .signalOffset = gin.get_sym_offset(rs_head_ptr),
                            .value = 1,
                        });
                }
                __syncwarp();
            }

            if constexpr (kWithScale) {
                // The final rank is reduced and scaled in registers
                constexpr int kNumVecBytes = sizeof(float4);
                const int src_rank_idx = (rank_idx + kNumRanks - 1) % kNumRanks;
                const auto rs_tail_ptr = rs_tail + src_rank_idx * kNumSlotsPerRank + slot_idx;
                if (ptx::elect_one_sync()) {
                    comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                        const auto tail_val = ptx::ld_acquire_sys(rs_tail_ptr);
                        if (tail_val >= rs_tail_target)
                            return true;
                        if (is_last_check) {
                            printf("DeepEP RDMA all-reduce receive timeout: rank %d, peer %d, "
                                   "chunk %d, tail target %lld, tail %lld\n",
                                   rank_idx, src_rank_idx, iterator.chunk_idx,
                                   static_cast<long long>(rs_tail_target), static_cast<long long>(tail_val));
                        }
                        return false;
                    });
                }
                __syncwarp();

                const auto recv_ptr = static_cast<const float4*>(chunks.get_chunk_ptr(src_rank_idx, slot_idx));
                const auto out_vec_ptr = reinterpret_cast<float4*>(out_ptr);
                const int num_vecs = iterator.num_chunk_bytes / kNumVecBytes;
                constexpr int kNumMaxVecs = kNumChunkBytes / kNumVecBytes;
                #pragma unroll
                for (int i = 0; i < math::constexpr_ceil_div(kNumMaxVecs, 32); ++ i) {
                    const int vec_idx = i * 32 + lane_idx;
                    const auto x = ptx::ld_with_gt_pred(out_vec_ptr + vec_idx, num_vecs, vec_idx);
                    const auto y = ptx::ld_with_gt_pred(recv_ptr + vec_idx, num_vecs, vec_idx);
                    const auto value = ptx::fmul4(ptx::fadd4(x, y), scale);
                    ptx::st_with_gt_pred(out_vec_ptr + vec_idx, value, num_vecs, vec_idx);
                }
                __syncwarp();
                if (ptx::elect_one_sync()) {
                    gin.signal<ncclTeamTagRail>(
                        src_rank_idx,
                        ncclGin_StrongVASignalAdd {
                            .signalWindow = window,
                            .signalOffset = gin.get_sym_offset(rs_head_ptr),
                            .value = 1,
                        });
                }
            }
            __syncwarp();

            // Broadcast the reduced chunk to every peer
            for (int i = lane_idx; i < kNumRanks - 1; i += 32) {
                const int dst_rank_idx = (rank_idx + i + 1) % kNumRanks;
                gin.put<ncclTeamTagRail>(out_ptr, out_ptr, iterator.num_chunk_bytes, dst_rank_idx);
            }
            __syncwarp();
        }
    }

    // Flush QPs and ensure the visiblity of preceding puts on responders
    comm::gpu_barrier<true, kNumRanks, 1,
                      kNumSMs, kNumThreads, kNumQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);
}

}  // namespace deep_ep::bucket

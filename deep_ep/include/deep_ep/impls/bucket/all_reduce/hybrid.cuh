#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/utils.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

namespace deep_ep::bucket {

template <int kNumTileBytes>
__device__ __forceinline__ void
hybrid_ar_broadcast(const int8_t* src, float4* dst, int num_bytes, int8_t* smem,
                    ptx::mbarrier* mbarrier, ptx::arrival_phase& phase) {
    if (ptx::elect_one_sync()) {
        const int count = min(kNumTileBytes, num_bytes);
        ptx::tma_load_1d(smem, src, mbarrier, count);
        ptx::mbarrier_arrive_and_set_tx(mbarrier, count);
    }
    // Steady state of the pipeline
    #pragma unroll 1
    for (int offset = 0; offset < num_bytes; offset += kNumTileBytes) {
        const int slot = (offset / kNumTileBytes) % 2;
        const int count = min(kNumTileBytes, num_bytes - offset);
        if (ptx::elect_one_sync()) {
            ptx::mbarrier_wait_and_flip_phase(mbarrier, phase);
            ptx::tma_store_wait_read();
            if (offset + kNumTileBytes < num_bytes) {
                const int next_count = min(kNumTileBytes, num_bytes - offset - kNumTileBytes);
                ptx::tma_load_1d(smem + (1 - slot) * kNumTileBytes, src + offset + kNumTileBytes, mbarrier, next_count);
                ptx::mbarrier_arrive_and_set_tx(mbarrier, next_count);
            }
        }
        __syncwarp();
        if (ptx::elect_one_sync()) {
            ptx::multimem_cp_async_bulk(dst + offset / sizeof(float4), smem + slot * kNumTileBytes, count);
            ptx::tma_store_commit();
        }
        __syncwarp();
    }
    if (ptx::elect_one_sync())
        ptx::tma_store_wait_read();
    __syncwarp();
}

template <int kNumRDMARanks, int kNumNVLRanks,
          int kNumSMs,
          int kNumMultimemWarpsPerGroup, int kNumIssueWarps, int kNumReduceWarps, int kNumBroadcastWarps,
          int kNumSmemBytesPerReduceWarp, int kNumSmemBytesPerBroadcastWarp,
          int kNumQPs,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx,
          int kNumChunkBytes,
          int kNumMultimemWarps = kNumMultimemWarpsPerGroup * kNumIssueWarps,
          int kNumThreads = (kNumMultimemWarps + kNumIssueWarps + kNumReduceWarps + kNumBroadcastWarps) * 32>
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_all_reduce(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                  void* workspace_ptr, void* buffer, void* buffer_multimem,
                  const float scale,
                  const int num_buckets,
                  const __grid_constant__ comm::BucketList buckets) {
    // Check slots
    constexpr int kNumGlobalIssueWarps = kNumSMs * kNumIssueWarps;
    constexpr int kNumGlobalReduceWarps = kNumSMs * kNumReduceWarps;
    constexpr int kNumSlots = layout::ChunkStorage::kNumBytes / kNumChunkBytes / kNumRDMARanks;
    constexpr int kNumSignalSlots = 2 * kNumRDMARanks * kNumSlots + kNumGlobalReduceWarps * kNumRDMARanks;
    EP_STATIC_ASSERT(kNumRDMARanks > 1 and kNumRDMARanks <= 128 and kNumNVLRanks > 1,
                     "Unsupported hybrid topology");
    EP_STATIC_ASSERT(kNumSMs >= 2 and kNumSlots > 0, "Invalid ring or SM count");
    EP_STATIC_ASSERT(kNumSmemBytesPerReduceWarp > 0 and kNumSmemBytesPerReduceWarp % kNumTMAAlignmentBytes == 0,
                     "Invalid shared memory size per reduce warp");
    EP_STATIC_ASSERT(kNumSignalSlots <= layout::AllReduceSignals::kNumTotalSlots,
                     "Hybrid all-reduce signals exceed the workspace");

    // Warp roles
    constexpr int kMultimemUnroll = 16;
    constexpr int kNumBroadcastTileBytes = kNumSmemBytesPerBroadcastWarp / 2;
    EP_STATIC_ASSERT(kNumSmemBytesPerBroadcastWarp > 0 and
                     kNumSmemBytesPerBroadcastWarp % (2 * kNumTMAAlignmentBytes) == 0,
                     "Invalid shared memory size per broadcast warp");
    constexpr int kNumEmissionLanes = 8;
    constexpr int kNumPartitions = kNumBroadcastWarps / kNumReduceWarps;
    EP_STATIC_ASSERT(kNumMultimemWarps % 4 == 0 and (kNumIssueWarps + kNumReduceWarps + kNumBroadcastWarps) % 4 == 0,
                     "Invalid warp groups");
    EP_STATIC_ASSERT(kNumBroadcastWarps >= kNumReduceWarps and kNumBroadcastWarps % kNumReduceWarps == 0 and
                     kNumPartitions <= kNumRDMARanks and math::constexpr_ceil_div(kNumRDMARanks, kNumPartitions) <= 32,
                     "Each broadcast lane must own at most one AG stream");

    // Registers
    constexpr int kNumMultimemRegisters = 176;
    constexpr int kNumOtherRegisters = 144;
    EP_STATIC_ASSERT(32 * (kNumMultimemWarps * kNumMultimemRegisters +
                          (kNumIssueWarps + kNumReduceWarps + kNumBroadcastWarps) * kNumOtherRegisters) <= 65536 - 2048,
                     "Register redistribution exceeds the CTA pool");

    // Barriers
    constexpr int kMultimemBarrierBaseIdx = 1;
    EP_STATIC_ASSERT(kNumIssueWarps > 0 and kMultimemBarrierBaseIdx + kNumIssueWarps - 1 < 16,
                     "Invalid named barrier count");

    // Indices
    const int sm_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int warp_idx = ptx::get_warp_idx();
    const int lane_idx = ptx::get_lane_idx();
    const int rdma_rank_idx = ncclTeamRail(nccl_dev_comm).rank;
    const int nvl_rank_idx = ncclTeamLsa(nccl_dev_comm).rank;

    // Init Gin
    constexpr int kNumChannels = kNumIssueWarps + kNumReduceWarps;
    const int channel_idx = warp_idx < kNumMultimemWarps ?
        warp_idx / kNumMultimemWarpsPerGroup : (warp_idx - kNumMultimemWarps) % kNumChannels;
    const auto [qp_idx, sharing_mode] =
        comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannels>(sm_idx, channel_idx);
    const comm::NCCLGin gin(nccl_dev_comm, window, qp_idx, sharing_mode);

    // Shared memory and mbarriers
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    __shared__ __align__(8) uint64_t mbarriers[kNumReduceWarps + kNumBroadcastWarps];

    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    auto& signals = workspace.signals[kContextIdx].all_reduce;
    auto& barrier_signals = workspace.signals[kContextIdx].barrier;
    const auto bcst_head = signals.counters;
    const auto rs_tail = bcst_head + kNumRDMARanks * kNumSlots;
    const auto ag_tail = rs_tail + kNumRDMARanks * kNumSlots;
    const auto chunks = layout::ChunkView2D<kNumRDMARanks, kNumSlots, kNumChunkBytes>{workspace.chunk_storage};

    // Every launch starts with fresh signal counters
    EP_STATIC_ASSERT(sizeof(layout::AllReduceSignals) % sizeof(int4) == 0 and
                     alignof(layout::AllReduceSignals) % alignof(int4) == 0,
                     "Signals must be a multiple of int4");
    constexpr int kNumSignalVecs = math::constexpr_ceil_div(kNumSignalSlots * sizeof(int64_t), sizeof(int4));
    for (int i = sm_idx * kNumThreads + thread_idx; i < kNumSignalVecs; i += kNumSMs * kNumThreads)
        reinterpret_cast<int4*>(&signals)[i] = make_int4(0, 0, 0, 0);
    if (thread_idx < kNumReduceWarps + kNumBroadcastWarps)
        ptx::mbarrier_init_with_fence(reinterpret_cast<ptx::mbarrier*>(mbarriers + thread_idx), 1);

    comm::gpu_barrier<true, kNumRDMARanks, kNumNVLRanks,
                      kNumSMs, kNumThreads, kNumQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, rdma_rank_idx, nvl_rank_idx,
        sm_idx, thread_idx);

    // Different warp roles
    if (warp_idx < kNumMultimemWarps) {
        // Intra-node reduction
        ptx::warpgroup_reg_realloc<kNumMultimemRegisters, kNumThreads>();
        const int group_idx = warp_idx / kNumMultimemWarpsPerGroup;
        const int group_thread_idx = (warp_idx % kNumMultimemWarpsPerGroup) * 32 + lane_idx;
        const int global_group_idx = sm_idx * kNumIssueWarps + group_idx;
        auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumNVLRanks * kNumGlobalIssueWarps>(
            num_buckets, buckets, nvl_rank_idx + kNumNVLRanks * global_group_idx);
        while (iterator.next()) {
            const auto src = math::advance_ptr<const float4>(buffer_multimem, iterator.bucket.offset + iterator.chunk_offset_in_bucket);
            const auto dst = math::advance_ptr<float4>(buffer, iterator.bucket.offset + iterator.chunk_offset_in_bucket);
            const int num_vecs = iterator.num_chunk_bytes / static_cast<int>(sizeof(float4));
            const auto load = [](const float4* ptr, const int& lhs = 1, const int& rhs = 0) {
                return ptx::multimem_ld_reduce_add_f32_with_gt_pred(ptr, lhs, rhs);
            };
            const auto store = [](float4* ptr, const float4& value, const int& lhs = 1, const int& rhs = 0) {
                ptx::st_with_gt_pred(ptr, value, lhs, rhs);
            };
            const auto process = [&](const float4& value) { return kWithScale ? ptx::fmul4(value, scale) : value; };
            utils::iterate_ld_st<kNumMultimemWarpsPerGroup, 32, kMultimemUnroll>(
                group_thread_idx, num_vecs, src, dst, load, store, process);
            // Notify the issue warp
            ptx::named_barrier_unaligned<(kNumMultimemWarpsPerGroup + 1) * 32>(kMultimemBarrierBaseIdx + group_idx);
        }
    } else {
        ptx::warpgroup_reg_realloc<kNumOtherRegisters, kNumThreads>();
        if (warp_idx < kNumMultimemWarps + kNumIssueWarps) {
            // RS Issue warp
            const int group_idx = warp_idx - kNumMultimemWarps;
            const int global_group_idx = sm_idx * kNumIssueWarps + group_idx;
            auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumNVLRanks * kNumGlobalIssueWarps>(
                num_buckets, buckets, nvl_rank_idx + kNumNVLRanks * global_group_idx);
            while (iterator.next()) {
                const int rail_seq = iterator.chunk_idx / kNumNVLRanks;
                const int owner = rail_seq % kNumRDMARanks;
                const int owner_seq = rail_seq / kNumRDMARanks;
                const int slot = owner_seq % kNumSlots;
                const int round = owner_seq / kNumSlots;
                if (ptx::elect_one_sync()) {
                    // Wait until the previous generation has been consumed
                    comm::timeout_while<kNumTimeoutCycles>([&](const bool& last) {
                        const auto head = ptx::ld_acquire_sys(bcst_head + owner * kNumSlots + slot);
                        if (last and head < round)
                            printf("Hybrid all_reduce credit timeout r=%d n=%d owner=%d chunk=%d slot=%d round=%d head=%lld\n",
                                   rdma_rank_idx, nvl_rank_idx, owner, iterator.chunk_idx, slot, round,
                                   static_cast<long long>(head));
                        return head >= round;
                    });
                }
                // Wait for the group's intra-node reduction
                ptx::named_barrier_unaligned<(kNumMultimemWarpsPerGroup + 1) * 32>(kMultimemBarrierBaseIdx + group_idx);
                if (ptx::elect_one_sync()) {
                    auto* tail = rs_tail + rdma_rank_idx * kNumSlots + slot;
                    if (owner == rdma_rank_idx) {
                        ptx::red_add_rel_gpu(tail, 1);
                    } else {
                        gin.put<ncclTeamTagRail>(
                            chunks.get_chunk_ptr(rdma_rank_idx, slot), math::advance_ptr<void>(buffer, iterator.bucket.offset + iterator.chunk_offset_in_bucket),
                            iterator.num_chunk_bytes, owner, 0,
                            ncclGin_StrongVASignalAdd{window, gin.get_sym_offset(tail), 1});
                    }
                }
                __syncwarp();
            }
        } else if (warp_idx < kNumMultimemWarps + kNumChannels) {
            // Reduce warps: Accumulate remote chunks and send fully reduced chunks back
            const int reduce_warp_idx = warp_idx - kNumMultimemWarps - kNumIssueWarps;
            const int global_reduce_warp_idx = sm_idx * kNumReduceWarps + reduce_warp_idx;
            const auto mbarrier = reinterpret_cast<ptx::mbarrier*>(mbarriers + reduce_warp_idx);
            const auto shared = smem + reduce_warp_idx * kNumSmemBytesPerReduceWarp;
            ptx::arrival_phase phase = 0;
            auto iterator = layout::ChunkIterator<kNumChunkBytes, kNumNVLRanks * kNumRDMARanks * kNumGlobalReduceWarps>(
                num_buckets, buckets, nvl_rank_idx + kNumNVLRanks * (rdma_rank_idx + kNumRDMARanks * global_reduce_warp_idx));
            while (iterator.next()) {
                const int owner_seq = iterator.chunk_idx / (kNumNVLRanks * kNumRDMARanks);
                const int slot = owner_seq % kNumSlots;
                const int target = owner_seq / kNumSlots + 1;
                const auto input = math::advance_ptr<void>(buffer, iterator.bucket.offset + iterator.chunk_offset_in_bucket);
                const auto out = chunks.get_chunk_ptr(rdma_rank_idx, slot);

                // Wait for all inputs before overwriting this slot's reduced result
                // NOTES: New arrivals also acknowledge consumption of the previous result
                for (int source = lane_idx; source < kNumRDMARanks; source += 32) {
                    comm::timeout_while<kNumTimeoutCycles>([&](const bool& last) {
                        const auto tail = ptx::ld_acquire_sys(rs_tail + source * kNumSlots + slot);
                        if (last and tail < target)
                            printf("Hybrid all_reduce RS timeout r=%d n=%d source=%d chunk=%d slot=%d target=%d tail=%lld\n",
                                   rdma_rank_idx, nvl_rank_idx, source, iterator.chunk_idx, slot, target,
                                   static_cast<long long>(tail));
                        return tail >= target;
                    });
                }
                __syncwarp();
                if (ptx::elect_one_sync()) {
                    #pragma unroll 1
                    for (int step = 0; step < kNumRDMARanks; ++ step) {
                        const int source = (rdma_rank_idx + step) % kNumRDMARanks;
                        const auto src = step == 0 ? input : chunks.get_chunk_ptr(source, slot);
                        #pragma unroll 1
                        for (int offset = 0; offset < iterator.num_chunk_bytes; offset += kNumSmemBytesPerReduceWarp) {
                            const int count = min(kNumSmemBytesPerReduceWarp, iterator.num_chunk_bytes - offset);
                            ptx::tma_store_wait();
                            ptx::tma_load_1d(shared, math::advance_ptr<const void>(src, offset), mbarrier, count);
                            ptx::mbarrier_arrive_and_set_tx(mbarrier, count);
                            ptx::mbarrier_wait_and_flip_phase(mbarrier, phase);
                            if (step == 0)
                                ptx::tma_store_1d(math::advance_ptr<void>(out, offset), shared, count);
                            else
                                ptx::tma_store_reduce_add_f32(math::advance_ptr<float>(out, offset), shared, count);
                            ptx::tma_store_commit();
                        }
                        ptx::tma_store_wait();
                    }
                    ptx::red_add_rel_gpu(ag_tail + global_reduce_warp_idx * kNumRDMARanks + rdma_rank_idx, 1);
                }
                __syncwarp();

                #pragma unroll 1
                for (int d = lane_idx; lane_idx < kNumEmissionLanes and d < kNumRDMARanks - 1; d += kNumEmissionLanes) {
                    const int peer = (rdma_rank_idx + 1 + d) % kNumRDMARanks;
                    gin.put<ncclTeamTagRail>(
                        math::advance_ptr<void>(buffer, iterator.bucket.offset + iterator.chunk_offset_in_bucket),
                        out, iterator.num_chunk_bytes, peer, 0,
                        ncclGin_StrongVASignalAdd{window, gin.get_sym_offset(ag_tail + global_reduce_warp_idx * kNumRDMARanks + rdma_rank_idx), 1});
                }
                __syncwarp();
            }
        } else {
            // Intra-node broadcast
            const int broadcast_warp_idx = warp_idx - kNumMultimemWarps - kNumChannels;
            const int global_reduce_warp_idx = sm_idx * kNumReduceWarps + broadcast_warp_idx % kNumReduceWarps;
            // Each broadcast warp watches a contiguous range of peers
            const int partition_idx = broadcast_warp_idx / kNumReduceWarps;
            const int peer_begin = partition_idx * kNumRDMARanks / kNumPartitions;
            const int peer_end = (partition_idx + 1) * kNumRDMARanks / kNumPartitions;
            const int peer = peer_begin + lane_idx;
            constexpr int kStride = kNumNVLRanks * kNumRDMARanks * kNumGlobalReduceWarps;
            auto iterator = layout::ChunkIterator<kNumChunkBytes, kStride>(
                num_buckets, buckets, nvl_rank_idx + kNumNVLRanks * (peer + kNumRDMARanks * global_reduce_warp_idx));
            iterator.next();
            const int length = peer < peer_end and iterator.chunk_idx < iterator.num_total_chunks
                ? (iterator.num_total_chunks - 1 - iterator.chunk_idx) / kStride + 1 : 0;
            const auto mbarrier = reinterpret_cast<ptx::mbarrier*>(mbarriers + kNumReduceWarps + broadcast_warp_idx);
            const auto shared = smem + kNumReduceWarps * kNumSmemBytesPerReduceWarp + broadcast_warp_idx * 2 * kNumBroadcastTileBytes;
            ptx::arrival_phase phase = 0;
            int delivered = 0;

            while (ptx::any(delivered < length)) {
                unsigned ready = 0;
                comm::timeout_while<kNumTimeoutCycles>([&](const bool& last) {
                    const auto arrived = delivered < length ? ptx::ld_acquire_sys(ag_tail + global_reduce_warp_idx * kNumRDMARanks + peer) : delivered;
                    ready = ptx::gather(arrived > delivered);
                    if (last and delivered < length and arrived <= delivered)
                        printf("Hybrid all_reduce AG timeout r=%d n=%d owner=%d j=%d delivered=%d length=%d tail=%lld\n",
                               rdma_rank_idx, nvl_rank_idx, peer, global_reduce_warp_idx, delivered, length,
                               static_cast<long long>(arrived));
                    return ready != 0;
                });
                __syncwarp();
                while (ready) {
                    const int selected_lane_idx = ptx::ffs(ready);
                    const int owner = peer_begin + selected_lane_idx;
                    int chunk_idx = 0;
                    int num_bytes = 0;
                    int64_t offset = 0;
                    if (lane_idx == selected_lane_idx) {
                        chunk_idx = iterator.chunk_idx;
                        num_bytes = iterator.num_chunk_bytes;
                        offset = iterator.bucket.offset + iterator.chunk_offset_in_bucket;
                        iterator.next();
                        ++ delivered;
                    }
                    chunk_idx = ptx::exchange(chunk_idx, selected_lane_idx);
                    num_bytes = ptx::exchange(num_bytes, selected_lane_idx);
                    offset = ptx::exchange(offset, selected_lane_idx);
                    const int slot = chunk_idx / (kNumNVLRanks * kNumRDMARanks) % kNumSlots;
                    const auto src = owner == rdma_rank_idx
                        ? static_cast<const int8_t*>(chunks.get_chunk_ptr(rdma_rank_idx, slot))
                        : math::advance_ptr<const int8_t>(buffer, offset);
                    hybrid_ar_broadcast<kNumBroadcastTileBytes>(
                        src, math::advance_ptr<float4>(buffer_multimem, offset), num_bytes, shared, mbarrier, phase);
                    if (ptx::elect_one_sync())
                        ptx::red_add_rel_gpu(bcst_head + owner * kNumSlots + slot, 1);
                    __syncwarp();
                    ready &= ready - 1;
                }
            }
        }
    }

    comm::gpu_barrier<true, kNumRDMARanks, kNumNVLRanks, kNumSMs, kNumThreads, kNumQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, rdma_rank_idx, nvl_rank_idx, sm_idx, thread_idx);
}

}  // namespace deep_ep::bucket

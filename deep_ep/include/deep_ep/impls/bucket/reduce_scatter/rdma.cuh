#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/utils.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

namespace deep_ep::bucket {


template <int kNumRanks, int kNumSMs,
          int kNumIssueWarps, int kNumReduceWarps,
          int kNumQPs,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx,
          int kNumIssueChunksPerBatch = 8,
          int kNumReduceChunksPerBatch = 8,
          int kNumWarps = kNumIssueWarps + kNumReduceWarps,
          int kNumThreads = kNumWarps * 32,
          int kNumGlobalIssueWarps = kNumSMs * kNumIssueWarps,
          int kNumGlobalReduceWarps = kNumSMs * kNumReduceWarps,
          int kNumGlobalThreads = kNumSMs * kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
rdma_reduce_scatter(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                    void* workspace_ptr, void* buffer,
                    const float scale,
                    const int num_buckets,
                    const __grid_constant__ comm::BucketList buckets) {
    // Check slots
    constexpr int kNumBytesPerChunk = 32 * 1024;
    constexpr int kNumElemsPerChunk = kNumBytesPerChunk / sizeof(float);
    constexpr int kNumSlotsPerRank = layout::ChunkStorage::kNumBytes / kNumBytesPerChunk / kNumRanks;

    EP_STATIC_ASSERT(kNumRanks * kNumSlotsPerRank <= layout::ReduceScatterSignals::kNumTotalSlots,
                     "Reduce-scatter slots exceed the signal workspace");
    EP_STATIC_ASSERT(kNumRanks <= kNumMaxRDMARanks, "Too many RDMA ranks");
    EP_STATIC_ASSERT(kNumIssueChunksPerBatch > 0 and kNumIssueChunksPerBatch <= 32,
                     "Issue chunk batch size must be between 1 and 32");
    EP_STATIC_ASSERT(kNumReduceChunksPerBatch > 0 and kNumReduceChunksPerBatch <= 32,
                     "Reduce chunk batch size must be between 1 and 32");

    // Registers
    constexpr int kNumIssueRegisters = 112;
    constexpr int kNumReduceRegisters = 144;
    constexpr int kNumInitialRegisters = (65536 / kNumThreads / 8) * 8;
    EP_STATIC_ASSERT(kNumIssueWarps % 4 == 0 and kNumReduceWarps % 4 == 0,
                     "Register roles must align to warp groups");
    EP_STATIC_ASSERT(kNumIssueWarps * kNumIssueRegisters + kNumReduceWarps * kNumReduceRegisters <=
                     kNumWarps * kNumInitialRegisters,
                     "Too many registers");

    // Indices
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto lane_idx = ptx::get_lane_idx();
    const auto warp_idx = ptx::get_warp_idx();
    const auto rank_idx = ncclTeamRail(nccl_dev_comm).rank;

    // Init Gin
    // TODO: find a better and more explanable way
    constexpr int kNumSendQPsPerWarp = 4;
    const int qp_idx = [&] {
        if (warp_idx < kNumIssueWarps) {
            // Send issue lanes stripe across the QPs assigned to their issue warp.
            return ((sm_idx * kNumIssueWarps + warp_idx) * kNumSendQPsPerWarp +
                    lane_idx % kNumSendQPsPerWarp) % kNumQPs;
        }

        // Place credit QPs after send QPs, wrapping when the pool is shared.
        return (kNumGlobalIssueWarps * kNumSendQPsPerWarp + sm_idx * kNumReduceWarps + warp_idx - kNumIssueWarps) % kNumQPs;
    }();
    const comm::NCCLGin gin(nccl_dev_comm, window, qp_idx);

    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    auto& bucket_signals = workspace.signals[kContextIdx];
    auto& barrier_signals = bucket_signals.barrier;
    auto& signals = bucket_signals.reduce_scatter;
    const auto chunks = layout::ChunkView2D<kNumRanks, kNumSlotsPerRank, kNumBytesPerChunk>{workspace.chunk_storage};

    // Every launch starts a fresh receive-credit ring.
    const auto global_thread_idx = sm_idx * kNumThreads + thread_idx;
    #pragma unroll
    for (int i = global_thread_idx; i < kNumRanks * kNumSlotsPerRank; i += kNumGlobalThreads) {
        const auto j = i / kNumSlotsPerRank, k = i % kNumSlotsPerRank;
        *signals.get_head_ptr<kNumSlotsPerRank>(j, k) = 0;
        *signals.get_tail_ptr<kNumSlotsPerRank>(j, k) = 0;
    }
    comm::gpu_barrier<true, kNumRanks, 1,
                   kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs,
                   kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);

    // Different warp roles
    if (warp_idx < kNumIssueWarps) {
        ptx::warpgroup_reg_realloc<kNumIssueRegisters, kNumThreads>();

        // Issuing by a *result* view, using multiple lanes to issue at the same time
        auto iterator = layout::ChunkIterator<kNumBytesPerChunk, kNumGlobalIssueWarps>(num_buckets, buckets, sm_idx * kNumIssueWarps + warp_idx);
        while (iterator.next()) {
            const auto slot_idx = iterator.chunk_idx % kNumSlotsPerRank;
            const auto head_target = iterator.chunk_idx / kNumSlotsPerRank;
            const auto tail_ptr = signals.get_tail_ptr<kNumSlotsPerRank>(rank_idx, slot_idx);

            #pragma unroll 1
            for (int peer_offset = 1; peer_offset < kNumRanks; peer_offset += kNumIssueChunksPerBatch) {
                const int num_batch_chunks = min(kNumIssueChunksPerBatch, kNumRanks - peer_offset);
                const auto dst_rank_idx = (rank_idx + peer_offset + lane_idx) % kNumRanks;
                const auto head_ptr = signals.get_head_ptr<kNumSlotsPerRank>(dst_rank_idx, slot_idx);
                comm::timeout_while<kNumTimeoutCycles>(lane_idx < num_batch_chunks, [=](const bool&) {
                    // NOTES: no need to do acquire
                    return ptx::ld_volatile<int64_t>(head_ptr) >= head_target;
                });
                __syncwarp();

                // Issue RDMA
                if (lane_idx < num_batch_chunks) {
                    const auto src_ptr = math::advance_ptr(buffer, iterator.bucket.offset + dst_rank_idx * iterator.bucket.num_bytes + iterator.chunk_offset_in_bucket);
                    gin.put<ncclTeamTagRail>(
                        chunks.get_chunk_ptr(rank_idx, slot_idx),
                        src_ptr, iterator.num_chunk_bytes, dst_rank_idx, 0,
                        ncclGin_StrongVASignalAdd {
                            .signalWindow = window,
                            .signalOffset = gin.get_sym_offset(tail_ptr),
                            .value = 1
                        });
                }
                __syncwarp();
            }
        }
    } else {
        ptx::warpgroup_reg_realloc<kNumReduceRegisters, kNumThreads>();

        auto iterator = layout::ChunkIterator<kNumBytesPerChunk, kNumGlobalReduceWarps>(num_buckets, buckets, sm_idx * kNumReduceWarps + warp_idx - kNumIssueWarps);
        while (iterator.next()) {
            const auto slot_idx = iterator.chunk_idx % kNumSlotsPerRank;
            const auto tail_target = iterator.chunk_idx / kNumSlotsPerRank + 1;
            const bool needs_credit = static_cast<int64_t>(iterator.chunk_idx) + kNumSlotsPerRank < iterator.num_total_chunks;

            // Reduce
            EP_STATIC_ASSERT(kNumElemsPerChunk % 256 == 0, "Unaligned chunk");
            constexpr int kNumVecsPerIter = 8;
            EP_STATIC_ASSERT(kNumElemsPerChunk % (32 * kNumVecsPerIter * 4) == 0, "Unaligned reduction iteration");
            const int num_vecs = iterator.num_chunk_bytes / static_cast<int>(sizeof(float4));
            const bool full_chunk = iterator.num_chunk_bytes == kNumBytesPerChunk;
            const auto dst_ptr = math::advance_ptr<float4>(buffer, iterator.bucket.offset + rank_idx * iterator.bucket.num_bytes + iterator.chunk_offset_in_bucket);
            #pragma unroll 1
            for (int peer_offset = 1; peer_offset < kNumRanks; peer_offset += kNumReduceChunksPerBatch) {
                const int num_batch_chunks = min(kNumReduceChunksPerBatch, kNumRanks - peer_offset);
                const bool is_last_batch = peer_offset + num_batch_chunks == kNumRanks;

                // Wait arrival in reverse cyclic order
                const auto wait_rank_idx = (rank_idx - peer_offset - (lane_idx % kNumRanks) + 2 * kNumRanks) % kNumRanks;
                const auto tail_ptr = signals.get_tail_ptr<kNumSlotsPerRank>(wait_rank_idx, slot_idx);
                comm::timeout_while<kNumTimeoutCycles>(lane_idx < num_batch_chunks, [=](const bool&) {
                    return ptx::ld_acquire_sys(tail_ptr) >= tail_target;
                });
                __syncwarp();

                #pragma unroll 1
                for (int i = 0; i < kNumElemsPerChunk / 4 / (32 * kNumVecsPerIter); ++ i) {
                    const auto vec_idx = i * 32 * kNumVecsPerIter + lane_idx;
                    float4 values[kNumVecsPerIter];
                    if (full_chunk) {
                        #pragma unroll
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            values[k] = ptx::ld_with_gt_pred(dst_ptr + vec_idx + k * 32);
                    } else {
                        #pragma unroll
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            values[k] = ptx::ld_with_gt_pred(dst_ptr + vec_idx + k * 32, num_vecs, vec_idx + k * 32);
                    }

                    #pragma unroll 1
                    for (int j = 0; j < num_batch_chunks; ++ j) {
                        const auto src_rank_idx = (rank_idx - peer_offset - j + 2 * kNumRanks) % kNumRanks;
                        const auto src_ptr = static_cast<const float4*>(
                            chunks.get_chunk_ptr(src_rank_idx, slot_idx)) + vec_idx;
                        float4 packed_values[kNumVecsPerIter];
                        if (full_chunk) {
                            #pragma unroll
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                packed_values[k] = ptx::ld_with_gt_pred(src_ptr + k * 32);
                        } else {
                            #pragma unroll
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                packed_values[k] = ptx::ld_with_gt_pred(src_ptr + k * 32, num_vecs, vec_idx + k * 32);
                        }

                        #pragma unroll
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            values[k] = ptx::fadd4(values[k], packed_values[k]);
                    }

                    // Scale
                    if constexpr (kWithScale) {
                        if (is_last_batch) {
                            #pragma unroll
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                values[k] = ptx::fmul4(values[k], scale);
                        }
                    }

                    // Store
                    if (full_chunk) {
                        #pragma unroll
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            ptx::st_with_gt_pred(dst_ptr + vec_idx + k * 32, values[k]);
                    } else {
                        #pragma unroll
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            ptx::st_with_gt_pred(dst_ptr + vec_idx + k * 32, values[k], num_vecs, vec_idx + k * 32);
                    }
                }
                __syncwarp();

                // Release slots of this batch; the final use of a slot needs no credit
                if (needs_credit) {
                    const auto head_ptr = signals.get_head_ptr<kNumSlotsPerRank>(rank_idx, slot_idx);
                    const auto release_rank_idx = (rank_idx - peer_offset - (lane_idx % kNumRanks) + 2 * kNumRanks) % kNumRanks;
                    if (lane_idx < num_batch_chunks)
                        gin.put_value<ncclTeamTagRail>(head_ptr, static_cast<int64_t>(tail_target), release_rank_idx);
                }
                __syncwarp();
            }
        }
    }

    // Ensure no later writes
    comm::gpu_barrier<true, kNumRanks, 1,
                      kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);
}

template <int kNumRanks, int kNumSMs,
          int kNumCastWarps, int kNumIssueWarps, int kNumReduceWarps,
          int kNumQPs,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx,
          int kNumIssueChunksPerBatch = 8,
          int kNumReduceChunksPerBatch = 8,
          int kNumCastWarpsPerGroup = 4,
          int kNumReduceWarpsPerGroup = 4,
          int kNumWarps = kNumCastWarps + kNumIssueWarps + kNumReduceWarps,
          int kNumThreads = kNumWarps * 32,
          int kNumCastGroups = kNumCastWarps / kNumCastWarpsPerGroup,
          int kNumReduceGroups = kNumReduceWarps / kNumReduceWarpsPerGroup,
          int kNumCastThreadsPerGroup = kNumCastWarpsPerGroup * 32,
          int kNumReduceThreadsPerGroup = kNumReduceWarpsPerGroup * 32,
          int kNumGlobalThreads = kNumSMs * kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
rdma_reduce_scatter_bf16(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                    void* workspace_ptr, void* buffer,
                    const float scale,
                    const int num_buckets,
                    const __grid_constant__ comm::BucketList buckets) {
    // Check slots
    constexpr int kNumCommBytesPerChunk = (kNumRanks <= 8 ? 64 : 32) * 1024;
    constexpr int kNumBytesPerChunk = 2 * kNumCommBytesPerChunk;
    constexpr int kNumElemsPerChunk = kNumBytesPerChunk / sizeof(float);
    constexpr int kNumSlotsPerRank = layout::ChunkStorage::kNumBytes / kNumCommBytesPerChunk / kNumRanks;

    EP_STATIC_ASSERT(kNumRanks * kNumSlotsPerRank <= layout::ReduceScatterSignals::kNumTotalSlots,
                     "Reduce-scatter slots exceed the signal workspace");
    EP_STATIC_ASSERT(kNumRanks <= kNumMaxRDMARanks, "Too many RDMA ranks");
    EP_STATIC_ASSERT(kNumIssueChunksPerBatch % kNumCastWarpsPerGroup == 0,
                     "Issue chunk batch size must be evenly divided across cast warps in a group");
    EP_STATIC_ASSERT(kNumIssueChunksPerBatch > 0 and kNumIssueChunksPerBatch <= 32,
                     "Issue chunk batch size must be between 1 and 32");
    EP_STATIC_ASSERT(kNumReduceChunksPerBatch > 0 and kNumReduceChunksPerBatch <= 32,
                     "Reduce chunk batch size must be between 1 and 32");
    EP_STATIC_ASSERT(kNumCastWarpsPerGroup > 0 and kNumCastWarps % kNumCastWarpsPerGroup == 0,
                     "Cast warps must be evenly divided across groups");
    EP_STATIC_ASSERT(kNumReduceWarpsPerGroup > 0 and kNumReduceWarps % kNumReduceWarpsPerGroup == 0,
                     "Reduce warps must be evenly divided across groups");
    EP_STATIC_ASSERT(kNumIssueWarps == kNumCastGroups + kNumReduceGroups,
                     "Each cast and reduce group requires one issue warp");
    constexpr int kNumGlobalCastGroups = (kNumSMs * kNumCastWarps) / kNumCastWarpsPerGroup;
    constexpr int kNumGlobalReduceGroups = (kNumSMs * kNumReduceWarps) / kNumReduceWarpsPerGroup;

    // Advancing to the next chunk must not depend on the credit delayed until its barrier.
    // This also guarantees that each group's final chunk needs no returned credits.
    EP_STATIC_ASSERT(kNumGlobalReduceGroups < kNumSlotsPerRank, "Pipelined reduce credits require more slots than global reduce groups");

    // Registers
    constexpr int kNumCastRegisters = 80;
    constexpr int kNumIssueRegisters = 80;
    constexpr int kNumReduceRegisters = 120;
    constexpr int kNumInitialRegisters = (65536 / kNumThreads / 8) * 8;
    EP_STATIC_ASSERT(kNumCastWarps % 4 == 0 and kNumIssueWarps % 4 == 0 and kNumReduceWarps % 4 == 0,
                     "Register roles must align to warp groups");
    EP_STATIC_ASSERT(kNumCastWarps * kNumCastRegisters + kNumIssueWarps * kNumIssueRegisters +
                     kNumReduceWarps * kNumReduceRegisters <= kNumWarps * kNumInitialRegisters,
                     "Too many registers");

    // Barriers
    constexpr int kCastBarrierBaseIdx = 1;
    constexpr int kReduceBarrierBaseIdx = kCastBarrierBaseIdx + kNumCastGroups;
    EP_STATIC_ASSERT(kCastBarrierBaseIdx + kNumCastGroups + kNumReduceGroups - 1 < 16, "Too many cast/reduce groups for named barriers");

    // Indices
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto lane_idx = ptx::get_lane_idx();
    const auto warp_idx = ptx::get_warp_idx();
    const auto global_warp_idx = sm_idx * kNumWarps + warp_idx;
    const auto rank_idx = ncclTeamRail(nccl_dev_comm).rank;

    // Init Gin
    // TODO: find a better and more explanable way
    constexpr int kNumSendQPsPerWarp = 4;
    const int qp_idx = [&] {
        if (warp_idx >= kNumCastWarps and warp_idx < kNumCastWarps + kNumCastGroups) {
            // Send issue lanes stripe across the QPs assigned to their cast group.
            const int cast_group_idx = warp_idx - kNumCastWarps;
            return ((sm_idx * kNumCastGroups + cast_group_idx) * kNumSendQPsPerWarp +
                    lane_idx % kNumSendQPsPerWarp) % kNumQPs;
        }

        if (warp_idx >= kNumCastWarps + kNumCastGroups and warp_idx < kNumCastWarps + kNumIssueWarps) {
            // Place credit QPs after send QPs, wrapping when the pool is shared.
            const int reduce_group_idx = warp_idx - kNumCastWarps - kNumCastGroups;
            return (kNumGlobalCastGroups * kNumSendQPsPerWarp + sm_idx * kNumReduceGroups + reduce_group_idx) % kNumQPs;
        }

        return 0;
    }();
    const comm::NCCLGin gin(nccl_dev_comm, window, qp_idx);

    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    auto& bucket_signals = workspace.signals[kContextIdx];
    auto& barrier_signals = bucket_signals.barrier;
    auto& signals = bucket_signals.reduce_scatter;
    const auto chunks = layout::ChunkView2D<kNumRanks, kNumSlotsPerRank, kNumCommBytesPerChunk>{workspace.chunk_storage};

    // Every launch starts a fresh receive-credit ring.
    #pragma unroll
    for (int i = sm_idx * kNumThreads + thread_idx; i < kNumRanks * kNumSlotsPerRank; i += kNumGlobalThreads) {
        const auto j = i / kNumSlotsPerRank, k = i % kNumSlotsPerRank;
        *signals.get_head_ptr<kNumSlotsPerRank>(j, k) = 0;
        *signals.get_tail_ptr<kNumSlotsPerRank>(j, k) = 0;
    }
    comm::gpu_barrier<true, kNumRanks, 1,
                   kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs,
                   kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);

    // Different warp roles
    if (warp_idx < kNumCastWarps) {
        ptx::warpgroup_reg_realloc<kNumCastRegisters, kNumThreads>();

        // A group owns a chunk offset; its cast warps keep disjoint destination batches.
        const int cast_group_idx = warp_idx / kNumCastWarpsPerGroup;
        const int cast_warp_idx_in_group = warp_idx % kNumCastWarpsPerGroup;
        constexpr int kNumCastChunksPerBatchPerWarp = kNumIssueChunksPerBatch / kNumCastWarpsPerGroup;
        auto iterator = layout::ChunkIterator<kNumBytesPerChunk, kNumGlobalCastGroups>(num_buckets, buckets, sm_idx * kNumCastGroups + cast_group_idx);
        while (iterator.next()) {
            const auto chunk_offset_bytes = iterator.chunk_offset_in_bucket;
            const int num_chunk_bytes = iterator.num_chunk_bytes;
            const bool full_chunk = num_chunk_bytes == kNumBytesPerChunk;
            const auto num_bucket_bytes = iterator.bucket.num_bytes;

            #pragma unroll 1
            for (int peer_offset = 1; peer_offset < kNumRanks; peer_offset += kNumIssueChunksPerBatch) {
                const int i = peer_offset + cast_warp_idx_in_group * kNumCastChunksPerBatchPerWarp;
                const int num_batch_chunks = max(0, min(kNumCastChunksPerBatchPerWarp, min(kNumRanks, peer_offset + kNumIssueChunksPerBatch) - i));

                // Convert this warp's destination chunks before handing the group batch to its issue warp.
                #pragma unroll 1
                for (int batch_idx = 0; batch_idx < num_batch_chunks; ++ batch_idx) {
                    const auto dst_rank_idx = (rank_idx + i + batch_idx) % kNumRanks;

                    // Pack this destination's FP32 chunk into BF16 in its first half.
                    constexpr int kNumVecsPerIter = 8;
                    EP_STATIC_ASSERT(kNumVecsPerIter > 0 and kNumElemsPerChunk % (256 * kNumVecsPerIter) == 0,
                                     "Unaligned send iteration");
                    const int num_vecs = num_chunk_bytes / static_cast<int>(sizeof(longlong4_t));
                    const auto chunk_ptr = math::advance_ptr<longlong4_t>(buffer, iterator.bucket.offset + dst_rank_idx * num_bucket_bytes + chunk_offset_bytes);
                    const uint32_t global_vec_idx_base = static_cast<uint32_t>((dst_rank_idx * num_bucket_bytes + chunk_offset_bytes) / static_cast<int64_t>(sizeof(longlong4_t)));
                    #pragma unroll 1
                    for (int j = 0; j < kNumElemsPerChunk / 256; j += kNumVecsPerIter) {
                        const auto vec_idx = j * 32 + lane_idx;
                        const uint32_t global_vec_idx = global_vec_idx_base + static_cast<uint32_t>(vec_idx);
                        const int num_remaining_vecs = num_vecs - vec_idx;
                        longlong4_t values[kNumVecsPerIter];
                        if (full_chunk) {
                            #pragma unroll
                            for (int k = 0; k < kNumVecsPerIter; ++k)
                                values[k] = ptx::ld_with_gt_pred(chunk_ptr + vec_idx + k * 32);
                        } else {
                            #pragma unroll
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                values[k] = ptx::ld_with_gt_pred(chunk_ptr + vec_idx + k * 32, num_remaining_vecs, k * 32);
                        }

                        float4 packed_values[kNumVecsPerIter];
                        #pragma unroll
                        for (int k = 0; k < kNumVecsPerIter; ++ k) {
                            #pragma unroll
                            for (int half_idx = 0; half_idx < 2; ++ half_idx) {
                                const auto& value = reinterpret_cast<const float4*>(&values[k])[half_idx];
                                // Derive rounding bits from the FP32 bit patterns and bucket-relative float4 vector index.
                                const auto& bits = reinterpret_cast<const uint4&>(value);
                                uint32_t h = bits.x * 0x5671D42Bu + bits.y * 0x9995E499u +
                                             bits.z * 0xACE1B8A5u + bits.w * 0xE153538Du +
                                             global_vec_idx * 2 + static_cast<uint32_t>(k * 2 * 32) + static_cast<uint32_t>(half_idx);
                                h ^= h >> 23;
                                h *= 0x7FEB352Du;
                                h ^= h >> 16;
                                uint32_t h1 = h * 0x846CA68Bu;
                                h1 ^= h1 >> 11;
                                uint32_t h2 = h * 0xD35A2D97u;
                                h2 ^= h2 >> 11;

                                // Use separate rounding bits for the two FP32 pairs.
                                const auto xy = ptx::cvt_rs_bf16x2(make_float2(value.x, value.y), h1);
                                const auto zw = ptx::cvt_rs_bf16x2(make_float2(value.z, value.w), h2);
                                auto& packed = reinterpret_cast<uint2*>(&packed_values[k])[half_idx];
                                packed.x = reinterpret_cast<const uint32_t&>(xy);
                                packed.y = reinterpret_cast<const uint32_t&>(zw);
                            }
                        }

                        // Every lane must finish reading before any lane overwrites FP32 data.
                        __syncwarp();

                        if (full_chunk) {
                            #pragma unroll
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                ptx::st_with_gt_pred(reinterpret_cast<float4*>(chunk_ptr) + vec_idx + k * 32, packed_values[k]);
                        } else {
                            #pragma unroll
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                ptx::st_with_gt_pred(reinterpret_cast<float4*>(chunk_ptr) + vec_idx + k * 32, packed_values[k], num_remaining_vecs, k * 32);
                        }
                    }
                    __syncwarp();
                }

                ptx::named_barrier_unaligned<kNumCastThreadsPerGroup + 32>(kCastBarrierBaseIdx + cast_group_idx);
            }
        }
    } else if (warp_idx < kNumCastWarps + kNumIssueWarps) {
        ptx::warpgroup_reg_realloc<kNumIssueRegisters, kNumThreads>();

        const int issue_warp_idx = warp_idx - kNumCastWarps;
        if (issue_warp_idx < kNumCastGroups) {
            // One send issue warp services all destination batches from its cast group.
            const int cast_group_idx = issue_warp_idx;

            auto iterator = layout::ChunkIterator<kNumBytesPerChunk, kNumGlobalCastGroups>(num_buckets, buckets, sm_idx * kNumCastGroups + cast_group_idx);
            while (iterator.next()) {
                const auto chunk_offset_bytes = iterator.chunk_offset_in_bucket;
                const int num_comm_chunk_bytes = iterator.num_chunk_bytes / 2;
                const auto num_bucket_bytes = iterator.bucket.num_bytes;
                const auto global_chunk_idx = iterator.chunk_idx;
                const auto slot_idx = global_chunk_idx % kNumSlotsPerRank;
                const auto head_target = global_chunk_idx / kNumSlotsPerRank;
                const auto tail_ptr = signals.get_tail_ptr<kNumSlotsPerRank>(rank_idx, slot_idx);

                #pragma unroll 1
                for (int peer_offset = 1; peer_offset < kNumRanks; peer_offset += kNumIssueChunksPerBatch) {
                    ptx::named_barrier_unaligned<kNumCastThreadsPerGroup + 32>(kCastBarrierBaseIdx + cast_group_idx);

                    // All peers in this cast group batch use distinct lanes of the issue warp.
                    const int num_batch_chunks = min(kNumIssueChunksPerBatch, kNumRanks - peer_offset);
                    const auto dst_rank_idx = (rank_idx + peer_offset + lane_idx) % kNumRanks;
                    const auto head_ptr = signals.get_head_ptr<kNumSlotsPerRank>(dst_rank_idx, slot_idx);
                    comm::timeout_while<kNumTimeoutCycles>(lane_idx < num_batch_chunks, [=](const bool&) {
                        // NOTES: no need to do acquire
                        return ptx::ld_volatile<int64_t>(head_ptr) >= head_target;
                    });
                    __syncwarp();

                    if (lane_idx < num_batch_chunks) {
                        const auto src_ptr = math::advance_ptr(buffer, iterator.bucket.offset + dst_rank_idx * num_bucket_bytes + chunk_offset_bytes);
                        gin.put<ncclTeamTagRail>(
                            chunks.get_chunk_ptr(rank_idx, slot_idx),
                            src_ptr, num_comm_chunk_bytes, dst_rank_idx, 0,
                            ncclGin_StrongVASignalAdd {
                                .signalWindow = window,
                                .signalOffset = gin.get_sym_offset(tail_ptr),
                                .value = 1
                            });
                    }
                    __syncwarp();
                }
            }
        } else {
            // One receive issue warp waits and returns credits for its reduce group.
            const int reduce_group_idx = issue_warp_idx - kNumCastGroups;
            auto iterator = layout::ChunkIterator<kNumBytesPerChunk, kNumGlobalReduceGroups>(num_buckets, buckets, sm_idx * kNumReduceGroups + reduce_group_idx);
            bool has_previous_credit = false;
            int previous_src_rank_idx = 0;
            int64_t previous_tail_target = 0;
            int64_t* previous_head_ptr = nullptr;
            while (iterator.next()) {
                const auto global_chunk_idx = iterator.chunk_idx;
                const auto slot_idx = global_chunk_idx % kNumSlotsPerRank;
                const auto tail_target = global_chunk_idx / kNumSlotsPerRank + 1;
                const bool needs_credit = static_cast<int64_t>(global_chunk_idx) + kNumSlotsPerRank < iterator.num_total_chunks;
                const auto head_ptr = signals.get_head_ptr<kNumSlotsPerRank>(rank_idx, slot_idx);

                #pragma unroll 1
                for (int peer_offset = 1; peer_offset < kNumRanks; peer_offset += kNumReduceChunksPerBatch) {
                    const int num_batch_chunks = min(kNumReduceChunksPerBatch, kNumRanks - peer_offset);
                    const auto src_rank_idx = (rank_idx - peer_offset - (lane_idx % kNumRanks) + 2 * kNumRanks) % kNumRanks;
                    const auto tail_ptr = signals.get_tail_ptr<kNumSlotsPerRank>(src_rank_idx, slot_idx);
                    comm::timeout_while<kNumTimeoutCycles>(lane_idx < num_batch_chunks, [=](const bool&) {
                        return ptx::ld_acquire_sys(tail_ptr) >= tail_target;
                    });
                    __syncwarp();

                    // Publish this batch's arrival and confirm the previous batch is consumed.
                    ptx::named_barrier_unaligned<kNumReduceThreadsPerGroup + 32>(kReduceBarrierBaseIdx + reduce_group_idx);

                    if (has_previous_credit)
                        gin.put_value<ncclTeamTagRail>(previous_head_ptr, previous_tail_target, previous_src_rank_idx);
                    __syncwarp();

                    // Keep the previous batch's slot and generation even across chunk/bucket boundaries.
                    has_previous_credit = needs_credit and lane_idx < num_batch_chunks;
                    previous_head_ptr = head_ptr;
                    previous_tail_target = static_cast<int64_t>(tail_target);
                    previous_src_rank_idx = src_rank_idx;
                }
            }
            // The last owned chunk cannot reuse its slot, so no final credit or group barrier is needed.
        }
    } else {
        ptx::warpgroup_reg_realloc<kNumReduceRegisters, kNumThreads>();

        const int reduce_warp_idx = warp_idx - kNumCastWarps - kNumIssueWarps;
        const int reduce_warp_idx_in_group = reduce_warp_idx % kNumReduceWarpsPerGroup;
        const int reduce_warpgroup_idx = reduce_warp_idx / kNumReduceWarpsPerGroup;
        auto iterator = layout::ChunkIterator<kNumBytesPerChunk, kNumGlobalReduceGroups>(num_buckets, buckets, sm_idx * kNumReduceGroups + reduce_warpgroup_idx);
        while (iterator.next()) {
            const auto chunk_offset_bytes = iterator.chunk_offset_in_bucket;
            const auto slot_idx = iterator.chunk_idx % kNumSlotsPerRank;
            const bool full_chunk = iterator.num_chunk_bytes == kNumBytesPerChunk;

            // Reduce
            constexpr int kNumVecsPerIter = 8;
            constexpr int kNumThreadsPerReduceGroup = kNumReduceWarpsPerGroup * 32;
            EP_STATIC_ASSERT(kNumVecsPerIter > 0 and kNumElemsPerChunk % (256 * kNumVecsPerIter * kNumReduceWarpsPerGroup) == 0,
                             "Unaligned reduction iteration");
            const int num_vecs = iterator.num_chunk_bytes / static_cast<int>(sizeof(longlong4_t));
            const auto num_bucket_bytes = iterator.bucket.num_bytes;
            const auto dst_ptr = math::advance_ptr<longlong4_t>(buffer, iterator.bucket.offset + rank_idx * num_bucket_bytes + chunk_offset_bytes);

            // Reduce remote peers in reverse cyclic order, storing partial FP32 sums between batches.
            #pragma unroll 1
            for (int peer_offset = 1; peer_offset < kNumRanks; peer_offset += kNumReduceChunksPerBatch) {
                const int num_batch_chunks = min(kNumReduceChunksPerBatch, kNumRanks - peer_offset);
                const bool is_last_batch = peer_offset + num_batch_chunks == kNumRanks;

                ptx::named_barrier_unaligned<kNumReduceThreadsPerGroup + 32>(kReduceBarrierBaseIdx + reduce_warpgroup_idx);

                #pragma unroll 1
                for (int i = 0; i < kNumElemsPerChunk / 8 / kNumThreadsPerReduceGroup; i += kNumVecsPerIter) {
                    const auto vec_idx = i * kNumThreadsPerReduceGroup + reduce_warp_idx_in_group * 32 + lane_idx;
                    const int num_remaining_vecs = num_vecs - vec_idx;
                    // The first batch loads the local contribution; later batches load the partial sum.
                    longlong4_t values[kNumVecsPerIter];
                    if (full_chunk) {
                        #pragma unroll
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            values[k] = ptx::ld_with_gt_pred(dst_ptr + vec_idx + k * kNumThreadsPerReduceGroup);
                    } else {
                        #pragma unroll
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            values[k] = ptx::ld_with_gt_pred(dst_ptr + vec_idx + k * kNumThreadsPerReduceGroup, num_remaining_vecs, k * kNumThreadsPerReduceGroup);
                    }

                    #pragma unroll 1
                    for (int j = 0; j < num_batch_chunks; ++ j) {
                        const auto src_rank_idx = (rank_idx - peer_offset - j + kNumRanks) % kNumRanks;
                        const auto src_ptr = static_cast<const float4*>(
                            chunks.get_chunk_ptr(src_rank_idx, slot_idx)) + vec_idx;
                        float4 packed_values[kNumVecsPerIter];
                        if (full_chunk) {
                            #pragma unroll
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                packed_values[k] = ptx::ld_with_gt_pred(src_ptr + k * kNumThreadsPerReduceGroup);
                        } else {
                            #pragma unroll
                            for (int k = 0; k < kNumVecsPerIter; ++ k)
                                packed_values[k] = ptx::ld_with_gt_pred(src_ptr + k * kNumThreadsPerReduceGroup, num_remaining_vecs, k * kNumThreadsPerReduceGroup);
                        }

                        const auto fp32x2_view = reinterpret_cast<float2*>(values);
                        #pragma unroll
                        for (int k = 0; k < kNumVecsPerIter; ++ k) {
                            #pragma unroll
                            for (int half_idx = 0; half_idx < 2; ++ half_idx) {
                                const auto& packed = reinterpret_cast<const uint2*>(&packed_values[k])[half_idx];
                                ptx::accumulate(fp32x2_view[k * 4 + half_idx * 2], reinterpret_cast<const nv_bfloat162&>(packed.x));
                                ptx::accumulate(fp32x2_view[k * 4 + half_idx * 2 + 1], reinterpret_cast<const nv_bfloat162&>(packed.y));
                            }
                        }
                    }

                    if constexpr (kWithScale) {
                        // Scale only the final sum, never an intermediate batch.
                        if (is_last_batch) {
                            #pragma unroll
                            for (int k = 0; k < kNumVecsPerIter; ++ k) {
                                #pragma unroll
                                for (int half_idx = 0; half_idx < 2; ++ half_idx) {
                                    auto& value = reinterpret_cast<float4*>(&values[k])[half_idx];
                                    value = ptx::fmul4(value, scale);
                                }
                            }
                        }
                    }

                    if (full_chunk) {
                        #pragma unroll
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            ptx::st_with_gt_pred(dst_ptr + vec_idx + k * kNumThreadsPerReduceGroup, values[k]);
                    } else {
                        #pragma unroll
                        for (int k = 0; k < kNumVecsPerIter; ++ k)
                            ptx::st_with_gt_pred(dst_ptr + vec_idx + k * kNumThreadsPerReduceGroup, values[k], num_remaining_vecs, k * kNumThreadsPerReduceGroup);
                    }
                }
            }
        }
    }

    // Ensure no later writes
    comm::gpu_barrier<true, kNumRanks, 1,
                      kNumSMs, kNumThreads, comm::kFlushAllAllocatedQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);
}

}  // namespace deep_ep::bucket

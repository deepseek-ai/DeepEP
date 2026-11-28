#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/bucket/chunk.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

namespace deep_ep::bucket {

template <int kNumRanks, int kNumSMs, int kNumThreads, int kNumVectorsPerThread,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx, bool kAccumulate>
__global__ void __launch_bounds__(kNumThreads, 1)
nvlink_reduce_scatter(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                      void* workspace_ptr,
                      void* buffer, void* buffer_multimem,
                      const float scale,
                      const int num_buckets,
                      const __grid_constant__ comm::BucketList buckets,
                      const __grid_constant__ comm::BucketList dst_buckets) {
    // Indices
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    const auto global_thread_idx = sm_idx * kNumThreads + thread_idx;
    const auto rank_idx = ncclTeamLsa(nccl_dev_comm).rank;

    // Barrier to ensure data is arrived
    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    auto& barrier_signals = workspace.signals[kContextIdx].barrier;
    const comm::NCCLGin gin(nccl_dev_comm, window, 0);
    comm::gpu_barrier<true, 1, kNumRanks,
                      kNumSMs, kNumThreads, 1,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, 0, rank_idx, sm_idx, thread_idx);

    // A chunk spans one warp for accumulation, or one grid iteration for in-place stores.
    constexpr int kNumWarps = kNumThreads / 32;
    constexpr int kNumThreadsPerChunk = kAccumulate ? 32 : kNumSMs * kNumThreads;
    constexpr int kNumChunkBytes = kNumVectorsPerThread * kNumThreadsPerChunk * sizeof(float4);
    constexpr int kChunkStride = kAccumulate ? kNumSMs * kNumWarps : 1;
    const int start_chunk_idx = kAccumulate ? warp_idx * kNumSMs + sm_idx : 0;
    const int thread_idx_in_chunk = kAccumulate ? lane_idx : global_thread_idx;

    // One shared-memory tile per warp, only used for accumulation.
    extern __shared__ float4 smem[];
    const auto smem_ptr = smem + warp_idx * 32 * kNumVectorsPerThread;

    auto iterator = layout::ChunkIterator<kNumChunkBytes, kChunkStride>(num_buckets, buckets, start_chunk_idx);
    while (iterator.next()) {
        const int num_vecs = iterator.num_chunk_bytes / sizeof(float4);
        if (not kAccumulate and thread_idx_in_chunk >= num_vecs)
            continue;

        const auto src_offset = iterator.bucket.offset + rank_idx * iterator.bucket.num_bytes + iterator.chunk_offset_in_bucket;
        const auto dst_offset = kAccumulate ? dst_buckets[iterator.bucket_idx].offset + iterator.chunk_offset_in_bucket : src_offset;
        const auto src = math::advance_ptr<const float4>(buffer_multimem, src_offset);
        const auto dst = math::advance_ptr<float4>(buffer, dst_offset);
        const bool full_chunk = iterator.num_chunk_bytes == kNumChunkBytes;

        // Load
        float4 values[kNumVectorsPerThread];
        if (full_chunk) {
            #pragma unroll
            for (int i = 0; i < kNumVectorsPerThread; ++ i)
                values[i] = ptx::multimem_ld_reduce_add_f32_with_gt_pred(src + thread_idx_in_chunk + i * kNumThreadsPerChunk);
        } else {
            #pragma unroll
            for (int i = 0; i < kNumVectorsPerThread; ++ i) {
                const int vec_idx = thread_idx_in_chunk + i * kNumThreadsPerChunk;
                values[i] = ptx::multimem_ld_reduce_add_f32_with_gt_pred(src + vec_idx, num_vecs, vec_idx);
            }
        }

        // Wait before reusing the shared-memory tile
        if constexpr (kAccumulate) {
            if (ptx::elect_one_sync())
                ptx::tma_store_wait_read<0>();
            __syncwarp();
        }

        // Scale and store
        #pragma unroll
        for (int i = 0; i < kNumVectorsPerThread; ++ i) {
            const int vec_idx = thread_idx_in_chunk + i * kNumThreadsPerChunk;
            const auto value = kWithScale ? ptx::fmul4(values[i], scale) : values[i];
            if constexpr (kAccumulate) {
                smem_ptr[vec_idx] = value;
            } else if (full_chunk) {
                ptx::st_with_gt_pred(dst + vec_idx, value);
            } else {
                ptx::st_with_gt_pred(dst + vec_idx, value, num_vecs, vec_idx);
            }
        }

        // Accumulate
        if constexpr (kAccumulate) {
            __syncwarp();
            if (ptx::elect_one_sync()) {
                ptx::tma_store_fence();
                ptx::tma_store_reduce_add_f32(reinterpret_cast<float*>(dst), smem_ptr, iterator.num_chunk_bytes);
                ptx::tma_store_commit();
            }
            __syncwarp();
        }
    }

    // Ensure no overlapping writes
    comm::gpu_barrier<true, 1, kNumRanks,
                      kNumSMs, kNumThreads, 1,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, 0, rank_idx, sm_idx, thread_idx);
}

}  // namespace deep_ep::bucket

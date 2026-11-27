#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/bucket/chunk.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

namespace deep_ep::bucket {

template <int kNumRanks, int kNumSMs, int kNumThreads,
          int64_t kNumTimeoutCycles, bool kWithScale, int kContextIdx>
__global__ void __launch_bounds__(kNumThreads, 1)
nvlink_all_reduce(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                  void* workspace_ptr,
                  void* buffer_multimem,
                  const float scale,
                  const int num_buckets,
                  const __grid_constant__ comm::BucketList buckets) {
    EP_STATIC_ASSERT(kNumThreads % 32 == 0, "Invalid thread count");

    // Indices
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();
    const auto rank_idx = ncclTeamLsa(nccl_dev_comm).rank;

    // Barrier to ensure data is arrived
    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    auto& barrier_signals = workspace.signals[kContextIdx].barrier;
    const comm::NCCLGin gin(nccl_dev_comm, window, 0);
    comm::gpu_barrier<true, 1, kNumRanks,
                      kNumSMs, kNumThreads, 1,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, 0, rank_idx, sm_idx, thread_idx);

    constexpr int kNumVectorsPerThread = 8;
    constexpr int kNumWarps = kNumThreads / 32;
    constexpr int kNumChunkBytes = kNumVectorsPerThread * 32 * static_cast<int>(sizeof(float4));
    constexpr int kChunkStride = kNumRanks * kNumSMs * kNumWarps;
    const int start_chunk_idx = (warp_idx * kNumSMs + sm_idx) * kNumRanks + rank_idx;

    // Fused reduce-scatter + multicast-store all-gather over this rank's shard slot
    auto iterator = layout::ChunkIterator<kNumChunkBytes, kChunkStride>(num_buckets, buckets, start_chunk_idx);
    while (iterator.next()) {
        const auto ptr = math::advance_ptr<float4>(
            buffer_multimem, iterator.bucket.offset + iterator.chunk_offset_in_bucket);
        float4 values[kNumVectorsPerThread];
        if (iterator.num_chunk_bytes == kNumChunkBytes) {
            #pragma unroll
            for (int i = 0; i < kNumVectorsPerThread; ++ i)
                values[i] = ptx::multimem_ld_reduce_add_f32_with_gt_pred(ptr + lane_idx + i * 32);
            #pragma unroll
            for (int i = 0; i < kNumVectorsPerThread; ++ i) {
                const auto value = kWithScale ? ptx::fmul4(values[i], scale) : values[i];
                ptx::multimem_st_f32_with_gt_pred(ptr + lane_idx + i * 32, value);
            }
        } else {
            const int num_vecs = iterator.num_chunk_bytes / static_cast<int>(sizeof(float4));
            #pragma unroll
            for (int i = 0; i < kNumVectorsPerThread; ++ i) {
                const int vec_idx = lane_idx + i * 32;
                values[i] = ptx::multimem_ld_reduce_add_f32_with_gt_pred(ptr + vec_idx, num_vecs, vec_idx);
            }
            #pragma unroll
            for (int i = 0; i < kNumVectorsPerThread; ++ i) {
                const int vec_idx = lane_idx + i * 32;
                const auto value = kWithScale ? ptx::fmul4(values[i], scale) : values[i];
                ptx::multimem_st_f32_with_gt_pred(ptr + vec_idx, value, num_vecs, vec_idx);
            }
        }
    }

    // Ensure no overlapping writes
    comm::gpu_barrier<true, 1, kNumRanks,
                      kNumSMs, kNumThreads, 1,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, 0, rank_idx, sm_idx, thread_idx);
}

}  // namespace deep_ep::bucket

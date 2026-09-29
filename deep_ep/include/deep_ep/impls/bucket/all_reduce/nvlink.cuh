#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/utils.cuh>
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
    // Indices
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
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

    // Fused reduce-scatter + multicast-store all-gather over this rank's shard slot
    for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
        // NOTES: the bucket covers the full tensor, each rank owns one shard of it
        const auto& bucket = buckets[bucket_idx];
        const auto num_shard_bytes = bucket.num_bytes / kNumRanks;
        const auto shard_offset = bucket.offset + rank_idx * num_shard_bytes;
        const auto ptr = math::advance_ptr<float4>(buffer_multimem, shard_offset);

        // TODO: merge iterations between buckets
        utils::iterate_ld_st<kNumSMs, kNumThreads, 8>(
            global_thread_idx,
            num_shard_bytes / static_cast<int64_t>(sizeof(float4)),
            ptr, ptr,
            [](const float4* src, const int& lhs = 1, const int& rhs = 0) { return ptx::multimem_ld_reduce_add_f32_with_gt_pred(src, lhs, rhs); },
            [](float4* dst, const float4& value, const int& lhs = 1, const int& rhs = 0) { ptx::multimem_st_f32_with_gt_pred(dst, value, lhs, rhs); },
            [&](const float4& v) { return kWithScale ? ptx::fmul4(v, scale) : v; });
    }

    // Ensure no overlapping writes
    comm::gpu_barrier<true, 1, kNumRanks,
                      kNumSMs, kNumThreads, 1,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, 0, rank_idx, sm_idx, thread_idx);
}

}  // namespace deep_ep::bucket

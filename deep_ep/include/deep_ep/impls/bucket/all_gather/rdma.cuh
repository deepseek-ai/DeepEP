#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/comm/handle.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

namespace deep_ep::bucket {

template <int kNumRanks, int kNumSMs, int64_t kNumTimeoutCycles, int kContextIdx,
          int kNumThreads = 512,
          int kNumWarps = kNumThreads / 32,
          int kNumGlobalWarps = kNumSMs * kNumWarps>
__global__ void __launch_bounds__(kNumThreads, 1)
rdma_all_gather(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                void* workspace_ptr, void* buffer,
                const int num_buckets,
                const __grid_constant__ comm::BucketList buckets) {
    // Indices
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto lane_idx = ptx::get_lane_idx();
    const auto warp_idx = ptx::get_warp_idx();
    const auto global_warp_idx = sm_idx * kNumWarps + warp_idx;
    const auto rank_idx = ncclTeamRail(nccl_dev_comm).rank;

    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    auto& barrier_signals = workspace.signals[kContextIdx].barrier;

    // Entry barrier: all ranks have their local shard in place.
    const comm::NCCLGin gin(nccl_dev_comm, window);
    comm::gpu_barrier<true, kNumRanks, 1,
                      kNumSMs, kNumThreads, 1,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, barrier_signals, rank_idx, 0, sm_idx, thread_idx);

    for (int dst_rank_idx = global_warp_idx; dst_rank_idx < kNumRanks; dst_rank_idx += kNumGlobalWarps) {
        if (dst_rank_idx == rank_idx)
            continue;

        for (int bucket_idx = lane_idx; bucket_idx < num_buckets; bucket_idx += 32) {
            const auto& [offset, num_bytes] = buckets[bucket_idx];
            const auto ptr = math::advance_ptr(buffer, offset + rank_idx * num_bytes);
            gin.put<ncclTeamTagRail>(ptr, ptr, num_bytes, dst_rank_idx);
        }
        __syncwarp();
    }
}

}  // namespace deep_ep::bucket

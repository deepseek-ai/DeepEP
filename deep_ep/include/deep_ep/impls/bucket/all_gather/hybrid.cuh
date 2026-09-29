#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/comm/handle.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

namespace deep_ep::bucket {

template <int kNumRDMARanks, int kNumNVLRanks, int kNumThreads,
          int64_t kNumTimeoutCycles, int kContextIdx,
          int kNumQPsPerChunk = layout::AllGatherSignals::kNumQPsPerChunk>
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_all_gather(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
                  void* workspace_ptr, void* buffer,
                  const int num_buckets, const int num_chunks, const int num_chunk_bytes,
                  const int rdma_rank_idx, const int nvl_rank_idx, const int rank_idx,
                  const __grid_constant__ comm::BucketList buckets) {
    // One SM per RDMA peer, one warp per QP
    EP_STATIC_ASSERT(kNumRDMARanks > 1 and kNumNVLRanks > 1, "Hybrid all-gather requires both RDMA and NVLink ranks");
    EP_STATIC_ASSERT(kNumNVLRanks <= kNumThreads and kNumQPsPerChunk * 32 <= kNumThreads, "Insufficient threads");

    // Indices: SM index is the destination RDMA rank, warp index is the QP index
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();

    // NOTES: all QPs of this SM are only used by this CTA, the QP index only matters for the issuing warps
    const comm::NCCLGin gin(nccl_dev_comm, window, warp_idx % kNumQPsPerChunk, NCCL_GIN_RESOURCE_SHARING_CTA);
    auto& bucket_signals = static_cast<layout::BucketWorkspace*>(workspace_ptr)->signals[kContextIdx];
    auto& signals = bucket_signals.all_gather;

    // Clear tails
    for (int i = sm_idx * kNumThreads + thread_idx; i < num_chunks; i += kNumRDMARanks * kNumThreads) {
        #pragma unroll
        for (int j = 0; j < kNumQPsPerChunk; ++ j)
            signals.tails[j][i] = 0;
    }

    // Entry barrier: make the cleared tails visible before any peer signals
    // NOTES: no flush is needed, the previous puts are already confirmed by the tail signals
    comm::gpu_barrier<true, kNumRDMARanks, kNumNVLRanks,
                      kNumRDMARanks, kNumThreads, kNumQPsPerChunk,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, false, true, true>(
        gin, bucket_signals.barrier,
        rdma_rank_idx, nvl_rank_idx, sm_idx, thread_idx);

    // Each warp is responsible for a QP
    if (sm_idx == rdma_rank_idx or warp_idx >= kNumQPsPerChunk)
        return;

    // Issue in order: each QP sends its own part of every chunk and signals once per chunk
    if (ptx::elect_one_sync()) {
        int chunk_idx = 0;
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& [bucket_offset, num_bucket_bytes] = buckets[bucket_idx];
            const auto shard_ptr = math::advance_ptr(buffer, bucket_offset + rank_idx * num_bucket_bytes);
            for (int64_t chunk_offset = 0; chunk_offset < num_bucket_bytes; chunk_offset += num_chunk_bytes, ++ chunk_idx) {
                // Split the chunk evenly across QPs, only the last chunk of a bucket may be uneven
                const auto num_bytes_in_chunk = static_cast<int>(min(num_bucket_bytes - chunk_offset, static_cast<int64_t>(num_chunk_bytes)));
                const auto num_bytes_per_qp = math::align(math::ceil_div(num_bytes_in_chunk, kNumQPsPerChunk), kNumRDMAAlignmentBytes);
                const auto qp_offset = min(warp_idx * num_bytes_per_qp, num_bytes_in_chunk);
                const auto num_bytes = min(num_bytes_per_qp, num_bytes_in_chunk - qp_offset);

                // NOTES: always signal even if this QP has nothing to send, so that the CE waits stay uniform
                const ncclGin_WeakVASignalInc remote_action {
                    .signalWindow = window,
                    .signalOffset = gin.get_sym_offset(signals.tails[warp_idx] + chunk_idx),
                };
                if (num_bytes > 0) {
                    const auto ptr = math::advance_ptr(shard_ptr, chunk_offset + qp_offset);
                    gin.put<ncclTeamTagRail>(ptr, ptr, num_bytes, sm_idx, 0, remote_action);
                } else {
                    gin.signal<ncclTeamTagRail>(sm_idx, remote_action);
                }
            }
        }
    }
}

}  // namespace deep_ep::bucket

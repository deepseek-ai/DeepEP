#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/comm/bucket.cuh>
#include <deep_ep/comm/handle.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

namespace deep_ep::bucket {

template <bool kIsScaleupNVLink, bool kUseChunkStaging,
          int kNumRDMARanks, int kNumNVLRanks, int kNumSMs, int kNumQPs,
          int64_t kNumTimeoutCycles, int kContextIdx,
          int kNumThreads = 256>
__global__ void __launch_bounds__(kNumThreads, 1)
ring_send_recv(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t window,
               void* workspace_ptr, void* buffer,
               const int num_buckets, const int num_chunks, const int num_chunk_bytes,
               const int rdma_rank_idx, const int nvl_rank_idx,
               const int dst_rdma_rank_idx,
               const __grid_constant__ comm::BucketList src_buckets,
               const __grid_constant__ comm::BucketList dst_buckets) {
    EP_STATIC_ASSERT(kNumRDMARanks > 1, "Ring send-recv kernel requires multiple RDMA ranks");
    EP_STATIC_ASSERT(not kUseChunkStaging or kNumNVLRanks > 1, "Chunk staging requires NVLink forwarding");
    EP_STATIC_ASSERT(kNumNVLRanks == 1 or kNumSMs >= 2, "Hybrid barrier requires at least 2 SMs");
    EP_STATIC_ASSERT(kNumQPs > 0 and kNumQPs * 32 <= kNumThreads, "Insufficient threads for QPs");

    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();

    const comm::NCCLGin gin(nccl_dev_comm, window, warp_idx % kNumQPs, NCCL_GIN_RESOURCE_SHARING_CTA);
    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    auto& bucket_signals = workspace.signals[kContextIdx];
    auto& signals = bucket_signals.ring_send_recv;

    // Clear tails before the entry barrier
    if constexpr (kUseChunkStaging) {
        for (int i = sm_idx * kNumThreads + thread_idx; i < num_chunks; i += kNumSMs * kNumThreads) {
            #pragma unroll
            for (int j = 0; j < kNumQPs; ++ j)
                signals.tails[j][i] = 0;
        }
    }

    // Entry barrier
    comm::gpu_barrier<kIsScaleupNVLink, kNumRDMARanks, kNumNVLRanks,
                      kNumSMs, kNumThreads, kNumQPs,
                      kNumTimeoutCycles, comm::kKernelBarrierTag, false, kUseChunkStaging, true>(
        gin, bucket_signals.barrier,
        rdma_rank_idx, nvl_rank_idx, sm_idx, thread_idx);

    if (sm_idx > 0 or dst_rdma_rank_idx < 0 or warp_idx >= kNumQPs)
        return;

    if (ptx::elect_one_sync()) {
        if constexpr (not kUseChunkStaging) {
            for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
                const auto& [src_offset, num_bucket_bytes] = src_buckets[bucket_idx];
                const auto dst_offset = dst_buckets[bucket_idx].offset;
                const auto num_bytes_per_qp = math::align<int64_t>(
                    math::ceil_div<int64_t>(num_bucket_bytes, kNumQPs), kNumRDMAAlignmentBytes);
                const auto qp_offset = min(static_cast<int64_t>(warp_idx) * num_bytes_per_qp, num_bucket_bytes);
                const auto num_bytes = min(num_bytes_per_qp, num_bucket_bytes - qp_offset);
                if (num_bytes > 0) {
                    gin.put<ncclTeamTagRail>(
                        math::advance_ptr(buffer, dst_offset + qp_offset),
                        math::advance_ptr(buffer, src_offset + qp_offset),
                        num_bytes, dst_rdma_rank_idx);
                }
            }
        } else {
            int chunk_idx = 0;
            int64_t stage_offset = 0;
            for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
                const auto& [src_offset, num_bucket_bytes] = src_buckets[bucket_idx];
                for (int64_t chunk_offset = 0; chunk_offset < num_bucket_bytes; chunk_offset += num_chunk_bytes, ++ chunk_idx) {
                    const auto num_bytes_in_chunk = static_cast<int>(
                        min(num_bucket_bytes - chunk_offset, static_cast<int64_t>(num_chunk_bytes)));
                    const auto num_bytes_per_qp = math::align(
                        math::ceil_div(num_bytes_in_chunk, kNumQPs), kNumRDMAAlignmentBytes);
                    const auto qp_offset = min(warp_idx * num_bytes_per_qp, num_bytes_in_chunk);
                    const auto num_bytes = min(num_bytes_per_qp, num_bytes_in_chunk - qp_offset);

                    // Always signal even if this QP has nothing to send, so the CE wait stays uniform
                    const ncclGin_WeakVASignalInc remote_action {
                        .signalWindow = window,
                        .signalOffset = gin.get_sym_offset(signals.tails[warp_idx] + chunk_idx),
                    };
                    if (num_bytes > 0) {
                        gin.put<ncclTeamTagRail>(
                            workspace.chunk_storage.data + stage_offset + chunk_offset + qp_offset,
                            math::advance_ptr(buffer, src_offset + chunk_offset + qp_offset),
                            num_bytes, dst_rdma_rank_idx, 0, remote_action);
                    } else {
                        gin.signal<ncclTeamTagRail>(dst_rdma_rank_idx, remote_action);
                    }
                }
                stage_offset += math::align<int64_t>(num_bucket_bytes, kNumRDMAAlignmentBytes);
            }
        }
    }
}

}  // namespace deep_ep::bucket

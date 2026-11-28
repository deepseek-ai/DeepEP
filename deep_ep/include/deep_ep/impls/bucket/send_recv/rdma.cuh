#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/bucket/send_recv.cuh>

namespace deep_ep::bucket {

template <int kNumRanks, int kNumSMs, int kNumThreads, int kNumQPs, int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(kNumThreads, 1)
rdma_send_recv(const ncclDevComm_t nccl_dev_comm,
               const ncclWindow_t window,
               void* storage_ptr,
               const int num_send_entries,
               const __grid_constant__ layout::SendRecvList send_entries) {
    EP_STATIC_ASSERT(kNumSMs == 1, "RDMA send-recv only supports one SM");
    EP_STATIC_ASSERT(kNumQPs == 1 or kNumQPs == 2, "RDMA send-recv only supports one or two QPs");
    EP_STATIC_ASSERT(kNumThreads % kNumQPs == 0, "kNumThreads must be divisible by kNumQPs");
    constexpr int kNumThreadsPerQP = kNumThreads / kNumQPs;
    EP_STATIC_ASSERT(layout::kNumMaxSendRecvEntries <= kNumThreadsPerQP,
                     "Num send recv entries must not exceed num threads per QP.");

    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto rank_idx = ncclTeamRail(nccl_dev_comm).rank;
    const auto qp_idx = thread_idx / kNumThreadsPerQP;
    const auto entry_idx = thread_idx % kNumThreadsPerQP;

    // NOTES: Only one CTA is launched, so resource sharing mode can be SHARING_CTA
    const comm::NCCLGin gin(nccl_dev_comm, window, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);

    // All ranks participate, even when their send list is empty.
    comm::scaleout_barrier_wo_local_sync<kNumRanks, kNumSMs, kNumThreads, kNumQPs,
                                        kNumTimeoutCycles, comm::kKernelBarrierTag, false>(
        gin, rank_idx, 0, 0, thread_idx);
    __syncthreads();

    // Issue send
    if (entry_idx < num_send_entries) {
        const auto& entry = send_entries[entry_idx];
        const auto num_bytes_per_qp = math::align<int64_t>(entry.num_bytes / kNumQPs, kNumRDMAAlignmentBytes);
        const auto offset = qp_idx * num_bytes_per_qp;
        const auto num_bytes = qp_idx + 1 == kNumQPs ?
            entry.num_bytes - offset : num_bytes_per_qp;
        // Skip self copy
        if (entry.src_rank_idx != entry.dst_rank_idx and num_bytes > 0)
            gin.put<ncclTeamTagRail>(
                math::advance_ptr(storage_ptr, entry.dst_offset + offset),
                math::advance_ptr(entry.src_ptr, offset),
                num_bytes, entry.dst_rank_idx);
    }
}

}  // namespace deep_ep::bucket

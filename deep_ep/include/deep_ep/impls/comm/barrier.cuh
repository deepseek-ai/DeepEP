#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/layout/common/barrier.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::comm {

template <bool kIsScaleupNVLink,
          int kNumSMs, int kNumThreads,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int64_t kNumTimeoutCycles,
          bool kFlushStores, bool kSequential>
__global__ void __launch_bounds__(kNumThreads, 1)
barrier_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window, void* signals,
             const int scaleout_rank_idx, const int scaleup_rank_idx) {
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);

    auto& barrier_signals = *static_cast<layout::BarrierSignals*>(signals);
    const auto gin = NCCLGin(nccl_dev_comm, nccl_window, 0);
    if constexpr (kSequential) {
        // Scaleout barrier
        if constexpr (kNumScaleoutRanks > 1)
            gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                        kNumSMs, kNumThreads, kFlushAllAllocatedQPs, kNumTimeoutCycles, kKernelBarrierTag,
                        kFlushStores, true, false>(
                gin, barrier_signals, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx, true, false);

        // Scaleup barrier, and it needs to flush the RDMA requests issued by scaleout barrier 
        gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                    kNumSMs, kNumThreads, kFlushAllAllocatedQPs, kNumTimeoutCycles, kKernelBarrierTag, kFlushStores, true, false>(
            gin, barrier_signals, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx, false, true);
    } else {
        gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                    kNumSMs, kNumThreads, kFlushAllAllocatedQPs, kNumTimeoutCycles, kKernelBarrierTag,
                    kFlushStores, true, false>(
            gin, barrier_signals, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
    }
}

}  // namespace deep_ep::comm

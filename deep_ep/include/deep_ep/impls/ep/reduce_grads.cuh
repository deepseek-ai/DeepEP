#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/ep/workspace.cuh>
#include <deep_ep/layout/ep/eplb.cuh>


namespace deep_ep::ep {

template <int kNumSMs,
          int kNumWarps,
          int kNumRanks,
          int kNumLocalExperts,
          int kNumRedundantExperts,
          int kNumChunkBytes,
          int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(kNumWarps * 32, 1)
lb_reduce_grads(const int* redundancy_mapping,
                const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                void* workspace,
                const __grid_constant__ layout::EPWeight weight) {
    // Indices
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx();

    // Checks
    constexpr int kNumThreads = kNumWarps * 32;
    EP_STATIC_ASSERT(kNumWarps > 0 and kNumWarps <= 32, "Invalid number of warps");

    // Shared memory
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    auto& smem_layout = *reinterpret_cast<layout::EPLBSharedMemoryLayout*>(smem);
    auto* tma_buffers = math::advance_ptr(smem, sizeof(smem_layout));
    auto* tma_buffer = math::advance_ptr(tma_buffers, warp_idx * kNumChunkBytes);
    auto* mbarrier = math::advance_ptr<ptx::mbarrier>(tma_buffers, kNumWarps * kNumChunkBytes) + warp_idx;

    // Communication
    const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window, 0);
    const auto workspace_layout = layout::EPWorkspaceLayout(workspace, 1, kNumRanks, kNumRanks * kNumLocalExperts);

    // Build tasks
    const int rank_idx = ncclTeamLsa(nccl_dev_comm).rank;
    smem_layout.build_tasks<kNumRanks, kNumLocalExperts, kNumRedundantExperts, kNumThreads, kNumWarps * kNumChunkBytes>(
        redundancy_mapping, rank_idx, thread_idx, tma_buffers);

    // Initialize barriers and chunk counter
    // NOTES: Initialize mbarriers after shared workspace reuse because they have a dedicated cache
    if (ptx::elect_one_sync()) {
        ptx::mbarrier_init_with_fence(mbarrier, 1);

        // Cleanup task fetch
        if (sm_idx == 0 and warp_idx == 0)
            *workspace_layout.get_lb_chunk_idx_counter_ptr() = 0;
    }
    __syncwarp();

    // Entry barrier
    comm::gpu_barrier<true, 1, kNumRanks,
                      kNumSMs, kNumThreads, 1, kNumTimeoutCycles, comm::kKernelBarrierTag, false, false, true>(
        gin, workspace_layout.get_barrier_signals(), 0, rank_idx, sm_idx, thread_idx);

    // Reduce
    if (ptx::elect_one_sync()) {
        auto iterator = layout::EPWeightIterator<1, kNumChunkBytes>(
            gin, smem_layout, &weight, workspace_layout.get_lb_chunk_idx_counter_ptr());
        ptx::arrival_phase phase = 0;
        while (iterator.next()) {
            // Load redundant gradients
            ptx::mbarrier_arrive_and_set_tx(mbarrier, iterator.num_chunk_bytes);
            ptx::tma_load_1d(tma_buffer, iterator.dst_ptr, mbarrier, iterator.num_chunk_bytes,
                             /* TODO: why this is faster? */ ptx::L2CacheHint::kEvictNormal);
            ptx::mbarrier_wait_and_flip_phase(mbarrier, phase);

            // Accumulate and release the shared buffer
            ptx::tma_store_reduce_add_f32(static_cast<float*>(iterator.src_ptr), tma_buffer, iterator.num_chunk_bytes);
            ptx::tma_store_commit();
            ptx::tma_store_wait_read<0>();
        }
    }
    __syncwarp();

    // Completion barrier
    comm::gpu_barrier<true, 1, kNumRanks,
                      kNumSMs, kNumThreads, 1, kNumTimeoutCycles, comm::kKernelBarrierTag, true, true, false>(
        gin, workspace_layout.get_barrier_signals(), 0, rank_idx, sm_idx, thread_idx);
}

}  // namespace deep_ep::ep

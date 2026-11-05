#pragma once

#include <algorithm>
#include <cooperative_groups.h>
#include <utility>

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/comm/handle.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/pp/workspace.cuh>

namespace deep_ep::pp {

// Returns `(local_idx, peer_idx)`. `local_idx` is this rank's buffer half for the peer (0 for the
// next rank, 1 for the previous rank) and `peer_idx` is the symmetric index the peer uses to
// address the same half.
template <int kNumRanks>
__device__ __forceinline__ std::pair<int, int> get_pp_buffer_offset(
    const int& src_rank_idx, const int& dst_rank_idx) {
    const auto next_rank_idx = (src_rank_idx + 1) % kNumRanks;
    return dst_rank_idx == next_rank_idx ? std::make_pair(0, 1) : std::make_pair(1, 0);
}

template <int kNumSMs,
          int kNumSmemBytes,
          int kNumStages = 2,
          int kNumTMABytesPerStage = math::constexpr_align<int, false>(
              (kNumSmemBytes - kNumStages * static_cast<int>(sizeof(ptx::mbarrier))) / kNumStages,
              kNumTMAAlignmentBytes),
          int kNumTMABlocksPerStage = kNumTMABytesPerStage / kNumTMAAlignmentBytes>
__device__ __forceinline__ void pp_tma_copy(
    void* src_ptr, void* dst_ptr,
    const int64_t& num_bytes, const int& sm_idx) {
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    const auto tma_buffers = smem;
    const auto mbarriers = reinterpret_cast<ptx::mbarrier*>(smem + kNumStages * kNumTMABytesPerStage);
    EP_STATIC_ASSERT(kNumTMABytesPerStage > 0, "Invalid shared memory bytes");
    EP_STATIC_ASSERT(kNumStages >= 2, "Need at least 2 stages for pipelining");

    // Init mbarriers
    ptx::arrival_phase phases[kNumStages];
    #pragma unroll
    for (int s = 0; s < kNumStages; ++ s)
        phases[s] = 0, ptx::mbarrier_init_with_fence(mbarriers + s, 1);

    // Work partitioning across SMs
    EP_DEVICE_ASSERT(num_bytes % kNumTMAAlignmentBytes == 0);
    const auto num_tma_blocks = num_bytes / kNumTMAAlignmentBytes;
    const auto num_tma_blocks_per_sm = math::ceil_div<int64_t>(num_tma_blocks, kNumSMs);
    const auto start_block_idx = sm_idx * num_tma_blocks_per_sm;
    const auto end_block_idx = std::min(start_block_idx + num_tma_blocks_per_sm, num_tma_blocks);
    const auto num_iterations = math::ceil_div<int64_t>(end_block_idx - start_block_idx, kNumTMABlocksPerStage);

    auto get_iter_info = [&](const int64_t& iter_idx) {
        const auto i = start_block_idx + iter_idx * kNumTMABlocksPerStage;
        const auto offset = i * kNumTMAAlignmentBytes;
        const auto num_transaction_bytes =
            std::min<int>(kNumTMABlocksPerStage, end_block_idx - i) * kNumTMAAlignmentBytes;
        return std::make_pair(offset, num_transaction_bytes);
    };

    // Fill pipeline: issue loads for the first kNumStages iterations
    for (int64_t iter_idx = 0; iter_idx < kNumStages and iter_idx < num_iterations; ++ iter_idx) {
        const auto [load_offset, num_load_bytes] = get_iter_info(iter_idx);
        ptx::tma_load_1d(
            tma_buffers + iter_idx * kNumTMABytesPerStage,
            math::advance_ptr(src_ptr, load_offset),
            mbarriers + iter_idx, num_load_bytes);
        ptx::mbarrier_arrive_and_set_tx(mbarriers + iter_idx, num_load_bytes);
    }

    for (int64_t iter_idx = 0; iter_idx < num_iterations; ++ iter_idx) {
        const auto stage_idx = static_cast<int>(iter_idx % kNumStages);
        const auto [store_offset, num_store_bytes] = get_iter_info(iter_idx);

        // Wait this stage's load and issue store
        ptx::mbarrier_wait_and_flip_phase(mbarriers + stage_idx, phases[stage_idx]);
        ptx::tma_store_1d(
            math::advance_ptr(dst_ptr, store_offset),
            tma_buffers + stage_idx * kNumTMABytesPerStage,
            num_store_bytes);
        ptx::tma_store_commit();

        // Prefetch: wait until this stage's store is completed, then issue next load
        const auto next_iter_idx = iter_idx + kNumStages;
        if (next_iter_idx < num_iterations) {
            ptx::tma_store_wait();
            const auto [load_offset, num_load_bytes] = get_iter_info(next_iter_idx);
            ptx::tma_load_1d(
                tma_buffers + stage_idx * kNumTMABytesPerStage,
                math::advance_ptr(src_ptr, load_offset),
                mbarriers + stage_idx, num_load_bytes);
            ptx::mbarrier_arrive_and_set_tx(mbarriers + stage_idx, num_load_bytes);
        }
    }

    // Drain all outstanding stores
    ptx::tma_store_wait();
}

// Send `x` to an adjacent rank. `x` is first staged into a local send slot (TMA), then pushed into
// the peer's recv slot through Gin. One slot is released back to us once the peer has consumed it.
template <int kNumSMs,
          int kNumRanks,
          int kNumSmemBytes,
          int kNumQPs,
          int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(32, 1)
pp_send_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
             void* x, const int64_t num_x_bytes,
             void* buffer, void* workspace,
             const int rank_idx, const int dst_rank_idx,
             const int64_t num_max_tensor_bytes,
             const int num_max_inflight_tensors) {
    const auto sm_idx = static_cast<int>(blockIdx.x);
    auto& pp = *static_cast<layout::PPSignals*>(workspace);
    const auto [local_idx_in_dst, dst_idx_in_local] = get_pp_buffer_offset<kNumRanks>(rank_idx, dst_rank_idx);

    EP_STATIC_ASSERT(kNumQPs > 0 and kNumQPs <= 2, "Invalid number of PP QPs");

    // Buffer offsets
    const auto send_count = pp.send_count[dst_idx_in_local];
    const auto slot_idx = send_count % num_max_inflight_tensors;
    auto send_buffer_ptr = math::advance_ptr(
        buffer, ((dst_idx_in_local + 2) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);
    auto recv_buffer_ptr = math::advance_ptr(
        buffer, ((local_idx_in_dst + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);

    // Wait buffer slot release and do TMA
    if (ptx::elect_one_sync()) {
        const auto release_target = send_count - num_max_inflight_tensors + 1;
        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
            if (ptx::ld_acquire_sys(&pp.release[dst_idx_in_local]) >= release_target)
                return true;
            if (is_last_check)
                printf("DeepEP PP send timeout, rank %d, peer %d, recv buffer is full\n", rank_idx, dst_rank_idx);
            return false;
        });
        pp_tma_copy<kNumSMs, kNumSmemBytes>(x, send_buffer_ptr, num_x_bytes, sm_idx);
    }
    cooperative_groups::this_grid().sync();

    // Issue RDMA put
    if (ptx::elect_one_sync()) {
        // Split buffer among QPs
        const auto num_bytes_per_qp = num_x_bytes / kNumQPs / kNumRDMAAlignmentBytes * kNumRDMAAlignmentBytes;
        for (int context_idx = sm_idx; context_idx < kNumQPs; context_idx += kNumSMs) {
            const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window, context_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
            const auto offset = context_idx * num_bytes_per_qp;
            const auto num_bytes = context_idx == kNumQPs - 1 ? num_x_bytes - offset : num_bytes_per_qp;
            const ncclGin_StrongVASignalAdd remote_action {
                .signalWindow = nccl_window,
                .signalOffset = gin.get_sym_offset(&pp.arrival[local_idx_in_dst][context_idx]),
                .value = 1,
            };
            // Always signal, as the receiver waits on every QP even if it has no data
            if (num_bytes > 0) {
                gin.put<ncclTeamTagWorld>(
                    math::advance_ptr(recv_buffer_ptr, offset), math::advance_ptr(send_buffer_ptr, offset),
                    num_bytes, dst_rank_idx, 0, remote_action);
            } else {
                gin.signal<ncclTeamTagWorld>(dst_rank_idx, remote_action);
            }
        }
        if (sm_idx == 0)
            pp.send_count[dst_idx_in_local] += 1;
    }
}

// Receive `x` from an adjacent rank. The data is staged in our recv slot by the peer, then copied
// out with TMA. Once copied, we release the slot back to the peer.
template <int kNumSMs,
          int kNumRanks,
          int kNumSmemBytes,
          int kNumQPs,
          int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(32, 1)
pp_recv_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
             void* x, const int64_t num_x_bytes,
             void* buffer, void* workspace,
             const int rank_idx, const int src_rank_idx,
             const int64_t num_max_tensor_bytes,
             const int num_max_inflight_tensors) {
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    auto& pp = *static_cast<layout::PPSignals*>(workspace);
    const auto [src_idx_in_local, local_idx_in_src] = get_pp_buffer_offset<kNumRanks>(src_rank_idx, rank_idx);
    EP_STATIC_ASSERT(kNumQPs > 0 and kNumQPs <= 2, "Invalid number of PP QPs");

    // Gin handle
    const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);

    // Buffer offsets
    const auto recv_count = pp.recv_count[src_idx_in_local];
    const auto slot_idx = recv_count % num_max_inflight_tensors;
    auto recv_buffer_ptr = math::advance_ptr(
        buffer, ((src_idx_in_local + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);

    // Wait arrival and copy from the buffer into a new tensor
    if (thread_idx < kNumQPs) {
        const auto& context_idx = thread_idx;
        const auto arrival_target = recv_count + 1;
        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
            const auto arrival = ptx::ld_acquire_sys(&pp.arrival[src_idx_in_local][context_idx]);
            if (arrival >= arrival_target)
                return true;
            if (is_last_check)
                printf("DeepEP PP recv timeout, rank %d, peer %d, context %d, arrival %lld, target %lld\n",
                       rank_idx, src_rank_idx, context_idx,
                       static_cast<long long>(arrival), static_cast<long long>(arrival_target));
            return false;
        });
    }
    __syncwarp();
    if (ptx::elect_one_sync()) {
        pp_tma_copy<kNumSMs, kNumSmemBytes>(recv_buffer_ptr, x, num_x_bytes, sm_idx);
    }
    cooperative_groups::this_grid().sync();

    // Release the slot back to the peer
    if (sm_idx == 0 and ptx::elect_one_sync()) {
        gin.signal<ncclTeamTagWorld>(
            src_rank_idx,
            ncclGin_StrongVASignalAdd {
                .signalWindow = nccl_window,
                .signalOffset = gin.get_sym_offset(&pp.release[local_idx_in_src]),
                .value = 1,
            });
        pp.recv_count[src_idx_in_local] += 1;
    }
}

}  // namespace deep_ep::pp

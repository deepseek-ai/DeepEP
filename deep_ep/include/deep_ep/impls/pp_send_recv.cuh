#pragma once

#include <cooperative_groups.h>
#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

template <int kNumRanks>
__device__ __forceinline__ std::pair<int, int> get_buffer_offset(
    const int& src_rank_idx, const int& dst_rank_idx) {
    const auto next_rank_idx = (src_rank_idx + 1) % kNumRanks;
    return dst_rank_idx == next_rank_idx ? std::make_pair(0, 1) : std::make_pair(1, 0);
}

template <int kNumQPs>
__device__ __forceinline__ std::pair<int64_t, int> get_pp_send_chunk(
    const int& qp_idx, const int64_t& num_bytes) {
    EP_STATIC_ASSERT(kNumQPs > 0, "Invalid number of QPs");

    const auto num_tma_blocks = num_bytes / ptx::kNumTMAAlignBytes;
    const auto num_tma_blocks_per_qp = math::ceil_div<int64_t>(num_tma_blocks, static_cast<int64_t>(kNumQPs));
    const auto start_block_idx = static_cast<int64_t>(qp_idx) * num_tma_blocks_per_qp;
    if (start_block_idx >= num_tma_blocks)
        return {0, 0};

    const auto end_block_idx = std::min(start_block_idx + num_tma_blocks_per_qp, num_tma_blocks);
    const auto offset = start_block_idx * ptx::kNumTMAAlignBytes;
    const auto chunk_bytes = static_cast<int>((end_block_idx - start_block_idx) * ptx::kNumTMAAlignBytes);
    return {offset, chunk_bytes};
}

template <int64_t kNumTimeoutCycles, typename timeout_print_t>
__device__ __forceinline__ void check_signal(
    const handle::NCCLGin& gin,
    const ncclGinSignal_t& signal_idx,
    const int64_t& target,
    const timeout_print_t& timeout_print) {
    const auto gdaki = static_cast<struct ncclGinGdakiGPUContext*>(gin.gin._ginHandle) + gin.gin.contextId;
    const auto signal_ptr = reinterpret_cast<int64_t*>(
        __ldg(reinterpret_cast<int64_t*>(&gdaki->signals_table.buffer))) + signal_idx;
    comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
        const auto signal = ptx::ld_acquire_sys<int64_t>(signal_ptr);
        if (signal >= target)
            return true;

        if (is_last_check)
            timeout_print();
        return false;
    });
}

template <int kNumSMs,
          int kNumSmemBytes,
          int kNumStages = 2,
          int kNumTMABytesPerStage = math::constexpr_align<int, false>(
              (kNumSmemBytes - kNumStages * sizeof(ptx::mbarrier)) / kNumStages, ptx::kNumTMAAlignBytes),
          int kNumTMABlocksPerStage = kNumTMABytesPerStage / ptx::kNumTMAAlignBytes>
__device__ __forceinline__ void tma_copy(
    void* src_ptr, void* dst_ptr,
    const int64_t& num_bytes, const int& sm_idx) {
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
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
    EP_DEVICE_ASSERT(num_bytes % ptx::kNumTMAAlignBytes == 0);
    const auto num_tma_blocks = num_bytes / ptx::kNumTMAAlignBytes;
    const auto num_tma_blocks_per_sm = math::ceil_div<int64_t>(num_tma_blocks, kNumSMs);
    const auto start_block_idx = sm_idx * num_tma_blocks_per_sm;
    const auto end_block_idx = std::min(start_block_idx + num_tma_blocks_per_sm, num_tma_blocks);
    const auto num_iterations = math::ceil_div<int64_t>(end_block_idx - start_block_idx, kNumTMABlocksPerStage);

    auto get_iter_info = [&](const int64_t& iter_idx) {
        const auto i = start_block_idx + iter_idx * kNumTMABlocksPerStage;
        const auto offset = i * ptx::kNumTMAAlignBytes;
        const auto num_transaction_bytes =
            std::min<int>(kNumTMABlocksPerStage, end_block_idx - i) * ptx::kNumTMAAlignBytes;
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
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);
    const auto [local_idx_in_dst, dst_idx_in_local] = get_buffer_offset<kNumRanks>(rank_idx, dst_rank_idx);

    // Gin handle
    const auto signal_gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);

    // Buffer offsets
    const auto send_count_ptr = workspace_layout.get_pp_send_count_ptr(dst_idx_in_local);
    const auto send_count = __ldg(send_count_ptr);
    const auto slot_idx = send_count % num_max_inflight_tensors;
    auto send_buffer_ptr = math::advance_ptr(
        buffer, ((dst_idx_in_local + 2) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);
    auto recv_buffer_ptr = math::advance_ptr(
        buffer, ((local_idx_in_dst + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);

    // Wait buffer slot release and do TMA
    if (ptx::elect_one_sync()) {
        check_signal<kNumTimeoutCycles>(
            signal_gin,
            static_cast<ncclGinSignal_t>(kNumRanks + dst_idx_in_local + 2),
            send_count - num_max_inflight_tensors + 1,
            // TODO: print more info, and control the SM who prints it
            []() { printf("DeepEP PP send timeout, recv buffer is full"); }
        );
        tma_copy<kNumSMs, kNumSmemBytes>(x, send_buffer_ptr, num_x_bytes, sm_idx);
    }
    cooperative_groups::this_grid().sync();

    // Issue RDMA puts across all requested QPs. Each QP signals completion independently,
    // because there is no cross-QP ordering guarantee.
    if (ptx::elect_one_sync()) {
        for (int qp_idx = sm_idx; qp_idx < kNumQPs; qp_idx += kNumSMs) {
            const auto [offset, chunk_bytes] = get_pp_send_chunk<kNumQPs>(qp_idx, num_x_bytes);
            const auto data_gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
            const auto recv_signal = static_cast<ncclGinSignal_t>(kNumRanks + local_idx_in_dst);

            if (chunk_bytes == 0) {
                data_gin.signal<ncclTeamTagWorld>(dst_rank_idx, ncclGin_SignalInc{recv_signal});
            } else {
                data_gin.put<ncclTeamTagWorld>(
                    math::advance_ptr(recv_buffer_ptr, offset),
                    math::advance_ptr(send_buffer_ptr, offset),
                    chunk_bytes, dst_rank_idx,
                    0,
                    ncclGin_SignalInc{recv_signal});
            }
        }
    }

    if (sm_idx == 0 and ptx::elect_one_sync()) {
        *send_count_ptr += 1;
    }
}

template <int kNumSMs,
          int kNumRanks,
          int kNumSmemBytes,
          int kNumQPs,
          int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(32, 1)
pp_recv_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
             void* x, int64_t num_x_bytes,
             void* buffer, void* workspace,
             const int rank_idx, const int src_rank_idx,
             const int64_t num_max_tensor_bytes,
             const int num_max_inflight_tensors) {
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);
    const auto [src_idx_in_local, local_idx_in_src] = get_buffer_offset<kNumRanks>(src_rank_idx, rank_idx);

    // QP 0 is used for release notification. Payload completion signals are
    // polled on the same QP/context that issued each RDMA put.
    const auto signal_gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);

    // Buffer offsets
    const auto recv_count_ptr = workspace_layout.get_pp_recv_count_ptr(src_idx_in_local);
    const auto recv_count = __ldg(recv_count_ptr);
    const auto slot_idx = recv_count % num_max_inflight_tensors;
    const auto recv_buffer_ptr = math::advance_ptr(
        buffer, ((src_idx_in_local + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);

    // Copy from the buffer into a new tensor
    if (ptx::elect_one_sync()) {
        for (int qp_idx = sm_idx; qp_idx < kNumQPs; qp_idx += kNumSMs) {
            const auto data_gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
            check_signal<kNumTimeoutCycles>(
                data_gin,
                static_cast<ncclGinSignal_t>(src_idx_in_local + kNumRanks),
                recv_count + 1,
                // TODO: print more info, and control the SM who prints it
                []() { printf("DeepEP PP recv timeout, recv buffer is empty\n"); }
            );
        }
    }
    cooperative_groups::this_grid().sync();

    if (ptx::elect_one_sync()) {
        tma_copy<kNumSMs, kNumSmemBytes>(recv_buffer_ptr, x, num_x_bytes, sm_idx);
    }
    cooperative_groups::this_grid().sync();

    // TODO: add a comment
    if (sm_idx == 0 and ptx::elect_one_sync()) {
        signal_gin.signal<ncclTeamTagWorld>(
            src_rank_idx,
            ncclGin_SignalInc(static_cast<ncclGinSignal_t>(kNumRanks + 2 + local_idx_in_src))
        );
        *recv_count_ptr += 1;
    }
}

} // namespace deep_ep::elastic

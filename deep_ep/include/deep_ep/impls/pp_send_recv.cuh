#pragma once

// MIT License
//
// Copyright (c) 2025 DeepSeek
// Changes and additions copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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

template <int64_t kNumTimeoutCycles, typename timeout_print_t>
__device__ __forceinline__ void check_signal(
    const handle::NCCLGin& gin,
    const ncclGinSignal_t& signal_idx,
    const int64_t& target,
    const timeout_print_t& timeout_print) {
    // TODO(NCCL): Using the official NCCL wait signal API, after they added timeout check.
    comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
        const auto signal = static_cast<int64_t>(
            gin.gin.readSignal(signal_idx, 64, cuda::memory_order_acquire));
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
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);

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
            gin,
            static_cast<ncclGinSignal_t>(kNumRanks + dst_idx_in_local + 2),
            send_count - num_max_inflight_tensors + 1,
            // TODO: print more info, and control the SM who prints it
            []() { printf("DeepEP PP send timeout, recv buffer is full"); }
        );
        tma_copy<kNumSMs, kNumSmemBytes>(x, send_buffer_ptr, num_x_bytes, sm_idx);
    }
    cooperative_groups::this_grid().sync();

    // Issue RDMA put
    if (sm_idx == 0 and ptx::elect_one_sync()) {
        gin.put<ncclTeamTagWorld>(
            recv_buffer_ptr,
            send_buffer_ptr,
            num_x_bytes, dst_rank_idx,
            0,
            // TODO: is this signal highly optimized?
            ncclGin_SignalInc(static_cast<ncclGinSignal_t>(local_idx_in_dst + kNumRanks)));
        *send_count_ptr += 1;
    }
}

template <int kNumSMs,
          int kNumRanks,
          int kNumSmemBytes,
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

    // Gin handle
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);

    // Buffer offsets
    const auto recv_count_ptr = workspace_layout.get_pp_recv_count_ptr(src_idx_in_local);
    const auto recv_count = __ldg(recv_count_ptr);
    const auto slot_idx = recv_count % num_max_inflight_tensors;
    const auto recv_buffer_ptr = math::advance_ptr(
        buffer, ((src_idx_in_local + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);

    // Copy from the buffer into a new tensor
    if (ptx::elect_one_sync()) {
        check_signal<kNumTimeoutCycles>(
            gin,
            static_cast<ncclGinSignal_t>(src_idx_in_local + kNumRanks),
            recv_count + 1,
            // TODO: print more info, and control the SM who prints it
            []() { printf("DeepEP PP recv timeout, recv buffer is empty\n"); }
        );
        tma_copy<kNumSMs, kNumSmemBytes>(recv_buffer_ptr, x, num_x_bytes, sm_idx);
    }
    cooperative_groups::this_grid().sync();

    // TODO: add a comment
    if (sm_idx == 0 and ptx::elect_one_sync()) {
        gin.signal<ncclTeamTagWorld>(
            src_rank_idx, ncclGin_SignalInc(static_cast<ncclGinSignal_t>(kNumRanks + local_idx_in_src + 2))
        );
        *recv_count_ptr += 1;
    }
}

} // namespace deep_ep::elastic

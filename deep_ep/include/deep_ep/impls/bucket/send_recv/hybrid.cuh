#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>
#include <deep_ep/layout/bucket/chunk.cuh>
#include <deep_ep/layout/bucket/send_recv.cuh>

namespace deep_ep::bucket {

template <int kNumNVLRanks>
__device__ __forceinline__ int32_t get_hybrid_target_signal(
    const int& entry_idx, const int& slot_step) {
    // Construct a unique signal for each step & entry
    return static_cast<int32_t>(
        slot_step * (kNumNVLRanks * layout::kNumMaxSendRecvEntries) + entry_idx + 1);
}

template <bool kIsScaleupNVLink,
          int kNumRDMARanks, int kNumNVLRanks,
          int kNumSMs, int kNumNVLWarps, int kNumForwardWarps,
          int kNumNVLChunkBytes, int kNumTMAStages,
          int kNumTotalSlots, int kNumSlotsPerEntry,
          int kNumQPs, int64_t kNumTimeoutCycles, int kContextIdx,
          int kNumThreads = (kNumNVLWarps + kNumForwardWarps) * 32>
__global__ void __launch_bounds__(kNumThreads, 1)
hybrid_send_recv(const ncclDevComm_t nccl_dev_comm,
                 const ncclWindow_t nccl_window,
                 void* workspace_ptr,
                 void* storage_ptr,
                 const int num_nvl_send_entries,
                 const int num_rdma_send_entries,
                 const __grid_constant__ layout::SendRecvList nvl_send_entries,
                 const __grid_constant__ layout::SendRecvList rdma_send_entries) {
    EP_STATIC_ASSERT(kNumNVLWarps > 0, "Hybrid send-recv requires at least one NVLink warp");
    EP_STATIC_ASSERT(kNumNVLChunkBytes % kNumTMAAlignmentBytes == 0,
                     "Hybrid send-recv TMA bytes must be aligned");
    EP_STATIC_ASSERT(kNumTMAStages > 0 and kNumTMAStages <= 32,
                     "Hybrid send-recv requires between 1 and 32 TMA stages");
    EP_STATIC_ASSERT(kNumQPs > 0, "Hybrid send-recv requires at least one QP");

    // Check slot distribution
    EP_STATIC_ASSERT(kNumTotalSlots > 0 and kNumTotalSlots <= layout::SendRecvSignals::kNumTotalSlots,
                     "Hybrid send-recv slots must fit within the signal capacity");
    EP_STATIC_ASSERT(kNumSlotsPerEntry > 0 and kNumTotalSlots % kNumSlotsPerEntry == 0,
                     "Hybrid send-recv slots must be divisible by slots per entry");
    EP_STATIC_ASSERT(kNumSMs * kNumForwardWarps * 32 >= kNumTotalSlots,
                     "Hybrid send-recv requires enough RDMA lanes for all slots");
    constexpr int kNumConcurrentEntries = kNumTotalSlots / kNumSlotsPerEntry;

    // Calculate chunk size according to the number of slots
    constexpr int kNumRDMAChunkBytes = layout::ChunkStorage::kNumBytes / kNumTotalSlots;
    EP_STATIC_ASSERT(kNumRDMAChunkBytes % kNumNVLChunkBytes == 0,
                     "Hybrid send-recv RDMA chunk size must be divisible by the TMA chunk size");
    EP_STATIC_ASSERT(kNumRDMAChunkBytes % kNumRDMAAlignmentBytes == 0,
                     "Hybrid send-recv RDMA chunk size must align with kNumRDMAAlignmentBytes");

    constexpr int kNumGlobalNVLWarps = kNumSMs * kNumNVLWarps;
    constexpr int kNumDirectRDMAQPs = std::min(kNumQPs, 2);

    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto lane_idx = ptx::get_lane_idx();
    const auto warp_idx = ptx::get_warp_idx();
    const auto global_warp_idx = sm_idx * kNumNVLWarps + warp_idx;
    const comm::NCCLGin gin(
        nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_GPU);
    const auto rdma_rank_idx = ncclTeamRail(nccl_dev_comm).rank;
    const auto nvl_rank_idx = ncclTeamLsa(nccl_dev_comm).rank;

    auto& workspace = *static_cast<layout::BucketWorkspace*>(workspace_ptr);
    auto& bucket_signals = workspace.signals[kContextIdx];
    auto& barrier_signals = bucket_signals.barrier;
    auto& forward_signals = bucket_signals.send_recv;

    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    const auto shared =
        layout::HybridSendRecvSharedLayout<kNumNVLWarps, kNumNVLChunkBytes, kNumTMAStages>(smem);
    auto* const num_forward_entries = shared.get_num_forward_entries_ptr();
    auto* const forward_entry_bases = shared.get_forward_entry_base_ptr(kNumNVLRanks);
    const auto chunks =
        layout::ChunkView1D<kNumTotalSlots, kNumRDMAChunkBytes>{workspace.chunk_storage};

    // Initialize counters and barriers
    const auto global_thread_idx = sm_idx * kNumThreads + thread_idx;
    if (warp_idx < kNumNVLWarps)
        for (int entry_idx = lane_idx; entry_idx < num_nvl_send_entries; entry_idx += 32)
            *shared.get_chunk_idx_ptr(entry_idx, warp_idx) = global_warp_idx;
    if (warp_idx < kNumNVLWarps and lane_idx < kNumTMAStages)
        ptx::mbarrier_init_with_fence(
            shared.get_mbarrier_ptr(warp_idx, lane_idx), 1);

    // Clear Signals
    for (int signal_idx = global_thread_idx; signal_idx < kNumTotalSlots;
         signal_idx += kNumSMs * kNumThreads) {
        forward_signals.head[signal_idx] = 0;
        forward_signals.tail[signal_idx] = 0;
    }

    // Find the entry range for each forwarding rank
    for (int dst_nvl_rank_idx = global_thread_idx; dst_nvl_rank_idx < kNumNVLRanks;
         dst_nvl_rank_idx += kNumSMs * kNumThreads) {
        int entry_offset = 0, num_entries = 0;
        for (int entry_idx = 0; entry_idx < num_nvl_send_entries; ++ entry_idx) {
            const auto& entry = nvl_send_entries[entry_idx];
            if (entry.dst_rank_idx / kNumNVLRanks != rdma_rank_idx and
                entry.dst_rank_idx % kNumNVLRanks == dst_nvl_rank_idx) {
                if (num_entries == 0)
                    entry_offset = entry_idx;
                forward_signals.forward_entries[entry_idx] = entry;
                ++ num_entries;
            }
        }
        forward_signals.forward_entry_offsets[dst_nvl_rank_idx] = entry_offset;
        forward_signals.num_forward_entries[dst_nvl_rank_idx] = num_entries;
    }

    // Synchronize to ensure transfer buffers can be reused
    comm::gpu_barrier<kIsScaleupNVLink, kNumRDMARanks, kNumNVLRanks,
                      kNumSMs, kNumThreads, kFlushAllAllocatedQPs, kNumTimeoutCycles,
                      comm::kKernelBarrierTag, true, true, true>(
        gin, barrier_signals, rdma_rank_idx, nvl_rank_idx, sm_idx, thread_idx);

    if (sm_idx == 0) {
        // Direct RDMA
        #pragma unroll
        for (int qp_idx = 0; qp_idx < kNumDirectRDMAQPs; ++ qp_idx) {
            const comm::NCCLGin gin(
                nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_GPU);
            for (int entry_idx = thread_idx; entry_idx < num_rdma_send_entries; entry_idx += kNumThreads) {
                const auto& entry = rdma_send_entries[entry_idx];
                const auto num_bytes_per_qp = math::align<int64_t>(
                    entry.num_bytes / kNumDirectRDMAQPs, kNumRDMAAlignmentBytes);
                const auto offset = qp_idx * num_bytes_per_qp;
                const auto num_bytes = qp_idx + 1 == kNumDirectRDMAQPs ?
                    entry.num_bytes - offset : num_bytes_per_qp;
                if (num_bytes > 0)
                    gin.put<ncclTeamTagRail>(
                        math::advance_ptr(storage_ptr, entry.dst_offset + offset),
                        math::advance_ptr(entry.src_ptr, offset),
                        num_bytes, entry.dst_rank_idx);
            }
        }
    }

    // Cache entry counts forwarded by this rank
    for (int src_nvl_rank_idx = thread_idx; src_nvl_rank_idx < kNumNVLRanks; src_nvl_rank_idx += kNumThreads)
        num_forward_entries[src_nvl_rank_idx] = gin.get_sym_ptr<ncclTeamTagLsa>(
            &forward_signals, src_nvl_rank_idx)->num_forward_entries[nvl_rank_idx];

    // Calculate the prefix sum
    // TODO: optimize to avoid the latency of O(num_nvl_ranks)
    for (int dst_nvl_rank_idx = thread_idx; dst_nvl_rank_idx < kNumNVLRanks;
         dst_nvl_rank_idx += kNumThreads) {
        int num_preceding_entries = 0;
        for (int src_nvl_rank_idx = 0; src_nvl_rank_idx < nvl_rank_idx; ++ src_nvl_rank_idx)
            num_preceding_entries += gin.get_sym_ptr<ncclTeamTagLsa>(
                &forward_signals, src_nvl_rank_idx)->num_forward_entries[dst_nvl_rank_idx];
        forward_entry_bases[dst_nvl_rank_idx] =
            num_preceding_entries - forward_signals.forward_entry_offsets[dst_nvl_rank_idx];
    }
    __syncthreads();

    if (warp_idx < kNumNVLWarps) {
        // Direct NVLink send and NVLink-to-RDMA forwarding
        ptx::arrival_phase phases[kNumTMAStages] = {};

        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
            bool has_pending = false;
            // Scan all entries
            for (int entry_base_idx = 0; entry_base_idx < num_nvl_send_entries; entry_base_idx += 32) {
                const auto entry_idx = entry_base_idx + lane_idx;
                bool ready = false;

                // Judge if the warp is ready to issue TMA
                if (entry_idx < num_nvl_send_entries) {
                    const auto& entry = nvl_send_entries[entry_idx];
                    const auto chunk_idx = *shared.get_chunk_idx_ptr(entry_idx, warp_idx);

                    // Calculate the dst slot idx and find out if it is empty
                    if (static_cast<int64_t>(chunk_idx) * kNumRDMAChunkBytes < entry.num_bytes) {
                        has_pending = true;
                        ready = entry.dst_rank_idx / kNumNVLRanks == rdma_rank_idx;
                        if (not ready) {
                            const auto forward_entry_idx = forward_entry_bases[entry.dst_rank_idx % kNumNVLRanks] + entry_idx;
                            const auto slot_step = chunk_idx / kNumSlotsPerEntry;
                            const auto target_signal =
                                get_hybrid_target_signal<kNumNVLRanks>(
                                    nvl_rank_idx * layout::kNumMaxSendRecvEntries + entry_idx, slot_step);
                            auto* const head_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(
                                forward_signals.get_head_ptr(forward_entry_idx % kNumConcurrentEntries * kNumSlotsPerEntry +
                                    chunk_idx % kNumSlotsPerEntry),
                                entry.dst_rank_idx % kNumNVLRanks);
                            ready = ptx::ld_acquire_sys(head_ptr) == target_signal;
                        }
                    }
                }

                // Issue ready entries one by one
                auto wip_mask = ptx::gather(ready);
                while (wip_mask) {
                    const auto issue_entry_idx = entry_base_idx + ptx::ffs(wip_mask);
                    wip_mask &= wip_mask - 1;
                    if (ptx::elect_one_sync()) {
                        const auto& entry = nvl_send_entries[issue_entry_idx];
                        auto* const chunk_idx_ptr = shared.get_chunk_idx_ptr(issue_entry_idx, warp_idx);
                        const auto chunk_idx = *chunk_idx_ptr;
                        const auto offset = static_cast<int64_t>(chunk_idx) * kNumRDMAChunkBytes;
                        const auto num_bytes = static_cast<int>(
                            min(static_cast<int64_t>(kNumRDMAChunkBytes), entry.num_bytes - offset));

                        // Copy to the destination
                        const bool do_forward = entry.dst_rank_idx / kNumNVLRanks != rdma_rank_idx;
                        const auto forward_entry_idx =
                            forward_entry_bases[entry.dst_rank_idx % kNumNVLRanks] + issue_entry_idx;
                        const auto slot_idx = forward_entry_idx % kNumConcurrentEntries * kNumSlotsPerEntry +
                            chunk_idx % kNumSlotsPerEntry;
                        auto* const dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(
                            do_forward ? chunks.get_chunk_ptr(slot_idx) :
                                math::advance_ptr(storage_ptr, entry.dst_offset + offset),
                            entry.dst_rank_idx % kNumNVLRanks);
                        int tma_chunk_idx = 0;
                        for (int64_t tma_offset = 0; tma_offset < num_bytes;
                             tma_offset += kNumNVLChunkBytes, ++ tma_chunk_idx) {
                            const auto buffer_idx = tma_chunk_idx % kNumTMAStages;
                            const auto num_tma_bytes = static_cast<int>(min(
                                static_cast<int64_t>(kNumNVLChunkBytes), num_bytes - tma_offset));
                            auto* const tma_buffer = shared.get_tma_buffer_ptr(warp_idx, buffer_idx);
                            auto* const mbar = shared.get_mbarrier_ptr(warp_idx, buffer_idx);

                            // NOTES: When NVLink bandwidth is full, store has larger latency than load
                            ptx::tma_store_wait_read<kNumTMAStages - 1>();
                            ptx::tma_load_1d(tma_buffer,
                                math::advance_ptr(entry.src_ptr, offset + tma_offset), mbar, num_tma_bytes);
                            ptx::mbarrier_arrive_and_set_tx(mbar, num_tma_bytes);
                            ptx::mbarrier_wait_and_flip_phase(mbar, phases[buffer_idx]);
                            ptx::tma_store_1d(math::advance_ptr(dst_ptr, tma_offset), tma_buffer, num_tma_bytes);
                            ptx::tma_store_commit();
                        }
                        ptx::tma_store_wait();

                        // Notify the dest that the chunk is ready
                        if (do_forward) {
                            const auto slot_step = chunk_idx / kNumSlotsPerEntry;
                            const auto target_signal =
                                get_hybrid_target_signal<kNumNVLRanks>(
                                    nvl_rank_idx * layout::kNumMaxSendRecvEntries + issue_entry_idx, slot_step);
                            auto* const tail_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(
                                forward_signals.get_tail_ptr(slot_idx),
                                entry.dst_rank_idx % kNumNVLRanks);
                            ptx::st_release_sys(tail_ptr, target_signal);
                        }

                        // Advance to the next chunk
                        *chunk_idx_ptr = chunk_idx + kNumGlobalNVLWarps;
                    }
                }
                __syncwarp();
            }

            // Check completion and timeout
            if (not ptx::any(has_pending))
                return true;
            if (is_last_check) {
                if (lane_idx == 0)
                    printf("DeepEP hybrid send-recv NVLink send timeout, RDMA: %d, NVLink: %d, "
                           "SM: %d, warp: %d\n",
                           rdma_rank_idx, nvl_rank_idx, sm_idx, warp_idx);
            }
            return false;
        });
    } else {
        // RDMA forwarding
        const auto qp_idx = thread_idx % kNumQPs;
        const comm::NCCLGin send_gin(
            nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_GPU);

        // Assign kNumSlotsPerEntry slots to each entry, shared across source ranks.
        const auto slot_idx = (sm_idx * kNumForwardWarps + warp_idx - kNumNVLWarps) * 32 + lane_idx;
        auto chunk_idx = slot_idx % kNumSlotsPerEntry;
        int src_nvl_rank_idx = 0;
        int src_entry_idx = slot_idx / kNumSlotsPerEntry;
        bool head_published = false;
        int entry_idx = 0;
        layout::SendRecvEntry forward_entry;
        const auto* src_signals = gin.get_sym_ptr<ncclTeamTagLsa>(
            &forward_signals, src_nvl_rank_idx);

        const auto load_entry = [&]() {
            // Carry the remaining stride across ranks using their actual entry counts.
            while (slot_idx < kNumTotalSlots and src_nvl_rank_idx < kNumNVLRanks and
                   src_entry_idx >= num_forward_entries[src_nvl_rank_idx]) {
                src_entry_idx -= num_forward_entries[src_nvl_rank_idx];
                if (++ src_nvl_rank_idx < kNumNVLRanks)
                    src_signals = gin.get_sym_ptr<ncclTeamTagLsa>(&forward_signals, src_nvl_rank_idx);
            }
            if (slot_idx < kNumTotalSlots and src_nvl_rank_idx < kNumNVLRanks) {
                entry_idx = src_signals->forward_entry_offsets[nvl_rank_idx] + src_entry_idx;
                forward_entry = src_signals->forward_entries[entry_idx];
            }
        };
        load_entry();

        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
            const bool has_pending =
                slot_idx < kNumTotalSlots and src_nvl_rank_idx < kNumNVLRanks;
            if (has_pending) {
                const auto offset = static_cast<int64_t>(chunk_idx) * kNumRDMAChunkBytes;

                if (offset >= forward_entry.num_bytes) {
                    // Move to the next entry
                    src_entry_idx += kNumConcurrentEntries;
                    load_entry();
                    chunk_idx = slot_idx % kNumSlotsPerEntry;
                    head_published = false;
                } else {
                    // Allow the sender to fill this slot
                    const auto slot_step = chunk_idx / kNumSlotsPerEntry;
                    const auto target_signal =
                        get_hybrid_target_signal<kNumNVLRanks>(
                            src_nvl_rank_idx * layout::kNumMaxSendRecvEntries + entry_idx, slot_step);
                    auto* const head_ptr =
                        forward_signals.get_head_ptr(slot_idx);
                    if (not head_published) {
                        ptx::st_release_sys(head_ptr, target_signal);
                        head_published = true;
                    }

                    // Check whether the NVLink copy has finished
                    const auto tail = ptx::ld_acquire_sys(
                        forward_signals.get_tail_ptr(slot_idx));
                    if (tail == target_signal) {
                        const auto num_bytes = static_cast<int64_t>(min(
                            static_cast<int64_t>(kNumRDMAChunkBytes),
                            forward_entry.num_bytes - offset));

                        // Forward the chunk to the destination over RDMA
                        send_gin.put<ncclTeamTagRail>(
                            math::advance_ptr(storage_ptr, forward_entry.dst_offset + offset),
                            chunks.get_chunk_ptr(slot_idx),
                            num_bytes, forward_entry.dst_rank_idx / kNumNVLRanks);

                        // Flush the particular QP before reusing the slot
                        ncclGinRequest_t request;
                        send_gin.flush_async<ncclTeamTagRail>(
                            forward_entry.dst_rank_idx / kNumNVLRanks, &request);
                        send_gin.wait(request);
                        chunk_idx += kNumSlotsPerEntry;
                        head_published = false;
                    }
                }
            }
            // Check completion and timeout
            if (not ptx::any(has_pending))
                return true;
            if (is_last_check) {
                if (lane_idx == 0)
                    printf("DeepEP hybrid send-recv forwarding timeout, RDMA: %d, NVLink: %d, "
                           "SM: %d\n",
                           rdma_rank_idx, nvl_rank_idx, sm_idx);
            }
            return false;
        });
    }

    // Synchronize to ensure arrival
    // NOTES: All other requests have been flushed, only needs to flush direct RDMA QPs
    comm::gpu_barrier<kIsScaleupNVLink, kNumRDMARanks, kNumNVLRanks,
                      kNumSMs, kNumThreads, kNumDirectRDMAQPs, kNumTimeoutCycles,
                      comm::kKernelBarrierTag, true, true, false>(
        gin, barrier_signals, rdma_rank_idx, nvl_rank_idx, sm_idx, thread_idx);
}

}  // namespace deep_ep::bucket

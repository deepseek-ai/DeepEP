#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/bucket/chunk.cuh>

namespace deep_ep::layout {

static constexpr int kNumMaxSendRecvEntries = 500;

struct alignas(32) SendRecvEntry {
    int src_rank_idx;
    int dst_rank_idx;
    void* src_ptr;
    int64_t dst_offset;
    int64_t num_bytes;
};

struct SendRecvList {
    SendRecvEntry entries[kNumMaxSendRecvEntries] = {};

    __forceinline__ __device__ __host__ SendRecvEntry& operator[](const int& entry_idx) {
        return entries[entry_idx];
    }

    __forceinline__ __device__ __host__ const SendRecvEntry& operator[](const int& entry_idx) const {
        return entries[entry_idx];
    }
};

struct alignas(kNumRDMAAlignmentBytes) SendRecvSignals {
    static constexpr int kNumTotalSlots = 1024;

    // Forwarding ranges in the host-sorted NVLink send list.
    int num_forward_entries[kNumMaxNVLRanks];
    int forward_entry_offsets[kNumMaxNVLRanks];
    SendRecvEntry forward_entries[kNumMaxSendRecvEntries];

    // Recv rank send head signals to indicate the slot is freed
    int32_t head[kNumTotalSlots];
    // Send rank send tail signals to indicate the slot is occupied
    int32_t tail[kNumTotalSlots];

    __forceinline__ __device__ int32_t* get_head_ptr(const int& slot_idx) {
        return &head[slot_idx];
    }

    __forceinline__ __device__ int32_t* get_tail_ptr(const int& slot_idx) {
        return &tail[slot_idx];
    }
};

template <int kNumNVLWarps, int kNumNVLChunkBytes, int kNumTMAStages>
struct HybridSendRecvSharedLayout {
    void* ptr;

    __forceinline__ __device__ __host__
    explicit HybridSendRecvSharedLayout(void* ptr): ptr(ptr) {}

    __forceinline__ __device__ __host__ static constexpr int64_t get_num_bytes(const int num_nvl_ranks) {
        return math::constexpr_align<int64_t>(
            static_cast<int64_t>(kNumNVLWarps) * kNumTMAStages * kNumNVLChunkBytes +
                static_cast<int64_t>(kNumNVLWarps) * kNumTMAStages * sizeof(ptx::mbarrier) +
                static_cast<int64_t>(kNumMaxSendRecvEntries) * kNumNVLWarps * sizeof(int32_t) +
                static_cast<int64_t>(num_nvl_ranks) * 2 * sizeof(int32_t),
            kNumTMAAlignmentBytes);
    }

    __forceinline__ __device__ __host__ ptx::mbarrier* get_mbarrier_ptr(
        const int warp_idx = 0, const int buffer_idx = 0) const {
        auto* const barriers = reinterpret_cast<ptx::mbarrier*>(
            get_tma_buffer_ptr(kNumNVLWarps));
        return barriers + warp_idx * kNumTMAStages + buffer_idx;
    }

    __forceinline__ __device__ __host__ int8_t* get_tma_buffer_ptr(
        const int warp_idx, const int buffer_idx = 0) const {
        return static_cast<int8_t*>(ptr) +
            static_cast<int64_t>(warp_idx * kNumTMAStages + buffer_idx) *
                kNumNVLChunkBytes;
    }

    __forceinline__ __device__ __host__ int32_t* get_chunk_idx_ptr(
        const int entry_idx = 0, const int warp_idx = 0) const {
        return reinterpret_cast<int32_t*>(get_mbarrier_ptr(kNumNVLWarps)) +
            entry_idx * kNumNVLWarps + warp_idx;
    }

    __forceinline__ __device__ __host__ int32_t* get_num_forward_entries_ptr() const {
        return get_chunk_idx_ptr(kNumMaxSendRecvEntries);
    }

    __forceinline__ __device__ __host__ int32_t* get_forward_entry_base_ptr(const int num_nvl_ranks) const {
        return get_num_forward_entries_ptr() + num_nvl_ranks;
    }
};

}  // namespace deep_ep::layout

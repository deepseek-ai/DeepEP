#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

namespace deep_ep::layout {

struct alignas(kNumRDMAAlignmentBytes) ReduceScatterSignals {
    static constexpr int kNumTotalSlots = 8192;
    int64_t heads[kNumTotalSlots];
    int64_t tails[kNumTotalSlots];

    template <int kNumSlotsPerRank>
    __forceinline__ __device__ int64_t* get_head_ptr(const int& rank_idx, const int& slot_idx) {
        return heads + rank_idx * kNumSlotsPerRank + slot_idx;
    }

    template <int kNumSlotsPerRank>
    __forceinline__ __device__ int64_t* get_tail_ptr(const int& rank_idx, const int& slot_idx) {
        return tails + rank_idx * kNumSlotsPerRank + slot_idx;
    }
};
EP_STATIC_ASSERT(sizeof(ReduceScatterSignals) <= kNumMaxSignalBytes, "Too many signals");

}  // namespace deep_ep::layout

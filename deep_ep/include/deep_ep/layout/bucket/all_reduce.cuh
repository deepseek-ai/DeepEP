#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

namespace deep_ep::layout {

struct alignas(kNumRDMAAlignmentBytes) AllReduceSignals {
    static constexpr int kNumTotalSlots = 16384;
    int64_t counters[kNumTotalSlots];
};
EP_STATIC_ASSERT(sizeof(AllReduceSignals) <= kNumMaxSignalBytes, "Too many signals");

}  // namespace deep_ep::layout

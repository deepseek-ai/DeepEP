#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

namespace deep_ep::layout {

struct alignas(kNumRDMAAlignmentBytes) CopyEngineSignals {
    static constexpr uint64_t kArrivedSignal = 1;

    uint64_t arrived;
    uint64_t prologue[kNumMaxNVLRanks];
    uint64_t epilogue[kNumMaxNVLRanks];
};
EP_STATIC_ASSERT(sizeof(CopyEngineSignals) <= kNumMaxSignalBytes, "Too many signals");

}  // namespace deep_ep::layout

#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

namespace deep_ep::layout {

struct alignas(kNumRDMAAlignmentBytes) CopyEngineSignals {
    uint64_t base[kNumMaxNVLRanks];
};
EP_STATIC_ASSERT(sizeof(CopyEngineSignals) <= kNumMaxSignalBytes, "Too many signals");

}  // namespace deep_ep::layout

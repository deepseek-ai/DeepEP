#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

namespace deep_ep::layout {

struct alignas(kNumRDMAAlignmentBytes) AllGatherSignals {
    static constexpr int kNumQPsPerChunk = 8;
    static constexpr int kNumMaxChunks = 8192;
    uint64_t tails[kNumQPsPerChunk][kNumMaxChunks];
};
EP_STATIC_ASSERT(sizeof(AllGatherSignals) <= kNumMaxSignalBytes, "Too many signals");

}  // namespace deep_ep::layout

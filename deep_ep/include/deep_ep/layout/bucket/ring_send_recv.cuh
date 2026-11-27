#pragma once

#include <cstdint>

#include <deep_ep/comm/bucket.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/layout/bucket/chunk.cuh>

namespace deep_ep::layout {

struct alignas(kNumRDMAAlignmentBytes) RingSendRecvSignals {
    static constexpr int kNumQPs = 2;
    static constexpr int kNumChunkBytes = 8 * 1024 * 1024;
    static constexpr int kNumMaxChunks = ChunkStorage::kNumBytes / kNumChunkBytes + comm::kNumMaxBuckets;
    EP_STATIC_ASSERT(kNumChunkBytes % (kNumQPs * kNumRDMAAlignmentBytes) == 0, "Invalid chunk alignment");
    uint64_t tails[kNumQPs][kNumMaxChunks];
};
EP_STATIC_ASSERT(sizeof(RingSendRecvSignals) <= kNumMaxSignalBytes, "Too many signals");

}  // namespace deep_ep::layout

#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/layout/common/barrier.cuh>

namespace deep_ep::layout {

// Pipeline-parallel send/recv state, indexed by the ring direction
struct alignas(kNumRDMAAlignmentBytes) PPSignals {
    BarrierSignals barrier;
    int64_t send_count[2] = {};
    int64_t recv_count[2] = {};
    int64_t arrival[2][2] = {};  // Ring direction, QP index
    int64_t release[2] = {};
};
EP_STATIC_ASSERT(sizeof(PPSignals) <= kNumMaxSignalBytes, "Too many signals");

}  // namespace deep_ep::layout

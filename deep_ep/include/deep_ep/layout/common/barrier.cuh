#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

namespace deep_ep::layout {

struct alignas(kNumRDMAAlignmentBytes) BarrierSignals {
    unsigned long long nvl_barrier_counter;
    int nvl_barrier_signals[2];

    __forceinline__ __device__ __host__ unsigned long long* get_nvl_barrier_counter_ptr() {
        return &nvl_barrier_counter;
    }

    __forceinline__ __device__ __host__ int* get_nvl_barrier_signal_ptr(const int& phase) {
        return nvl_barrier_signals + phase;
    }
};
EP_STATIC_ASSERT(sizeof(BarrierSignals) <= kNumMaxSignalBytes, "Too many signals");

}  // namespace deep_ep::layout

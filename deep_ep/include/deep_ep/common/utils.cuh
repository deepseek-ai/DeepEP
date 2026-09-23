#pragma once

#include <cstdint>

namespace deep_ep::utils {

struct Identity {
    template <typename value_t>
    __device__ __forceinline__ value_t operator()(value_t value) const {
        return value;
    }
};

template <int kNumThreadGroups, int kNumThreadsPerGroup, int kNumElemsPerThread,
          typename index_t, typename ld_ptr_t, typename st_ptr_t,
          typename ld_func_t, typename st_func_t,
          typename process_func_t = Identity,
          int kThreadStride = kNumThreadGroups * kNumThreadsPerGroup>
__device__ __forceinline__ void iterate_ld_st(const int& global_thread_idx,
                                              const index_t& num_elems,
                                              const ld_ptr_t& ld_ptr, const st_ptr_t& st_ptr,
                                              const ld_func_t& ld_func,
                                              const st_func_t& st_func,
                                              const process_func_t& process_func = process_func_t{}) {
    constexpr int kStride = kNumElemsPerThread * kThreadStride;
    constexpr int kMaxElemOffset = (kNumElemsPerThread - 1) * kThreadStride;
    using ld_value_t = decltype(ld_func(ld_ptr));
    using st_value_t = decltype(process_func(ld_func(ld_ptr)));

    // Main loop
    index_t elem_begin_idx = global_thread_idx;
    for (; elem_begin_idx + kMaxElemOffset < num_elems; elem_begin_idx += kStride) {
        const auto ld_base = ld_ptr + elem_begin_idx;
        ld_value_t ld_values[kNumElemsPerThread];
        #pragma unroll
        for (int i = 0; i < kNumElemsPerThread; ++ i) {
            const auto elem_offset = i * kThreadStride;
            ld_values[i] = ld_func(ld_base + elem_offset);
        }
        st_value_t st_values[kNumElemsPerThread];
        #pragma unroll
        for (int i = 0; i < kNumElemsPerThread; ++ i) {
            st_values[i] = process_func(ld_values[i]);
        }
        const auto st_base = st_ptr + elem_begin_idx;
        #pragma unroll
        for (int i = 0; i < kNumElemsPerThread; ++ i) {
            const auto elem_offset = i * kThreadStride;
            st_func(st_base + elem_offset, st_values[i]);
        }
    }

    // Tail
    if (elem_begin_idx < num_elems) {
        const auto ld_base = ld_ptr + elem_begin_idx;
        const auto num_remaining_elems = static_cast<int>(num_elems - elem_begin_idx);
        ld_value_t ld_values[kNumElemsPerThread];
        #pragma unroll
        for (int i = 0; i < kNumElemsPerThread; ++ i) {
            const auto elem_offset = i * kThreadStride;
            ld_values[i] = ld_func(ld_base + elem_offset, num_remaining_elems, elem_offset);
        }
        st_value_t st_values[kNumElemsPerThread];
        #pragma unroll
        for (int i = 0; i < kNumElemsPerThread; ++ i) {
            st_values[i] = process_func(ld_values[i]);
        }
        const auto st_base = st_ptr + elem_begin_idx;
        #pragma unroll
        for (int i = 0; i < kNumElemsPerThread; ++ i) {
            const auto elem_offset = i * kThreadStride;
            st_func(st_base + elem_offset, st_values[i], num_remaining_elems, elem_offset);
        }
    }
}

}  // namespace deep_ep::utils

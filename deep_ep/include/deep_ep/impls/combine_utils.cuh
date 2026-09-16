#pragma once

// MIT License
//
// Copyright (c) 2025 DeepSeek
// Changes and additions copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <stdexcept>


namespace deep_ep::elastic {

// Packing for `token_map_at_dispatch[token][k]`. Written by dispatch, read by the combine epilogue.
// Layout (int32):
//   bits [30:26] — dst_scaleout_rank (5 bits, 0..31)
//   bits [25:12] — slot within `recv[dst_scaleout_rank][channel × slots_per_channel + slot]` (14 bits, 0..16383)
//   bits [11:0]  — channel index (12 bits, 0..4095)
// Field widths add up to 31, leaving bit 31 clear on any valid entry. The epilogue reuses the
// `-1`/`< 0` sentinel convention that `compute_topk_slots` produces for trailing (unused) `topk_slot_idx`
// entries — `combine_reduce` predicates loads via `ldg_with_gez_pred`'s sign check. Keeping bit 31
// clear on valid entries preserves this shared sentinel invariant so no extra masking is needed.
constexpr int kCombineRecvMapChannelBits = 12;
constexpr int kCombineRecvMapSlotBits    = 14;
constexpr int kCombineRecvMapRankBits    = 5;
constexpr int kCombineRecvMapChannelMask = (1 << kCombineRecvMapChannelBits) - 1;
constexpr int kCombineRecvMapSlotMask    = (1 << kCombineRecvMapSlotBits) - 1;
constexpr int kCombineRecvMapRankMask    = (1 << kCombineRecvMapRankBits) - 1;
constexpr int kCombineRecvMapSlotShift    = kCombineRecvMapChannelBits;
constexpr int kCombineRecvMapRankShift    = kCombineRecvMapChannelBits + kCombineRecvMapSlotBits;

__device__ __host__ __forceinline__
int pack_combine_recv_addr(const int& rank, const int& slot, const int& channel) {
    return (rank << kCombineRecvMapRankShift) | (slot << kCombineRecvMapSlotShift) | channel;
}

__device__ __host__ __forceinline__
void unpack_combine_recv_addr(const int& packed, int& rank, int& slot, int& channel) {
    rank    = (packed >> kCombineRecvMapRankShift) & kCombineRecvMapRankMask;
    slot    = (packed >> kCombineRecvMapSlotShift) & kCombineRecvMapSlotMask;
    channel = packed & kCombineRecvMapChannelMask;
}

template <bool kAllowMultipleReduction, int kNumRanks, int kNumTopk>
constexpr bool use_rank_layout() {
    if constexpr (not kAllowMultipleReduction)
        return false;
    return kNumRanks <= kNumTopk;
}

template <bool kAllowMultipleReduction, int kNumRanks, int kNumTopk>
constexpr int get_num_tokens_in_layout() {
    return use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>() ? kNumRanks : kNumTopk;
}

template <int kLength, int kMaxUnrollFactor, int kWarpSize = 32>
constexpr int get_max_unroll_factor() {
    for (int i = kMaxUnrollFactor; i >= 1; -- i)
        if (kLength % (kWarpSize * i) == 0)
            return i;
    throw std::logic_error("Invalid length, cannot find unrolling factor");
}

// Determine the vector type for combine loads/stores based on arch and hidden size alignment
template <int kHiddenBytes>
struct CombineVecTraits {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    // On SM100+, use `longlong4_t` (32 bytes) if hidden is aligned, otherwise fall back to `int4` (16 bytes)
    // NOTES: we observe some performance degrade with `longlong4_t`, temporarily disable it
    static constexpr bool kUseLonglong4 = false;
    using vec_t = std::conditional_t<kUseLonglong4, longlong4_t, int4>;
#else
    using vec_t = int4;
#endif
};

template <int kNumValidTopk, typename fetch_func_t>
__device__ __forceinline__
void compute_topk_slots(int (&topk_slot_idx)[kNumValidTopk], uint32_t mask,
                        const fetch_func_t& fetch_func) {
    #pragma unroll
    for (int k = 0; k < kNumValidTopk; ++ k) {
        const int lowest_idx = __ffs(mask) - 1;
        // Here we perform the exchange unconditionally to avoid `BRA.DIV`
        const auto fetched = fetch_func(lowest_idx);
        mask &= mask - 1;
        topk_slot_idx[k] = lowest_idx >= 0 ? fetched : -1;
    }
}

template <int kHiddenVec, int kUnrollFactor, int kNumExpectedTopk, int kNumValidTopk,
          typename vec_t, typename get_src_buffer_ptr_func_t, typename wait_buffer_func_t>
__device__ __forceinline__
void combine_reduce(const int& lane_idx, int (&topk_slot_idx)[kNumValidTopk],
                    vec_t* dst_buffer_ptr,
                    const get_src_buffer_ptr_func_t& get_src_buffer_ptr_func,
                    const wait_buffer_func_t& wait_buffer_func,
                    vec_t* bias_0 = nullptr, vec_t* bias_1 = nullptr) {
    constexpr int kNumElemsPerVec = sizeof(vec_t) / sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(kNumElemsPerVec % 2 == 0, "Invalid number of elements");
    EP_STATIC_ASSERT(kHiddenVec % (kUnrollFactor * 32) == 0, "Invalid unrolling");

    // We use BF16 add as much as possible, as casting is slow
    const bool enable_hadd_bypass =
        (bias_0 == nullptr and bias_1 == nullptr) and
        (kNumValidTopk <= 2 or topk_slot_idx[2] < 0);
    EP_STATIC_ASSERT(kNumValidTopk > 0, "Invalid top-k");

    if (enable_hadd_bypass) {
        #pragma unroll 1
        for (int i = 0; i < kHiddenVec / (kUnrollFactor * 32); ++ i) {
            // Read values 0
            const auto slot_0 = topk_slot_idx[0];
            const auto src_base_ptr_0 = get_src_buffer_ptr_func(slot_0);
            vec_t values_0[kUnrollFactor] = {};
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                values_0[j] = ptx::ldg_with_gez_pred(
                    src_base_ptr_0 + (i * (kUnrollFactor * 32) + j * 32 + lane_idx), slot_0);
            }

            // Read values 1
            vec_t values_1[kUnrollFactor] = {};
            const auto slot_1 = kNumValidTopk == 1 ? -1 : topk_slot_idx[1];
            const auto src_base_ptr_1 = get_src_buffer_ptr_func(slot_1);
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                values_1[j] = ptx::ldg_with_gez_pred(
                    src_base_ptr_1 + (i * (kUnrollFactor * 32) + j * 32 + lane_idx), slot_1);
            }

            // Wait buffer releases for the first write
            if (i == 0)
                wait_buffer_func();

            // Reduce into shared memory
            const auto bf162_view_0 = reinterpret_cast<nv_bfloat162*>(values_0);
            const auto bf162_view_1 = reinterpret_cast<nv_bfloat162*>(values_1);
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                #pragma unroll
                for (int l = 0; l < kNumElemsPerVec / 2; ++ l)
                    bf162_view_0[j * (kNumElemsPerVec / 2) + l] += bf162_view_1[j * (kNumElemsPerVec / 2) + l];
                dst_buffer_ptr[i * (kUnrollFactor * 32) + j * 32 + lane_idx] = values_0[j];
            }
        }
    } else {
        #pragma unroll 1
        for (int i = 0; i < kHiddenVec / (kUnrollFactor * 32); ++ i) {
            // Add bias
            float2 reduced[kUnrollFactor * kNumElemsPerVec / 2] = {};
            const auto add_bias = [&](const vec_t* base_ptr) {
                // Read
                vec_t values[kUnrollFactor];
                #pragma unroll
                for (int j = 0; j < kUnrollFactor; ++ j)
                    values[j] = ptx::ldg(base_ptr + i * (kUnrollFactor * 32) + j * 32 + lane_idx);

                // Reduce
                const auto bf162_view = reinterpret_cast<nv_bfloat162*>(values);
                #pragma unroll
                for (int j = 0; j < kUnrollFactor * kNumElemsPerVec / 2; ++ j)
                    ptx::accumulate(reduced[j], bf162_view[j]);
            };
            bias_0 != nullptr ? add_bias(bias_0) : void();
            bias_1 != nullptr ? add_bias(bias_1) : void();

            #pragma unroll
            for (int k = 0; k < kNumValidTopk; ++ k) {
                // We have a limitation on `k` to reduce the branch instruction count
                if (k >= kNumExpectedTopk and topk_slot_idx[k] < 0)
                    break;

                // Read values
                const auto src_base_ptr = get_src_buffer_ptr_func(topk_slot_idx[k]);
                vec_t values[kUnrollFactor] = {};
                #pragma unroll
                for (int j = 0; j < kUnrollFactor; ++ j) {
                    values[j] = ptx::ldg_with_gez_pred(
                        src_base_ptr + (i * (kUnrollFactor * 32) + j * 32 + lane_idx), topk_slot_idx[k]);
                }

                // Reduce
                const auto bf162_view = reinterpret_cast<nv_bfloat162*>(values);
                #pragma unroll
                for (int j = 0; j < kUnrollFactor * kNumElemsPerVec / 2; ++ j)
                    ptx::accumulate(reduced[j], bf162_view[j]);
            }

            // Wait buffer releases for the first write
            if (i == 0)
                wait_buffer_func();

            // Cast into shared memory
            #pragma unroll
            for (int j = 0; j < kUnrollFactor; ++ j) {
                vec_t casted_value;
                auto bf162_view = reinterpret_cast<nv_bfloat162*>(&casted_value);
                #pragma unroll
                for (int l = 0; l < kNumElemsPerVec / 2; ++ l)
                    bf162_view[l] = __float22bfloat162_rn(reduced[j * (kNumElemsPerVec / 2) + l]);
                dst_buffer_ptr[i * (kUnrollFactor * 32) + j * 32 + lane_idx] = casted_value;
            }
        }
    }
}

}  // namespace deep_ep::elastic

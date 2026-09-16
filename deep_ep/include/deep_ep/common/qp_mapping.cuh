#pragma once

// MIT License
//
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Pure-integer channel<->QP mapping math shared by `get_qp_mode` and
// `get_qp_signal_id` in `comm.cuh`. Kept free of CUDA/NCCL types so the exact
// shipped logic also compiles as plain host C++ for the unit test in
// `tests/cpp/test_qp_mapping.cpp`.

#if defined(__CUDACC__)
#define QP_MAPPING_HD __host__ __device__ __forceinline__
#else
#define QP_MAPPING_HD inline
#endif

namespace deep_ep::elastic::comm {

// Result of a balanced contiguous partition: which bin an item lands in, its
// 0-based index within that bin, and the total size of that bin.
struct QPSlot {
    int bin;
    int local;
    int bin_size;
};

// Balanced contiguous partition of `n` items across `q` bins.
//
// The first `n % q` bins own `ceil(n/q)` items; the remaining bins own
// `floor(n/q)`. Items are assigned in monotonic, contiguous blocks (item 0 ->
// bin 0, ...), so a producer's consecutive items stay grouped on one/adjacent
// bin. Properties (all asserted in the unit test):
//   * Balanced:   any two bins differ in size by <= 1.
//   * No idle bin when q <= n: every bin index in [0, q) owns >= 1 item.
//   * Contiguous & monotonic: bin index is non-decreasing in `idx`.
//   * Backward-compatible: when `n % q == 0` the split is 0, every item takes
//     the `else` path, and `bin == idx / (n/q)` -- byte-identical to the old
//     ceil-block mapping for even divisions (all shipped power-of-2 configs).
//
// Callers guarantee `q >= 1`. `base == 0` (i.e. `q > n`) is safe: then
// `rem == n`, `split == n`, and every valid `idx` in [0, n) takes the `if`
// branch, so the `else` branch never divides by `base`.
QP_MAPPING_HD constexpr QPSlot balanced_partition(int idx, int n, int q) {
    const int base = n / q;
    const int rem = n % q;                    // number of "big" (base+1) bins
    const int split = rem * (base + 1);       // items owned by the leading big bins
    if (idx < split)
        return QPSlot{idx / (base + 1), idx % (base + 1), base + 1};
    const int d = idx - split;
    return QPSlot{rem + d / base, d % base, base};
}

// QP index for a data/notify channel. Mirrors `get_qp_mode` (minus the NCCL
// resource-sharing mode, which stays in `comm.cuh`).
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps>
QP_MAPPING_HD constexpr int channel_to_qp(int sm_idx, int channel_in_sm_idx,
                                          bool is_notify_warp = false) {
    static_assert(kNumQPs >= 1,
                  "kNumQPs must be >= 1. With 0 the kNumQPs == 1 fast path is skipped, "
                  "kNumSMs <= kNumAvailableQPs is false, and balanced_partition() is called "
                  "with q == 0, i.e. n / 0. Compiled for the host this specialization exits on "
                  "SIGFPE; in a GPU run it produced corrupted output and no reported fault.");
    // Only one QP
    if constexpr (kNumQPs == 1)
        return 0;

    // The notify warp always uses 1 SM and 1 QP
    if (is_notify_warp)
        return 0;

    constexpr int kQPStartIdx = static_cast<int>(kWithNotifyWarps);
    constexpr int kNumAvailableQPs = kNumQPs - kQPStartIdx;
    if constexpr (kNumSMs <= kNumAvailableQPs) {
        // More QPs than SMs: an SM owns `num_qps_in_sm` QPs, strided across SMs at
        // sm_idx + offset*kNumSMs. Within the SM, balance the SM's channels across
        // its local QPs (contiguous blocks, remainder spread one-per-QP).
        const int num_qps_in_sm = (kNumAvailableQPs / kNumSMs) + (sm_idx < (kNumAvailableQPs % kNumSMs));
        const int local_qp_idx = balanced_partition(channel_in_sm_idx, kNumChannelsPerSM, num_qps_in_sm).bin;
        return kQPStartIdx + sm_idx + local_qp_idx * kNumSMs;
    } else {
        // Fewer QPs than SMs: all SMs share all QPs. Balance the global channels
        // across the QPs (contiguous blocks, remainder spread one-per-QP), so an
        // SM's channels stay grouped and no QP is left idle.
        constexpr int kNumChannels = kNumSMs * kNumChannelsPerSM;
        const int global_channel_idx = sm_idx * kNumChannelsPerSM + channel_in_sm_idx;
        return kQPStartIdx + balanced_partition(global_channel_idx, kNumChannels, kNumAvailableQPs).bin;
    }
}

// Per-channel signal id within its QP (0-based index among the channels sharing
// that QP). Mirrors `get_qp_signal_id`. Uses the same balanced partition as
// `channel_to_qp`, so `(qp, signal_id)` is unique per channel and
// `signal_id < ceil(channels / qps)` -- within the tuner's signal budget.
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM, bool kWithNotifyWarps>
QP_MAPPING_HD constexpr int channel_to_signal_id(int sm_idx, int channel_in_sm_idx) {
    static_assert(kNumQPs >= 1,
                  "kNumQPs must be >= 1. With 0 the kNumQPs == 1 fast path is skipped, "
                  "kNumSMs <= kNumAvailableQPs is false, and balanced_partition() is called "
                  "with q == 0, i.e. n / 0. Compiled for the host this specialization exits on "
                  "SIGFPE; in a GPU run it produced corrupted output and no reported fault.");
    if constexpr (kNumQPs == 1)
        return sm_idx * kNumChannelsPerSM + channel_in_sm_idx;

    constexpr int kQPStartIdx = static_cast<int>(kWithNotifyWarps);
    constexpr int kNumAvailableQPs = kNumQPs - kQPStartIdx;
    if constexpr (kNumSMs <= kNumAvailableQPs) {
        const int num_qps_in_sm = (kNumAvailableQPs / kNumSMs) + (sm_idx < (kNumAvailableQPs % kNumSMs));
        return balanced_partition(channel_in_sm_idx, kNumChannelsPerSM, num_qps_in_sm).local;
    } else {
        constexpr int kNumChannels = kNumSMs * kNumChannelsPerSM;
        const int global_channel_idx = sm_idx * kNumChannelsPerSM + channel_in_sm_idx;
        return balanced_partition(global_channel_idx, kNumChannels, kNumAvailableQPs).local;
    }
}

}  // namespace deep_ep::elastic::comm

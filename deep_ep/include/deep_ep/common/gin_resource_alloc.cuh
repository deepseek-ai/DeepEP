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

#include <algorithm>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>

namespace deep_ep::elastic::gin_alloc {

// NCCL GIN: provider resource budget.
static constexpr int kTotalQPBudget              = 256;
static constexpr int kMaxGinContextBudget        = 17;

// Design constant: ScaleOut warps per SM used by the auto-tuner's budget math.
// Matches the buffer's `!prefer_overlap_with_compute` cap on `num_channels_per_sm`
// (one ScaleOut warp per channel). Under-provisions vs. the runtime peak (which
// can go up to `kNumMaxChannelsPerSM = 8`); worst case just means heavier
// warps-per-context sharing, not a correctness issue.
static constexpr int kMaxWarpsPerSM              = 4;

// Upper bound on ScaleOut warps used by any single kernel: 55 SMs × 4 warps = 220.
//
// Each ScaleOut warp (== channel) needs its own dedicated signal: e.g. when
// warp0 (channel0) signals its peer channel0 warp on a remote node, it uses a
// signal reserved for that channel. That same signal is reused for the channel
// across all nodes in the rail group, so it is shared rail-wide — the signal
// count scales with the number of channels (warps), not with the number of
// nodes.
static constexpr int kMaxSM           = (kTotalQPBudget - 2*kMaxGinContextBudget)/kMaxWarpsPerSM;
static constexpr int kMaxScaleoutWarps           = kMaxSM * kMaxWarpsPerSM;

struct GinResourceConfig {
    int gin_indexed_signals_cnt;      // NGinIndexedSignalsPerGinContext (per NCCL context)
    int gin_context_cnt;              // ScaleOut contexts + one Notify context
};

static constexpr int kMinGinContextCnt           = 2;
static constexpr int kMaxGinContextCnt           = kMaxGinContextBudget;

// Default context count (== default QP count). 11 contexts -> 21 signals/context.
// Contexts and signals-per-context are inversely coupled through
// `gin_indexed_signals_for`, so more QPs means a smaller per-context signal budget. The
// equivalent alternatives are {5, 6, 7, 8, 9, 14}; everything else loses a part somewhere.
// Notably 12, 15, 16 and 17 all drop to 3 parts at 12 SMs -- 17 (the provider maximum) leaves
// only 13 signals/context, and its per-SM QP split puts 4 channels on the busiest QP.
static constexpr int kDefaultGinContextCnt       = 11;

// Per-context indexed-signal budget, workaround for current limitations in provider.
__forceinline__ __device__ __host__ constexpr int gin_indexed_signals_for(int gin_context_cnt) {
    return (kTotalQPBudget - 2 * gin_context_cnt) / gin_context_cnt;
}

__forceinline__ __device__ __host__ constexpr GinResourceConfig make_gin_resources(int gin_context_cnt) {
    return GinResourceConfig{gin_indexed_signals_for(gin_context_cnt), gin_context_cnt};
}

// Each ScaleOut warp (== channel) needs its own dedicated indexed signal id, so the total
// signal budget (ctx * signals/ctx) must cover the worst-case warp count for EVERY legal
// context count, not just the default. The tightest points are ctx = 13 and ctx = 17, both at
// 221 against the 220-warp ceiling -- one signal of slack. Do not raise `kMaxSM` /
// `kMaxWarpsPerSM`, widen the context range, or lower `kTotalQPBudget` without re-checking.
__forceinline__ __host__ constexpr bool all_gin_context_counts_cover_warps() {
    for (int ctx = kMinGinContextCnt; ctx <= kMaxGinContextCnt; ++ ctx)
        if (ctx * gin_indexed_signals_for(ctx) < kMaxScaleoutWarps)
            return false;
    return true;
}
static_assert(all_gin_context_counts_cover_warps(),
              "GIN layout cannot give each ScaleOut warp a dedicated signal id "
              "for every legal context count");

// Preferred (and workspace-sizing) maximum for per-part signalling.
static constexpr int kMaxParts = 4;

static constexpr int kScaleoutSlotRoundingReserve = kMaxParts;

struct GinPartAllocation {
    int num_parts;           // per-channel part count (>= 1)
    int num_channels_per_sm;
};

// Worst-case number of channels that land on a single GIN context (QP). MUST match the
// signal-id assignment in `channel_to_signal_id` (`qp_mapping.cuh`) exactly, because the
// per-channel signal id is its offset within its QP's block and the provisioned budget has
// to cover the largest such offset.
//
// QP assignment is two-level, so there are two regimes:
//   * num_sms <= num_available_qps: each SM owns its own block of QPs
//     (`num_qps_in_sm = avail / num_sms`, plus one for the first `avail % num_sms` SMs) and
//     balances only its OWN channels across them. The worst SM is one WITHOUT the remainder
//     bonus, so it hosts `ceil(channels_per_sm / (avail / num_sms))` channels per QP.
//   * num_sms >  num_available_qps: all SMs share all QPs, and the global balanced partition
//     gives `ceil(num_channels / avail)` channels per QP.
//
// NOTE: the first regime is NOT `ceil(num_sms * channels_per_sm / avail)`. That form
// under-counts, because spare QPs owned by OTHER SMs cannot absorb this SM's channels. Using
// it there provisions too few signals, and the kernel then signals ids outside the
// provisioned range -- which fails silently: no counts ever arrive and dispatch times out in
// the CPU wait with all-zero received counts.
__forceinline__ __device__ __host__ constexpr int channels_per_context(
        int num_sms, int num_available_qps, int num_channels_per_sm) {
    const int avail = num_available_qps > 1 ? num_available_qps : 1;
    const int sms = num_sms > 1 ? num_sms : 1;
    if (sms <= avail) {
        const int num_qps_in_sm = avail / sms;
        return math::constexpr_ceil_div(num_channels_per_sm,
                                        num_qps_in_sm > 1 ? num_qps_in_sm : 1);
    }
    return math::constexpr_ceil_div(sms * num_channels_per_sm, avail);
}

// Per-part signal allocation: pick the largest num_parts (up to kMaxParts) that fits
//   channels_per_context(...) * num_parts <= gin_indexed_signals_cnt
// at the requested channels/SM, then reduce channels_per_sm until the budget holds.
__forceinline__ __device__ __host__ constexpr GinPartAllocation compute_part_allocation(
        const GinResourceConfig& cfg, int num_sms, int num_available_qps, int num_channels_per_sm) {
    const int gin_signals = cfg.gin_indexed_signals_cnt;
    const int channels_per_ctx = channels_per_context(num_sms, num_available_qps, num_channels_per_sm);
    const int budget_parts = gin_signals / (channels_per_ctx > 1 ? channels_per_ctx : 1);
    GinPartAllocation alloc{};
    alloc.num_parts = budget_parts < kMaxParts ? budget_parts : kMaxParts;
    alloc.num_parts = alloc.num_parts > 1 ? alloc.num_parts : 1;
    alloc.num_channels_per_sm = num_channels_per_sm;
    while (alloc.num_channels_per_sm > 1 and
           static_cast<long long>(channels_per_context(num_sms, num_available_qps,
                                                      alloc.num_channels_per_sm)) * alloc.num_parts > gin_signals)
        --alloc.num_channels_per_sm;
#ifndef __CUDA_ARCH__
    if (alloc.num_channels_per_sm < num_channels_per_sm)
        printf("[WARN] DeepEP GIN signal budget reduced the number of channels per SM "
               "from %d to %d\n", num_channels_per_sm, alloc.num_channels_per_sm);
#endif
#ifndef __CUDA_ARCH__
    EP_HOST_ASSERT(static_cast<long long>(channels_per_context(num_sms, num_available_qps,
                                                               alloc.num_channels_per_sm)) * alloc.num_parts <= gin_signals and
                   "GIN signal budget cannot host even 1 part-signal per channel "
                   "at 1 channel/SM. Reduce --num-sms or num_allocated_qps.");
#endif
    return alloc;
}

// Kernel-side entry points: derive the per-channel part count (and verify the launched
// channel count) as compile-time constants from the provisioned indexed-signal budget.
__forceinline__ __device__ __host__ constexpr int constexpr_num_parts(
        int gin_signals, int num_sms, int num_qps, bool with_notify, int channels_per_sm) {
    GinResourceConfig cfg{};
    cfg.gin_indexed_signals_cnt = gin_signals;
    const int avail = (num_qps - (with_notify ? 1 : 0)) > 0 ? (num_qps - (with_notify ? 1 : 0)) : 1;
    return compute_part_allocation(cfg, num_sms > 0 ? num_sms : 1, avail, channels_per_sm).num_parts;
}

__forceinline__ __device__ __host__ constexpr int constexpr_channels_per_sm(
        int gin_signals, int num_sms, int num_qps, bool with_notify, int channels_per_sm) {
    GinResourceConfig cfg{};
    cfg.gin_indexed_signals_cnt = gin_signals;
    const int avail = (num_qps - (with_notify ? 1 : 0)) > 0 ? (num_qps - (with_notify ? 1 : 0)) : 1;
    return compute_part_allocation(cfg, num_sms > 0 ? num_sms : 1, avail, channels_per_sm).num_channels_per_sm;
}

}  // namespace deep_ep::elastic::gin_alloc

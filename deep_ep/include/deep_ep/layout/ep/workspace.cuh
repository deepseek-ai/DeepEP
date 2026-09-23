#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/layout/common/barrier.cuh>

namespace deep_ep::layout {

struct alignas(kNumRDMAAlignmentBytes) EPSignals {
    static constexpr int kNumMaxExperts = 2048;
    static constexpr int kNumMaxExpertsPerRank = 512;
    static constexpr int kNumMaxChannels = 1280;

    BarrierSignals barrier_signals;
    int64_t notify_reduction[kNumMaxRanks + kNumMaxExperts];
    int64_t scaleup_rank_expert_count[2][kNumMaxRanks + kNumMaxExperts];
    int scaleup_atomic_sender_count[kNumMaxRanks];
    int scaleout_rank_expert_count[2][kNumMaxRanks + kNumMaxExperts];
    int64_t scaleout_channel_signaled_tail[kNumMaxChannels][kNumMaxRanks];
    int channel_scaleup_tail[kNumMaxChannels][kNumMaxRanks];
    int lb_chunk_idx_counter;
};
EP_STATIC_ASSERT(sizeof(EPSignals) <= kNumMaxSignalBytes, "Too many signals");

struct EPWorkspaceLayout {
    EPSignals& signals;

    int num_ranks;
    int num_scaleout_ranks, num_scaleup_ranks;
    int num_experts, num_experts_per_rank;

    __forceinline__ __device__ __host__
    EPWorkspaceLayout(void* base,
                      const int& num_scaleout_ranks,
                      const int& num_scaleup_ranks,
                      const int& num_experts):
        signals(*static_cast<EPSignals*>(base)),
        num_ranks(num_scaleout_ranks * num_scaleup_ranks),
        num_scaleout_ranks(num_scaleout_ranks),
        num_scaleup_ranks(num_scaleup_ranks),
        num_experts(num_experts),
        num_experts_per_rank(num_experts / num_ranks) {
        EP_UNIFIED_ASSERT(num_experts % num_ranks == 0);
        EP_UNIFIED_ASSERT(num_ranks <= kNumMaxRanks);
        EP_UNIFIED_ASSERT(num_experts <= EPSignals::kNumMaxExperts);
        EP_UNIFIED_ASSERT(num_experts_per_rank <= EPSignals::kNumMaxExpertsPerRank);
    }

    __forceinline__ __device__ __host__ static constexpr int64_t get_num_bytes() {
        return sizeof(EPSignals);
    }

    __forceinline__ __device__ __host__ BarrierSignals& get_barrier_signals() const {
        return signals.barrier_signals;
    }

    __forceinline__ __device__ __host__ int64_t* get_notify_reduction_workspace_ptr() const {
        return signals.notify_reduction;
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_expert_count_ptr() const {
        return signals.scaleup_rank_expert_count[not kIsSendBuffer];
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_count_ptr() const {
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>();
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_expert_count_ptr() const {
        return get_scaleup_rank_expert_count_ptr<kIsSendBuffer>() + num_scaleup_ranks;
    }

    __forceinline__ __device__ __host__ int* get_scaleup_atomic_sender_counter() const {
        return signals.scaleup_atomic_sender_count;
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_expert_count_ptr() const {
        return signals.scaleout_rank_expert_count[not kIsSendBuffer];
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_count_ptr(
        const int& scaleout_rank_idx = 0, const int& scaleup_rank_idx = 0) const {
        return get_scaleout_rank_expert_count_ptr<kIsSendBuffer>() +
               scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx;
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_expert_count_ptr(
        const int& scaleout_rank_idx = 0, const int& expert_idx = 0) const {
        return get_scaleout_rank_expert_count_ptr<kIsSendBuffer>() + num_ranks +
               scaleout_rank_idx * num_scaleup_ranks * num_experts_per_rank + expert_idx;
    }

    __forceinline__ __device__ __host__ int64_t* get_scaleout_channel_signaled_tail_ptr(
        const int& channel_idx, const int& scaleout_rank_idx) const {
        return &signals.scaleout_channel_signaled_tail[channel_idx][scaleout_rank_idx];
    }

    __forceinline__ __device__ __host__ int* get_channel_scaleup_tail_ptr(
        const int& channel_idx, const int& scaleup_rank_idx) const {
        return &signals.channel_scaleup_tail[channel_idx][scaleup_rank_idx];
    }

    __forceinline__ __device__ __host__ int* get_lb_chunk_idx_counter_ptr() const {
        return &signals.lb_chunk_idx_counter;
    }
};

}  // namespace deep_ep::layout

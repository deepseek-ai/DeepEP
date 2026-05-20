#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic::layout {

struct WorkspaceLayout {
    void* workspace;

    int num_ranks;
    int num_scaleout_ranks, num_scaleup_ranks;
    int num_experts, num_experts_per_rank;

    // We want to fix the layout position for all settings,
    // so that one buffer can be reused for all cases
    static constexpr int kNumMaxRanks = 1024;
    static constexpr int kNumMaxExperts = 2048;
    static constexpr int kNumMaxExpertsPerRank = 256;
    static constexpr int kNumMaxInflightAGRS = 32;

    static constexpr int64_t kNumBarrierSignalBytes = 16;

    __forceinline__ __device__ __host__
    WorkspaceLayout(void* workspace,
                    const int& num_scaleout_ranks,
                    const int& num_scaleup_ranks,
                    const int& num_experts):
        workspace(workspace),
        num_ranks(num_scaleout_ranks * num_scaleup_ranks),
        num_scaleout_ranks(num_scaleout_ranks),
        num_scaleup_ranks(num_scaleup_ranks),
        num_experts(num_experts) {
        num_experts_per_rank = num_experts / num_ranks;
        EP_UNIFIED_ASSERT(num_experts % num_ranks == 0);
        EP_UNIFIED_ASSERT(num_ranks <= kNumMaxRanks);
        EP_UNIFIED_ASSERT(num_experts <= kNumMaxExperts);
        EP_UNIFIED_ASSERT(num_experts_per_rank <= kNumMaxExpertsPerRank);
    }

    static int64_t get_num_bytes() {
        // Pure NVLink scaleup barrier signals
        int64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;

        // Notify reduction workspace
        num_bytes += (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t);

        // Scaleup notify threads
        // Rank send/recv count
        num_bytes += kNumMaxRanks * sizeof(int64_t) * 2;
        // Expert send/recv count
        num_bytes += kNumMaxExperts * sizeof(int64_t) * 2;

        // Scaleup atomic sender count
        num_bytes += kNumMaxRanks * sizeof(int);

        // Scaleout notify threads
        // Rank send/recv count
        num_bytes += kNumMaxRanks * sizeof(int) * 2;
        // Expert send/recv count
        num_bytes += kNumMaxExperts * sizeof(int) * 2;

        // Scaleout channel metadata (finish flag and tails)
        num_bytes += kNumMaxRanks * kNumMaxChannels * sizeof(int64_t);

        // Channel aggregated into the scaleup domains
        // Also reused for channel scaleup tail
        num_bytes += kNumMaxRanks * kNumMaxChannels * sizeof(int);

        // Rank send/recv count, for PP prev/next ranks
        num_bytes += 2 * 2 * sizeof(int64_t);

        // AGRS signals
        num_bytes += (kNumMaxInflightAGRS + 1) * kNumMaxRanks * sizeof(int);

        return num_bytes;
    }

    __forceinline__ __device__ __host__ unsigned long long* get_nvl_barrier_counter_ptr() const {
        return static_cast<unsigned long long*>(workspace);
    }

    __forceinline__ __device__ __host__ int* get_nvl_barrier_signal_ptr(const int& phase) const {
        return math::advance_ptr<int>(workspace, (2 + phase) * sizeof(int));
    }

    __forceinline__ __device__ __host__ int64_t* get_notify_reduction_workspace_ptr() const {
        return math::advance_ptr<int64_t>(workspace, kNumBarrierSignalBytes);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int64_t* get_scaleup_rank_expert_count_ptr() const {
        const auto base_ptr =
            math::advance_ptr<int64_t>(get_notify_reduction_workspace_ptr(), (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t));
        return base_ptr + (kIsSendBuffer ? 0 : kNumMaxRanks + kNumMaxExperts);
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
        return math::advance_ptr<int>(
            get_scaleup_rank_expert_count_ptr<true>(), 2 * (kNumMaxRanks + kNumMaxExperts) * sizeof(int64_t));
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_expert_count_ptr() const {
        const auto base_ptr =
            math::advance_ptr<int>(get_scaleup_atomic_sender_counter(), kNumMaxRanks * sizeof(int));
        return base_ptr + (kIsSendBuffer ? 0 : kNumMaxRanks + kNumMaxExperts);
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_rank_count_ptr(
        const int& scaleout_rank_idx = 0, const int& scaleup_rank_idx = 0) const {
        const auto base_ptr = get_scaleout_rank_expert_count_ptr<kIsSendBuffer>();
        return base_ptr + scaleout_rank_idx * num_scaleup_ranks + scaleup_rank_idx;
    }

    template <bool kIsSendBuffer>
    __forceinline__ __device__ __host__ int* get_scaleout_expert_count_ptr(
        const int& scaleout_rank_idx = 0, const int& expert_idx = 0) const {
        const auto base_ptr = get_scaleout_rank_expert_count_ptr<kIsSendBuffer>() + num_ranks;
        return base_ptr + scaleout_rank_idx * (num_scaleup_ranks * num_experts_per_rank) + expert_idx;
    }

    __forceinline__ __device__ __host__ int64_t* get_scaleout_channel_signaled_tail_ptr(
        const int& channel_idx, const int& scaleout_rank_idx) const {
        const auto base_ptr = math::advance_ptr<int64_t>(
            get_scaleout_rank_expert_count_ptr<true>(),
            (kNumMaxRanks + kNumMaxExperts) * sizeof(int) * 2);
        return base_ptr + (channel_idx * num_scaleout_ranks + scaleout_rank_idx);
    }

    __forceinline__ __device__ __host__ int* get_channel_scaleup_tail_ptr(
        const int& channel_idx, const int& scaleup_rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_scaleout_channel_signaled_tail_ptr(0, 0),
            kNumMaxRanks * kNumMaxChannels * sizeof(int64_t));
        return base_ptr + (channel_idx * num_scaleup_ranks + scaleup_rank_idx);
    }

    __forceinline__ __device__ __host__ int64_t* get_pp_send_count_ptr(const int& offset) const {
        const auto base_ptr = math::advance_ptr<int64_t>(
            get_channel_scaleup_tail_ptr(0, 0),
            kNumMaxRanks * kNumMaxChannels * sizeof(int));
        return base_ptr + offset;
    }

    __forceinline__ __device__ __host__ int64_t* get_pp_recv_count_ptr(const int& offset) const {
        const auto base_ptr = math::advance_ptr<int64_t>(
            get_pp_send_count_ptr(0), 2 * sizeof(int64_t));
        return base_ptr + offset;
    }

    __forceinline__ __device__ __host__ int* get_agrs_recv_signal_ptr(const int& slot, const int& rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_pp_recv_count_ptr(0), 2 * sizeof(int64_t));
        return base_ptr + slot * kNumMaxRanks + rank_idx;
    }

    __forceinline__ __device__ __host__ int* get_agrs_session_signal_ptr(const int& rank_idx) const {
        const auto base_ptr = math::advance_ptr<int>(
            get_agrs_recv_signal_ptr(0, 0), kNumMaxInflightAGRS * kNumMaxRanks * sizeof(int));
        return base_ptr + rank_idx;
    }
};

struct TokenLayout {
    int num_hidden_bytes, num_sf_bytes;
    // NOTES: the top-k index is always 32-bit
    bool with_metadata;
    int num_topk, num_metadata_bytes;
    void* base;

    __forceinline__ __device__ __host__
    TokenLayout(const int& num_hidden_bytes, const int& num_sf_bytes,
                const int& num_topk, const bool& with_metadata, void* base = nullptr) :
        num_hidden_bytes(num_hidden_bytes),
        num_sf_bytes(num_sf_bytes),
        // Metadata includes: top-k indices, weight and source rank/token index
        with_metadata(with_metadata),
        num_topk(num_topk),
        num_metadata_bytes(num_topk * (sizeof(int) + sizeof(float)) +
                           (with_metadata ? (1 + num_topk) * sizeof(int) : 0)),
        base(base) {
        EP_STATIC_ASSERT(sizeof(int) == sizeof(float), "Invalid size assumption");
        EP_UNIFIED_ASSERT(num_hidden_bytes % ptx::kNumTMAAlignBytes == 0);
    }

    template <bool kWithMBarrier, typename dtype_t = int>
    __forceinline__ __device__ __host__ dtype_t get_num_bytes() const {
        const auto num_bytes = math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_sf_bytes, ptx::kNumTMAAlignBytes) +
                               math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes) +
                               math::align<int>(kWithMBarrier ? sizeof(ptx::mbarrier) : 0, ptx::kNumTMAAlignBytes);
        return static_cast<dtype_t>(num_bytes);
    }

    __forceinline__ __device__ __host__ void* get_base_ptr() const {
        return base;
    }

    __forceinline__ __device__ __host__ void set_base_ptr(void* ptr) {
        base = ptr;
    }

    __forceinline__ __device__ __host__ void* get_hidden_ptr() const {
        return get_base_ptr();
    }

    __forceinline__ __device__ __host__ sf_pack_t* get_sf_ptr() const {
        return math::advance_ptr<sf_pack_t>(base, math::align(num_hidden_bytes, ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__ int* get_metadata_ptr() const {
        return math::advance_ptr<int>(get_sf_ptr(), math::align(num_sf_bytes, ptx::kNumTMAAlignBytes));
    }

    __forceinline__ __device__ __host__ int* get_topk_idx_ptr() const {
        return get_metadata_ptr();
    }

    __forceinline__ __device__ __host__ float* get_topk_weights_ptr() const {
        return math::advance_ptr<float>(get_metadata_ptr(), num_topk * sizeof(int));
    }

    __forceinline__ __device__ __host__ int* get_src_token_global_idx_ptr() const {
        return math::advance_ptr<int>(get_topk_weights_ptr(), num_topk * sizeof(float));
    }

    __forceinline__ __device__ __host__ int* get_linked_list_idx_ptr() const {
        return get_src_token_global_idx_ptr() + 1;
    }

    __forceinline__ __device__ ptx::mbarrier* get_mbarrier_ptr() const {
        return math::advance_ptr<ptx::mbarrier>(get_metadata_ptr(), math::align(num_metadata_bytes, ptx::kNumTMAAlignBytes));
    }
};

template <bool kWithMBarrier>
struct BufferLayout {
    TokenLayout token_layout;
    int num_ranks;
    int num_max_tokens_per_rank;

    void* base;

    __forceinline__ __device__ __host__
    BufferLayout(const TokenLayout& token_layout,
                 const int& num_ranks,
                 const int& max_num_tokens_per_rank,
                 void* base = nullptr) :
        token_layout(token_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(max_num_tokens_per_rank),
        base(base) {}

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_token() const {
        return token_layout.get_num_bytes<kWithMBarrier, int64_t>();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * get_num_bytes_per_token();
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    __forceinline__ __device__ __host__
    void* get_buffer_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    __forceinline__ __device__ __host__
    BufferLayout get_rank_buffer(const int& rank_idx) const {
        return BufferLayout(token_layout,
                            1, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_rank() * rank_idx);
    }

    template <int kNumTokensPerChannel>
    __forceinline__ __device__ __host__
    BufferLayout get_channel_buffer(const int& channel_idx) const {
        EP_UNIFIED_ASSERT(num_max_tokens_per_rank % kNumTokensPerChannel == 0);
        return BufferLayout(token_layout,
                            // Do not use `num_max_tokens_per_rank / kNumTokensPerChannel` as the false stride
                            num_ranks, num_max_tokens_per_rank,
                            static_cast<int8_t*>(base) + get_num_bytes_per_token() * kNumTokensPerChannel * channel_idx);
    }

    __forceinline__ __device__ __host__
    TokenLayout get_token_buffer(const int& token_idx, const bool& global = false) const {
        EP_UNIFIED_ASSERT(num_ranks == 1 or global);
        return TokenLayout(token_layout.num_hidden_bytes, token_layout.num_sf_bytes, token_layout.num_topk, token_layout.with_metadata,
                           static_cast<int8_t*>(base) + token_layout.get_num_bytes<kWithMBarrier, int64_t>() * token_idx);
    }
};

}  // namespace deep_ep::elastic

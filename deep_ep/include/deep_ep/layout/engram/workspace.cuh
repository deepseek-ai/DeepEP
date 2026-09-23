#pragma once

#include <cstdint>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/layout/common/barrier.cuh>

namespace deep_ep::layout {

struct alignas(kNumRDMAAlignmentBytes) EngramWorkspace {
    static constexpr int kNumMaxLayers = 128;
    static constexpr int kNumMaxPeerContexts = 2048;

    BarrierSignals barrier;
    uint32_t issue_counters[kNumMaxLayers][kNumMaxPeerContexts];

    __forceinline__ __device__ __host__
    uint32_t* get_issue_counter_ptr(const int layer_idx, const int peer_context_idx) {
        return &issue_counters[layer_idx][peer_context_idx];
    }
};
EP_STATIC_ASSERT(sizeof(EngramWorkspace) <= kNumMaxSignalBytes, "Too many signals");

// The Engram buffer layout.
// The GPU segment gives each layer a fixed-size region in one receive buffer:
//   [layer 0: num_max_tokens * num_entries_per_token entries] ... [layer (L - 1): ...]
// Each rank's CPU/GPU RDMA storage shard packs the per-layer storages back-to-back:
//   [layer 0: entries of layer 0] ... [layer (L - 1): entries of layer (L - 1)]
template <int kNumRanks, int kNumSFShards, int kHiddenBytes, int kEntriesPerToken, int... kEntriesPack>
struct EngramLayout {
    static constexpr int kNumLayers = sizeof...(kEntriesPack);
    static constexpr int kNumSFOwnersPerShard = kNumRanks / kNumSFShards;
    static constexpr int kNumHiddenBytes = kHiddenBytes;
    static constexpr int kNumEntriesPerToken = kEntriesPerToken;

    __forceinline__ __device__ __host__
    static constexpr int get_num_entries(const int& layer_idx) {
        constexpr int kNumEntriesPerLayer[kNumLayers] = {kEntriesPack...};
        return kNumEntriesPerLayer[layer_idx];
    }

    __forceinline__ __device__ __host__
    static constexpr int64_t get_num_prefix_entries(const int& layer_idx) {
        int64_t num_prefix_entries = 0;
        for (int i = 0; i < layer_idx; ++ i)
            num_prefix_entries += get_num_entries(i);
        return num_prefix_entries;
    }

    __forceinline__ __device__ __host__
    static int64_t get_storage_byte_offset(const int& layer_idx, const int& entry_idx) {
        return (get_num_prefix_entries(layer_idx) + entry_idx) * kNumHiddenBytes;
    }

    __forceinline__ __device__ __host__
    static int64_t get_recv_byte_offset(const int& layer_idx, const int& entry_idx, const int& num_max_tokens) {
        return (layer_idx * kNumEntriesPerToken * num_max_tokens + entry_idx) * kNumHiddenBytes;
    }

    __forceinline__ __device__ __host__
    static constexpr int get_num_sf_rows_per_shard(const int& layer_idx) {
        return kNumSFOwnersPerShard * get_num_entries(layer_idx);
    }

    __forceinline__ __device__ __host__
    static int64_t get_sf_row_idx(const int& layer_idx, const int& global_idx) {
        return kNumSFOwnersPerShard * get_num_prefix_entries(layer_idx) +
               global_idx % get_num_sf_rows_per_shard(layer_idx);
    }

    __forceinline__ __device__ __host__
    static int get_sf_shard_idx(const int& layer_idx, const int& global_idx) {
        return global_idx / get_num_sf_rows_per_shard(layer_idx);
    }
};

}  // namespace deep_ep::layout

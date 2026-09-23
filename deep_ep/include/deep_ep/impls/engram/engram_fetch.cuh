#pragma once

#include <cuda/atomic>
#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/engram/workspace.cuh>


namespace deep_ep::engram {

template <int kNumRDMAPeers, int kNumGinContexts, int kNumIssueWarps>
struct EngramIssueWarpLayout {
    int rdma_peer_idx;
    int gin_context_idx;
    int peer_context_idx;
    int num_peer_context_warps;
    int peer_warp_idx;
    int num_peer_warps;

    __device__ __forceinline__ explicit EngramIssueWarpLayout(const int issue_warp_idx) {
        constexpr int kNumPeerContexts = kNumRDMAPeers * kNumGinContexts;

        peer_context_idx = issue_warp_idx % kNumPeerContexts;
        rdma_peer_idx = peer_context_idx % kNumRDMAPeers;
        gin_context_idx = peer_context_idx / kNumRDMAPeers;
        num_peer_context_warps = math::ceil_div(kNumIssueWarps - peer_context_idx, kNumPeerContexts);

        peer_warp_idx = issue_warp_idx / kNumRDMAPeers;
        num_peer_warps = math::ceil_div(kNumIssueWarps - rdma_peer_idx, kNumRDMAPeers);
    }

    __device__ __forceinline__ int get_num_active_peer_context_warps(const int num_token_tiles) const {
        const auto num_token_warps = gin_context_idx < num_token_tiles
            ? math::ceil_div(num_token_tiles - gin_context_idx, kNumGinContexts) : 0;
        return num_token_warps < num_peer_context_warps ? num_token_warps : num_peer_context_warps;
    }
};

template <int kNumSMs, int kNumQPs,
          int kNumSFWarpsPerSM, int kNumIssueWarpsPerSM,
          typename engram_layout_t,
          int kNumSFPacks,
          int kNumMaxTokens,
          int kNumRDMAPeers,
          int kNumRanksPerRDMAPeer,
          int kFlushDepth,
          int64_t kNumTimeoutCycles,
          int kNumSFWarps = kNumSMs * kNumSFWarpsPerSM,
          int kNumIssueWarps = kNumSMs * kNumIssueWarpsPerSM,
          int kNumThreads = (kNumSFWarpsPerSM + kNumIssueWarpsPerSM) * 32>
__global__ void __launch_bounds__(kNumThreads, 1)
engram_fetch_impl(
    const ncclDevComm_t nccl_dev_comm,
    const ncclWindow_t nccl_window,
    const ncclWindow_t* storage_windows,
    void* fetched,
    void* workspace_ptr,
    int* indices,
    ncclGinRequest_t* last_gin_requests,
    const sf_pack_t* sf_shards,
    sf_pack_t* fetched_sf,
    const int sf_token_stride,
    const int sf_hidden_stride,
    const int sf_layer_stride,
    const int num_tokens
) {
    constexpr int kWarpSize = 32;
    constexpr int kNumLayers = engram_layout_t::kNumLayers;
    constexpr int kNumEntriesPerToken = engram_layout_t::kNumEntriesPerToken;

    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto lane_idx = ptx::get_lane_idx();

    // Two warp-sized halves form one circular pending-request buffer per issue warp.
    __shared__ int64_t pending_requests[kNumIssueWarpsPerSM][2 * kWarpSize];

    if (warp_idx < kNumIssueWarpsPerSM) {
        constexpr int kNumPeerContexts = kNumRDMAPeers * kNumQPs;
        const auto issue_warp_idx = sm_idx * kNumIssueWarpsPerSM + warp_idx;
        const auto num_token_tiles = math::ceil_div(num_tokens, kWarpSize);
        const auto issue_layout = EngramIssueWarpLayout<
            kNumRDMAPeers, kNumQPs, kNumIssueWarps>(issue_warp_idx);
        if (issue_layout.peer_warp_idx >= num_token_tiles)
            return;

        const auto num_active_peer_context_warps =
            issue_layout.get_num_active_peer_context_warps(num_token_tiles);
        const auto rdma_peer_rank = issue_layout.rdma_peer_idx * kNumRanksPerRDMAPeer;
        const auto doorbell_threshold = kFlushDepth >= num_active_peer_context_warps
            ? kFlushDepth / num_active_peer_context_warps : 1;
        const auto gin = comm::NCCLGin(
            nccl_dev_comm, nccl_window, issue_layout.gin_context_idx, NCCL_GIN_RESOURCE_SHARING_GPU);
        auto& workspace = *static_cast<layout::EngramWorkspace*>(workspace_ptr);
        auto* warp_pending_requests = pending_requests[warp_idx];

        const auto num_entries = num_tokens * kNumEntriesPerToken;
        #pragma unroll
        for (int layer_idx = 0; layer_idx < kNumLayers; ++ layer_idx) {
            int num_issued_requests = 0;
            int num_pending_requests = 0;
            int pending_head = 0;
            const auto num_entries_per_rank = engram_layout_t::get_num_entries(layer_idx);
            const auto peer_global_idx_begin = issue_layout.rdma_peer_idx * kNumRanksPerRDMAPeer * num_entries_per_rank;
            const auto peer_global_idx_end = (issue_layout.rdma_peer_idx + 1) * kNumRanksPerRDMAPeer * num_entries_per_rank;

            const auto issue_batch = [&](const int batch_size, const bool is_last_batch = false) {
                const auto rings_doorbell = is_last_batch or num_issued_requests % doorbell_threshold + batch_size >= doorbell_threshold;
                num_issued_requests += batch_size;
                const auto options =
                    (rings_doorbell ? ncclGinOptFlagsDefault : ncclGinOptFlagsAggregateRequests) |
                    (batch_size == kWarpSize ? ncclGinOptFlagsWarpGet : ncclGinOptFlagsDefault);

                if (lane_idx < batch_size) {
                    const auto request = warp_pending_requests[pending_head + lane_idx];
                    const auto entry_idx = static_cast<int>(request);
                    const auto global_idx = static_cast<int>(request >> 32);
                    const auto peer_local_idx = global_idx - peer_global_idx_begin;
                    const auto intra_peer_rank_idx = peer_local_idx / num_entries_per_rank;
                    const auto local_entry_idx = peer_local_idx % num_entries_per_rank;
                    const auto src_byte_offset = engram_layout_t::get_storage_byte_offset(layer_idx, local_entry_idx);
                    const auto dst_byte_offset = engram_layout_t::get_recv_byte_offset(
                        layer_idx, entry_idx, kNumMaxTokens);
                    gin.get<ncclTeamTagWorld, ncclCoopThread, ncclGin_SegmentMixed>(
                        storage_windows[intra_peer_rank_idx], src_byte_offset,
                        nccl_window, gin.get_sym_offset(math::advance_ptr(fetched, dst_byte_offset)),
                        engram_layout_t::kNumHiddenBytes,
                        rdma_peer_rank,
                        options);
                }
            };

            for (int token_tile_idx = issue_layout.peer_warp_idx;
                 token_tile_idx < num_token_tiles;
                 token_tile_idx += issue_layout.num_peer_warps) {
                const auto first_entry_idx = token_tile_idx * kWarpSize * kNumEntriesPerToken;

                for (int round_idx = 0; round_idx < kNumEntriesPerToken; ++round_idx) {
                    const auto entry_idx = first_entry_idx + round_idx * kWarpSize + lane_idx;
                    const auto global_idx = entry_idx < num_entries
                        ? __ldg(indices + layer_idx * num_entries + entry_idx)
                        : -1;
                    const auto belongs_to_peer =
                        global_idx >= peer_global_idx_begin && global_idx < peer_global_idx_end;
                    const auto peer_mask = __ballot_sync(0xffffffffu, belongs_to_peer);

                    if (belongs_to_peer) {
                        const auto lower_lanes_mask = (1u << lane_idx) - 1;
                        const auto pending_idx =
                            (pending_head + num_pending_requests +
                             __popc(peer_mask & lower_lanes_mask)) &
                            (2 * kWarpSize - 1);
                        warp_pending_requests[pending_idx] =
                            (static_cast<int64_t>(global_idx) << 32) | entry_idx;
                    }
                    __syncwarp();
                    num_pending_requests += __popc(peer_mask);

                    // Keep the final batch to ring the doorbell before flush_async
                    if (num_pending_requests > kWarpSize) {
                        issue_batch(kWarpSize);
                        pending_head ^= kWarpSize;
                        num_pending_requests -= kWarpSize;
                    }
                }
            }

            if (num_pending_requests > 0) {
                issue_batch(num_pending_requests, true);
                __syncwarp();
            }

            if (ptx::elect_one_sync()) {
                auto* issue_counter = workspace.get_issue_counter_ptr(
                    layer_idx, issue_layout.peer_context_idx);
                const auto request_ptr =
                    last_gin_requests + layer_idx * kNumPeerContexts + issue_layout.peer_context_idx;
                if (num_active_peer_context_warps == 1) {
                    gin.flush_async<ncclTeamTagWorld, ncclCoopThread>(rdma_peer_rank, request_ptr);
                } else {
                    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> num_arrivals(*issue_counter);

                    // Publish this issue warp's last_issued_get update. The last warp
                    // alone acquires all earlier arrivals before flushing the QP.
                    const auto arrival_idx = num_arrivals.fetch_add(1, cuda::memory_order_release);
                    if (arrival_idx + 1 == num_active_peer_context_warps) {
                        ptx::fence_acquire_gpu();
                        gin.flush_async<ncclTeamTagWorld, ncclCoopThread>(
                            rdma_peer_rank, request_ptr);
                        ptx::st_relaxed_gpu(issue_counter, 0u);
                    } else if (layer_idx + 1 < kNumLayers) {
                        comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                            if (ptx::ld_volatile<uint32_t>(issue_counter) == 0)
                                return true;

                            if (is_last_check) {
                                printf("DeepEP Engram issue barrier timeout, peer: %d, "
                                       "gin context: %d, layer: %d\n",
                                       issue_layout.rdma_peer_idx, issue_layout.gin_context_idx, layer_idx);
                            }
                            return false;
                        });
                    }
                }
            }
            __syncwarp();
        }
    } else if constexpr (kNumSFPacks > 0) {
        constexpr int kNumPacksPerVec = sizeof(uint64_t) / sizeof(sf_pack_t);
        constexpr int kNumVecs = kNumSFPacks / kNumPacksPerVec;
        EP_STATIC_ASSERT(kNumSFPacks % kNumPacksPerVec == 0, "Invalid number of SF packs");

        const auto sf_warp_idx = sm_idx * kNumSFWarpsPerSM + warp_idx - kNumIssueWarpsPerSM;
        const auto num_entries = num_tokens * kNumEntriesPerToken;
        const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window);

        const auto copy_sf_entry = [&](const int layer_idx, const int entry_idx) {
            const auto global_idx = __ldg(indices + layer_idx * num_entries + entry_idx);
            if (global_idx < 0)
                return;

            const auto sf_row_idx = engram_layout_t::get_sf_row_idx(layer_idx, global_idx);
            const auto sf_shard_idx = engram_layout_t::get_sf_shard_idx(layer_idx, global_idx);
            const auto* sf_shard = gin.get_sym_ptr<ncclTeamTagLsa>(const_cast<sf_pack_t*>(sf_shards), sf_shard_idx);
            const auto* src = reinterpret_cast<const uint64_t*>(sf_shard + sf_row_idx * kNumSFPacks);

            uint64_t values[kNumVecs];
            #pragma unroll
            for (int i = 0; i < kNumVecs; ++ i)
                values[i] = __ldg(src + i);

            auto* dst = fetched_sf + layer_idx * sf_layer_stride +
                (entry_idx / kNumEntriesPerToken) * sf_token_stride +
                (entry_idx % kNumEntriesPerToken) * kNumSFPacks * sf_hidden_stride;
            #pragma unroll
            for (int i = 0; i < kNumVecs; ++ i) {
                #pragma unroll
                for (int j = 0; j < kNumPacksPerVec; ++ j)
                    dst[(i * kNumPacksPerVec + j) * sf_hidden_stride] = reinterpret_cast<const sf_pack_t*>(&values[i])[j];
            }
        };

        #pragma unroll
        for (int layer_idx = 0; layer_idx < kNumLayers; ++layer_idx) {
            const auto num_token_tiles = math::ceil_div(num_tokens, kWarpSize);
            for (int i = sf_warp_idx; i < kNumEntriesPerToken * num_token_tiles; i += kNumSFWarps) {
                const auto entry_slot_idx = i / num_token_tiles;
                const auto token_tile_idx = i % num_token_tiles;
                const auto token_idx = token_tile_idx * kWarpSize + lane_idx;
                if (token_idx < num_tokens)
                    copy_sf_entry(layer_idx, token_idx * kNumEntriesPerToken + entry_slot_idx);
            }
        }
    }
}

}  // namespace deep_ep::engram

#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

template <int kNumQPs,
          int kNumEntriesPerRank,
          int kHidden,
          int kNumRDMAPeers,
          int kNumRanksPerRDMAPeer,
          size_t kIntraRankStorageStride,
          int kNumThreads,
          typename team_t,
          int kNumWarps = kNumThreads / 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16)>
__global__ void __launch_bounds__(kNumThreads, 1)
engram_fetch_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                  void* storage, void* fetched, int* indices,
                  ncclGinRequest_t* last_gin_requests,
                  const int num_tokens) {
    const auto qp_idx = static_cast<int>(blockIdx.x);
    const auto warp_idx = ptx::get_warp_idx();
    const auto global_warp_idx = qp_idx * kNumWarps + warp_idx;
    const auto thread_idx = static_cast<int>(threadIdx.x);

    // Gin handle
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);

    __shared__ bool sent_to_peer[kNumRDMAPeers];
    EP_STATIC_ASSERT(kNumRDMAPeers <= kNumThreads, "Too many RDMA peers");
    if (thread_idx < kNumRDMAPeers)
        sent_to_peer[thread_idx] = false;
    __syncthreads();

    // Issue RDMA
    const auto issue_rdma_get = [=](const int& token_idx, const int& peer_idx,
                                    const int64_t& src_byte_offset, const int& extra_options = 0) {
        gin.get<team_t, ncclCoopThread, ncclGin_SegmentMixed>(math::advance_ptr(storage, src_byte_offset),
                        math::advance_ptr(fetched, static_cast<int64_t>(token_idx) * kNumHiddenBytes),
                        kNumHiddenBytes, peer_idx, extra_options);
    };

    // Each warp fetches one token cooperatively via RDMA gin.get
    // TODO: deal with padded tokens
    if (ptx::elect_one_sync()) {
        #pragma unroll 4
        for (int i = global_warp_idx; i < num_tokens; i += kNumQPs * kNumWarps) {
            const auto global_idx = __ldg(indices + i);
            const auto owner_rank_idx = global_idx / kNumEntriesPerRank;
            const auto local_entry_idx = global_idx % kNumEntriesPerRank;

            // Route owner rank to RDMA peer and intra-peer rank
            const auto peer_idx = owner_rank_idx / kNumRanksPerRDMAPeer;
            const auto intra_peer_rank_idx = owner_rank_idx % kNumRanksPerRDMAPeer;

            // Byte offset into storage layout
            const auto src_byte_offset = static_cast<int64_t>(intra_peer_rank_idx) * kIntraRankStorageStride +
                                         static_cast<int64_t>(local_entry_idx) * kNumHiddenBytes;

            // Delay ring DB
            issue_rdma_get(i, peer_idx, src_byte_offset, ncclGinOptFlagsAggregateRequests);
            sent_to_peer[peer_idx] = true;
        }
    }
    __syncthreads();

    // Issue flush per peer we sent to; its unconditional DB ring flushes all
    // prior aggregated gets on the same QP.
    if (ptx::elect_one_sync()) {
        for (int i = warp_idx; i < kNumRDMAPeers; i += kNumWarps) {
            const auto request_ptr = last_gin_requests + qp_idx * kNumRDMAPeers + i;
            if (sent_to_peer[i]) {
                gin.flush_async<team_t, ncclCoopThread>(i, request_ptr);
            } else {
                EP_STATIC_ASSERT(sizeof(ncclGinRequest_t) == sizeof(int4), "Invalid request size");
                *reinterpret_cast<int4*>(request_ptr) = make_int4(0, 0, 0, 0);
            }
        }
    }
}

} // namespace deep_ep::elastic

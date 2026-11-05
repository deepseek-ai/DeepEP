#pragma once

#include <deep_ep/comm/handle.cuh>
#include <deep_ep/common/compiled.cuh>


namespace deep_ep::engram {

template <int kNumRDMAPeers, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
engram_fetch_wait_impl(const ncclDevComm_t nccl_dev_comm,
                       const ncclWindow_t nccl_window,
                       ncclGinRequest_t* last_gin_requests,
                       const int num_active_gin_contexts) {
    const auto thread_idx = static_cast<int>(threadIdx.x);
    EP_STATIC_ASSERT(sizeof(ncclGinRequest_t) == sizeof(int4), "Invalid request size");
    const auto num_active_peer_contexts = num_active_gin_contexts * kNumRDMAPeers;

    for (int peer_context_idx = thread_idx; peer_context_idx < num_active_peer_contexts; peer_context_idx += kNumThreads) {
        const auto context_idx = peer_context_idx / kNumRDMAPeers;
        const auto gin = comm::NCCLGin(
            nccl_dev_comm, nccl_window, context_idx, NCCL_GIN_RESOURCE_SHARING_THREAD);

        auto last_gin_req_int4 = __ldg(reinterpret_cast<int4*>(last_gin_requests + peer_context_idx));
        auto last_gin_req = *reinterpret_cast<ncclGinRequest_t*>(&last_gin_req_int4);
        gin.wait(last_gin_req, cuda::memory_order_relaxed);
    }
}

}  // namespace deep_ep::engram

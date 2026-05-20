#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>


namespace deep_ep::elastic {

template <int kNumRDMAPeers, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
engram_fetch_wait_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                       ncclGinRequest_t* last_gin_requests) {
    const auto qp_idx = static_cast<int>(blockIdx.x);
    const auto thread_idx = static_cast<int>(threadIdx.x);

    // Gin handle
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);

    // Wait for all RDMA gets to complete
    for (int i = thread_idx; i < kNumRDMAPeers; i += kNumThreads) {
        EP_STATIC_ASSERT(sizeof(ncclGinRequest_t) == sizeof(int4), "Invalid request size");
        auto last_gin_req_int4 = __ldg(reinterpret_cast<int4*>(last_gin_requests + qp_idx * kNumRDMAPeers + i));
        if (last_gin_req_int4.x != 0 or last_gin_req_int4.y != 0 or
            last_gin_req_int4.z != 0 or last_gin_req_int4.w != 0) {
            auto last_gin_req = *reinterpret_cast<ncclGinRequest_t*>(&last_gin_req_int4);
            gin.wait(last_gin_req);
        }
    }
}

} // namespace deep_ep::elastic

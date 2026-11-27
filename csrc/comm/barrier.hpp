#pragma once

#include "stream.hpp"
#include "../kernels/comm/api.hpp"
#include "../kernels/comm/barrier.hpp"

namespace deep_ep::comm {

static void barrier(const Context& group,
                    void* signals,
                    const at::cuda::CUDAStream& stream,
                    const int64_t& num_gpu_timeout_cycles,
                    const bool& with_cpu_sync,
                    const bool& sequential,
                    const bool& flush_stores = true,
                    const bool& wait_comm_stream = false,
                    const int& num_qps = kFlushAllAllocatedQPs) {
    // Sync
    const bool wait_comm = wait_comm_stream and
        not (is_stream_capturing(stream) and not is_stream_capturing(get_comm_stream()));
    if (with_cpu_sync) {
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    } else if (wait_comm) {
        stream_wait(stream, get_comm_stream());
    }

    // Launch
    launch_barrier(
        group.dev_comm, group.window,
        signals,
        group.scaleout_rank_idx, group.scaleup_rank_idx,
        group.num_scaleout_ranks, group.num_scaleup_ranks,
        num_qps,
        num_gpu_timeout_cycles,
        group.is_scaleup_nvlink,
        flush_stores, sequential,
        stream);

    // Sync
    if (with_cpu_sync) {
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
    } else if (wait_comm) {
        // To work with APIs with previous events
        stream_wait(get_comm_stream(), stream);
        // Rejoin comm_stream into the capture stream so CUDA Graph capture does not leave it dangling
        if (is_stream_capturing(stream))
            stream_wait(stream, get_comm_stream());
    }
}

}  // namespace deep_ep::comm

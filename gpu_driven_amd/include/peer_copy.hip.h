// peer_copy.hip
#pragma once

#include "ring_buffer.hip.h"
#include <hip/hip_runtime.h>

template <typename X, typename Y, typename Z = decltype(X() + Y())>
__host__ __device__ constexpr Z divUp(X x, Y y) {
  return (x + y - 1) / y;
}

hipError_t launch_peer_bulk_copy(void* dst_ptr, int dst_dev, void* src_ptr,
                                  int src_dev, size_t bytes,
                                  hipStream_t stream = 0);

hipError_t launch_peer_bulk_copy2(CopyTask const* host_tasks, int num_tasks,
                                   hipStream_t stream, int src_device,
                                   CopyTask*& d_tasks);

__global__ void peer_copy_kernel_vec_batched(CopyTask const* __restrict__ tasks,
                                             int num_tasks,
                                             int tasks_per_block);

template <int PIPE_DEPTH,  // same as kPipelineDepth
          typename VecT>   // 16 B per transaction
__global__ void peer_copy_kernel_vec_pipelined(
    CopyTask const* __restrict__ tasks, int num_tasks, int tasks_per_block);

HostToDeviceNVlinkBuffer* initialize_ring_buffer_for_nvlink_forwarding(
    hipStream_t stream);

bool post_copy_task(HostToDeviceNVlinkBuffer* rb, CopyTask const* host_tasks,
                    int num_tasks, hipStream_t stream, int src_device,
                    CopyTask*& d_tasks);
#pragma once

#include <cstdint>
#include <format>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/ep/eplb.cuh>

#include "../../runtime/jit.hpp"
#include "../comm/api.hpp"

namespace deep_ep::ep {

static void launch_lb_reduce_grads(const comm::Context& context,
                                   float* redundant_expert_grads,
                                   float* expert_grads,
                                   const int* redundancy_mapping,
                                   const int& num_redundant_experts,
                                   const int& num_local_experts,
                                   const int64_t& hidden,
                                   const int& num_sms,
                                   const at::cuda::CUDAStream& stream) {
    // Each warp owns one chunk buffer
    constexpr int num_chunk_bytes = 24 * 1024;
    constexpr int num_bytes_per_warp = num_chunk_bytes + sizeof(ptx::mbarrier);
    constexpr int num_smem_layout_bytes = sizeof(layout::EPLBSharedMemoryLayout);
    const int num_smem_bytes = jit->device.get_num_smem_bytes();
    const int num_warps = (num_smem_bytes - num_smem_layout_bytes) / num_bytes_per_warp;
    const int num_threads = num_warps * 32;
    EP_HOST_ASSERT(num_smem_layout_bytes + num_bytes_per_warp <= num_smem_bytes);
    EP_HOST_ASSERT(num_warps <= 32);

    // Gradient descriptor
    const layout::EPWeight weight = {redundant_expert_grads, expert_grads, hidden * static_cast<int64_t>(sizeof(float))};
    EP_HOST_ASSERT(weight.num_bytes_per_expert % kNumTMAAlignmentBytes == 0);

    // Compile
    const auto kernel = jit->compile("ep_lb_reduce_grads", std::format(R"(
#include <deep_ep/impls/ep/reduce_grads.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::lb_reduce_grads<{}, {}, {}, {}, {}, {}, {}>);
}}
)", num_sms, num_warps, context.num_nvl_ranks, num_local_experts, num_redundant_experts,
    num_chunk_bytes, context.num_gpu_timeout_cycles));

    // Launch
    jit->launch(
        kernel, {
            .stream = stream.stream(),
            .num_smem_bytes = num_smem_bytes,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
            .cluster_dim = dim3(2 - (num_sms % 2), 1, 1),
            .cooperative = true,
        },
        redundancy_mapping, context.dev_comm, context.window, context.workspace,
        weight
    );
}

}  // namespace deep_ep::ep

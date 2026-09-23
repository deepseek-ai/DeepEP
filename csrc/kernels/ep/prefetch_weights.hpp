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

static void launch_lb_prefetch_weights(const comm::Context& context,
                                       const int& num_weights,
                                       const layout::EPWeightList& weights,
                                       const int* redundancy_mapping,
                                       const int& num_redundant_experts,
                                       const int& num_local_experts,
                                       const int& num_sms,
                                       const at::cuda::CUDAStream& stream) {
    // Each warp owns one fixed-size chunk buffer and handles both its load and store
    constexpr int num_chunk_bytes = 12 * 1024;
    constexpr int num_bytes_per_warp = num_chunk_bytes + sizeof(ptx::mbarrier);
    const int num_smem_bytes = jit->device.get_num_smem_bytes();
    constexpr int num_smem_layout_bytes = sizeof(layout::EPLBSharedMemoryLayout);
    EP_HOST_ASSERT(num_smem_layout_bytes + num_bytes_per_warp <= num_smem_bytes);
    const int num_warps = (num_smem_bytes - num_smem_layout_bytes) / num_bytes_per_warp;
    EP_HOST_ASSERT(num_warps <= 32);
    const int num_threads = num_warps * 32;

    // Compile
    const auto kernel = jit->compile("ep_lb_prefetch_weights", std::format(R"(
#include <deep_ep/impls/ep/prefetch_weights.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::lb_prefetch_weights<{}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", num_sms, num_warps, context.num_nvl_ranks, num_local_experts, num_redundant_experts,
    num_weights, num_chunk_bytes, context.num_gpu_timeout_cycles));

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
        weights
    );
}

}  // namespace deep_ep::ep

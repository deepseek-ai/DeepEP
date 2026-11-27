#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <string>

#include <ATen/cuda/CUDAContext.h>

#include <deep_ep/layout/ep/eplb.cuh>

#include "../../runtime/jit.hpp"
#include "../comm/api.hpp"

namespace deep_ep::ep {

static void launch_lb_get_redundancy_mapping(const comm::Context& context, const torch::Tensor& topk_idx,
                                             const torch::Tensor& redundancy_mapping, const torch::Tensor& remapped_topk_idx,
                                             const int& num_experts, const int& num_routes, const int& token_alignment,
                                             const int& num_sms, const int& num_redundant_experts_per_rank,
                                             const at::cuda::CUDAStream& stream) {
    constexpr int num_warps = 32;
    constexpr int num_threads = num_warps * 32;
    const int num_main_experts_per_rank = num_experts / context.num_ranks;
    const int num_domain_experts = context.num_nvl_ranks * num_main_experts_per_rank;
    const int qp_idx = context.num_rdma_ranks > 1 ? context.num_allocated_qps - 1 : 0;
    const int num_smem_ints = std::max({
        num_warps * num_experts,
        2 * num_threads,
        3 * num_domain_experts + 2 * context.num_nvl_ranks,
        3 * num_experts,
    });
    const int num_smem_bytes = num_smem_ints * static_cast<int>(sizeof(int));
    EP_HOST_ASSERT(num_smem_bytes <= jit->device.get_num_smem_bytes());
    EP_HOST_ASSERT(math::ceil_div<int64_t>(num_routes, num_threads) * num_experts <=
                   layout::LBPrepareSignals::kNumMaxPrepareRoutes);

    // Compile
    const auto kernel = jit->compile("lb_get_redundancy_mapping", std::format(R"(
#include <deep_ep/impls/ep/lb_get_redundancy_mapping.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::lb_get_redundancy_mapping<{}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", context.num_rdma_ranks, context.num_nvl_ranks, num_main_experts_per_rank, num_redundant_experts_per_rank,
        token_alignment, num_sms, num_threads, context.num_gpu_timeout_cycles));

    // Launch
    jit->launch(kernel, {
                            .stream = stream.stream(),
                            .num_smem_bytes = num_smem_bytes,
                            .grid_dim = dim3(num_sms, 1, 1),
                            .block_dim = dim3(num_threads, 1, 1),
                            .cooperative = true,
                        },
                topk_idx.data_ptr(), num_routes, context.rank_idx, qp_idx, context.dev_comm, context.window,
                redundancy_mapping.data_ptr<int>(), remapped_topk_idx.data_ptr(), context.workspace);
}

}  // namespace deep_ep::ep

#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>

#include "../comm/api.hpp"
#include "../../runtime/jit.hpp"

namespace deep_ep::pp {

static int get_num_qps(const comm::Context& context) {
    const auto num_qps = std::min(2, context.num_allocated_qps);
    return num_qps;
}

static void launch_pp_send(
    const comm::Context& context,
    void* x, const int64_t& num_x_bytes,
    const int& dst_rank_idx,
    const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors,
    const int& num_sms,
    const at::cuda::CUDAStream& stream) {
    constexpr int kNumThreads = 32;
    const auto num_qps = get_num_qps(context);
    const auto num_smem_bytes = jit->device.get_num_smem_bytes();
    EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms());

    // Compile
    const auto kernel = jit->compile("pp_send", std::format(R"(
#include <deep_ep/impls/pp/pp_send_recv.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::pp::pp_send_impl<{}, {}, {}, {}, {}>);
}}
)", num_sms, context.num_ranks, num_smem_bytes, num_qps, context.num_gpu_timeout_cycles));

    // Launch
    jit->launch(
        kernel, {
            .stream = stream,
            .num_smem_bytes = num_smem_bytes,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(kNumThreads, 1, 1),
            .cooperative = true,
        },
        context.dev_comm, context.window,
        x, num_x_bytes,
        context.buffer, context.workspace,
        context.rank_idx, dst_rank_idx,
        num_max_tensor_bytes, num_max_inflight_tensors
    );
}

static void launch_pp_recv(
    const comm::Context& context,
    void* x, const int64_t& num_x_bytes,
    const int& src_rank_idx,
    const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors,
    const int& num_sms,
    const at::cuda::CUDAStream& stream) {
    constexpr int kNumThreads = 32;
    const auto num_qps = get_num_qps(context);
    const auto num_smem_bytes = jit->device.get_num_smem_bytes();
    EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms());

    // Compile
    const auto kernel = jit->compile("pp_recv", std::format(R"(
#include <deep_ep/impls/pp/pp_send_recv.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::pp::pp_recv_impl<{}, {}, {}, {}, {}>);
}}
)", num_sms, context.num_ranks, num_smem_bytes, num_qps, context.num_gpu_timeout_cycles));

    // Launch
    jit->launch(
        kernel, {
            .stream = stream,
            .num_smem_bytes = num_smem_bytes,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(kNumThreads, 1, 1),
            .cooperative = true,
        },
        context.dev_comm, context.window,
        x, num_x_bytes,
        context.buffer, context.workspace,
        context.rank_idx, src_rank_idx,
        num_max_tensor_bytes, num_max_inflight_tensors
    );
}

}  // namespace deep_ep::pp

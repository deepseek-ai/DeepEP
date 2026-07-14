#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/layout.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

class PPSendRuntime final : public jit::LaunchRuntime<PPSendRuntime> {
public:
    struct Args {
        // Templated arguments
        int num_ranks;
        int num_smem_bytes;
        int64_t num_timeout_cycles;

        // Parameters
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* x;
        int64_t num_x_bytes;
        void* buffer;
        void* workspace;
        int rank_idx, dst_rank_idx;
        int64_t num_max_tensor_bytes;
        int num_max_inflight_tensors;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/pp_send_recv.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&pp_send_impl<{}, {}, {}, {}>);
}}
)", args.launch_args.grid_dim.first,
    args.num_ranks,
    args.num_smem_bytes,
    args.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            args.x, args.num_x_bytes,
            args.buffer, args.workspace,
            args.rank_idx, args.dst_rank_idx,
            args.num_max_tensor_bytes, args.num_max_inflight_tensors
        ));
    }
};

static void launch_pp_send(const jit::NoRefPtr& nccl_dev_comm,
                           const ncclWindow_t& nccl_window,
                           void* x, const int64_t& num_x_bytes,
                           void* buffer, void* workspace,
                           const int& rank_idx, const int& dst_rank_idx, const int& num_ranks,
                           const int64_t& num_max_tensor_bytes,
                           const int num_max_inflight_tensors,
                           const int& num_sms,
                           const int64_t& num_timeout_cycles,
                           const int& num_smem_bytes,
                           const at::cuda::CUDAStream& stream) {
    // Generate, build and launch
    const PPSendRuntime::Args args = {
        .num_ranks = num_ranks,
        .num_smem_bytes = num_smem_bytes,
        .num_timeout_cycles = num_timeout_cycles,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .x = x,
        .num_x_bytes = num_x_bytes,
        .buffer = buffer,
        .workspace = workspace,
        .rank_idx = rank_idx,
        .dst_rank_idx = dst_rank_idx,
        .num_max_tensor_bytes = num_max_tensor_bytes,
        .num_max_inflight_tensors = num_max_inflight_tensors,
        .launch_args = jit::LaunchArgs(num_sms, 32, num_smem_bytes, 1, true)
    };
    const auto code = PPSendRuntime::generate(args);
    const auto runtime = jit::compiler->build("pp_send", code);
    PPSendRuntime::launch(runtime, args, stream);
}

class PPRecvRuntime final : public jit::LaunchRuntime<PPRecvRuntime> {
public:
    struct Args {
        // Templated arguments
        int num_ranks;
        int num_smem_bytes;
        int64_t num_timeout_cycles;

        // Parameters
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* x;
        int64_t num_x_bytes;
        void* buffer;
        void* workspace;
        int rank_idx;
        int src_rank_idx;
        int64_t num_max_tensor_bytes;
        int num_max_inflight_tensors;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/pp_send_recv.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&pp_recv_impl<{}, {}, {}, {}>);
}}
)", args.launch_args.grid_dim.first,
    args.num_ranks,
    args.num_smem_bytes,
    args.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            args.x, args.num_x_bytes,
            args.buffer, args.workspace,
            args.rank_idx, args.src_rank_idx,
            args.num_max_tensor_bytes,
            args.num_max_inflight_tensors
        ));
    }
};

static void launch_pp_recv(const jit::NoRefPtr& nccl_dev_comm,
                           const ncclWindow_t& nccl_window,
                           void* x,
                           const int64_t& num_x_bytes,
                           void* buffer, void* workspace,
                           const int& rank_idx, const int& src_rank_idx, const int& num_ranks,
                           const int64_t& num_max_tensor_bytes,
                           const int& num_max_inflight_tensors,
                           const int& num_sms,
                           const int64_t& num_timeout_cycles,
                           const int& num_smem_bytes,
                           const at::cuda::CUDAStream& stream) {
    // Generate, build and launch
    const PPRecvRuntime::Args args = {
        .num_ranks = num_ranks,
        .num_smem_bytes = num_smem_bytes,
        .num_timeout_cycles = num_timeout_cycles,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .x = x,
        .num_x_bytes = num_x_bytes,
        .buffer = buffer,
        .workspace = workspace,
        .rank_idx = rank_idx,
        .src_rank_idx = src_rank_idx,
        .num_max_tensor_bytes = num_max_tensor_bytes,
        .num_max_inflight_tensors = num_max_inflight_tensors,
        .launch_args = jit::LaunchArgs(num_sms, 32, num_smem_bytes, 1, true)
    };
    const auto code = PPRecvRuntime::generate(args);
    const auto runtime = jit::compiler->build("pp_recv", code);
    PPRecvRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic

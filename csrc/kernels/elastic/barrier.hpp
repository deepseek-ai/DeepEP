#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

class BarrierRuntime final : public jit::LaunchRuntime<BarrierRuntime> {
public:
    struct Args {
        // Templated arguments
        bool is_scaleup_nvlink;
        int num_scaleout_ranks, num_scaleup_ranks;
        int64_t num_timeout_cycles;

        // Parameters
        ncclDevComm_t nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* workspace;
        int scaleout_rank_idx, scaleup_rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/barrier.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&barrier_impl<{}, {}, {}, {}, {}, {}>);
}}
)",                        args.is_scaleup_nvlink,
                           args.launch_args.grid_dim.first, args.launch_args.num_threads,
                           args.num_scaleout_ranks, args.num_scaleup_ranks,
                           args.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            args.workspace, args.scaleout_rank_idx, args.scaleup_rank_idx
        ));
    }
};

static void launch_barrier(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
                           void* workspace,
                           const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                           const int64_t& num_timeout_cycles,
                           const bool& is_scaleup_nvlink,
                           const at::cuda::CUDAStream& stream) {
    // Number of threads equals to the number of ranks
    constexpr auto kNumThreads = 512;

    // Generate, build and launch
    // NOTES: also use 1 SM in hybrid kernel to ensure global sync.
    const auto num_sms = 1;
    const BarrierRuntime::Args args = {
        .is_scaleup_nvlink = is_scaleup_nvlink,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .num_timeout_cycles = num_timeout_cycles,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .workspace = workspace,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .launch_args = jit::LaunchArgs(num_sms, kNumThreads, 0, 1, true)};
    const auto code = BarrierRuntime::generate(args);
    const auto runtime = jit::compiler->build("barrier", code);
    BarrierRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic

#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/layout.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

class EngramFetchRuntime final : public jit::LaunchRuntime<EngramFetchRuntime> {
public:
    struct Args {
        // Templated arguments
        int num_entries_per_rank;
        int hidden;
        int num_scaleout_ranks, num_scaleup_ranks;
        int64_t num_cpu_bytes_per_rank;
        int num_qps;
        bool allow_hybrid_mode;

        // Parameters
        ncclDevComm_t nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* storage;
        void* fetched;
        int* indices;
        ncclGinRequest_t* last_gin_requests;
        int num_tokens;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        int num_rdma_peers, num_ranks_per_rdma_peer;
        std::string team_tag;
        if (args.allow_hybrid_mode) {
            num_rdma_peers = args.num_scaleout_ranks;
            num_ranks_per_rdma_peer = args.num_scaleup_ranks;
            team_tag = "ncclTeamTagRail";
        } else {
            num_rdma_peers = args.num_scaleout_ranks * args.num_scaleup_ranks;
            num_ranks_per_rdma_peer = 1;
            team_tag = "ncclTeamTagWorld";
        }
        auto func_name = fmt::format("engram_fetch_impl<{}, {}, {}, {}, {}, {}, {}, {}>",
            args.num_qps, args.num_entries_per_rank, args.hidden,
            num_rdma_peers, num_ranks_per_rdma_peer,
            args.num_cpu_bytes_per_rank, args.launch_args.num_threads, team_tag);

        return fmt::format(R"(
#include <deep_ep/impls/engram_fetch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", func_name);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            args.storage, args.fetched,
            args.indices,
            args.last_gin_requests,
            args.num_tokens
        ));
    }
};

static void launch_engram_fetch(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
                                void* storage, void* fetched,
                                int* indices,
                                ncclGinRequest_t* last_gin_requests,
                                const int& num_entries_per_rank, const int& hidden,
                                const int& num_tokens,
                                const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                const int64_t& num_cpu_bytes_per_rank,
                                const int& num_qps,
                                const bool& allow_hybrid_mode,
                                const at::cuda::CUDAStream& stream) {
    constexpr int kNumEngramFetchThreads = 1024;

    // Generate, build and launch
    const EngramFetchRuntime::Args args = {
        .num_entries_per_rank = num_entries_per_rank,
        .hidden = hidden,
        .num_scaleout_ranks = num_scaleout_ranks,
        .num_scaleup_ranks = num_scaleup_ranks,
        .num_cpu_bytes_per_rank = num_cpu_bytes_per_rank,
        .num_qps = num_qps,
        .allow_hybrid_mode = allow_hybrid_mode,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .storage = storage,
        .fetched = fetched,
        .indices = indices,
        .last_gin_requests = last_gin_requests,
        .num_tokens = num_tokens,
        .launch_args = jit::LaunchArgs(num_qps, kNumEngramFetchThreads)};
    const auto code = EngramFetchRuntime::generate(args);
    const auto runtime = jit::compiler->build("engram_fetch", code);
    EngramFetchRuntime::launch(runtime, args, stream);
}

class EngramFetchWaitRuntime final : public jit::LaunchRuntime<EngramFetchWaitRuntime> {
public:
    struct Args {
        // Templated arguments
        int num_scaleout_ranks, num_scaleup_ranks;
        bool allow_hybrid_mode;

        ncclDevComm_t nccl_dev_comm;
        ncclWindow_t nccl_window;
        ncclGinRequest_t* last_gin_requests;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const int num_rdma_peers = args.allow_hybrid_mode
            ? args.num_scaleout_ranks
            : args.num_scaleout_ranks * args.num_scaleup_ranks;
        auto func_name = fmt::format("engram_fetch_wait_impl<{}, {}>",
            num_rdma_peers, args.launch_args.num_threads);

        return fmt::format(R"(
#include <deep_ep/impls/engram_fetch_wait.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", func_name);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            args.last_gin_requests
        ));
    }
};

static void launch_engram_fetch_wait(ncclGinRequest_t* last_gin_requests,
                                     const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
                                     const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                     const int& num_qps,
                                     const bool& allow_hybrid_mode,
                                     const at::cuda::CUDAStream& stream) {
    constexpr int kNumEngramFetchWaitThreads = 1024;

    // Generate, build and launch
    const EngramFetchWaitRuntime::Args args = {
        .num_scaleout_ranks = num_scaleout_ranks,
        .num_scaleup_ranks = num_scaleup_ranks,
        .allow_hybrid_mode = allow_hybrid_mode,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .last_gin_requests = last_gin_requests,
        .launch_args = jit::LaunchArgs(num_qps, kNumEngramFetchWaitThreads)};
    const auto code = EngramFetchWaitRuntime::generate(args);
    const auto runtime = jit::compiler->build("engram_fetch_wait", code);
    EngramFetchWaitRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic

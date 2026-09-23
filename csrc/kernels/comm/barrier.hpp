#pragma once

#include <cstdint>
#include <format>

#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../../runtime/jit.hpp"

namespace deep_ep::comm {

static void launch_barrier(const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window,
                           void* signals,
                           const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                           const int64_t& num_timeout_cycles,
                           const bool& is_scaleup_nvlink,
                           const bool& flush_stores, const bool& sequential,
                           const at::cuda::CUDAStream& stream) {
    // Number of threads equals to the number of ranks
    constexpr auto kNumThreads = 512;

    // NOTES: only the parallel hybrid kernel needs 2 SMs; the sequential mode does scaleout and
    // scaleup one after another, so a single SM is sufficient.
    const auto num_sms = (not sequential and num_scaleout_ranks > 1) ? 2 : 1;

    // Compile
    const auto kernel = jit->compile("barrier", std::format(R"(
#include <deep_ep/impls/comm/barrier.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::comm::barrier_impl<{}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", is_scaleup_nvlink, num_sms, kNumThreads,
        num_scaleout_ranks, num_scaleup_ranks,
        num_timeout_cycles, flush_stores, sequential));

    // Launch
    jit->launch(
        kernel, {
            .stream = stream.stream(),
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(kNumThreads, 1, 1),
            .cooperative = true,
        },
        nccl_dev_comm, nccl_window,
        signals, scaleout_rank_idx, scaleup_rank_idx
    );
}

}  // namespace deep_ep::comm

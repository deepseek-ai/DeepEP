#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../../runtime/jit.hpp"

namespace deep_ep::engram {

static void launch_engram_fetch(const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window,
                                ncclWindow_t* storage_windows, void* fetched, void* workspace,
                                int* indices,
                                ncclGinRequest_t* last_gin_requests,
                                void* sf_shards, void* fetched_sf,
                                const int& sf_token_stride, const int& sf_hidden_stride,
                                const int& sf_layer_stride,
                                const int& num_sf_shards,
                                const std::vector<int>& num_entries_per_layer,
                                const int& hidden, const int& elem_size, const int& num_sf_packs,
                                const int& num_entries_per_token,
                                const int& num_tokens, const int& num_max_tokens,
                                const int& num_rdma_peers, const int& num_ranks_per_rdma_peer,
                                const int& num_qps, const int& flush_depth,
                                const int64_t& num_timeout_cycles,
                                const at::cuda::CUDAStream& stream) {
    constexpr int kNumSFWarpsPerSM = 4;
    const auto num_sms = jit->device.get_num_sms();

    const auto num_sf_warps_per_sm = num_sf_packs > 0 ? kNumSFWarpsPerSM : 0;
    const auto num_issue_warps_per_sm = 32 - num_sf_warps_per_sm;

    const auto num_threads = (num_sf_warps_per_sm + num_issue_warps_per_sm) * 32;
    EP_HOST_ASSERT(flush_depth > 0);
    EP_HOST_ASSERT(sf_hidden_stride != 1);
    EP_HOST_ASSERT(num_sms * num_issue_warps_per_sm >= num_rdma_peers * num_qps);

    std::string num_entries_per_layer_string;
    num_entries_per_layer_string.reserve(num_entries_per_layer.size() * 12);
    for (const auto num_entries : num_entries_per_layer) {
        if (not num_entries_per_layer_string.empty())
            num_entries_per_layer_string += ", ";
        num_entries_per_layer_string += std::to_string(num_entries);
    }
    const auto func_name = std::format(
        "engram_fetch_impl<{}, {}, {}, {}, deep_ep::layout::EngramLayout<{}, {}, {}, {}, {}>, {}, {}, {}, {}, {}, {}>",
        num_sms, num_qps, num_sf_warps_per_sm, num_issue_warps_per_sm,
        num_rdma_peers * num_ranks_per_rdma_peer, num_sf_shards,
        hidden * elem_size, num_entries_per_token,
        num_entries_per_layer_string,
        num_sf_packs, num_max_tokens,
        num_rdma_peers, num_ranks_per_rdma_peer,
        flush_depth, num_timeout_cycles);

    // Compile
    const auto kernel = jit->compile("engram_fetch", std::format(R"(
#include <deep_ep/impls/engram/engram_fetch.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::engram::{});
}}
)", func_name));

    // Launch
    jit->launch(
        kernel, {
            .stream = stream.stream(),
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
        },
        nccl_dev_comm, nccl_window,
        storage_windows, fetched, workspace,
        indices,
        last_gin_requests,
        static_cast<sf_pack_t*>(sf_shards), static_cast<sf_pack_t*>(fetched_sf),
        sf_token_stride, sf_hidden_stride, sf_layer_stride,
        num_tokens
    );
}

static void launch_engram_fetch_wait(ncclGinRequest_t* last_gin_requests,
                                     const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window,
                                     const int& num_rdma_peers,
                                     const int& num_active_gin_contexts,
                                     const at::cuda::CUDAStream& stream) {
    constexpr int kNumEngramFetchWaitThreads = 1024;

    // Compile
    const auto func_name = std::format(
        "engram_fetch_wait_impl<{}, {}>", num_rdma_peers, kNumEngramFetchWaitThreads);
    const auto kernel = jit->compile("engram_fetch_wait", std::format(R"(
#include <deep_ep/impls/engram/engram_fetch_wait.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::engram::{});
}}
)", func_name));

    // Launch
    jit->launch(
        kernel, {
            .stream = stream.stream(),
            .grid_dim = dim3(1, 1, 1),
            .block_dim = dim3(kNumEngramFetchWaitThreads, 1, 1),
        },
        nccl_dev_comm, nccl_window,
        last_gin_requests,
        num_active_gin_contexts
    );
}

}  // namespace deep_ep::engram

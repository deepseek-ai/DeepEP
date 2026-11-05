#pragma once

#include <cstdint>
#include <format>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>

#include <deep_ep/comm/bucket.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>

#include "../../runtime/jit.hpp"

namespace deep_ep::bucket {

static void launch_reduce_scatter(
    const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& window,
    void* workspace, void* buffer, void* buffer_multimem,
    const int& num_buckets, const comm::BucketList& buckets,
    const int& num_rdma_ranks, const int& num_nvl_ranks,
    const int& num_sms, const int& num_qps,
    const int64_t& num_timeout_cycles, const float& scale,
    const std::string& comm_precision, const int& context_idx,
    const at::cuda::CUDAStream& stream) {
    // Checks
    EP_HOST_ASSERT(comm_precision == "fp32" or comm_precision == "bf16");
    EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets);
    EP_HOST_ASSERT(num_rdma_ranks > 0 and num_nvl_ranks > 0);
    EP_HOST_ASSERT(num_rdma_ranks > 1 or num_nvl_ranks > 1);
    EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms());
    EP_HOST_ASSERT((comm_precision != "bf16" or (num_rdma_ranks > 1 and num_nvl_ranks == 1)) and
                   "BF16 communication is only supported for RDMA-only reduce-scatter");

    if (num_rdma_ranks > 1 and num_nvl_ranks > 1) {
        // Hybrid RDMA and NVLink
        EP_HOST_ASSERT(num_sms > 1 and "Hybrid reduce-scatter requires at least 2 SMs");

        constexpr int kNumMultimemWarps = 8;
        constexpr int kNumMultimemWarpsPerGroup = 4;
        constexpr int kNumIssueWarps = 2;
        constexpr int kNumReduceWarps = 2;
        constexpr int kNumSmemBytesPerReduceWarp = 64 * 1024;
        constexpr int kNumThreads = (kNumMultimemWarps + kNumIssueWarps + kNumReduceWarps) * 32;
        constexpr int kNumSmemBytes = kNumReduceWarps * kNumSmemBytesPerReduceWarp;
        EP_HOST_ASSERT(kNumSmemBytes + kNumReduceWarps * sizeof(uint64_t) <= jit->device.get_num_smem_bytes() and
                       "Hybrid reduce-scatter TMA buffers exceed the shared-memory limit");

        // Compile
        const auto kernel = jit->compile("bucket_reduce_scatter", std::format(R"(
#include <deep_ep/impls/bucket/reduce_scatter/hybrid.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::hybrid_reduce_scatter<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",                         num_rdma_ranks, num_nvl_ranks,
                            num_sms,
                            kNumMultimemWarps,
                            kNumMultimemWarpsPerGroup,
                            kNumIssueWarps,
                            kNumReduceWarps,
                            kNumSmemBytesPerReduceWarp,
                            num_qps,
                            num_timeout_cycles, scale != 1.0f,
                            context_idx));

        jit->launch(
            kernel, {
                .stream = stream,
                .num_smem_bytes = kNumSmemBytes,
                .grid_dim = dim3(num_sms, 1, 1),
                .block_dim = dim3(kNumThreads, 1, 1),
                .cluster_dim = dim3(2 - (num_sms % 2), 1, 1),
                .cooperative = true,
            },
            nccl_dev_comm, window,
            workspace, buffer, buffer_multimem,
            scale, num_buckets,
            buckets
        );
    } else if (num_rdma_ranks > 1 and comm_precision == "bf16") {
        const int num_cast_warps = num_rdma_ranks < 4 ? 4 : 8;
        const int num_issue_warps = num_rdma_ranks < 4 ? 8 : 4;
        const int num_reduce_warps = num_rdma_ranks < 4 ? 4 : 8;
        const int cast_group = num_rdma_ranks < 4 ? 1 : 4;
        const int reduce_group = num_rdma_ranks < 4 ? 1 : 4;
        const int issue_batch = num_rdma_ranks < 4 ? 1 : (num_rdma_ranks < 16 ? 4 : 8);
        const int reduce_batch = num_rdma_ranks < 4 ? 1 : 8;
        const int num_threads = (num_cast_warps + num_issue_warps + num_reduce_warps) * 32;

        // Compile
        const auto kernel = jit->compile("bucket_reduce_scatter", std::format(R"(
#include <deep_ep/impls/bucket/reduce_scatter/rdma.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::rdma_reduce_scatter_bf16<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",                         num_rdma_ranks, num_sms, num_cast_warps, num_issue_warps, num_reduce_warps, num_qps,
                            num_timeout_cycles, scale != 1.0f, context_idx,
                            issue_batch, reduce_batch, cast_group, reduce_group),
            {.arch = jit->device.get_arch(/* use_arch_family = */ false)});

        jit->launch(
            kernel, {
                .stream = stream,
                .grid_dim = dim3(num_sms, 1, 1),
                .block_dim = dim3(num_threads, 1, 1),
                .cluster_dim = dim3(2 - (num_sms % 2), 1, 1),
                .cooperative = true,
            },
            nccl_dev_comm, window,
            workspace, buffer,
            scale, num_buckets,
            buckets
        );
    } else if (num_rdma_ranks > 1) {
        // RDMA only
        constexpr int kNumIssueWarps = 8;
        constexpr int kNumReduceWarps = 8;
        constexpr int kNumThreads = (kNumIssueWarps + kNumReduceWarps) * 32;
        const int num_batch_chunks = num_rdma_ranks >= 8 ? 8 : num_rdma_ranks;

        // Compile
        const auto kernel = jit->compile("bucket_reduce_scatter", std::format(R"(
#include <deep_ep/impls/bucket/reduce_scatter/rdma.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::rdma_reduce_scatter<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",                         num_rdma_ranks, num_sms,
                            kNumIssueWarps, kNumReduceWarps, num_qps,
                            num_timeout_cycles, scale != 1.0f, context_idx,
                            num_batch_chunks, num_batch_chunks));

        jit->launch(
            kernel, {
                .stream = stream,
                .grid_dim = dim3(num_sms, 1, 1),
                .block_dim = dim3(kNumThreads, 1, 1),
                .cluster_dim = dim3(2 - (num_sms % 2), 1, 1),
                .cooperative = true,
            },
            nccl_dev_comm, window,
            workspace, buffer,
            scale, num_buckets,
            buckets
        );
    } else {
        // NVLink only
        constexpr int kNumThreads = 1024;

        // Compile
        const auto kernel = jit->compile("bucket_reduce_scatter", std::format(R"(
#include <deep_ep/impls/bucket/reduce_scatter/nvlink.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::nvlink_reduce_scatter<{}, {}, {}, {}, {}, {}>);
}}
)",                         num_nvl_ranks, num_sms, kNumThreads,
                            num_timeout_cycles, scale != 1.0f, context_idx));

        jit->launch(
            kernel, {
                .stream = stream,
                .grid_dim = dim3(num_sms, 1, 1),
                .block_dim = dim3(kNumThreads, 1, 1),
                .cluster_dim = dim3(2 - (num_sms % 2), 1, 1),
                .cooperative = true,
            },
            nccl_dev_comm, window,
            workspace, buffer, buffer_multimem,
            scale, num_buckets,
            buckets
        );
    }
}

}  // namespace deep_ep::bucket

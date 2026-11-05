#pragma once

#include <cstdint>
#include <format>

#include <ATen/cuda/CUDAContext.h>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

#include "../../runtime/jit.hpp"
#include "../comm/api.hpp"

namespace deep_ep::bucket {

static void launch_all_reduce(
    const comm::Context& context,
    const int& num_buckets, const comm::BucketList& buckets,
    const int& context_idx, const int& num_sms,
    const int64_t& num_timeout_cycles, const float& scale,
    const at::cuda::CUDAStream& stream) {
    // Checks
    EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets);
    EP_HOST_ASSERT(num_sms > 0 and num_sms <= jit->device.get_num_sms());

    if (context.num_rdma_ranks > 1 and context.num_nvl_ranks > 1) {
        EP_HOST_ASSERT(context.buffer_multimem != nullptr);
        EP_HOST_ASSERT(num_sms >= 2 and "Hybrid all-reduce requires at least 2 SMs");
        EP_HOST_ASSERT(context.num_rdma_ranks <= 128 and context.num_rdma_ranks * context.num_nvl_ranks <= 1024);

        const int rdma = context.num_rdma_ranks, nvl = context.num_nvl_ranks;
        const int mm_warps = rdma <= 32 ? 8 : 4;
        constexpr int issue_warps = 1;
        constexpr int reduce_warps = 1;
        const int min_broadcast_warps = reduce_warps * math::ceil_div(rdma, 32);
        const int broadcast_warps = math::align(issue_warps + reduce_warps + min_broadcast_warps, 4) - issue_warps - reduce_warps;
        const int num_chunk_bytes = (rdma <= 4 ? 256 : 64) * 1024;
        // Limit total SMEM usage when more broadcast warps are needed
        const int num_reduce_smem_bytes = (broadcast_warps <= 2 ? 64 : 32) * 1024;
        const int num_broadcast_smem_bytes = (broadcast_warps <= 2 ? 32 : 8) * 1024;
        const int num_threads = (mm_warps * issue_warps + issue_warps + reduce_warps + broadcast_warps) * 32;
        const int num_smem_bytes = reduce_warps * num_reduce_smem_bytes + broadcast_warps * num_broadcast_smem_bytes;
        EP_HOST_ASSERT(num_smem_bytes + (reduce_warps + broadcast_warps) * sizeof(uint64_t) <= jit->device.get_num_smem_bytes());
        const int num_slots = layout::ChunkStorage::kNumBytes / num_chunk_bytes / rdma;
        EP_HOST_ASSERT(2 * rdma * num_slots + num_sms * reduce_warps * rdma <= layout::AllReduceSignals::kNumTotalSlots and
                       "Hybrid all-reduce signals exceed the workspace");

        // Compile
        const auto kernel = jit->compile("bucket_all_reduce", std::format(R"(
#include <deep_ep/impls/bucket/all_reduce/hybrid.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::hybrid_all_reduce<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", rdma, nvl, num_sms, mm_warps, issue_warps, reduce_warps, broadcast_warps, num_reduce_smem_bytes, num_broadcast_smem_bytes,
            context.num_allocated_qps, num_timeout_cycles, scale != 1.0f, context_idx, num_chunk_bytes));
        // Launch
        const auto options = deep_jit::cuda::LaunchOptions {
            .stream = stream,
            .num_smem_bytes = num_smem_bytes,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
            .cluster_dim = dim3(2 - (num_sms % 2), 1, 1),
            .cooperative = true,
        };
        jit->launch(
            kernel, options,
            context.dev_comm, context.window,
            context.workspace, context.buffer, context.buffer_multimem,
            scale, num_buckets, buckets
        );
    } else if (context.num_rdma_ranks == 1) {
        // NVLink only
        EP_HOST_ASSERT(context.num_nvl_ranks > 1 and context.buffer_multimem != nullptr);

        // Compile
        constexpr int kNumThreads = 1024;
        const auto kernel = jit->compile("bucket_all_reduce", std::format(R"(
#include <deep_ep/impls/bucket/all_reduce/nvlink.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::nvlink_all_reduce<{}, {}, {}, {}, {}, {}>);
}}
)", context.num_nvl_ranks, num_sms, kNumThreads,
            num_timeout_cycles, scale != 1.0f, context_idx));

        // Launch
        const auto options = deep_jit::cuda::LaunchOptions {
            .stream = stream,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(kNumThreads, 1, 1),
            .cluster_dim = dim3(2 - (num_sms % 2), 1, 1),
            .cooperative = true,
        };
        jit->launch(
            kernel, options,
            context.dev_comm, context.window,
            context.workspace, context.buffer_multimem,
            scale, num_buckets,
            buckets
        );
    } else {
        // RDMA only
        constexpr int kNumIssueWarps = 8;
        constexpr int kNumReduceWarps = 8;
        constexpr int kNumSmemBytesPerReduceWarp = 16 * 1024;
        constexpr int kNumThreads = (kNumIssueWarps + kNumReduceWarps) * 32;
        constexpr int kNumSmemBytes = kNumReduceWarps * kNumSmemBytesPerReduceWarp;
        EP_HOST_ASSERT(kNumSmemBytes + kNumReduceWarps * sizeof(uint64_t) <= jit->device.get_num_smem_bytes() and
                       "RDMA all-reduce TMA buffers exceed the shared-memory limit");

        // Compile
        const auto kernel = jit->compile("bucket_all_reduce", std::format(R"(
#include <deep_ep/impls/bucket/all_reduce/rdma.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::rdma_all_reduce<{}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", context.num_rdma_ranks, num_sms,
            kNumIssueWarps, kNumReduceWarps, kNumSmemBytesPerReduceWarp,
            context.num_allocated_qps,
            num_timeout_cycles, scale != 1.0f, context_idx));

        // Launch
        const auto options = deep_jit::cuda::LaunchOptions {
            .stream = stream,
            .num_smem_bytes = kNumSmemBytes,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(kNumThreads, 1, 1),
            .cluster_dim = dim3(2 - (num_sms % 2), 1, 1),
            .cooperative = true,
        };
        jit->launch(
            kernel, options,
            context.dev_comm, context.window,
            context.workspace, context.buffer,
            scale, num_buckets,
            buckets
        );
    }
}

}  // namespace deep_ep::bucket

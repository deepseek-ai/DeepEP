#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <vector>

#include <ATen/cuda/CUDAContext.h>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>
#include <deep_ep/layout/bucket/send_recv.cuh>

#include "../../comm/stream.hpp"
#include "../../kernels/comm/api.hpp"
#include "../../runtime/jit.hpp"
#include "../../utils/event.hpp"
#include "../driver/api.hpp"

namespace deep_ep::bucket {

static bool use_rdma_send_recv(const comm::Context& context) {
    return context.num_rdma_ranks > 1 and context.num_nvl_ranks == 1;
}

static bool use_hybrid_send_recv(const comm::Context& context) {
    return context.num_rdma_ranks > 1 and context.num_nvl_ranks > 1;
}

static void launch_send_recv(const comm::Context& context,
                             const int num_nvl_send_entries,
                             const int num_rdma_send_entries,
                             const layout::SendRecvList& nvl_send_entries,
                             const layout::SendRecvList& rdma_send_entries,
                             const int& context_idx,
                             int num_sms,
                             const at::cuda::CUDAStream& stream) {
    if (use_hybrid_send_recv(context)) {
        EP_HOST_ASSERT(num_sms >= 2 and num_sms <= jit->device.get_num_sms());

        // Three NVLink warps allow each warp to double-buffer 32-KiB TMA chunks within shared memory budget
        constexpr int num_nvl_warps = 3;
        constexpr int num_total_slots = layout::SendRecvSignals::kNumTotalSlots;
        // NOTES: Achieve balance between single-entry throughput and multi-entry congestion
        const int num_slots_per_entry = num_total_slots / (2 * std::min(8u,
            std::bit_ceil(static_cast<unsigned int>(std::ceil(std::sqrt(context.num_rdma_ranks))))));
        const int num_forward_warps = math::ceil_div(num_total_slots, num_sms * 32);
        constexpr int num_tma_chunk_bytes = 32 * 1024;
        constexpr int num_tma_stages = 2;
        const int num_qps = context.num_allocated_qps;
        const int num_threads = (num_nvl_warps + num_forward_warps) * 32;
        const auto num_smem_bytes =
            layout::HybridSendRecvSharedLayout<
                num_nvl_warps, num_tma_chunk_bytes, num_tma_stages>::
                get_num_bytes(context.num_nvl_ranks);
        EP_HOST_ASSERT(num_smem_bytes <= jit->device.get_num_smem_bytes() and
                       "Hybrid send-recv TMA buffers exceed the shared-memory limit");
        const auto kernel = jit->compile("bucket_hybrid_send_recv", std::format(R"(
#include <deep_ep/impls/bucket/send_recv/hybrid.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::hybrid_send_recv<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", context.is_scaleup_nvlink, context.num_rdma_ranks, context.num_nvl_ranks,
            num_sms, num_nvl_warps, num_forward_warps, num_tma_chunk_bytes,
            num_tma_stages, num_total_slots, num_slots_per_entry, num_qps,
            context.num_gpu_timeout_cycles, context_idx));
        jit->launch(
            kernel, {
                .stream = stream.stream(),
                .num_smem_bytes = num_smem_bytes,
                .grid_dim = dim3(num_sms, 1, 1),
                .block_dim = dim3(num_threads, 1, 1),
                .cooperative = true,
            },
            context.dev_comm, context.window, context.workspace, context.buffer,
            num_nvl_send_entries, num_rdma_send_entries,
            nvl_send_entries, rdma_send_entries);
        return;
    } else if (use_rdma_send_recv(context)) {
        // Pure RDMA
        EP_HOST_ASSERT(num_sms == 0);
        num_sms = 1;
        const int num_qps = std::min(context.num_allocated_qps, 2);
        const int num_threads =
            math::constexpr_align(layout::kNumMaxSendRecvEntries, 128) * num_qps;
        const auto kernel = jit->compile("bucket_rdma_send_recv", std::format(R"(
#include <deep_ep/impls/bucket/send_recv/rdma.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::rdma_send_recv<{}, {}, {}, {}, {}>);
}}
)", context.num_rdma_ranks, num_sms, num_threads, num_qps, context.num_gpu_timeout_cycles));
        jit->launch(
            kernel, {
                .stream = stream.stream(),
                .grid_dim = dim3(num_sms, 1, 1),
                .block_dim = dim3(num_threads, 1, 1),
            },
            context.dev_comm, context.window, context.buffer,
            num_rdma_send_entries, rdma_send_entries);

        // Handle self copy separately
        for (int entry_idx = 0; entry_idx < num_rdma_send_entries; ++ entry_idx) {
            const auto& entry = rdma_send_entries[entry_idx];
            auto* const dst_ptr = math::advance_ptr(context.buffer, entry.dst_offset);
            if (entry.src_rank_idx == entry.dst_rank_idx and entry.src_ptr != dst_ptr)
                CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                    dst_ptr, entry.src_ptr, entry.num_bytes, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        // Pure NVLink
        EP_HOST_ASSERT(num_sms == 0);
        EP_HOST_ASSERT(context.num_rdma_ranks == 1);

        // Build copy descriptors
        std::vector<size_t> sizes;
        std::vector<void*> dst_ptrs, src_ptrs;
        for (int entry_idx = 0; entry_idx < num_nvl_send_entries; ++ entry_idx) {
            const auto& entry = nvl_send_entries[entry_idx];
            src_ptrs.push_back(entry.src_ptr);
            dst_ptrs.push_back(context.get_sym_ptr(
                math::advance_ptr(context.buffer, entry.dst_offset), entry.dst_rank_idx));
            sizes.push_back(entry.num_bytes);
        }

        // Wait arrival & push & write
        auto& workspace = *static_cast<layout::BucketWorkspace*>(context.workspace);
        driver::copy_engine_push_synced(context, workspace.signals[context_idx].copy_engine, dst_ptrs, src_ptrs, sizes, stream);
    }
}

}  // namespace deep_ep::bucket

#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

#include "../../runtime/jit.hpp"
#include "../comm/api.hpp"
#include "../driver/api.hpp"

namespace deep_ep::bucket {

static void launch_ring_send_recv(
    const comm::Context& context,
    const int& num_buckets,
    const comm::BucketList& src_buckets,
    const comm::BucketList& dst_buckets,
    const int& shift,
    const bool& cyclic,
    const int& context_idx,
    bool& flush_stores,
    const at::cuda::CUDAStream& stream) {
    EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets);
    EP_HOST_ASSERT(context.num_ranks > 1);
    EP_HOST_ASSERT(cyclic ? (shift % context.num_ranks != 0) :
                            (shift != 0 and shift > -context.num_ranks and shift < context.num_ranks));

    const auto mod_pos = [](const int& a, const int& m) { return ((a % m) + m) % m; };
    const int norm_shift = cyclic ? mod_pos(shift, context.num_ranks) : shift;
    const int dst_rank_idx = cyclic ? (context.rank_idx + norm_shift) % context.num_ranks : (context.rank_idx + shift);
    const bool has_dst = dst_rank_idx >= 0 and dst_rank_idx < context.num_ranks;

    const auto build_nvl_copies = [&](const int& peer_nvl_rank_idx,
                                      std::vector<void*>& dst_ptrs,
                                      std::vector<void*>& src_ptrs,
                                      std::vector<size_t>& sizes) {
        dst_ptrs.reserve(num_buckets);
        src_ptrs.reserve(num_buckets);
        sizes.reserve(num_buckets);
        void* const peer_buffer = context.get_sym_ptr(context.buffer, peer_nvl_rank_idx);
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& [src_offset, num_bytes] = src_buckets[bucket_idx];
            const auto dst_offset = dst_buckets[bucket_idx].offset;
            src_ptrs.push_back(math::advance_ptr(context.buffer, src_offset));
            dst_ptrs.push_back(math::advance_ptr(peer_buffer, dst_offset));
            sizes.push_back(num_bytes);
        }
    };

    if (context.num_rdma_ranks == 1) {
        // Pure NVLink
        std::vector<void*> dst_ptrs, src_ptrs;
        std::vector<size_t> sizes;
        if (has_dst)
            build_nvl_copies(dst_rank_idx, dst_ptrs, src_ptrs, sizes);
        auto& workspace = *static_cast<layout::BucketWorkspace*>(context.workspace);
        driver::copy_engine_push_synced(context, workspace.signals[context_idx].copy_engine, dst_ptrs, src_ptrs, sizes, stream);
        flush_stores = false;
        return;
    }

    // Pure RDMA and Hybrid
    constexpr int kNumThreads = 256;
    constexpr int num_chunk_bytes = layout::RingSendRecvSignals::kNumChunkBytes;
    const int num_sms = context.num_nvl_ranks > 1 ? 2 : 1;
    const int num_qps = std::min(context.num_allocated_qps, layout::RingSendRecvSignals::kNumQPs);

    const int dst_rdma_rank_idx = has_dst ? (dst_rank_idx / context.num_nvl_ranks) : -1;
    const int dst_nvl_rank_idx = has_dst ? (dst_rank_idx % context.num_nvl_ranks) : -1;
    const bool send_rdma = has_dst and (dst_rdma_rank_idx != context.rdma_rank_idx);
    const bool send_nvl = has_dst and (dst_rdma_rank_idx == context.rdma_rank_idx);
    const bool use_chunk_staging = context.num_nvl_ranks > 1 and (shift % context.num_nvl_ranks != 0);

    // Determine whether this rank forwards an incoming cross-rail RDMA stream to a local NVL peer
    const int fwd_dst_nvl_rank_idx = mod_pos(context.nvl_rank_idx + norm_shift, context.num_nvl_ranks);
    bool do_forward = false;
    int num_chunks = 0;
    if (use_chunk_staging) {
        const int fwd_dst_rank_idx = context.rdma_rank_idx * context.num_nvl_ranks + fwd_dst_nvl_rank_idx;
        const int fwd_src_rank_idx = cyclic ?
            mod_pos(fwd_dst_rank_idx - norm_shift, context.num_ranks) :
            (fwd_dst_rank_idx - shift);
        do_forward = fwd_src_rank_idx >= 0 and fwd_src_rank_idx < context.num_ranks and
                     fwd_src_rank_idx / context.num_nvl_ranks != context.rdma_rank_idx;

        int64_t total_staged_bytes = 0;
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            num_chunks += static_cast<int>(math::ceil_div<int64_t>(src_buckets[bucket_idx].num_bytes, num_chunk_bytes));
            total_staged_bytes += math::align<int64_t>(src_buckets[bucket_idx].num_bytes, kNumRDMAAlignmentBytes);
        }
        EP_HOST_ASSERT(num_chunks <= layout::RingSendRecvSignals::kNumMaxChunks and "Too many chunks");
        EP_HOST_ASSERT(total_staged_bytes <= layout::ChunkStorage::kNumBytes and
                       "Total ring_send_recv size exceeds ChunkStorage capacity for cross-rail forwarding");
    }

    // Compile & launch
    const auto kernel = jit->compile("bucket_ring_send_recv", std::format(R"(
#include <deep_ep/impls/bucket/ring_send_recv.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::ring_send_recv<{}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", context.is_scaleup_nvlink, use_chunk_staging, context.num_rdma_ranks, context.num_nvl_ranks,
        num_sms, num_qps, context.num_gpu_timeout_cycles, context_idx));

    jit->launch(
        kernel, {
            .stream = stream,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(kNumThreads, 1, 1),
            .cooperative = true,
        },
        context.dev_comm, context.window,
        context.workspace, context.buffer,
        num_buckets, do_forward ? num_chunks : 0, num_chunk_bytes,
        context.rdma_rank_idx, context.nvl_rank_idx,
        send_rdma ? dst_rdma_rank_idx : -1,
        src_buckets, dst_buckets);

    // Intra-node direct pushes over NVLink via Copy Engine
    if (send_nvl) {
        std::vector<void*> dst_ptrs, src_ptrs;
        std::vector<size_t> sizes;
        build_nvl_copies(dst_nvl_rank_idx, dst_ptrs, src_ptrs, sizes);
        driver::copy_engine_push(stream, dst_ptrs, src_ptrs, sizes);
    }

    // Forward arrived cross-rail RDMA chunks over NVLink via Copy Engine
    if (do_forward) {
        auto& workspace = *static_cast<layout::BucketWorkspace*>(context.workspace);
        auto& signals = workspace.signals[context_idx].ring_send_recv;
        void* const fwd_peer_buffer = context.get_sym_ptr(context.buffer, fwd_dst_nvl_rank_idx);
        std::vector<uint64_t*> wait_signals(num_qps);
        std::vector<void*> dst_ptrs(1), src_ptrs(1);
        std::vector<size_t> sizes(1);
        int chunk_idx = 0;
        int64_t stage_offset = 0;
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& [dst_offset, num_bucket_bytes] = dst_buckets[bucket_idx];
            for (int64_t chunk_offset = 0; chunk_offset < num_bucket_bytes; chunk_offset += num_chunk_bytes, ++ chunk_idx) {
                for (int qp_idx = 0; qp_idx < num_qps; ++ qp_idx)
                    wait_signals[qp_idx] = signals.tails[qp_idx] + chunk_idx;
                driver::copy_engine_wait(stream, wait_signals, 1);

                const auto num_bytes = static_cast<size_t>(
                    std::min<int64_t>(num_bucket_bytes - chunk_offset, num_chunk_bytes));
                src_ptrs[0] = workspace.chunk_storage.data + stage_offset + chunk_offset;
                dst_ptrs[0] = math::advance_ptr(fwd_peer_buffer, dst_offset + chunk_offset);
                sizes[0] = num_bytes;
                driver::copy_engine_push(stream, dst_ptrs, src_ptrs, sizes);
            }
            stage_offset += math::align<int64_t>(num_bucket_bytes, kNumRDMAAlignmentBytes);
        }
        EP_HOST_ASSERT(chunk_idx == num_chunks);
    }

    // Exit barrier is in the epilogue
    flush_stores = not use_chunk_staging;
}

}  // namespace deep_ep::bucket

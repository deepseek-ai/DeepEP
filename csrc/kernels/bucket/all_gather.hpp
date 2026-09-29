#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
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

static void launch_all_gather(
    const comm::Context& context,
    const int& num_buckets, const comm::BucketList& buckets,
    const std::optional<comm::BucketList>& src_buckets,
    const int& context_idx,
    uint64_t& seq_idx,
    const float& nvl_bandwidth, const float& rdma_bandwidth,
    const at::cuda::CUDAStream& stream) {
    EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets);
    EP_HOST_ASSERT((context.num_rdma_ranks == 1 or not src_buckets.has_value()) and
                   "Out-of-buffer inputs are only supported by the pure NVLink path");

    if (context.num_rdma_ranks > 1 and context.num_nvl_ranks == 1) {
        // Pure RDMA
        constexpr int kNumThreads = 512;
        const auto num_sms = jit->device.get_num_sms();

        // Compile
        const auto kernel = jit->compile("bucket_rdma_all_gather", std::format(R"(
#include <deep_ep/impls/bucket/all_gather/rdma.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::rdma_all_gather<{}, {}, {}, {}>);
}}
)", context.num_rdma_ranks, num_sms, context.num_gpu_timeout_cycles, context_idx));

        // Launch
        jit->launch(
            kernel, {
                .stream = stream,
                .grid_dim = dim3(num_sms, 1, 1),
                .block_dim = dim3(kNumThreads, 1, 1),
                .cooperative = true,
            },
            context.dev_comm, context.window,
            context.workspace, context.buffer,
            num_buckets, buckets
        );
    } else if (context.num_rdma_ranks == 1) {
        // Pure NVLink
        auto& workspace = *static_cast<layout::BucketWorkspace*>(context.workspace);
        auto& signals = workspace.signals[context_idx].copy_engine;
        EP_HOST_ASSERT(context.num_nvl_ranks <= kNumMaxNVLRanks and
                       "Number of NVLink ranks exceeds the maximum workspace size");

        // Build copy descriptors
        std::vector<size_t> sizes;
        std::vector<void*> dst_ptrs, src_ptrs;
        std::vector<uint64_t*> write_signals, wait_signals;
        write_signals.reserve(context.num_nvl_ranks - 1);
        wait_signals.reserve(context.num_nvl_ranks - 1);
        for (int rank_offset = 1; rank_offset < context.num_nvl_ranks; ++ rank_offset) {
            // Wait/write signals are same for prologue and epilogue
            const auto peer_rank_idx = (context.nvl_rank_idx + rank_offset) % context.num_nvl_ranks;
            write_signals.push_back(
                static_cast<uint64_t*>(context.get_sym_ptr(signals.base, peer_rank_idx)) + context.nvl_rank_idx);
            wait_signals.push_back(signals.base + peer_rank_idx);

            // Create copies
            for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
                const auto& [offset, num_bytes] = buckets[bucket_idx];
                const auto shard_offset = offset + context.nvl_rank_idx * num_bytes;
                const auto src_offset = src_buckets.has_value() ? (*src_buckets)[bucket_idx].offset : shard_offset;
                dst_ptrs.push_back(context.get_sym_ptr(math::advance_ptr(context.buffer, shard_offset), peer_rank_idx));
                src_ptrs.push_back(math::advance_ptr(context.buffer, src_offset));
                sizes.push_back(num_bytes);
            }
        }

        // Out-of-buffer inputs also need a local copy
        for (int bucket_idx = 0; src_buckets.has_value() and bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& [offset, num_bytes] = buckets[bucket_idx];
            const auto shard_offset = offset + context.nvl_rank_idx * num_bytes;
            if ((*src_buckets)[bucket_idx].offset == shard_offset)
                continue;

            dst_ptrs.push_back(math::advance_ptr(context.buffer, shard_offset));
            src_ptrs.push_back(math::advance_ptr(context.buffer, (*src_buckets)[bucket_idx].offset));
            sizes.push_back(num_bytes);
        }

        // Wait arrival & push & write
        driver::copy_engine_sync(stream, driver::increase_seq_idx(stream, seq_idx), write_signals, wait_signals);
        driver::copy_engine_push(stream, dst_ptrs, src_ptrs, sizes);
        driver::copy_engine_sync(stream, driver::increase_seq_idx(stream, seq_idx), write_signals, wait_signals);
    } else if (context.num_rdma_ranks > 1 and context.num_nvl_ranks > 1) {
        // Hybrid: one SM issues RDMA to one peer with `kNumQPsPerChunk` QPs, the CE forwards arrived chunks over NVLink
        constexpr int kNumThreads = 256;
        constexpr int kNumQPsPerChunk = layout::AllGatherSignals::kNumQPsPerChunk;
        EP_HOST_ASSERT(context.num_rdma_ranks <= jit->device.get_num_sms() and "Insufficient SMs for RDMA peers");
        EP_HOST_ASSERT(kNumQPsPerChunk <= context.num_allocated_qps and "Insufficient QPs");

        int64_t num_shard_bytes = 0;
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx)
            num_shard_bytes += buckets[bucket_idx].num_bytes;

        // Choose the number of chunks by estimating the single-stream CE timeline: the CE pushes the local shard, then
        // forwards each chunk wave (one chunk from every RDMA peer) once it lands, paying a launch gap per wave.
        // The chunk is floored so that each QP puts at least 512 KiB, where the NIC efficiency is stable
        constexpr int kNumMinChunkBytes = 4 * 1024 * 1024;
        const auto push_t = num_shard_bytes * (context.num_nvl_ranks - 1) / nvl_bandwidth;
        const auto rdma_t = num_shard_bytes * (context.num_rdma_ranks - 1) / rdma_bandwidth;
        const auto forward_t = push_t * (context.num_rdma_ranks - 1);
        int num_best_chunks = 1;
        float best_t = std::numeric_limits<float>::infinity();
        for (int i = 1; i <= std::min<int64_t>(64, num_shard_bytes / kNumMinChunkBytes); ++ i) {
            // CE-bound: forward all waves after the local push or the first wave; RDMA-bound: forward the last wave
            // NOTES: `10e-6` is CE overhead
            const auto t = std::max(std::max(push_t, rdma_t / i) + forward_t + i * (10e-6f), rdma_t + forward_t / i);
            if (t < best_t)
                best_t = t, num_best_chunks = i;
        }
        // Calculate in 64 bits, then bound each chunk before narrowing its byte count.
        const auto num_chunk_bytes = static_cast<int>(std::clamp<int64_t>(
            math::align<int64_t>(
                math::ceil_div<int64_t>(num_shard_bytes, num_best_chunks),
                kNumQPsPerChunk * kNumRDMAAlignmentBytes),
            4 * 1024, layout::ChunkStorage::kNumBytes
        ));

        // One completion signal per chunk on every QP
        int num_chunks = 0;
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx)
            num_chunks += static_cast<int>(math::ceil_div<int64_t>(buckets[bucket_idx].num_bytes, num_chunk_bytes));
        EP_HOST_ASSERT(num_chunks <= layout::AllGatherSignals::kNumMaxChunks and "Too many chunks");

        // Compile
        const auto kernel = jit->compile("bucket_hybrid_all_gather", std::format(R"(
#include <deep_ep/impls/bucket/all_gather/hybrid.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::bucket::hybrid_all_gather<{}, {}, {}, {}, {}>);
}}
)", context.num_rdma_ranks, context.num_nvl_ranks, kNumThreads, context.num_gpu_timeout_cycles, context_idx));

        // Launch
        jit->launch(
            kernel, {
                .stream = stream,
                .grid_dim = dim3(context.num_rdma_ranks, 1, 1),
                .block_dim = dim3(kNumThreads, 1, 1),
                .cooperative = true,
            },
            context.dev_comm, context.window,
            context.workspace, context.buffer,
            num_buckets, num_chunks, num_chunk_bytes, context.rdma_rank_idx, context.nvl_rank_idx, context.rank_idx,
            buckets
        );

        // NVLink peers' buffers
        std::vector<void*> lsa_buffers;
        lsa_buffers.reserve(context.num_nvl_ranks - 1);
        for (int rank_offset = 1; rank_offset < context.num_nvl_ranks; ++ rank_offset) {
            const int peer_rank_idx = (context.nvl_rank_idx + rank_offset) % context.num_nvl_ranks;
            lsa_buffers.push_back(context.get_sym_ptr(context.buffer, peer_rank_idx));
        }

        std::vector<size_t> sizes;
        std::vector<void*> dst_ptrs, src_ptrs;
        const auto push_to_nvl_peers = [&](const int64_t& src_offset, const size_t& num_bytes) {
            const auto src_ptr = math::advance_ptr(context.buffer, src_offset);
            for (const auto& lsa_buffer: lsa_buffers) {
                src_ptrs.push_back(src_ptr);
                dst_ptrs.push_back(math::advance_ptr(lsa_buffer, src_offset));
                sizes.push_back(num_bytes);
            }
        };
        const auto flush_pushes = [&]() {
            driver::copy_engine_push(stream, dst_ptrs, src_ptrs, sizes);
            src_ptrs.clear();
            dst_ptrs.clear();
            sizes.clear();
        };

        // Local shards over NVLink
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& [offset, num_bytes] = buckets[bucket_idx];
            push_to_nvl_peers(offset + context.rank_idx * num_bytes, num_bytes);
        }
        flush_pushes();

        // Forward RDMA results over NVLink, chunk by chunk
        auto& signals = static_cast<layout::BucketWorkspace*>(context.workspace)->signals[context_idx].all_gather;
        std::vector<uint64_t*> wait_signals(kNumQPsPerChunk);
        int chunk_idx = 0;
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& [offset, num_bucket_bytes] = buckets[bucket_idx];
            for (int64_t chunk_offset = 0; chunk_offset < num_bucket_bytes; chunk_offset += num_chunk_bytes, ++ chunk_idx) {
                // Wait for all RDMA peers to finish this chunk on every QP
                for (int qp_idx = 0; qp_idx < kNumQPsPerChunk; ++ qp_idx)
                    wait_signals[qp_idx] = signals.tails[qp_idx] + chunk_idx;
                driver::copy_engine_wait(stream, wait_signals, context.num_rdma_ranks - 1);

                const auto num_bytes = std::min<int64_t>(num_bucket_bytes - chunk_offset, num_chunk_bytes);
                for (int rank_offset = 1; rank_offset < context.num_rdma_ranks; ++ rank_offset) {
                    const int peer_rank_idx = (context.rdma_rank_idx + rank_offset) % context.num_rdma_ranks;
                    const auto src_rank_idx = peer_rank_idx * context.num_nvl_ranks + context.nvl_rank_idx;
                    push_to_nvl_peers(offset + src_rank_idx * num_bucket_bytes + chunk_offset, num_bytes);
                }
                flush_pushes();
            }
        }
        EP_HOST_ASSERT(chunk_idx == num_chunks);

        // Exit barrier is in the epilogue
    }
}

}  // namespace deep_ep::bucket

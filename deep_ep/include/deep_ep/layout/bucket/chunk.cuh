#pragma once

#include <cstdint>

#include <deep_ep/comm/bucket.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>

namespace deep_ep::layout {

struct alignas(kNumRDMAAlignmentBytes) ChunkStorage {
    static constexpr int kNumBytes = 256 * 1024 * 1024;
    EP_STATIC_ASSERT(kNumBytes % kNumRDMAAlignmentBytes == 0, "Invalid RDMA alignment");

    int8_t data[kNumBytes];
};

template <int kNumChunks, int kNumBytesPerChunk>
struct ChunkView1D {
    EP_STATIC_ASSERT(kNumChunks > 0, "Invalid number of chunks");
    EP_STATIC_ASSERT(kNumBytesPerChunk > 0 and kNumBytesPerChunk % kNumRDMAAlignmentBytes == 0,
                     "Invalid chunk size");
    EP_STATIC_ASSERT(kNumChunks * kNumBytesPerChunk <= ChunkStorage::kNumBytes,
                     "Chunk view exceeds the storage capacity");

    ChunkStorage& storage;

    __forceinline__ __device__ void* get_chunk_ptr(const int& chunk_idx) const {
        const int chunk_offset = chunk_idx * kNumBytesPerChunk;
        return storage.data + chunk_offset;
    }
};

template <int kNumRanks, int kNumChunksPerRank, int kNumBytesPerChunk>
struct ChunkView2D {
    EP_STATIC_ASSERT(kNumRanks > 0, "Invalid number of ranks");
    EP_STATIC_ASSERT(kNumChunksPerRank > 0, "Invalid number of chunks per rank");
    EP_STATIC_ASSERT(kNumBytesPerChunk > 0 and kNumBytesPerChunk % kNumRDMAAlignmentBytes == 0,
                     "Invalid chunk size");
    EP_STATIC_ASSERT(kNumRanks * kNumChunksPerRank * kNumBytesPerChunk <= ChunkStorage::kNumBytes,
                     "Chunk view exceeds the storage capacity");

    ChunkStorage& storage;

    __forceinline__ __device__ void* get_chunk_ptr(
        const int& rank_idx, const int& chunk_idx) const {
        const int global_chunk_idx = rank_idx * kNumChunksPerRank + chunk_idx;
        const int chunk_offset = global_chunk_idx * kNumBytesPerChunk;
        return storage.data + chunk_offset;
    }
};

// Iterates over all buckets as one flat sequence of `kNumBytesPerChunk`-sized chunks, e.g.
//     [bucket 0: chunk 0, 1, ..., n0 - 1][bucket 1: chunk n0, n0 + 1, ..., n0 + n1 - 1]...
// Each iterator visits chunks `start_chunk_idx, start_chunk_idx + kStride, ...`, so `kStride` iterators
// with distinct starts in `[0, kStride)` cover every chunk exactly once
template <int kNumBytesPerChunk, int kStride>
struct ChunkIterator {
    const int num_buckets;
    const comm::BucketList& buckets;
    int num_total_chunks = 0;           // Total across all buckets, including each bucket's partial tail

    // States, valid after a successful `next()`
    comm::Bucket bucket;
    int chunk_idx;                      // Global chunk index across all buckets
    int num_chunk_bytes;                // Only the last chunk of a bucket may be smaller than `kNumBytesPerChunk`
    int64_t chunk_offset_in_bucket;     // Byte offset of this chunk within `bucket`

    // NOTES: `int` is enough for chunk indices, as a chunk is normally at least several KB, which covers several TB in total
    __forceinline__ __device__ ChunkIterator(const int& num_buckets, const comm::BucketList& buckets, const int& start_chunk_idx):
        num_buckets(num_buckets), buckets(buckets), chunk_idx(start_chunk_idx - kStride) {
        for (int i = 0; i < num_buckets; ++ i)
            num_total_chunks += static_cast<int>(math::ceil_div<int64_t>(buckets[i].num_bytes, kNumBytesPerChunk));
    }

    __forceinline__ __device__ bool next() {
        // Move chunk
        chunk_idx += kStride;

        // Locate bucket
        while (bucket_idx < num_buckets) {
            bucket = buckets[bucket_idx];
            const int bucket_end_chunk_idx = bucket_start_chunk_idx + static_cast<int>(math::ceil_div<int64_t>(bucket.num_bytes, kNumBytesPerChunk));
            if (chunk_idx < bucket_end_chunk_idx) {
                chunk_offset_in_bucket = static_cast<int64_t>(chunk_idx - bucket_start_chunk_idx) * kNumBytesPerChunk;
                // NOTES: only the last chunk may be partial, so the remainder fits in `int` without a 64-bit `min`
                const bool is_last_chunk = chunk_idx + 1 == bucket_end_chunk_idx;
                num_chunk_bytes = is_last_chunk ? static_cast<int>(bucket.num_bytes - chunk_offset_in_bucket) : kNumBytesPerChunk;
                return true;
            }

            // Move into the next bucket
            bucket_start_chunk_idx = bucket_end_chunk_idx;
            bucket_idx += 1;
        }
        return false;
    }

private:
    int bucket_idx = 0, bucket_start_chunk_idx = 0;
};

}  // namespace deep_ep::layout

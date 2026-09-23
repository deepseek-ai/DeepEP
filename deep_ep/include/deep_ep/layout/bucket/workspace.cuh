#pragma once

#include <deep_ep/layout/bucket/all_gather.cuh>
#include <deep_ep/layout/bucket/all_reduce.cuh>
#include <deep_ep/layout/common/barrier.cuh>
#include <deep_ep/layout/bucket/chunk.cuh>
#include <deep_ep/layout/bucket/copy_engine.cuh>
#include <deep_ep/layout/bucket/reduce_scatter.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

namespace deep_ep::layout {

struct alignas(kNumRDMAAlignmentBytes) BucketSignals {
    BarrierSignals barrier;
    CopyEngineSignals copy_engine;
    AllGatherSignals all_gather;
    ReduceScatterSignals reduce_scatter;
    AllReduceSignals all_reduce;
};
EP_STATIC_ASSERT(sizeof(BucketSignals) * kNumMaxContexts <= kNumMaxSignalBytes, "Too many signals");

struct alignas(kNumRDMAAlignmentBytes) BucketWorkspace {
    BucketSignals signals[kNumMaxContexts];
    ChunkStorage chunk_storage;
};

}  // namespace deep_ep::layout

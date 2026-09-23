#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>

#ifdef __CUDACC__
#include <cub/block/block_scan.cuh>
#include <deep_ep/comm/handle.cuh>
#endif

namespace deep_ep::layout {

// One exchange pair between an original expert and a redundant expert slot
struct EPLBTask {
    int local_expert_idx;
    int dst_nvl_rank_idx;
    int redundant_expert_idx;
};

struct alignas(kNumTMAAlignmentBytes) EPLBSharedMemoryLayout {
    static constexpr int kNumBytes = 32 * 1024;
    static constexpr int kNumMaxTasks = (kNumBytes - sizeof(int)) / sizeof(EPLBTask);

    int num_tasks;
    EPLBTask tasks[kNumMaxTasks];

#ifdef __CUDACC__
    template <int kNumRanks, int kNumLocalExperts, int kNumRedundantExperts, int kNumThreads, int kNumWorkspaceBytes>
    __forceinline__ __device__ void build_tasks(const int* redundancy_mapping, const int& rank_idx, const int& thread_idx,
                                               void* smem_workspace_ptr) {
        EP_STATIC_ASSERT(kNumLocalExperts > 0, "Invalid number of local experts");

        // Reuse TMA storage for the block scan
        using BlockScan = cub::BlockScan<int, kNumThreads>;
        EP_STATIC_ASSERT(sizeof(typename BlockScan::TempStorage) <= kNumWorkspaceBytes, "Insufficient shared-memory workspace");
        auto& smem_workspace = *static_cast<typename BlockScan::TempStorage*>(smem_workspace_ptr);

        // Compact local expert exchanges in mapping order
        int num_found_tasks = 0;
        for (int base = 0; base < kNumRanks * kNumRedundantExperts; base += kNumThreads) {
            const int i = base + thread_idx;
            const int expert_idx = i < kNumRanks * kNumRedundantExperts ? redundancy_mapping[i] : -1;
            const int is_local = expert_idx >= 0 and expert_idx / kNumLocalExperts == rank_idx;
            int task_idx, num_iter_found_tasks;
            BlockScan(smem_workspace).ExclusiveSum(is_local, task_idx, num_iter_found_tasks);
            if (is_local) {
                task_idx += num_found_tasks;
                EP_DEVICE_ASSERT(task_idx < kNumMaxTasks);
                tasks[task_idx] = {
                    .local_expert_idx = expert_idx % kNumLocalExperts,
                    .dst_nvl_rank_idx = i / kNumRedundantExperts,
                    .redundant_expert_idx = i % kNumRedundantExperts
                };
            }
            num_found_tasks += num_iter_found_tasks;
            __syncthreads();
        }

        // Publish task count
        if (ptx::get_warp_idx() == 0 and ptx::elect_one_sync())
            num_tasks = num_found_tasks;
        __syncthreads();
    }
#endif
};
EP_STATIC_ASSERT(sizeof(EPLBSharedMemoryLayout) <= EPLBSharedMemoryLayout::kNumBytes, "LB metadata exceeds shared-memory capacity");

static constexpr int kNumMaxWeightEntries = 16;

struct EPWeight {
    void* redundant_expert_weights;
    void* expert_weights;
    int64_t num_bytes_per_expert;
};

struct EPWeightList {
    EPWeight weights[kNumMaxWeightEntries] = {};

    __forceinline__ __device__ __host__ EPWeight& operator[](const int& weight_idx) {
        return weights[weight_idx];
    }

    __forceinline__ __device__ __host__ const EPWeight& operator[](const int& weight_idx) const {
        return weights[weight_idx];
    }
};

#ifdef __CUDACC__
// Claim chunks in weight, chunk, task order
template <int kNumWeights, int kNumBytesPerChunk>
struct EPWeightIterator {
    const comm::NCCLGin& gin;
    const EPLBSharedMemoryLayout& smem_layout;
    const EPWeight* weights;
    const int num_tasks;
    int* chunk_idx_counter;

    // States, valid after a successful `next()`
    int num_chunk_bytes;
    void *src_ptr, *dst_ptr;

    __forceinline__ __device__ EPWeightIterator(const comm::NCCLGin& gin, const EPLBSharedMemoryLayout& smem_layout,
                                                const EPWeight* weights,
                                                int* chunk_idx_counter):
        gin(gin), smem_layout(smem_layout), weights(weights), num_tasks(smem_layout.num_tasks),
        chunk_idx_counter(chunk_idx_counter) {
    }

    __forceinline__ __device__ bool next() {
        // Claim a chunk
        int chunk_idx = atomicAdd(chunk_idx_counter, 1) - num_skipped_chunks;

        // Advance weight monotonically
        while (weight_idx < kNumWeights) {
            const int num_chunks = static_cast<int>(math::ceil_div<int64_t>(weights[weight_idx].num_bytes_per_expert, kNumBytesPerChunk)) * num_tasks;
            if (chunk_idx < num_chunks)
                break;

            chunk_idx -= num_chunks;
            num_skipped_chunks += num_chunks;
            weight_idx ++;
        }
        if (weight_idx == kNumWeights)
            return false;

        // Locate task and resolve pointers
        const auto& [redundant_expert_weights, expert_weights, num_bytes_per_expert] = weights[weight_idx];
        const auto& [local_expert_idx, dst_nvl_rank_idx, redundant_expert_idx] = smem_layout.tasks[chunk_idx % num_tasks];
        const int64_t chunk_offset = static_cast<int64_t>(chunk_idx / num_tasks) * kNumBytesPerChunk;
        num_chunk_bytes = static_cast<int>(min(static_cast<int64_t>(kNumBytesPerChunk), num_bytes_per_expert - chunk_offset));
        src_ptr = math::advance_ptr(expert_weights, local_expert_idx * num_bytes_per_expert + chunk_offset);
        dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(math::advance_ptr(redundant_expert_weights,
            redundant_expert_idx * num_bytes_per_expert + chunk_offset), dst_nvl_rank_idx);
        return true;
    }

private:
    int weight_idx = 0, num_skipped_chunks = 0;
};
#endif

}  // namespace deep_ep::layout

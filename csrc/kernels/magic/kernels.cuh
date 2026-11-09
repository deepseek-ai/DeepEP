#pragma once

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <nccl.h>
#include <nccl_device.h>
#include <type_traits>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/ptx.cuh>

#include "ptx.cuh"

namespace deep_ep::magic::kernels {

namespace common_ptx = deep_ep::ptx;

using namespace deep_ep::magic::ptx;

static constexpr unsigned kNumStages = 2;
static constexpr unsigned kMaxWorldSize = 64;
// Reserve a 128-byte header region to keep the payload 128-byte aligned.
static constexpr unsigned kHeaderBytes = 128;
static constexpr unsigned kAllToAllThreads = 64;
static constexpr unsigned kAllToAllWarps = kAllToAllThreads / 32;
static constexpr unsigned kReduceThreads = 256;

__host__ __device__ __forceinline__ unsigned align_bytes(unsigned bytes, unsigned alignment) {
    return (bytes + alignment - 1) / alignment * alignment;
}

struct MagicSlotLayout {
    size_t rank_slot_bytes;
    unsigned row_bytes, lsa_bulk_rows, gin_bulk_rows;
    unsigned lsa_first_rank, lsa_size;

    __host__ __device__ __forceinline__ bool is_lsa_peer(unsigned peer) const {
        return peer >= lsa_first_rank && peer < lsa_first_rank + lsa_size;
    }
    __host__ __device__ __forceinline__ unsigned gin_bulk_bytes() const { return gin_bulk_rows * row_bytes; }
    __host__ __device__ __forceinline__ unsigned gin_warp_smem_bytes() const {
        return align_bytes(gin_bulk_bytes() + sizeof(common_ptx::mbarrier), kHeaderBytes);
    }
    __host__ __device__ __forceinline__ size_t row_offset(unsigned peer, unsigned row_idx) const {
        const unsigned bulk_rows = is_lsa_peer(peer) ? lsa_bulk_rows : gin_bulk_rows;
        const size_t bulk_bytes = static_cast<size_t>(bulk_rows) * row_bytes;
        const size_t bulk_stride = (bulk_bytes + kHeaderBytes - 1) / kHeaderBytes * kHeaderBytes;
        return kHeaderBytes + (row_idx / bulk_rows) * bulk_stride + static_cast<size_t>(row_idx % bulk_rows) * row_bytes;
    }
};

struct MagicRowLayout {
    unsigned token_bytes, scale_bytes, topk;

    __host__ __device__ __forceinline__ unsigned topk_offset() const { return token_bytes + scale_bytes; }
    __host__ __device__ __forceinline__ unsigned row_bytes() const {
        return align_bytes(topk_offset() + topk * (sizeof(int32_t) + sizeof(float)), sizeof(uint4));
    }
    __host__ __device__ __forceinline__ unsigned num_topk_vectors() const { return (row_bytes() - topk_offset()) / sizeof(uint4); }
};

static constexpr unsigned kMagicX = 0x4e344617u;
static constexpr unsigned kMagicY = 0x9978b15du;
static constexpr unsigned kMagicZ = 0xf2e48b2bu;
static constexpr unsigned kMagicW = 0xbea6b5e2u;

__host__ __device__ __forceinline__ uint4 magic_uint4() {
    return make_uint4(kMagicX, kMagicY, kMagicZ, kMagicW);
}

__device__ __forceinline__ bool is_magic(const uint4& value) {
    return (value.x == kMagicX && value.y == kMagicY) || (value.z == kMagicZ && value.w == kMagicW);
}

__device__ __forceinline__ unsigned claim_cta_task(unsigned* counter, unsigned& shared_task_idx) {
    __syncthreads();
    if (threadIdx.x == 0)
        shared_task_idx = atomicAdd(counter, 1);
    __syncthreads();
    return shared_task_idx;
}

__device__ __forceinline__ bool mark_send_task_done(unsigned& num_completed_tasks, unsigned num_tasks) {
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> ref(num_completed_tasks);
    return ref.fetch_add(1, cuda::memory_order_acq_rel) + 1 == num_tasks;
}

__device__ __forceinline__ uint2 poll_recv_header(const uint2* header_ptr, uint32_t header_seq) {
    uint2 header;
    do {
        header = ld_volatile_global(header_ptr);
    } while (header.y != header_seq);
    return header;
}

__device__ __forceinline__ uint4 recv_vector(uint4* src, const uint64_t* completion_slot, uint64_t seq_value, bool& completion_seen) {
    uint4 value = ld_volatile_global(src);
    while (!completion_seen && is_magic(value)) {
        completion_seen = ld_acquire_sys_global(completion_slot) == seq_value;
        value = ld_volatile_global(src);
    }
    st_volatile_global(magic_uint4(), src);
    return value;
}

__device__ __forceinline__ void copy_vectors(const uint4* src, uint4* dst, unsigned num_vectors) {
    for (unsigned i = threadIdx.x; i < num_vectors; i += blockDim.x)
        dst[i] = src[i];
}

__device__ __forceinline__ void st_volatile_vectors(const uint4* src, uint4* dst, unsigned num_vectors) {
    for (unsigned i = threadIdx.x; i < num_vectors; i += blockDim.x)
        st_volatile_global(src[i], dst + i);
}

__device__ __forceinline__ void recv_vectors(
    uint4* src, uint4* dst, unsigned num_vectors, const uint64_t* completion_slot, uint64_t seq_value, bool& completion_seen) {
    for (unsigned i = threadIdx.x; i < num_vectors; i += blockDim.x)
        dst[i] = recv_vector(src + i, completion_slot, seq_value, completion_seen);
}

__device__ __forceinline__ unsigned combine_staging_row_idx(
    const unsigned* staging_row_indices, unsigned max_tokens_per_rank, unsigned peer, unsigned row_idx) {
    return staging_row_indices[static_cast<size_t>(peer) * max_tokens_per_rank + row_idx];
}

struct DispatchOperationArgs {
    const uint8_t* x;
    const uint8_t* x_scale;
    const int32_t* topk_ids;
    const float* topk_weights;
    unsigned* staging_row_indices;
    unsigned* token_row_counts;
    unsigned num_tokens;
    unsigned max_tokens_per_rank;
    unsigned num_local_experts;
    uint8_t* recv_x;
    uint8_t* recv_x_scale;
    int32_t* recv_topk_ids;
    float* recv_topk_weights;
    int32_t* psum_recv_per_rank;
    unsigned* recv_peer_row_start;
    MagicRowLayout row;

    template <int kNumRanks>
    __device__ __forceinline__ unsigned token_idx(unsigned peer, unsigned row_idx) const {
        return combine_staging_row_idx(staging_row_indices, max_tokens_per_rank, peer, row_idx) / kNumRanks;
    }
    template <int kNumRanks>
    __device__ __forceinline__ unsigned topk_peer(unsigned token_idx, unsigned topk_idx) const {
        const int32_t expert = topk_ids[static_cast<size_t>(token_idx) * row.topk + topk_idx];
        return expert < 0 ? kNumRanks : expert / num_local_experts;
    }
    __device__ __forceinline__ unsigned tma_row_bytes() const { return row.topk_offset(); }
    __device__ __forceinline__ const uint4* x_row(unsigned token_idx) const {
        return reinterpret_cast<const uint4*>(x + static_cast<size_t>(token_idx) * row.token_bytes);
    }
    __device__ __forceinline__ const uint4* x_scale_row(unsigned token_idx) const {
        return reinterpret_cast<const uint4*>(x_scale + static_cast<size_t>(token_idx) * row.scale_bytes);
    }
    __device__ __forceinline__ uint4 load_topk_vector(unsigned token_idx, unsigned vector_idx) const {
        const unsigned topk_idx = vector_idx * 2;
        const size_t offset = static_cast<size_t>(token_idx) * row.topk + topk_idx;
        const auto* ids = topk_ids + offset;
        const auto* weights = topk_weights + offset;
        if (row.topk % 2 == 0) {
            const uint2 id_pair = *reinterpret_cast<const uint2*>(ids);
            const uint2 weight_pair = *reinterpret_cast<const uint2*>(weights);
            return make_uint4(id_pair.x, weight_pair.x, id_pair.y, weight_pair.y);
        }
        const bool has_pair = topk_idx + 1 < row.topk;
        return make_uint4(static_cast<uint32_t>(ids[0]),
                          __float_as_uint(weights[0]),
                          has_pair ? static_cast<uint32_t>(ids[1]) : 0,
                          has_pair ? __float_as_uint(weights[1]) : 0);
    }
    __device__ __forceinline__ uint4 load_vector(unsigned token_idx, unsigned vector_idx) const {
        const unsigned num_x_vectors = row.token_bytes / sizeof(uint4);
        const unsigned topk_vector_offset = row.topk_offset() / sizeof(uint4);
        if (vector_idx < num_x_vectors)
            return x_row(token_idx)[vector_idx];
        if (vector_idx < topk_vector_offset)
            return x_scale_row(token_idx)[vector_idx - num_x_vectors];
        return load_topk_vector(token_idx, vector_idx - topk_vector_offset);
    }
    __device__ __forceinline__ void store_topk_vector(unsigned out_row, unsigned vector_idx, const uint4& value) const {
        const unsigned topk_idx = vector_idx * 2;
        const size_t offset = static_cast<size_t>(out_row) * row.topk + topk_idx;
        auto* ids = recv_topk_ids + offset;
        auto* weights = recv_topk_weights + offset;
        if (row.topk % 2 == 0) {
            *reinterpret_cast<uint2*>(ids) = make_uint2(value.x, value.z);
            *reinterpret_cast<uint2*>(weights) = make_uint2(value.y, value.w);
        } else {
            ids[0] = static_cast<int32_t>(value.x);
            weights[0] = __uint_as_float(value.y);
            if (topk_idx + 1 < row.topk) {
                ids[1] = static_cast<int32_t>(value.z);
                weights[1] = __uint_as_float(value.w);
            }
        }
    }
};

struct CombineOperationArgs {
    const uint8_t* x;
    const unsigned* recv_peer_row_start;
    uint8_t* staging;
    const unsigned* staging_row_indices;
    unsigned* dispatch_send_counts;
    unsigned row_bytes;
    unsigned max_tokens_per_rank;

    __device__ __forceinline__ unsigned tma_row_bytes() const { return row_bytes; }
    __device__ __forceinline__ const uint4* x_row(unsigned peer, unsigned row_idx) const {
        return reinterpret_cast<const uint4*>(x + static_cast<size_t>(recv_peer_row_start[peer] + row_idx) * row_bytes);
    }
};

template <int kNumRanks>
__device__ __forceinline__ void dispatch_prologue(const DispatchOperationArgs& args, unsigned* send_counts) {
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const unsigned num_warps = gridDim.x * blockDim.x / 32;
    const unsigned lower_lane_mask = (1u << lane) - 1;
    for (unsigned token_idx = warp_idx; token_idx < args.num_tokens; token_idx += num_warps) {
        const unsigned peer = lane < args.row.topk ? args.topk_peer<kNumRanks>(token_idx, lane) : kNumRanks;
        const unsigned same_peer_lanes = __match_any_sync(0xffffffffu, peer);
        const bool is_peer_leader = peer < kNumRanks && (same_peer_lanes & lower_lane_mask) == 0;
        const unsigned peer_leader_mask = __ballot_sync(0xffffffffu, is_peer_leader);
        if (lane == 0)
            args.token_row_counts[token_idx] = __popc(peer_leader_mask);
        if (is_peer_leader) {
            const unsigned row_idx = atomicAdd(send_counts + peer, 1);
            args.staging_row_indices[static_cast<size_t>(peer) * args.max_tokens_per_rank + row_idx] =
                token_idx * kNumRanks + __popc(peer_leader_mask & lower_lane_mask);
        }
    }
}

template <int kNumRanks>
__device__ __forceinline__ void lsa_send_bulk(
    const DispatchOperationArgs& args, unsigned peer, unsigned first_row, unsigned num_rows, uint4* dst) {
    const unsigned num_vectors = args.row.row_bytes() / sizeof(uint4);
    for (unsigned i = 0; i < num_rows; ++i) {
        const unsigned token_idx = args.token_idx<kNumRanks>(peer, first_row + i);
        for (unsigned j = threadIdx.x; j < num_vectors; j += blockDim.x)
            st_volatile_global(args.load_vector(token_idx, j), dst + j);
        dst += num_vectors;
    }
}

template <int kNumRanks>
__device__ __forceinline__ void tma_load_bulk(
    const DispatchOperationArgs& args, unsigned peer, unsigned first_row, unsigned num_rows, uint8_t* smem, common_ptx::mbarrier* barrier) {
    const auto& row = args.row;
    const unsigned i = threadIdx.x % 32;
    if (i < num_rows) {
        const unsigned token_idx = args.token_idx<kNumRanks>(peer, first_row + i);
        auto* dst = smem + static_cast<size_t>(i) * row.row_bytes();
        common_ptx::tma_load_1d(dst, args.x_row(token_idx), barrier, row.token_bytes, common_ptx::L2CacheHint::kEvictNormal);
        if (row.scale_bytes != 0)
            common_ptx::tma_load_1d(
                dst + row.token_bytes, args.x_scale_row(token_idx), barrier, row.scale_bytes, common_ptx::L2CacheHint::kEvictNormal);
    }
    constexpr unsigned kTopkLanes = 4;
    for (unsigned row_idx = i / kTopkLanes; row_idx < num_rows; row_idx += 32 / kTopkLanes) {
        const unsigned token_idx = args.token_idx<kNumRanks>(peer, first_row + row_idx);
        auto* dst = reinterpret_cast<uint4*>(smem + static_cast<size_t>(row_idx) * row.row_bytes() + row.topk_offset());
        for (unsigned vector_idx = i % kTopkLanes; vector_idx < row.num_topk_vectors(); vector_idx += kTopkLanes)
            dst[vector_idx] = args.load_topk_vector(token_idx, vector_idx);
    }
    common_ptx::tma_store_fence();
}

template <int kNumRanks>
__device__ __forceinline__ void dispatch_epilogue(const DispatchOperationArgs& args,
                                                  unsigned peer,
                                                  unsigned row_idx,
                                                  unsigned out_row,
                                                  bool is_remote,
                                                  uint4* src,
                                                  const uint64_t* completion_slot,
                                                  uint64_t seq_value) {
    const auto& row = args.row;
    const unsigned num_x_vectors = row.token_bytes / sizeof(uint4), num_scale_vectors = row.scale_bytes / sizeof(uint4);
    auto* recv_x = reinterpret_cast<uint4*>(args.recv_x + static_cast<size_t>(out_row) * row.token_bytes);
    auto* recv_x_scale = reinterpret_cast<uint4*>(args.recv_x_scale + static_cast<size_t>(out_row) * row.scale_bytes);
    if (!is_remote) {
        const unsigned token_idx = args.token_idx<kNumRanks>(peer, row_idx);
        copy_vectors(args.x_row(token_idx), recv_x, num_x_vectors);
        copy_vectors(args.x_scale_row(token_idx), recv_x_scale, num_scale_vectors);
        for (unsigned i = (threadIdx.x + blockDim.x - 32) % blockDim.x; i < row.num_topk_vectors(); i += blockDim.x)
            args.store_topk_vector(out_row, i, args.load_topk_vector(token_idx, i));
        return;
    }
    bool completion_seen = false;
    recv_vectors(src, recv_x, num_x_vectors, completion_slot, seq_value, completion_seen);
    recv_vectors(src + num_x_vectors, recv_x_scale, num_scale_vectors, completion_slot, seq_value, completion_seen);
    src += row.topk_offset() / sizeof(uint4);
    for (unsigned i = (threadIdx.x + blockDim.x - 32) % blockDim.x; i < row.num_topk_vectors(); i += blockDim.x)
        args.store_topk_vector(out_row, i, recv_vector(src + i, completion_slot, seq_value, completion_seen));
}

template <int kNumRanks>
__device__ __forceinline__ void lsa_send_bulk(
    const CombineOperationArgs& args, unsigned peer, unsigned first_row, unsigned num_rows, uint4* dst) {
    st_volatile_vectors(args.x_row(peer, first_row), dst, num_rows * (args.row_bytes / sizeof(uint4)));
}

template <int kNumRanks>
__device__ __forceinline__ void tma_load_bulk(
    const CombineOperationArgs& args, unsigned peer, unsigned first_row, unsigned num_rows, uint8_t* smem, common_ptx::mbarrier* barrier) {
    if (threadIdx.x % 32 == 0)
        common_ptx::tma_load_1d(smem,
                                args.x_row(peer, first_row),
                                barrier,
                                num_rows * args.row_bytes,
                                common_ptx::L2CacheHint::kEvictNormal);
}

__device__ __forceinline__ void combine_epilogue(const CombineOperationArgs& args,
                                                 unsigned peer,
                                                 unsigned row_idx,
                                                 bool is_remote,
                                                 uint4* src,
                                                 const uint64_t* completion_slot,
                                                 uint64_t seq_value) {
    const unsigned staging_row_idx = combine_staging_row_idx(args.staging_row_indices, args.max_tokens_per_rank, peer, row_idx);
    auto* dst = reinterpret_cast<uint4*>(args.staging + static_cast<size_t>(staging_row_idx) * args.row_bytes);
    const unsigned num_vectors = args.row_bytes / sizeof(uint4);
    if (!is_remote) {
        copy_vectors(args.x_row(peer, row_idx), dst, num_vectors);
        return;
    }
    bool completion_seen = false;
    recv_vectors(src, dst, num_vectors, completion_slot, seq_value, completion_seen);
}

struct GinSendBuffer {
    uint8_t* rows;
    common_ptx::mbarrier* barrier;
    unsigned row_bytes, tma_row_bytes;
    common_ptx::arrival_phase phase;

    __device__ __forceinline__ void init(uint8_t* smem, const MagicSlotLayout& slot, unsigned tma_row_bytes) {
        rows = smem + threadIdx.x / 32 * slot.gin_warp_smem_bytes();
        row_bytes = slot.row_bytes;
        this->tma_row_bytes = tma_row_bytes;
        barrier = reinterpret_cast<common_ptx::mbarrier*>(rows + slot.gin_bulk_bytes());
        phase = 0;
        if (threadIdx.x % 32 == 0)
            common_ptx::mbarrier_init_with_fence(barrier);
        __syncwarp();
    }

    template <int kNumRanks, typename OperationArgs>
    __device__ __forceinline__ void copy_bulk(
        const OperationArgs& operation_args, unsigned peer, unsigned first_row, unsigned num_rows, uint8_t* dst) {
        tma_load_bulk<kNumRanks>(operation_args, peer, first_row, num_rows, rows, barrier);
        if (threadIdx.x % 32 == 0) {
            common_ptx::mbarrier_arrive_and_set_tx(barrier, num_rows * tma_row_bytes);
            common_ptx::mbarrier_wait_and_flip_phase(barrier, phase);
        }
        __syncwarp();
        if (threadIdx.x % 32 == 0)
            common_ptx::tma_store_1d(dst, rows, num_rows * row_bytes);
        common_ptx::tma_store_commit();
        common_ptx::tma_store_wait();
        __syncwarp();
    }
};

__global__ void initialize_magic_workspace_kernel(void* recv_workspace, size_t bytes) {
    const size_t thread_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t num_vectors = bytes / sizeof(uint4);
    auto* vectors = static_cast<uint4*>(recv_workspace);
    const uint4 value = magic_uint4();
    for (size_t offset = thread_idx; offset < num_vectors; offset += stride)
        vectors[offset] = value;
}

struct MagicMemoryRegion {
    ncclWindow_t window;
    void* base;
};

struct MagicCommState {
    ncclDevComm_t dev_comm;
    MagicMemoryRegion send_staging;
    MagicMemoryRegion recv_workspace;
    MagicMemoryRegion completion;
};

struct MagicAllToAllState {
    uint64_t seq;
    unsigned lsa_send_task_counter;
    unsigned send_completed_tasks[kMaxWorldSize];
    // High 32 bits count rows; low 32 bits count peers.
    unsigned long long recv_row_and_peer_counter;
};

template <int kNumRanks, typename OperationArgs>
struct MagicAllToAllArgs {
    MagicCommState comm_state;
    MagicAllToAllState* state;
    void* lsa_peer_workspaces[kNumRanks];
    uint64_t* lsa_peer_completion_ptrs[kNumRanks];
    unsigned* send_counts;
    unsigned* recv_counts;
    MagicSlotLayout slot;
    OperationArgs operation_args;
    unsigned num_lsa_send_ctas;
    unsigned num_gin_send_ctas;
};

template <int kNumRanks, bool kUseGin, typename OperationArgs>
__global__ void magic_all_to_all_kernel(MagicAllToAllArgs<kNumRanks, OperationArgs> args) {
    constexpr bool kIsDispatch = std::is_same_v<OperationArgs, DispatchOperationArgs>;
    extern __shared__ __align__(kHeaderBytes) uint8_t smem[];
    __shared__ unsigned shared_task_idx;
    const unsigned rank = args.comm_state.dev_comm.rank;
    const unsigned lsa_size = kUseGin ? args.comm_state.dev_comm.lsaSize : kNumRanks;
    const unsigned lsa_first_rank = rank / lsa_size * lsa_size;
    const unsigned num_send_ctas = args.num_lsa_send_ctas + args.num_gin_send_ctas;
    const bool is_lsa_send = blockIdx.x < args.num_lsa_send_ctas;
    const bool is_recv = blockIdx.x >= num_send_ctas;
    auto& state = *args.state;
    const uint64_t seq_value = state.seq + 1;
    const uint32_t header_seq = static_cast<uint32_t>(seq_value);
    const unsigned stage = static_cast<unsigned>((seq_value - 1) % kNumStages);
    const unsigned stage_slot_offset = stage * kNumRanks;
    auto slot_ptr = [&](void* base, unsigned slot_idx) {
        return reinterpret_cast<uint4*>(static_cast<uint8_t*>(base) + slot_idx * args.slot.rank_slot_bytes);
    };
    auto slot_row_ptr = [&](void* base, unsigned slot_idx, unsigned peer, unsigned row_idx) {
        return slot_ptr(base, slot_idx) + args.slot.row_offset(peer, row_idx) / sizeof(uint4);
    };

    if (blockIdx.x == 0) {
        for (unsigned i = threadIdx.x; i < kNumRanks; i += blockDim.x) {
            state.send_completed_tasks[i] = 0;
            if constexpr (!kIsDispatch)
                args.operation_args.dispatch_send_counts[i] = 0;
        }
        if (threadIdx.x == 0) {
            state.lsa_send_task_counter = 0;
            state.recv_row_and_peer_counter = 0;
        }
    }
    if constexpr (kIsDispatch)
        dispatch_prologue<kNumRanks>(args.operation_args, args.send_counts);
    cooperative_groups::this_grid().sync();
    if (blockIdx.x == 0 && threadIdx.x == 0)
        state.seq = seq_value;

    if (is_lsa_send) {
        if (blockIdx.x == 0) {
            for (unsigned peer = lsa_first_rank + threadIdx.x; peer < lsa_first_rank + lsa_size; peer += blockDim.x) {
                if (peer != rank)
                    st_volatile_global(make_uint4(args.send_counts[peer], header_seq, 0, 0),
                                       slot_ptr(args.lsa_peer_workspaces[peer], stage_slot_offset + rank));
            }
        }
        unsigned max_count = 0;
        for (unsigned peer = lsa_first_rank + threadIdx.x % 32; peer < lsa_first_rank + lsa_size; peer += 32) {
            if (peer != rank)
                max_count = max(max_count, args.send_counts[peer]);
        }
        max_count = __reduce_max_sync(0xffffffffu, max_count);
        const unsigned bulk_rows = args.slot.lsa_bulk_rows;
        const unsigned max_num_bulks = (max_count + bulk_rows - 1) / bulk_rows;
        for (;;) {
            const unsigned task_idx = claim_cta_task(&state.lsa_send_task_counter, shared_task_idx);
            const unsigned peer = lsa_first_rank + (rank + 1 + task_idx) % lsa_size;
            const unsigned bulk_idx = task_idx / lsa_size;
            if (bulk_idx >= max_num_bulks)
                break;
            const unsigned count = args.send_counts[peer];
            const unsigned first_row = bulk_idx * bulk_rows;
            if (first_row >= count || peer == rank)
                continue;
            const unsigned num_rows = min(bulk_rows, count - first_row);
            auto* dst = slot_row_ptr(args.lsa_peer_workspaces[peer], stage_slot_offset + rank, peer, first_row);
            lsa_send_bulk<kNumRanks>(args.operation_args, peer, first_row, num_rows, dst);
            __syncthreads();
            const unsigned num_peer_bulks = (count + bulk_rows - 1) / bulk_rows;
            if (threadIdx.x == 0 && mark_send_task_done(state.send_completed_tasks[peer], num_peer_bulks))
                st_release_sys_global(seq_value, args.lsa_peer_completion_ptrs[peer] + stage_slot_offset + rank);
        }
    } else if (!is_recv) {
        if constexpr (kUseGin) {
            const unsigned num_gin_peers = kNumRanks - lsa_size;
            const unsigned gin_idx = (blockIdx.x - args.num_lsa_send_ctas) * kAllToAllWarps + threadIdx.x / 32;
            if (gin_idx >= num_gin_peers)
                return;
            const unsigned peer = (lsa_first_rank + lsa_size + gin_idx) % kNumRanks;
            GinSendBuffer send_buffer;
            send_buffer.init(smem, args.slot, args.operation_args.tma_row_bytes());
            const ncclGin gin(args.comm_state.dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_THREAD);
            const auto team = ncclTeamWorld(args.comm_state.dev_comm);
            const unsigned bulk_rows = args.slot.gin_bulk_rows;
            const unsigned count = args.send_counts[peer];
            const unsigned num_peer_bulks = max(1, (count + bulk_rows - 1) / bulk_rows);
            const unsigned src_slot = stage_slot_offset + peer, dst_slot = stage_slot_offset + rank;
            if (threadIdx.x % 32 == 0)
                slot_ptr(args.comm_state.send_staging.base, src_slot)[0] = make_uint4(count, header_seq, 0, 0);
            for (unsigned bulk_idx = 0; bulk_idx < num_peer_bulks; ++bulk_idx) {
                const unsigned first_row = bulk_idx * bulk_rows;
                const unsigned num_rows = min(bulk_rows, count - first_row);
                if (num_rows != 0)
                    send_buffer.copy_bulk<kNumRanks>(
                        args.operation_args,
                        peer,
                        first_row,
                        num_rows,
                        reinterpret_cast<uint8_t*>(slot_row_ptr(args.comm_state.send_staging.base, src_slot, peer, first_row)));
                if (threadIdx.x % 32 == 0) {
                    const size_t offset = bulk_idx == 0 ? 0 : args.slot.row_offset(peer, first_row);
                    const size_t bytes = (bulk_idx == 0 ? kHeaderBytes : 0) + static_cast<size_t>(num_rows) * args.slot.row_bytes;
                    gin.put(team,
                            peer,
                            args.comm_state.recv_workspace.window,
                            dst_slot * args.slot.rank_slot_bytes + offset,
                            args.comm_state.send_staging.window,
                            src_slot * args.slot.rank_slot_bytes + offset,
                            bytes);
                }
            }
            if (threadIdx.x % 32 == 0)
                gin.signal(team, peer, ncclGin_VASignalAdd{args.comm_state.completion.window, dst_slot * sizeof(uint64_t), kNumStages});
        }
    } else {
        __shared__ unsigned shared_num_recv_rows, shared_recv_row_start;
        const unsigned peer = blockIdx.x - num_send_ctas;
        const bool is_remote = peer != rank;
        if (threadIdx.x == 0) {
            const auto* header_ptr =
                reinterpret_cast<const uint2*>(slot_ptr(args.lsa_peer_workspaces[rank], stage_slot_offset + peer));
            const unsigned num_recv_rows = is_remote ? poll_recv_header(header_ptr, header_seq).x : args.send_counts[rank];
            unsigned recv_row_start = 0;
            if constexpr (kIsDispatch) {
                const unsigned long long recv_allocation =
                    atomicAdd(&state.recv_row_and_peer_counter, (static_cast<unsigned long long>(num_recv_rows) << 32) | 1ull);
                const unsigned peer_order_idx = static_cast<unsigned>(recv_allocation);
                recv_row_start = static_cast<unsigned>(recv_allocation >> 32);
                args.recv_counts[peer] = num_recv_rows;
                args.operation_args.recv_peer_row_start[peer] = recv_row_start;
                if (args.operation_args.psum_recv_per_rank != nullptr)
                    args.operation_args.psum_recv_per_rank[peer_order_idx] = static_cast<int32_t>(recv_row_start + num_recv_rows);
            }
            shared_num_recv_rows = num_recv_rows;
            shared_recv_row_start = recv_row_start;
        }
        __syncthreads();
        const unsigned num_recv_rows = shared_num_recv_rows;
        const unsigned recv_row_start = shared_recv_row_start;
        const auto* completion_slot = static_cast<const uint64_t*>(args.comm_state.completion.base) + stage_slot_offset + peer;
        for (unsigned row_idx = 0; row_idx < num_recv_rows; ++row_idx) {
            auto* src = slot_row_ptr(args.lsa_peer_workspaces[rank], stage_slot_offset + peer, peer, row_idx);
            if constexpr (kIsDispatch)
                dispatch_epilogue<kNumRanks>(
                    args.operation_args, peer, row_idx, recv_row_start + row_idx, is_remote, src, completion_slot, seq_value);
            else
                combine_epilogue(args.operation_args, peer, row_idx, is_remote, src, completion_slot, seq_value);
        }
    }
}

template <bool kIsQuantized>
__global__ void magic_combine_reduce_kernel(unsigned world_size,
                                            unsigned row_bytes,
                                            unsigned hidden_size,
                                            unsigned quant_block_size,
                                            const uint8_t* staging,
                                            const unsigned* token_row_counts,
                                            __nv_bfloat16* out) {
    constexpr unsigned kElemsPerVec = sizeof(uint4) / (kIsQuantized ? sizeof(int8_t) : sizeof(__nv_bfloat16));
    constexpr unsigned kOutElemsPerVec = sizeof(uint4) / sizeof(__nv_bfloat16);
    const unsigned token_idx = blockIdx.x;
    const unsigned num_rows = token_row_counts[token_idx];
    const uint8_t* rows = staging + static_cast<size_t>(token_idx) * world_size * row_bytes;
    auto* dst = reinterpret_cast<uint4*>(out + static_cast<size_t>(token_idx) * hidden_size);
    for (unsigned vector_idx = threadIdx.x; vector_idx < hidden_size / kElemsPerVec; vector_idx += blockDim.x) {
        float acc[kElemsPerVec] = {};
        for (unsigned row_idx = 0; row_idx < num_rows; ++row_idx) {
            const uint8_t* src = rows + static_cast<size_t>(row_idx) * row_bytes;
            const uint4 value = reinterpret_cast<const uint4*>(src)[vector_idx];
            if constexpr (kIsQuantized) {
                const auto* elems = reinterpret_cast<const int8_t*>(&value);
                const float scale = reinterpret_cast<const float*>(src + hidden_size)[vector_idx * kElemsPerVec / quant_block_size];
                for (unsigned j = 0; j < kElemsPerVec; ++j)
                    acc[j] += elems[j] * scale;
            } else {
                const auto* elems = reinterpret_cast<const __nv_bfloat16*>(&value);
                for (unsigned j = 0; j < kElemsPerVec; ++j)
                    acc[j] += __bfloat162float(elems[j]);
            }
        }
        for (unsigned v = 0; v < kElemsPerVec / kOutElemsPerVec; ++v) {
            uint4 result;
            auto* result_elems = reinterpret_cast<__nv_bfloat16*>(&result);
            for (unsigned j = 0; j < kOutElemsPerVec; ++j)
                result_elems[j] = __float2bfloat16_rn(acc[v * kOutElemsPerVec + j]);
            dst[vector_idx * (kElemsPerVec / kOutElemsPerVec) + v] = result;
        }
    }
}

}

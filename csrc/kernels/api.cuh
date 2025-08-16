#pragma once

#include <vector>
#include <iostream>
#include "nixl_types.h"
#include "exception.cuh"

namespace deep_ep {

// Intranode runtime
namespace intranode {

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream);

} // namespace intranode

// Internode runtime
namespace internode {

void *alloc(size_t size, size_t alignment);

void free(void *ptr);

} // namespace internode

// Layout kernels
namespace layout {

void get_dispatch_layout(const int64_t* topk_idx,
                         int* num_tokens_per_rank, int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert, bool* is_token_in_rank,
                         int num_tokens, int num_topk, int num_ranks, int num_experts,
                         cudaStream_t stream);

} // namespace layout

// Intranode kernels
namespace intranode {

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     int num_tokens, const bool* is_token_in_rank, int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy, int num_memset_int, int expert_alignment,
                     void** buffer_ptrs, int** barrier_signal_ptrs, int rank,
                     cudaStream_t stream, int num_sms);

void cached_notify_dispatch(const int* rank_prefix_matrix, int num_memset_int,
                            void** buffer_ptrs, int** barrier_signal_ptrs, int rank, int num_ranks,
                            cudaStream_t stream);

void dispatch(void* recv_x, float* recv_x_scales, int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights, int* recv_channel_offset,
              int* send_head, const void* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
              const bool* is_token_in_rank, const int* channel_prefix_matrix,
              int num_tokens, int num_worst_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
              int scale_token_stride, int scale_hidden_stride,
              void** buffer_ptrs, int rank, int num_ranks,
              cudaStream_t stream, int num_sms,
              int num_max_send_tokens, int num_recv_buffer_tokens);

void cached_notify_combine(void** buffer_ptrs, int* send_head, int num_channels, int num_recv_tokens, int num_memset_int,
                           int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream);

void combine(cudaDataType_t type,
             void* recv_x, float* recv_topk_weights,
             const void* x, const float* topk_weights,
             const void* bias_0, const void* bias_1,
             const int* src_idx, const int* rank_prefix_matrix, const int* channel_prefix_matrix,
             int* send_head, int num_tokens, int num_recv_tokens, int hidden, int num_topk,
             void** buffer_ptrs, int rank, int num_ranks,
             cudaStream_t stream, int num_sms,
             int num_max_send_tokens, int num_recv_buffer_tokens);

} // namespace intranode

// Internode kernels
namespace internode {
struct gpu_channel_nixl_ctx {
    //arrays of elements per remote rdma rank
    nixlGpuXferReqH *data_request_handles;
    nixlGpuXferReqH *remote_head_counter_handles;
    uint64_t *local_head_counters;
    uint64_t *local_tail_counters;

    uint64_t *last_barrier_counter;
    uint64_t *local_barrier_counter_ptr;
    nixlGpuXferReqH *remote_barrier_handles;

    __device__ inline int get_local_head_counter(int rank) {
        return local_head_counters[rank];
    }

    __device__ inline int get_local_tail_counter(int rank) {
        return local_tail_counters[rank];
    }

    __device__ inline nixlGpuXferReqH get_remote_head_handle(int rank) {
        return remote_head_counter_handles[rank];
    }

    __device__ inline nixlGpuXferReqH get_remote_data_handle(int rank) {
        return data_request_handles[rank];
    }
};

struct gpu_nixl_ctx {
    struct gpu_channel_nixl_ctx *channel_ctxs; // [num_channels]
    int num_channels;
    int num_rdma_ranks;
    int rank;
};

int get_source_meta_bytes();

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_rdma_rank, int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     const bool* is_token_in_rank, int num_tokens, int num_channels,
                     int hidden_int4, int num_scales, int num_topk, int expert_alignment,
                     int* rdma_channel_prefix_matrix, int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix, int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs, int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs, int rank,
                     cudaStream_t stream, int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                     bool low_latency_mode, internode::gpu_nixl_ctx nixl_ctx);

void dispatch(void* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights, void* recv_src_meta,
              const void* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
              int* send_rdma_head, int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix, int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix, const int* recv_gbl_rank_prefix_sum,
              const bool* is_token_in_rank,
              int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,
              int scale_token_stride, int scale_hidden_stride,
              void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
              int rank, int num_ranks, bool is_cached_dispatch,
              cudaStream_t stream, int num_channels, bool low_latency_mode, internode::gpu_nixl_ctx nixl_ctx);

void cached_notify(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights,
                   int num_ranks, int num_channels, int num_combined_tokens, int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, int* combined_nvl_head,
                   void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs, int num_max_nvl_chunked_recv_tokens,
                   int** barrier_signal_ptrs, int rank, cudaStream_t stream,
                   int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                   bool is_cached_dispatch, bool low_latency_mode, internode::gpu_nixl_ctx nixl_ctx);

void combine(cudaDataType_t type,
             void* combined_x, float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x, const float* topk_weights,
             const void* bias_0, const void* bias_1,
             const int* combined_rdma_head, const int* combined_nvl_head,
             const void* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, const int* gbl_channel_prefix_matrix,
             int num_tokens, int num_combined_tokens, int hidden, int num_topk,
             void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
             int rank, int num_ranks, cudaStream_t stream, int num_channels, bool low_latency_mode, internode::gpu_nixl_ctx nixl_ctx);
} // namespace internode

// Internode low-latency kernels
namespace internode_ll {
struct gpu_nixl_ctx {
    uint64_t *local_counters; // [local_expert_id][src_rank]
    uint64_t *clean_counters; // Counters to be cleaned for the next iteration
    nixlGpuXferReqH *remote_counter_reqs; // [local_expert_id,dest_rank]
    nixlGpuXferReqH *batch_reqs; // [local_expert_id,dest_rank]
    uint64_t *local_sync_counters; // [src_rank]
    nixlGpuXferReqH *remote_sync_counters; // [dest_rank]
    void **rdma_p2p_ptrs; // [num_ranks]
    uint64_t **counters_p2p_ptrs; // [num_ranks]
    void *rdma_buffer_ptr;
    int num_local_experts;
    int num_ranks;
    int rank;

    /* Double buffering considerations are handled by the caller */
    __device__ inline void *rdma_p2p_ptr_get(uint64_t ptr, int dst_rank) {
        if (rdma_p2p_ptrs[dst_rank] == nullptr)
            return nullptr;

        return (void *)(reinterpret_cast<uint64_t>(rdma_p2p_ptrs[dst_rank]) + batch_offset_get(ptr));
    }

    /* Double buffering considerations are handled by nixl_ctx */
    __device__ inline uint64_t *counter_p2p_ptr_get(int local_expert_idx, int dst_rank) {
        if (counters_p2p_ptrs[dst_rank] == nullptr)
            return nullptr;

        return counters_p2p_ptrs[dst_rank] + (local_expert_idx * num_ranks + rank);
    }

    __device__ inline uint64_t *local_counter_get(int local_expert_idx, int src_rank) {
        return &local_counters[local_expert_idx * num_ranks + src_rank];
    }

    __device__ inline nixlGpuXferReqH remote_counter_get(int local_expert_idx, int dest_rank) {
        return remote_counter_reqs[local_expert_idx * num_ranks + dest_rank];
    }

    __device__ inline nixlGpuXferReqH remote_sync_counter_get(int dest_rank) {
        return remote_sync_counters[dest_rank];
    }

    __device__ inline uint64_t *local_sync_counter_get(int src_rank) {
        return &local_sync_counters[src_rank];
    }

    __device__ inline nixlGpuXferReqH batch_get(int local_expert_idx, int dest_rank) {
        return batch_reqs[local_expert_idx * num_ranks + dest_rank];
    }

    __device__ inline size_t batch_offset_get(uint64_t ptr) {
        return ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr);
    }

    __device__ inline void clean_counters_warp(int lane_id) {
#ifdef __CUDACC__
        #pragma unroll
#endif
        for (int i = lane_id; i < num_ranks * num_local_experts; i += 32)
            clean_counters[i] = 0;
    }
};

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                              int* clean_1, int num_clean_int_1,
                              cudaStream_t stream);

void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
              const void* x, const int64_t* topk_idx,
              int* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0,
              void* workspace, int num_device_sms,
              cudaStream_t stream, int phases, internode_ll::gpu_nixl_ctx nixl_ctx);

void combine(void* combined_x,
             void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
             const void* x, const int64_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             bool use_logfmt,
             void* workspace, int num_device_sms,
             cudaStream_t stream, int phases, bool zero_copy, internode_ll::gpu_nixl_ctx nixl_ctx);

void sync(internode_ll::gpu_nixl_ctx nixl_ctx, cudaStream_t stream);

} // namespace internode_ll

} // namespace deep_ep

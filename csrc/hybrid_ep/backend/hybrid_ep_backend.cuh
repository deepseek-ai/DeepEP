// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#pragma once

#include "utils.cuh"
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda/ptx>
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
#include "doca_gpunetio_host.h"
#include "doca_gpunetio_device.h"
#include "infiniband/verbs.h"
#include "infiniband/mlx5dv.h"
#endif

namespace hybrid_ep{

/*enum DATA_TYPE{
  HYBRID_EP_DATA_TYPE_FP32,
  HYBRID_EP_DATA_TYPE_FP16,
  HYBRID_EP_DATA_TYPE_BF16,
  HYBRID_EP_DATA_TYPE_FP8
};*/

/*template<int NUM_OF_BOOL_TO_REDUCE>
struct bool_any_reduction_type{};

template<> struct bool_any_reduction_type<8> { using Type = uint64_t; };
template<> struct bool_any_reduction_type<4> { using Type = uint32_t; };
template<> struct bool_any_reduction_type<2> { using Type = uint16_t; };
template<> struct bool_any_reduction_type<1> { using Type = uint8_t; };*/

template<int NUM_OF_BOOL_TO_REDUCE>
using Reduce_t =
  typename std::conditional<NUM_OF_BOOL_TO_REDUCE % 8 == 0, uint64_t,
    typename std::conditional<NUM_OF_BOOL_TO_REDUCE % 4 == 0, uint32_t,
      typename std::conditional<NUM_OF_BOOL_TO_REDUCE % 2 == 0, uint16_t, uint8_t
      >::type
    >::type
  >::type;

template<int NUM_OF_BYTES_TO_COPY>
using Copy_t =
  typename std::conditional<NUM_OF_BYTES_TO_COPY % 16 == 0, uint4,
    typename std::conditional<NUM_OF_BYTES_TO_COPY % 8 == 0, uint2,
      typename std::conditional<NUM_OF_BYTES_TO_COPY % 4 == 0, uint32_t,
        typename std::conditional<NUM_OF_BYTES_TO_COPY % 2 == 0, uint16_t, uint8_t
        >::type
      >::type
    >::type
  >::type;

enum scan_state{
  EMPTY = 0, 
  PRIV_SUM = 1 
};

struct tmp_state_t{
  scan_state state;
  int32_t value;
};

// Generic warp group for warp-specializaion.
template<int NUM_WARPS,
         int STARTING_WARPS>
struct warp_group{
  __host__ __device__ static constexpr int size(){ return 32 * NUM_WARPS; }
  __host__ __device__ static constexpr int warp_size(){ return NUM_WARPS; }

  __host__ __device__ static int thread_rank(){ return threadIdx.x - (32 * STARTING_WARPS); }
  __host__ __device__ static int warp_rank(){ return thread_rank() / 32; }
};

template<typename TOKEN_DATA_TYPE,
         int NUM_OF_STAGES,
         int HIDDEN_DIM, 
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         bool FORWARD_DISPATCH>
struct dispatch_kernel_dynamic_shared_memory_buffer_t{};

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint8_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, true>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint8_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory Prob buffer. Only used in FW dispatch. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_prob_buffer[NUM_OF_STAGES][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory scaling factor buffer. Only when using FP8 token. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_scaling_factor_buffer[NUM_OF_STAGES][HIDDEN_DIM / 128];
  // Shared memory attn_to_rdma_map buffer, Should be 16B alignment.
  alignas(16) bool attn_to_rdma_map_buffer[NUM_OF_TOKENS_PER_CHUNK * (NUM_OF_NODES - 1)];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
  // Shared memory mr info for dispatch. (Mr info can be cached in shared memory, while qp info can't be cached.)
  alignas(8) dispatch_memory_region_info_t dispatch_memory_region_info[NUM_OF_NODES - 1];
  // Num of tx messages.
  uint32_t inter_node_num_of_write_per_node[NUM_OF_NODES - 1];
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint16_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, true>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory Prob buffer. Only used in FW dispatch. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_prob_buffer[NUM_OF_STAGES][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory attn_to_rdma_map buffer, Should be 16B alignment.
  alignas(16) bool attn_to_rdma_map_buffer[NUM_OF_TOKENS_PER_CHUNK * (NUM_OF_NODES - 1)];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
  // Shared memory mr info for dispatch. (Mr info can be cached in shared memory, while qp info can't be cached.)
  alignas(8) dispatch_memory_region_info_t dispatch_memory_region_info[NUM_OF_NODES - 1];
  // Num of tx messages.
  uint32_t inter_node_num_of_write_per_node[NUM_OF_NODES - 1];
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint8_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, false>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint8_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory scaling factor buffer. Only when using FP8 token. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_scaling_factor_buffer[NUM_OF_STAGES][HIDDEN_DIM / 128];
  // Shared memory attn_to_rdma_map buffer, Should be 16B alignment.
  alignas(16) bool attn_to_rdma_map_buffer[NUM_OF_TOKENS_PER_CHUNK * (NUM_OF_NODES - 1)];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
  // Shared memory mr info for dispatch. (Mr info can be cached in shared memory, while qp info can't be cached.)
  alignas(8) dispatch_memory_region_info_t dispatch_memory_region_info[NUM_OF_NODES - 1];
  // Num of tx messages.
  uint32_t inter_node_num_of_write_per_node[NUM_OF_NODES - 1];
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint16_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, false>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory attn_to_rdma_map buffer, Should be 16B alignment.
  alignas(16) bool attn_to_rdma_map_buffer[NUM_OF_TOKENS_PER_CHUNK * (NUM_OF_NODES - 1)];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
  // Shared memory mr info for dispatch. (Mr info can be cached in shared memory, while qp info can't be cached.)
  alignas(8) dispatch_memory_region_info_t dispatch_memory_region_info[NUM_OF_NODES - 1];
  // Num of tx messages.
  uint32_t inter_node_num_of_write_per_node[NUM_OF_NODES - 1];
};
#endif

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint8_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, 1, true>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint8_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory Prob buffer. Only used in FW dispatch. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_prob_buffer[NUM_OF_STAGES][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory scaling factor buffer. Only when using FP8 token. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_scaling_factor_buffer[NUM_OF_STAGES][HIDDEN_DIM / 128];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint16_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, 1, true>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory Prob buffer. Only used in FW dispatch. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_prob_buffer[NUM_OF_STAGES][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint8_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, 1, false>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint8_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory scaling factor buffer. Only when using FP8 token. Should be 16B alignment so can be used with TMA. 128B is too strict.
  alignas(16) float intra_node_scaling_factor_buffer[NUM_OF_STAGES][HIDDEN_DIM / 128];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
};

template<int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE> 
struct dispatch_kernel_dynamic_shared_memory_buffer_t<uint16_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, 1, false>{
  // Shared memory token buffer. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_buffer[NUM_OF_STAGES][HIDDEN_DIM];
  // Shared memory mbarrier that protect token entry, 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_buffer[NUM_OF_STAGES][2]; 
};

template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         bool BACKWARD_COMBINE>
struct combine_kernel_dynamic_shared_memory_buffer_t{};

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES>
struct combine_kernel_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, 
                                                     NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, true>{
  // Shared memory token buffer for intra node red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for intra node red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];
  // Shared memory token buffer for inter node red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for inter node red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];

  // Shared memory prob buffer for intra node red warp group G2S data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float intra_node_prob_G2S_buffer[NUM_OF_STAGES_G2S][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory prob buffer for intra node red warp group S2G data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float intra_node_prob_S2G_buffer[NUM_OF_STAGES_S2G][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory prob buffer for inter node red warp group G2S data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float inter_node_prob_G2S_buffer[NUM_OF_STAGES_G2S][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory prob buffer for inter node red warp group S2G data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float inter_node_prob_S2G_buffer[NUM_OF_STAGES_S2G][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES];

  // Shared memory mbarrier that protect intra node red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];
  // Shared memory mbarrier that protect inter node red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t inter_node_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];
  // Shared memory mbarrier that maintain producer->consumer relationship between intra-node red warp group and rdma warp group. 1 per chunk, Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_to_rdma_mbarrier_buffer[NUM_OF_NODES - 1][MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK];

  // Shared memory mr info for combine. (Mr info can be cached in shared memory, while qp info can't be cached.)
  alignas(8) combine_memory_region_info_t combine_memory_region_info[NUM_OF_NODES - 1];

  // Num of tx messages.
  uint32_t inter_node_num_of_write_per_node[NUM_OF_NODES - 1];

  // Endgroup flag for each token entry in G2S buffer. true means that this token is the last token of a intra-node reduction group, otherwise not.
  bool intra_node_flag_G2S_buffer[NUM_OF_STAGES_G2S];
  bool inter_node_flag_G2S_buffer[NUM_OF_STAGES_G2S];
};
#endif

template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE>
struct combine_kernel_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, 
                                                     NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, 1, true>{
  // Shared memory token buffer for inter node red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for inter node red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];

  // Shared memory prob buffer for inter node red warp group G2S data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float inter_node_prob_G2S_buffer[NUM_OF_STAGES_G2S][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
  // Shared memory prob buffer for inter node red warp group S2G data movement. Should be 16B alignment so can be used with TMA. 128B is too strict.
  // Only used in BW combine.
  alignas(16) float inter_node_prob_S2G_buffer[NUM_OF_STAGES_S2G][NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];

  // Shared memory mbarrier that protect inter node red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t inter_node_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];

  // Endgroup flag for each token entry in G2S buffer. true means that this token is the last token of a intra-node reduction group, otherwise not.
  bool inter_node_flag_G2S_buffer[NUM_OF_STAGES_G2S];
};

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES>
struct combine_kernel_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, 
                                                     NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, false>{
  // Shared memory token buffer for intra node red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for intra node red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t intra_node_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];
  // Shared memory token buffer for inter node red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for inter node red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];

  // Shared memory mbarrier that protect intra node red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];
  // Shared memory mbarrier that protect inter node red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t inter_node_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];
  // Shared memory mbarrier that maintain producer->consumer relationship between intra-node red warp group and rdma warp group. 1 per chunk, Should be 8B alignment(natural alignment).
  alignas(8) uint64_t intra_node_to_rdma_mbarrier_buffer[NUM_OF_NODES - 1][MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK];

  // Shared memory mr info for combine. (Mr info can be cached in shared memory, while qp info can't be cached.)
  alignas(8) combine_memory_region_info_t combine_memory_region_info[NUM_OF_NODES - 1];

  // Num of tx messages.
  uint32_t inter_node_num_of_write_per_node[NUM_OF_NODES - 1];

  // Endgroup flag for each token entry in G2S buffer. true means that this token is the last token of a intra-node reduction group, otherwise not.
  bool intra_node_flag_G2S_buffer[NUM_OF_STAGES_G2S];
  bool inter_node_flag_G2S_buffer[NUM_OF_STAGES_G2S];
};
#endif

template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE>
struct combine_kernel_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, 
                                                     NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, 1, false>{
  // Shared memory token buffer for inter node red warp group G2S data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_G2S_buffer[NUM_OF_STAGES_G2S][HIDDEN_DIM];
  // Shared memory token buffer for inter node red warp group S2G data movement. Should be 128B alignment for optimal perf for TMA.
  alignas(128) uint16_t inter_node_token_S2G_buffer[NUM_OF_STAGES_S2G][HIDDEN_DIM];

  // Shared memory mbarrier that protect inter node red warp group G2S token entry. 1st for producer->consumer, 2nd for consumer->producer. Should be 8B alignment(natural alignment).
  alignas(8) uint64_t inter_node_mbarrier_G2S_buffer[NUM_OF_STAGES_G2S][2];

  // Endgroup flag for each token entry in G2S buffer. true means that this token is the last token of a intra-node reduction group, otherwise not.
  bool inter_node_flag_G2S_buffer[NUM_OF_STAGES_G2S];
};


// Data structure for kernel parameter for dispatch kernel.
template<typename TOKEN_DATA_TYPE>
struct dispatch_kernel_param_t{
  // Input buffers. These buffers are local buffers.
  const TOKEN_DATA_TYPE* attn_input_token;
  const float* attn_input_prob; // Needed by expert layer, so only valid in forward dispatch.
  const float* attn_input_token_scaling_factor; // If input token is FP8 dtype, we need scaling factor for tokens.
  // Output buffers. These buffers are both local and remote buffers.
  TOKEN_DATA_TYPE* expert_output_token[MAX_NUM_OF_RANKS_PER_NODE];
  float* expert_output_prob[MAX_NUM_OF_RANKS_PER_NODE]; // Only valid in forward dispatch.
  float* expert_output_scaling_factor[MAX_NUM_OF_RANKS_PER_NODE]; // Only valid for FP8 token type.
  // Internal temp buffers. These buffers are local buffers.
  const TOKEN_DATA_TYPE* rdma_inter_node_group_token;
  const float* rdma_inter_node_group_prob; // Only valid in forward dispatch.
  const float* rdma_inter_node_group_scaling_factor; // Only valid for FP8 token type.
  uint64_t* rdma_inter_node_group_flags; // For RDMA Atomic flags.
  uint32_t* intra_node_write_completion_flags; // For intra-node S2G write completion notification.
  // Metadata buffers. These buffers are local buffers.
  const bool* rdma_to_attn_map;
  const bool* attn_to_rdma_map;
  const int32_t* sparse_to_dense_map;
  uint64_t* expected_rdma_flag_value;
  uint32_t* expected_intra_node_flag_value;
  int local_rank;
  int node_rank;
  // The number of token output by attn layer on a rank/GPU.
  int num_of_tokens_per_rank;
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  // qp info and mr info
  struct doca_gpu_dev_verbs_qp **d_qps_gpu;
  struct dispatch_memory_region_info_t *mr_info;
#endif
};

// Data structure for kernel parameter for combine kernel.
struct combine_kernel_param_t{
  // Input buffers. These buffers are both local and remote buffers.
  uint16_t* expert_input_token[MAX_NUM_OF_RANKS_PER_NODE];
  float* expert_input_prob[MAX_NUM_OF_RANKS_PER_NODE];
  // Output buffers. These buffers are local buffers.
  uint16_t* attn_output_token;
  float* attn_output_prob;
  // Internal temp buffers. These buffers are local buffers.
  uint16_t* rdma_intra_node_red_token;
  float* rdma_intra_node_red_prob;
  const uint16_t* rdma_inter_node_group_token;
  const float* rdma_inter_node_group_prob;
  uint64_t* rdma_inter_node_group_flags;
  uint32_t* intra_node_write_completion_flags; // For intra-node src ready notification.
  // Metadata buffers. These buffers are local buffers.
  const bool* rdma_to_attn_map;
  const bool* attn_to_rdma_map;
  const int32_t* sparse_to_dense_map;
  uint64_t* expected_rdma_flag_value;
  uint32_t* expected_intra_node_flag_value;
  int node_rank;
  // The number of token output by attn layer on a rank/GPU.
  int num_of_tokens_per_rank;
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  // qp info and mr info
  struct doca_gpu_dev_verbs_qp **d_qps_gpu;
  struct combine_memory_region_info_t *mr_info;
#endif
};

// Each CUDA block has sixteen named barriers numbered 0..15.
// __syncthreads(); will use the 0 named barriers, so we want to avoid that.
// We want to use 1 for intra-node reduction warp group, 2 for inter-node reduction warp group, 3 for RDMA warp group. 
inline __device__ void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id = 0) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
// Device function for inter-node node2node(RDMA) warp for dispatch kernel. There can be only 1 inter-node warp per CUDA block!
template<typename INTER_NODE_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH>
inline __device__ void N2N_warp_group_device_function(const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const bool *attn_to_rdma_map,
                                                      doca_gpu_dev_verbs_qp **d_qps_gpu,
                                                      struct dispatch_memory_region_info_t *mr_info,
                                                      SMEM_TYPE* smem_buffer_ptr)
{
  // Load attn_to_rdma_map using LDG.128. Each token will need 1 bool from this map.
  // using attn_to_rdma_map_load_t = uint4;
  int NUM_OF_CHUNKS_PER_RANK = (num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
  constexpr int WQE_NUM_RATIO = 1 + std::is_same<TOKEN_DATA_TYPE, uint8_t>::value + FORWARD_DISPATCH;
  // constexpr int NUN_OF_ATTN_TO_RDMA_MAP_LOAD_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK * (NUM_OF_NODES - 1) / sizeof(attn_to_rdma_map_load_t);
  static_assert(INTER_NODE_GROUP::size() == 32, "INTER_NODE_GROUP should be 1 warp.");
  static_assert(INTER_NODE_GROUP::size() >= NUM_OF_NODES - 1, "mr_info should be loaded at once.");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % INTER_NODE_GROUP::size() == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of 32.");
  // static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(attn_to_rdma_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of sizeof(attn_to_rdma_map_load_t).");
  // The (NUM_OF_NODES - 1) queue pairs of one block were arranged together.
  int block_offset = blockIdx.x * (NUM_OF_NODES - 1);
  // Loading mr_infos to shared memory.
  struct dispatch_memory_region_info_t *smem_mr_info_ptr = nullptr;
  uint32_t *smem_inter_node_num_of_write_per_node_ptr = nullptr;
  if constexpr(NUM_OF_NODES != 1) {
    smem_mr_info_ptr = smem_buffer_ptr->dispatch_memory_region_info;
    smem_inter_node_num_of_write_per_node_ptr = smem_buffer_ptr->inter_node_num_of_write_per_node;
    if (INTER_NODE_GROUP::thread_rank() < NUM_OF_NODES - 1) {
      smem_mr_info_ptr[INTER_NODE_GROUP::thread_rank()] = mr_info[INTER_NODE_GROUP::thread_rank() + block_offset];
      smem_inter_node_num_of_write_per_node_ptr[INTER_NODE_GROUP::thread_rank()] = 0;
    }
    __syncwarp();
  }
  // For each chunk.
  for (int chunk_idx = blockIdx.x; chunk_idx < NUM_OF_CHUNKS_PER_RANK; chunk_idx += NUM_OF_BLOCKS) {
    int chunk_base_token_idx = chunk_idx * NUM_OF_TOKENS_PER_CHUNK;
    int token_range = NUM_OF_TOKENS_PER_CHUNK;
    // Attn_to_rdma_map cached in shared memory.
    bool *smem_attn_to_rdma_map_ptr = nullptr;
    // Reading one chunk of attn_to_rdma_map into shared memory.
    if constexpr(NUM_OF_NODES != 1) {
      smem_attn_to_rdma_map_ptr = smem_buffer_ptr->attn_to_rdma_map_buffer;
      if (chunk_base_token_idx + token_range > num_of_tokens_per_rank) {
        token_range = num_of_tokens_per_rank - chunk_base_token_idx;
      }
      for (int map_load_idx = INTER_NODE_GROUP::thread_rank();
           map_load_idx < token_range * (NUM_OF_NODES - 1);
           map_load_idx += INTER_NODE_GROUP::size()) {
        smem_attn_to_rdma_map_ptr[map_load_idx] = attn_to_rdma_map[chunk_base_token_idx * (NUM_OF_NODES - 1) + map_load_idx];
      }
      __syncwarp();
    }
    // For each remote.
    for (int idx = 0; idx < NUM_OF_NODES - 1; ++idx) {
      int remote_idx = (idx + node_rank) % (NUM_OF_NODES - 1);
      // Queue pair for the current block to the current remote.
      struct doca_gpu_dev_verbs_qp *qp = d_qps_gpu[remote_idx + block_offset];
      // Real remote node rank.
      int remote_node_rank = remote_idx < node_rank ? remote_idx : remote_idx + 1;
      // Calculating total num of tokens need to be sent to the current remote.
      int num_of_tokens_need_write = 0;
      for (int token_idx_in_chunk = INTER_NODE_GROUP::thread_rank();
           token_idx_in_chunk < token_range;
           token_idx_in_chunk += INTER_NODE_GROUP::size()) {
        num_of_tokens_need_write += smem_attn_to_rdma_map_ptr[remote_idx + token_idx_in_chunk * (NUM_OF_NODES - 1)];
      }
      num_of_tokens_need_write = __reduce_add_sync(0xffffffff, num_of_tokens_need_write);
      int total_write_cnt = num_of_tokens_need_write * WQE_NUM_RATIO + 1;
      // Getting wqe slots.
      uint64_t base_wqe_idx = 0;
      if (INTER_NODE_GROUP::thread_rank() == 0) {
        base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(qp, total_write_cnt);
        smem_inter_node_num_of_write_per_node_ptr[remote_idx] += total_write_cnt;
      }
      base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);
      uint64_t curr_wqe_idx = base_wqe_idx;
      // For the current chunk to the current remote.
      for (int token_idx_in_chunk = INTER_NODE_GROUP::thread_rank();
           token_idx_in_chunk < NUM_OF_TOKENS_PER_CHUNK;
           token_idx_in_chunk += INTER_NODE_GROUP::size()) {
        int64_t token_idx = token_idx_in_chunk + chunk_base_token_idx;
        bool need_write = false;
        if (token_idx_in_chunk < token_range) {
          need_write = smem_attn_to_rdma_map_ptr[remote_idx + token_idx_in_chunk * (NUM_OF_NODES - 1)];
        }
        uint32_t write_map = __ballot_sync(0xffffffff, need_write);
        uint32_t partial_write_map = ((1 << INTER_NODE_GROUP::thread_rank()) - 1) & write_map;
        int write_cnt = __popc(write_map);
        int write_idx = __popc(partial_write_map);
        if (need_write) {
          // Constructing wqes for tokens.
          uint64_t my_wqe_idx = curr_wqe_idx + write_idx;
          struct doca_gpu_dev_verbs_wqe *token_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, my_wqe_idx);
          doca_gpu_dev_verbs_wqe_prepare_write(qp, token_wqe_ptr, my_wqe_idx,
                                                    DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                                    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                                                    smem_mr_info_ptr[remote_idx].token_raddr + token_idx * HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE),
                                                    smem_mr_info_ptr[remote_idx].token_rkey,
                                                    smem_mr_info_ptr[remote_idx].token_laddr + token_idx * HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE),
                                                    smem_mr_info_ptr[remote_idx].token_lkey,
                                                    HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE));
          // Constructing wqes for probs.
          if constexpr(FORWARD_DISPATCH) {
            my_wqe_idx += write_cnt;
            struct doca_gpu_dev_verbs_wqe *prob_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, my_wqe_idx);
            doca_gpu_dev_verbs_wqe_prepare_write(qp, prob_wqe_ptr, my_wqe_idx,
                                                      DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                                      DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                                                      smem_mr_info_ptr[remote_idx].prob_raddr + token_idx * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float),
                                                      smem_mr_info_ptr[remote_idx].prob_rkey,
                                                      smem_mr_info_ptr[remote_idx].prob_laddr + (token_idx * NUM_OF_NODES + remote_node_rank) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float),
                                                      smem_mr_info_ptr[remote_idx].prob_lkey,
                                                      (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
          }
          // Constructing wqes for scaling factor(Only for FP8 token).
          if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
            my_wqe_idx += write_cnt;
            struct doca_gpu_dev_verbs_wqe *sf_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, my_wqe_idx);
            doca_gpu_dev_verbs_wqe_prepare_write(qp, sf_wqe_ptr, my_wqe_idx,
                                                      DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                                      DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                                                      smem_mr_info_ptr[remote_idx].scaling_factor_raddr + token_idx * (HIDDEN_DIM / 128) * sizeof(float),
                                                      smem_mr_info_ptr[remote_idx].scaling_factor_rkey,
                                                      smem_mr_info_ptr[remote_idx].scaling_factor_laddr + token_idx * (HIDDEN_DIM / 128) * sizeof(float),
                                                      smem_mr_info_ptr[remote_idx].scaling_factor_lkey,
                                                      (HIDDEN_DIM / 128) * sizeof(float));
          }
        }
        curr_wqe_idx += write_cnt * WQE_NUM_RATIO;
        __syncwarp(0xffffffff);
      }
      if (INTER_NODE_GROUP::thread_rank() == 0) {
        // Construct wqe for flag.
        struct doca_gpu_dev_verbs_wqe *flag_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, curr_wqe_idx);
        doca_gpu_dev_verbs_wqe_prepare_atomic(qp, flag_wqe_ptr, curr_wqe_idx,
                                                   DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                                   DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                   smem_mr_info_ptr[remote_idx].flag_raddr + chunk_idx * sizeof(uint64_t),
                                                   smem_mr_info_ptr[remote_idx].flag_rkey,
                                                   smem_mr_info_ptr[remote_idx].flag_laddr + chunk_idx * sizeof(uint64_t),
                                                   smem_mr_info_ptr[remote_idx].flag_lkey,
                                                   sizeof(uint64_t), 1, 0);
        // Post send and poll cqs.
        doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(qp, base_wqe_idx, curr_wqe_idx);
        doca_gpu_dev_verbs_submit_db<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                                          DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                          DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
                                          DOCA_GPUNETIO_VERBS_QP_SQ>(qp, (curr_wqe_idx + 1));
      }
      __syncwarp();
    }
  }
  if (INTER_NODE_GROUP::thread_rank() < NUM_OF_NODES - 1) {
    struct doca_gpu_dev_verbs_qp *qp = d_qps_gpu[block_offset + INTER_NODE_GROUP::thread_rank()];
    int status = doca_gpu_dev_verbs_poll_cq<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                                                 DOCA_GPUNETIO_VERBS_QP_SQ>(
                                                 doca_gpu_dev_verbs_qp_get_cq_sq(qp),
                                                 smem_inter_node_num_of_write_per_node_ptr[INTER_NODE_GROUP::thread_rank()]);
    assert(status >= 0);
  }
}
#endif

// Device function for intra-node G2S warp for dispatch kernel. There can be only 1 intra-node G2S warp per CUDA block!
template<typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES, 
         int HIDDEN_DIM,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH>
inline __device__ void G2S_warp_group_device_function(const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const uint64_t* expected_flag_value,
                                                      const bool* rdma_to_attn_map,
                                                      const TOKEN_DATA_TYPE* attn_input_token, 
                                                      const float* attn_input_prob,
                                                      const float* attn_input_token_scaling_factor,
                                                      const TOKEN_DATA_TYPE* rdma_inter_node_group_token,
                                                      const float* rdma_inter_node_group_prob,
                                                      const float* rdma_inter_node_group_scaling_factor,
                                                      const uint64_t* rdma_inter_node_group_flags,
                                                      SMEM_TYPE* smem_buffer_ptr)
{
  // Load rdma_to_attn_map using LDG.128. Each token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK; 
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  int stage = 0;
  uint32_t consumer_parity = 1;

  // Only 1 thread within the G2S warp will be active, other threads will just exit.
  if(elect_sync(~0)){
    // Loop through all data chunk. Data(chunk) parallel between multiple CUDA blocks.
    for(int i = blockIdx.x; i < num_of_chunks_per_rank; i += NUM_OF_BLOCKS){
      // How many rdma_to_attn load iter for this chunk.
      int num_of_routing_info_load_iter_for_current_chunk;
      // How many token for this chunk.
      int current_chunk_size;
      if(remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1){
        num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
        current_chunk_size = remainder_chunk_size;
      }else{
        num_of_routing_info_load_iter_for_current_chunk = NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }
      for(int j = 0; j < NUM_OF_NODES; j++){
        // The current node been processed. For each chunk id, node_id order is local_node, local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
        int node_id = node_rank >= j ? node_rank - j : node_rank + NUM_OF_NODES - j;
        // The tile id within the rdma buffers for the current node id. Because rdma buffers only have NUM_OF_NODES - 1 tile.
        int rdma_buffer_tile_id = node_id > node_rank ? node_id - 1 : node_id;
        // Check if the chunk of this node is ready to be consumed.
        // The chunks of local node is the attn input buffers, which are always ready to be consumed.
        // The chunks of remote node is the rdma_inter_node_group buffers, which is produced by remote RDMA Write operation. Should poll the flag produced by remote RDMA Atomic FA before consumed.
        if(node_id != node_rank){
          const uint64_t* flag_location = rdma_inter_node_group_flags + (rdma_buffer_tile_id * num_of_chunks_per_rank + i);
          uint64_t rdma_flag = 0;
          do{
            rdma_flag = 0;
            // Need a strong system-scope load to observe external RDMA Atomic result.
            asm volatile("ld.relaxed.sys.global.b64 %0, [%1];"
                         : "=l"(rdma_flag)
                         : "l"(__cvta_generic_to_global(flag_location))
                         : "memory");
          }while(rdma_flag != *expected_flag_value);
        }
        // Load every token and its properties from Global to Shared. Only load tokens that is needed by this node.
        const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map + 
                                                                         (node_id * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK));
        const TOKEN_DATA_TYPE* token_load_base_addr;
        const float* prob_load_base_addr;
        const float* scaling_factor_load_base_addr;
        // For other node's attn token and properties, read from rdma_inter_node_group buffers.
        // For this node's attn token and properties, read from attn input buffers.
        if(node_id != node_rank){
          int chunk_first_token_id = rdma_buffer_tile_id * num_of_tokens_per_rank + i * NUM_OF_TOKENS_PER_CHUNK;
          token_load_base_addr = rdma_inter_node_group_token + chunk_first_token_id * HIDDEN_DIM;
          if constexpr(FORWARD_DISPATCH){
            prob_load_base_addr = rdma_inter_node_group_prob + chunk_first_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);
          }
          if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
            scaling_factor_load_base_addr = rdma_inter_node_group_scaling_factor + chunk_first_token_id * (HIDDEN_DIM / 128);
          }
        }else{
          int chunk_first_token_id = i * NUM_OF_TOKENS_PER_CHUNK;
          token_load_base_addr = attn_input_token + chunk_first_token_id * HIDDEN_DIM;
          if constexpr(FORWARD_DISPATCH){
            prob_load_base_addr = attn_input_prob + chunk_first_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES);
          }
          if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
            scaling_factor_load_base_addr = attn_input_token_scaling_factor + chunk_first_token_id * (HIDDEN_DIM / 128);
          }
        }
        //#pragma unroll
        for(int k = 0; k < num_of_routing_info_load_iter_for_current_chunk; k++){
          rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[k];
          #pragma unroll
          for(int n = 0; n < NUM_OF_TOKENS_PER_LOAD_ITER; n++){
            int current_token_id = k * NUM_OF_TOKENS_PER_LOAD_ITER + n;
            // If the current token is out-of-bound, then just end this load iter.
            if(current_token_id >= current_chunk_size){
              break;
            }
            bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + n);
            // If a token is needed by this node(i.e. any expert of this node), load the token and its properties to shared memory entry.
            if(token_needed_by_this_node){
              // Wait until shared memory has free entry.
              while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->intra_node_mbarrier_buffer[stage][1], consumer_parity)){}
              // Issue TMA to load current token and its properties from global to shared memory.
              uint32_t total_tx_size = 0;
              // Load token.
              cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                       cuda::ptx::space_global,
                                       reinterpret_cast<void*>(&smem_buffer_ptr->intra_node_token_buffer[stage][0]),
                                       reinterpret_cast<const void*>(token_load_base_addr + (current_token_id * HIDDEN_DIM)),
                                       (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE)),
                                       &smem_buffer_ptr->intra_node_mbarrier_buffer[stage][0]);

              total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE));

              // Optionally load prob(Only in FW dispatch).
              if constexpr(FORWARD_DISPATCH){
                // rdma_inter_node_group prob buffers and attn prob buffers will have different prob vec length.
                const float* prob_load_token_addr;
                if(node_id != node_rank){
                  prob_load_token_addr = prob_load_base_addr + (current_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));
                }else{
                  prob_load_token_addr = prob_load_base_addr + (current_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES)) + 
                                                               (node_rank * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));
                }
                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         reinterpret_cast<void*>(&smem_buffer_ptr->intra_node_prob_buffer[stage][0]),
                                         reinterpret_cast<const void*>(prob_load_token_addr),
                                         (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)),
                                         &smem_buffer_ptr->intra_node_mbarrier_buffer[stage][0]);

                total_tx_size += (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
              }

              // Optionally load scaling factor(Only for FP8 token).
              if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         reinterpret_cast<void*>(&smem_buffer_ptr->intra_node_scaling_factor_buffer[stage][0]),
                                         reinterpret_cast<const void*>(scaling_factor_load_base_addr + (current_token_id * (HIDDEN_DIM / 128))),
                                         (uint32_t)((HIDDEN_DIM / 128) * sizeof(float)),
                                         &smem_buffer_ptr->intra_node_mbarrier_buffer[stage][0]);

                total_tx_size += (uint32_t)((HIDDEN_DIM / 128) * sizeof(float));
              }

              cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                   cuda::ptx::scope_cta,
                                                   cuda::ptx::space_shared,
                                                   &smem_buffer_ptr->intra_node_mbarrier_buffer[stage][0],
                                                   total_tx_size);

              stage += 1;
              if(stage == NUM_OF_STAGES){
                stage = 0;
                consumer_parity ^= 1;
              }
            }
          }
        }
      }
    }
  }
}

// Device function for intra-node S2G warp for dispatch kernel. There can be only 1 intra-node S2G warp per CUDA block!
template<typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES, 
         int HIDDEN_DIM,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH>
inline __device__ void S2G_warp_group_device_function(const int local_rank,
                                                      const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const bool* rdma_to_attn_map,
                                                      const int32_t* sparse_to_dense_map,
                                                      TOKEN_DATA_TYPE* const* remote_expert_output_token,
                                                      float* const* remote_expert_output_prob,
                                                      float* const* remote_expert_output_scaling_factor,
                                                      SMEM_TYPE* smem_buffer_ptr)
{
  // Load rdma_to_attn_map using LDG.128. Each token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  // Load sparse_to_dense_map according to the NUM_OF_RANKS_PER_NODE.
  using sparse_to_dense_map_load_t = Copy_t<NUM_OF_RANKS_PER_NODE * sizeof(int32_t)>;
  constexpr int NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_INPUT_TOKEN = (NUM_OF_RANKS_PER_NODE * sizeof(int32_t)) / sizeof(sparse_to_dense_map_load_t);
  constexpr int NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER = sizeof(sparse_to_dense_map_load_t) / sizeof(int32_t);
  
  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  int stage = 0;
  uint32_t producer_parity = 0;

  // Only 1 thread within the S2G warp will be active, other threads will just exit.
  if(elect_sync(~0)){
    // Loop through all data chunk. Data(chunk) parallel between multiple CUDA blocks.
    for(int i = blockIdx.x; i < num_of_chunks_per_rank; i += NUM_OF_BLOCKS){
      // How many rdma_to_attn load iter for this chunk.
      int num_of_routing_info_load_iter_for_current_chunk;
      // How many token for this chunk.
      int current_chunk_size;
      if(remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1){
        num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
        current_chunk_size = remainder_chunk_size;
      }else{
        num_of_routing_info_load_iter_for_current_chunk = NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }
      for(int j = 0; j < NUM_OF_NODES; j++){
        // The current node been processed. For each chunk id, node_id order is local_node, local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
        int node_id = node_rank >= j ? node_rank - j : node_rank + NUM_OF_NODES - j;
        // Store every token and its properties from Shared to Global. Only store tokens that is needed by this node.
        const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map + 
                                                                         (node_id * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK));

        const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (node_id * num_of_tokens_per_rank + i * NUM_OF_TOKENS_PER_CHUNK) * NUM_OF_RANKS_PER_NODE;

        //#pragma unroll
        for(int k = 0; k < num_of_routing_info_load_iter_for_current_chunk; k++){
          rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[k];
          #pragma unroll
          for(int n = 0; n < NUM_OF_TOKENS_PER_LOAD_ITER; n++){
            int current_token_id = k * NUM_OF_TOKENS_PER_LOAD_ITER + n;
            // If the current token is out-of-bound, then just end this load iter.
            if(current_token_id >= current_chunk_size){
              break;
            }
            bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + n);
            if(token_needed_by_this_node){
              const sparse_to_dense_map_load_t* sparse_to_dense_map_load_addr = reinterpret_cast<const sparse_to_dense_map_load_t*>
                                                                                (sparse_to_dense_map_load_base_addr + (k * NUM_OF_TOKENS_PER_LOAD_ITER + n) * NUM_OF_RANKS_PER_NODE);
              // Wait until token entry within the shared memory has been produced.
              while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->intra_node_mbarrier_buffer[stage][0], producer_parity)){}

              // This token entry will be multicast to all ranks within this node which need this token and its properties.
              // The current implementation do the multicast by issue each unicast separately(we call it a unicast group). If NVLS can be used, we should use it here. 
              #pragma unroll
              for(int m = 0; m < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_INPUT_TOKEN; m++){
                // Load sparse_to_dense_map.
                sparse_to_dense_map_load_t sparse_to_dense_map_data = sparse_to_dense_map_load_addr[m];
                #pragma unroll
                for(int t = 0; t < NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER; t++){
                  int32_t output_buffer_index = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data) + t);
                  // Only unicast to this rank if it need the current token.
                  if(output_buffer_index != -1){
                    int remote_rank_id = m * NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER + t;
                    // Store the token from shared to remote global.
                    TOKEN_DATA_TYPE* remote_token_addr = remote_expert_output_token[remote_rank_id] + (output_buffer_index * HIDDEN_DIM);
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                             cuda::ptx::space_shared,
                                             reinterpret_cast<void*>(remote_token_addr),
                                             reinterpret_cast<const void*>(&smem_buffer_ptr->intra_node_token_buffer[stage][0]),
                                             (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE)));

                    // Store the prob from shared to remote global for FW dispatch.
                    if constexpr(FORWARD_DISPATCH){
                      float* remote_prob_addr = remote_expert_output_prob[remote_rank_id] + (output_buffer_index * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));
                      cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                               cuda::ptx::space_shared,
                                               reinterpret_cast<void*>(remote_prob_addr),
                                               reinterpret_cast<const void*>(&smem_buffer_ptr->intra_node_prob_buffer[stage][0]),
                                               (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)));

                    }

                    // Store the scaling factor from shared to remote global for FP8 tokens.
                    if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
                      float* remote_scaling_factor_addr = remote_expert_output_scaling_factor[remote_rank_id] + (output_buffer_index * (HIDDEN_DIM / 128));
                      cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                               cuda::ptx::space_shared,
                                               reinterpret_cast<void*>(remote_scaling_factor_addr),
                                               reinterpret_cast<const void*>(&smem_buffer_ptr->intra_node_scaling_factor_buffer[stage][0]),
                                               (uint32_t)((HIDDEN_DIM / 128) * sizeof(float)));

                    }
                  }
                }
              }
              // Commit the previous issued S2G TMA instructions for the same shared memory token entry to a bulk async copy group.
              cuda::ptx::cp_async_bulk_commit_group();
              // Wait for previous commited TMA instructions to finish reading the shared memory, so the shared memory can be reused by the producer.
              cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
              // Notify the producer warp to load next token entry to the shared memory as the shared memory can be reused.
              cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_mbarrier_buffer[stage][1]);
              // Goto next token entry in shared memory.
              stage += 1;
              if(stage == NUM_OF_STAGES){
                stage = 0;
                producer_parity ^= 1;
              }
            }
          }
        }
      }
    }
    // All S2G TMA operations for all tokens assigned to this CUDA block have been issued.
    // If the synchronization for output buffer for current rank is on host-side(i.e. cudaStreamSynchronize + MPI_Barrier etc.), then all CUDA block can exit. 
    // The result of output buffer for current rank is not ready when the dipatch kernel is completed, a Barrier within the node is needed.
    // Otherwise, the S2G warp of the first CUDA block must wait for all writes to the local output buffer complete before exit. So kernel completion means the output buffers for current rank is ready. 
    /*if constexpr(DEVICE_SIDE_SYNC){
      // Wait for all previous issued TMA instructions to complete writing to remote global memory.
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      // Atomically add 1 to the remote flag on remote ranks within the node to notify the remote rank.
      for(int i = 0; i < NUM_OF_RANKS_PER_NODE; i++){
        // red.release.sys.global.add.u32          [a], 1; 
        asm volatile("red.release.sys.global.add.u32 [%0], %1;"
                     :
                     : "l"(__cvta_generic_to_global(&remote_write_completion_flags[i][local_rank])) , "n"(1)
                     : "memory");
      }
      if(blockIdx.x == 0){
        // Wait for all flags on local rank to reach the expected value before exit.
        for(int i = 0; i < NUM_OF_RANKS_PER_NODE; i++){
          uint32_t flag_data = 0;
          do{
            flag_data = 0;
            // Need a strong system-scope load to observe peer ranks' Atomic result.
            asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                         : "=r"(flag_data)
                         : "l"(__cvta_generic_to_global(&remote_write_completion_flags[local_rank][i]))
                         : "memory");
          }while(flag_data != expected_flag_value);
        }
      }
    }*/
  }
}

// Device function for intra-node G2S warp for combine kernel. There can be only 1 such warp per CUDA block!
template<typename SMEM_TYPE,
         int NUM_OF_STAGES_G2S, 
         int HIDDEN_DIM, 
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool BACKWARD_COMBINE>
inline __device__ void intra_node_G2S_warp_group_device_function(const int node_rank,
                                                                 const int num_of_tokens_per_rank, 
                                                                 const bool* rdma_to_attn_map,
                                                                 const int32_t* sparse_to_dense_map, 
                                                                 uint16_t* const* remote_expert_input_token,
                                                                 float* const* remote_expert_input_prob,
                                                                 SMEM_TYPE* smem_buffer_ptr)
{
  // Load rdma_to_attn_map using LDG.128. Each dst token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  // Load sparse_to_dense_map according to the NUM_OF_RANKS_PER_NODE.
  using sparse_to_dense_map_load_t = Copy_t<NUM_OF_RANKS_PER_NODE * sizeof(int32_t)>;
  constexpr int NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN = (NUM_OF_RANKS_PER_NODE * sizeof(int32_t)) / sizeof(sparse_to_dense_map_load_t);
  constexpr int NUM_OF_INPUT_TOKENS_PER_LOAD_ITER = sizeof(sparse_to_dense_map_load_t) / sizeof(int32_t);

  // The intra node reduction warp group of each CUDA block produce a chunk at a time.
  // The chunk order is: first produce the same chunk id for all other nodes id, then produce following chunk id.
  // (i.e. chunk 0 for node + 1, node + 2, ... node - 1, then chunk 1 for node + 1, node + 2, ... node - 1)
  // The RDMA warp group of a CUDA block will consume the chunk by the same order. So each CUDA block will produce and consume the same set of chunks id.
  // The reason to distribute chunk in this order is that the inter-node reduction will need the same chunk id from all other nodes, so we need to produce and send chunks in this order.

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // Total number of chunks to produce for RDMA warps to consume.
  const int total_num_of_chunks = (NUM_OF_NODES - 1) * num_of_chunks_per_rank;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // Token stage id and phase.
  int token_stage = 0;
  uint32_t token_consumer_parity = 1;

  // Only 1 thread within the intra-node G2S warp will be active, other threads will just exit.
  if(elect_sync(~0)){
    // Iterate through all chunks assigned to this block.
    for(int i = blockIdx.x; i < total_num_of_chunks; i += NUM_OF_BLOCKS){
      // Which node this chunk will be sent to.
      int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
      // What is the chunk id of this chunk for the node it will be sent to.
      int chunk_id = i / (NUM_OF_NODES - 1);
      // How many rdma_to_attn load iter for this chunk.
      int num_of_routing_info_load_iter_for_current_chunk;
      // How many token for this chunk.
      int current_chunk_size;
      if(remainder_chunk_size != 0 && chunk_id == num_of_chunks_per_rank - 1){
        num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
        current_chunk_size = remainder_chunk_size;
      }else{
        num_of_routing_info_load_iter_for_current_chunk = NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }
    
      const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map + 
                                                                         (node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK));

      const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (node_id * num_of_tokens_per_rank + chunk_id * NUM_OF_TOKENS_PER_CHUNK) * NUM_OF_RANKS_PER_NODE;
    
      // Iterate through all dst tokens within this chunk.
      for(int j = 0; j < num_of_routing_info_load_iter_for_current_chunk; j++){
        rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[j];
        #pragma unroll
        for(int k = 0; k < NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER; k++){
          int current_token_id = j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k;
          // If the current token is out-of-bound, then just end this load iter.
          if(current_token_id >= current_chunk_size){
            break;
          }
          // Check whether this dst token is needed by this node. If not needed, just skip.
          bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + k);
          // If this dst token is needed by this node, load the sparse_to_dense map and load the src token for this dst token.
          if(token_needed_by_this_node){
            const sparse_to_dense_map_load_t* sparse_to_dense_map_load_addr = reinterpret_cast<const sparse_to_dense_map_load_t*>
                                                                              (sparse_to_dense_map_load_base_addr + (j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k) * NUM_OF_RANKS_PER_NODE);
            // Load sparse_to_dense map for this dst token(i.e. a row in sparse_to_dense map).
            sparse_to_dense_map_load_t sparse_to_dense_map_data[NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN];
            // First load sparse_to_dense map and decide the last src token within this row.
            int last_src_token_id = 0;
            #pragma unroll
            for(int n = 0; n < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; n++){
              sparse_to_dense_map_data[n] = sparse_to_dense_map_load_addr[n];
              #pragma unroll
              for(int m = 0; m < NUM_OF_INPUT_TOKENS_PER_LOAD_ITER; m++){
                int32_t sparse_to_dense_map_value = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data[n]) + m);
                if(sparse_to_dense_map_value != -1){
                  last_src_token_id = n * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER + m;
                }
              }
            }

            // Then issue all G2S TMA for this row.
            #pragma unroll
            for(int n = 0; n < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; n++){
              #pragma unroll
              for(int m = 0; m < NUM_OF_INPUT_TOKENS_PER_LOAD_ITER; m++){
                int32_t sparse_to_dense_map_value = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data[n]) + m);
                if(sparse_to_dense_map_value != -1){
                  int current_src_token_id = n * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER + m;
                  // Wait until current token entry within the shared memory has been consumed.
                  while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[token_stage][1], token_consumer_parity)){}

                  uint32_t total_tx_size = 0;
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(&smem_buffer_ptr->intra_node_token_G2S_buffer[token_stage][0]),
                                           reinterpret_cast<const void*>(remote_expert_input_token[current_src_token_id] + (sparse_to_dense_map_value * HIDDEN_DIM)),
                                           (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)),
                                           &smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[token_stage][0]);

                  total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));

                  if constexpr(BACKWARD_COMBINE){
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                             cuda::ptx::space_global,
                                             reinterpret_cast<void*>(&smem_buffer_ptr->intra_node_prob_G2S_buffer[token_stage][0]),
                                             reinterpret_cast<const void*>(remote_expert_input_prob[current_src_token_id] + (sparse_to_dense_map_value * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE))),
                                             (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)),
                                             &smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[token_stage][0]);

                    total_tx_size += (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
                  }

                  if(current_src_token_id == last_src_token_id){
                    smem_buffer_ptr->intra_node_flag_G2S_buffer[token_stage] = true;
                  }
                  else{
                    smem_buffer_ptr->intra_node_flag_G2S_buffer[token_stage] = false;
                  }

                  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                       cuda::ptx::scope_cta,
                                                       cuda::ptx::space_shared,
                                                       &smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[token_stage][0],
                                                       total_tx_size);

                  // Goto next token entry in shared memory.
                  token_stage += 1;
                  if(token_stage == NUM_OF_STAGES_G2S){
                    token_stage = 0;
                    token_consumer_parity ^= 1;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// Device function for intra-node reduction warp group for combine kernel.
template<typename INTRA_NODE_RED_GROUP,
         typename SMEM_TYPE,
         int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
         bool BACKWARD_COMBINE>
inline __device__ void intra_node_red_warp_group_device_function(const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const bool* rdma_to_attn_map,
                                                                 uint16_t* rdma_intra_node_red_token,
                                                                 float* rdma_intra_node_red_prob,
                                                                 SMEM_TYPE* smem_buffer_ptr)
{
  // Load rdma_to_attn_map using LDG.128. Each dst token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  // Load sparse_to_dense_map according to the NUM_OF_RANKS_PER_NODE.
  /*using sparse_to_dense_map_load_t = Copy_t<NUM_OF_RANKS_PER_NODE * sizeof(int32_t)>;
  constexpr int NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN = (NUM_OF_RANKS_PER_NODE * sizeof(int32_t)) / sizeof(sparse_to_dense_map_load_t);
  constexpr int NUM_OF_INPUT_TOKENS_PER_LOAD_ITER = sizeof(sparse_to_dense_map_load_t) / sizeof(int32_t);*/

  // Processing token using BF16x2 intruction, HIDDEN_DIM must be multiple of 2.
  static_assert(HIDDEN_DIM % 2 == 0, "HIDDEN_DIM must be multiple of 2.");
  constexpr int NUM_OF_ELEMENT_PER_THREAD = (HIDDEN_DIM / 2) / INTRA_NODE_RED_GROUP::size();
  // Processing prob using fp32.
  constexpr int NUM_OF_PROB_VEC_ELEMENT_PER_THREAD = ((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE - 1) / INTRA_NODE_RED_GROUP::size()) + 1;
  //static_assert(INTRA_NODE_RED_GROUP::size() >= NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE, "The size of intra-node reduction warp group must not be smaller than prob size.");

  // The intra node reduction warp group of each CUDA block produce a chunk at a time.
  // The chunk order is: first produce the same chunk id for all other nodes id, then produce following chunk id.
  // (i.e. chunk 0 for node + 1, node + 2, ... node - 1, then chunk 1 for node + 1, node + 2, ... node - 1)
  // The RDMA warp group of a CUDA block will consume the chunk by the same order. So each CUDA block will produce and consume the same set of chunks id.
  // The reason to distribute chunk in this order is that the inter-node reduction will need the same chunk id from all other nodes, so we need to produce and send chunks in this order.

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // Total number of chunks to produce for RDMA warps to consume.
  const int total_num_of_chunks = (NUM_OF_NODES - 1) * num_of_chunks_per_rank;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // Src token stage id and phase.
  int token_stage = 0;
  uint32_t token_producer_parity = 0;

  // Dst token stage id.
  int dst_token_stage = 0;

  // Whether there are S2G TMA operations of a previous chunk's dst token in-flight(unfinished).
  bool outstanding_in_flight_chunk = false;

  // rdma_remote_node_id and chunk_id for previous chunk.
  int last_chunk_id;
  int last_rdma_remote_node_id;

  // Iterate through all chunks assigned to this block.
  for(int i = blockIdx.x; i < total_num_of_chunks; i += NUM_OF_BLOCKS){
    // Which node this chunk will be sent to.
    int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
    // What is the chunk id of this chunk for the node it will be sent to.
    int chunk_id = i / (NUM_OF_NODES - 1);
    // Which node this chunk belongs to in output rdma reduction buffers.
    int rdma_remote_node_id = node_id > node_rank ? node_id - 1 : node_id;
    // How many rdma_to_attn load iter for this chunk.
    int num_of_routing_info_load_iter_for_current_chunk;
    // How many token for this chunk.
    int current_chunk_size;
    if(remainder_chunk_size != 0 && chunk_id == num_of_chunks_per_rank - 1){
      num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
      current_chunk_size = remainder_chunk_size;
    }else{
      num_of_routing_info_load_iter_for_current_chunk = NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK;
      current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
    }

    const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map + 
                                                                      (node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK));

    uint16_t* rdma_intra_node_red_token_base_ptr = rdma_intra_node_red_token + (rdma_remote_node_id * num_of_tokens_per_rank + chunk_id * NUM_OF_TOKENS_PER_CHUNK) * HIDDEN_DIM;
    float* rdma_intra_node_red_prob_base_ptr;
    if constexpr(BACKWARD_COMBINE){
      rdma_intra_node_red_prob_base_ptr = rdma_intra_node_red_prob + 
                                          (rdma_remote_node_id * num_of_tokens_per_rank + chunk_id * NUM_OF_TOKENS_PER_CHUNK) * 
                                          (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);
    }

    // How many dst token entry of current chunk have been in-flight.
    int additional_in_flight_s2g = 0;
    // Iterate through all dst tokens within this chunk.
    for(int j = 0; j < num_of_routing_info_load_iter_for_current_chunk; j++){
      rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[j];
      #pragma unroll
      for(int k = 0; k < NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER; k++){
        // Check whether there is a previous chunk's dst token S2G in-flight and also current chunk already has NUM_OF_ADDITIONAL_IN_FLIGHT_S2G dst token S2G in-flight.
        // If so, wait for previous chunk's S2G finish and notify the RDMA warp groups.
        if(outstanding_in_flight_chunk && (additional_in_flight_s2g == NUM_OF_ADDITIONAL_IN_FLIGHT_S2G)){
          if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
            if(elect_sync(~0)){
              // Wait for previous chunk's S2G finish.
              cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<NUM_OF_ADDITIONAL_IN_FLIGHT_S2G>{});
              // Notify the rdma warp group.
              if constexpr(NUM_OF_NODES != 1){
                cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[last_rdma_remote_node_id][last_chunk_id]);
              }
            }
          }
          outstanding_in_flight_chunk = false;
        }
        int current_token_id = j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k;
        // If the current token is out-of-bound, then just end this load iter.
        if(current_token_id >= current_chunk_size){
          break;
        }
        // Check whether this dst token is needed by this node. If not needed, just skip.
        bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + k);
        // If this dst token is needed by this node, which means this dst token will have at least 1 src token within the shread memory.
        // Then, load the src token for this dst token from shared memory and accumulate it to the accumulator.
        if(token_needed_by_this_node){
          // Accumulator for this dst token. Token must be accumulated in FP32.
          float2 acc_token_fp32[NUM_OF_ELEMENT_PER_THREAD];
          // Optional Accumulator for this dst token prob.
          float acc_prob[NUM_OF_PROB_VEC_ELEMENT_PER_THREAD];
          // End reduction group flag.
          bool last_src_token = false;
          // Init accumulator.
          #pragma unroll
          for(int n = 0; n < NUM_OF_ELEMENT_PER_THREAD; n++){
            acc_token_fp32[n].x = 0.0f;
            acc_token_fp32[n].y = 0.0f;
          }
          #pragma unroll
          for(int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++){
            acc_prob[n] = 0.0f;
          }

          // Continue loading src token for this dst token and reduce them to accumulator until all src token for this dst token have been accumulated.
          do{
            // Base address for current token and prob(optional) in shared memory.
            __nv_bfloat162* load_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(&smem_buffer_ptr->intra_node_token_G2S_buffer[token_stage][0]);
            float* load_prob_base_ptr;
            if constexpr(BACKWARD_COMBINE){
              load_prob_base_ptr = &smem_buffer_ptr->intra_node_prob_G2S_buffer[token_stage][0];
            }

            // Wait until current src token ready in shared memory.
            if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
              if(elect_sync(~0)){
                while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[token_stage][0], token_producer_parity)){}
              }
            }
            arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);

            // Accumulate token and prob(optional).
            #pragma unroll
            for(int n = 0; n < NUM_OF_ELEMENT_PER_THREAD; n++){
              int element_id = (n * INTRA_NODE_RED_GROUP::size()) + INTRA_NODE_RED_GROUP::thread_rank();
              __nv_bfloat162 src_data = load_token_base_ptr[element_id];
              float2 src_data_fp32 = __bfloat1622float2(src_data);
              acc_token_fp32[n].x += src_data_fp32.x;
              acc_token_fp32[n].y += src_data_fp32.y;      
            }

            if constexpr(BACKWARD_COMBINE){
              #pragma unroll
              for(int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++){
                int element_id = INTRA_NODE_RED_GROUP::thread_rank() + n * INTRA_NODE_RED_GROUP::size();
                if(element_id < NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE){
                  float src_data = load_prob_base_ptr[element_id];
                  acc_prob[n] += src_data;
                }
              }
            }

            // Check flag for last src token.
            last_src_token = smem_buffer_ptr->intra_node_flag_G2S_buffer[token_stage];

            // Make sure all warp group have finished loading the token entry and accumulate it to the register accumulator.
            // Then notify the producer warp to load next token entry to the shared memory as the shared memory can be reused.
            arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);
            if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
              if(elect_sync(~0)){
                cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[token_stage][1]);
              }
            }
            
            // Goto next src token entry.
            token_stage += 1;
            if(token_stage == NUM_OF_STAGES_G2S){
              token_stage = 0;
              token_producer_parity ^= 1;
            }

          }while(!last_src_token);

          // Base address for current dst token and prob(optional) in shared memory.
          __nv_bfloat162* store_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(&smem_buffer_ptr->intra_node_token_S2G_buffer[dst_token_stage][0]);
          float* store_prob_base_ptr;
          if constexpr(BACKWARD_COMBINE){
            store_prob_base_ptr = &smem_buffer_ptr->intra_node_prob_S2G_buffer[dst_token_stage][0];
          }

          // Let the TMA thread to wait for previously issued TMA S2G operations finish reading this entry.
          if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
            if(elect_sync(~0)){
              cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_STAGES_S2G - 1>{});
            }
          }
          // Make sure all threads within the red warp group have wait for previously issued TMA S2G operations finish reading this entry before storing new data to this entry.
          arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);
          
          // Store the token.
          #pragma unroll
          for(int n = 0; n < NUM_OF_ELEMENT_PER_THREAD; n++){
            int element_id = (n * INTRA_NODE_RED_GROUP::size()) + INTRA_NODE_RED_GROUP::thread_rank();
            // Convert accumulated token back to BF16 and store the result back to shared memory token entry.
            store_token_base_ptr[element_id] = __float22bfloat162_rn(acc_token_fp32[n]);
          }

          // Store the prob(optional).
          if constexpr(BACKWARD_COMBINE){
            #pragma unroll
            for(int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++){
              int element_id = INTRA_NODE_RED_GROUP::thread_rank() + n * INTRA_NODE_RED_GROUP::size();
              if(element_id < NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE){
                store_prob_base_ptr[element_id] = acc_prob[n];
              }
            }
          }

          // Make sure the shared memory stored by current thread is visible by async proxy.
          cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);

          // Make sure all threads within the red warp group have finished storing the current token entry and making it visible to async proxy.
          arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);

          // Let the TMA thread to issue S2G TMA operations for current token entry.
          if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
            if(elect_sync(~0)){
              uint16_t* current_token_addr = rdma_intra_node_red_token_base_ptr + (j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k) * HIDDEN_DIM;
              // Store the token from shared to global.
              cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                       cuda::ptx::space_shared,
                                       reinterpret_cast<void*>(current_token_addr),
                                       reinterpret_cast<const void*>(&smem_buffer_ptr->intra_node_token_S2G_buffer[dst_token_stage][0]),
                                       (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)));

              // Store the prob from shared to global(Optional).
              if constexpr(BACKWARD_COMBINE){
                float* current_prob_addr = rdma_intra_node_red_prob_base_ptr + (j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);
                cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                         cuda::ptx::space_shared,
                                         reinterpret_cast<void*>(current_prob_addr),
                                         reinterpret_cast<const void*>(&smem_buffer_ptr->intra_node_prob_S2G_buffer[dst_token_stage][0]),
                                         (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)));

              }
              // Commit S2G TMA operations for this dst token into a bulk async copy group.
              cuda::ptx::cp_async_bulk_commit_group();
            }
          }

          // Goto next dst token entry.
          dst_token_stage += 1;
          if(dst_token_stage == NUM_OF_STAGES_S2G){
            dst_token_stage = 0;
          }

          // Another token entry's S2G in-flight.
          additional_in_flight_s2g += 1;
        }
      }
    }
    // If the current chunk does not have NUM_OF_ADDITIONAL_IN_FLIGHT_S2G dst token entry in-flight, which is possible of rdma_to_attn map is really sparse.
    // We need to wait for both previous and current chunks' dst token entry S2G to finish and notify the RDMA warp group.
    if(outstanding_in_flight_chunk){
      if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
        if(elect_sync(~0)){
          // Wait for all previous chunk's(i.e. previous and current chunk) S2G finish.
          cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
          // Notify the rdma warp group.
          if constexpr(NUM_OF_NODES != 1){
            cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[last_rdma_remote_node_id][last_chunk_id]);
            cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[rdma_remote_node_id][chunk_id]);
          }
        }
      }
      outstanding_in_flight_chunk = false;
    }else{ // Otherwise, the current chunks is in-flight.
      outstanding_in_flight_chunk = true;
    }

    // Update last chunk's id.
    last_rdma_remote_node_id = rdma_remote_node_id;
    last_chunk_id = chunk_id;
  }

  // When all chunks have been processed, we need to check whether the last chunk is still in-flight.
  // If so, wait for it and notify RDMA warp group.
  if(outstanding_in_flight_chunk){
    if(INTRA_NODE_RED_GROUP::warp_rank() == 0){
      if(elect_sync(~0)){
        // Wait for the last chunk's S2G finish.
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
        // Notify the rdma warp group.
        if constexpr(NUM_OF_NODES != 1){
          cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[last_rdma_remote_node_id][last_chunk_id]);
        }
      }
    }
  }
}

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
// Device function for inter-node node2node(RDMA) warp for combine kernel. There can be only 1 inter-node warp per CUDA block!
template<typename INTER_NODE_RDMA_GROUP,
         typename SMEM_TYPE,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool BACKWARD_COMBINE>
inline __device__ void inter_node_N2N_warp_group_device_function(const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const bool* rdma_to_attn_map,
                                                                 doca_gpu_dev_verbs_qp **d_qps_gpu,
                                                                 struct combine_memory_region_info_t *mr_info,
                                                                 SMEM_TYPE* smem_buffer_ptr)
{
  // Load rdma_to_attn_map using LDG.128. Each token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  constexpr int WQE_NUM_RATIO = 1 + BACKWARD_COMBINE;
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  static_assert(INTER_NODE_RDMA_GROUP::size() == 32, "INTER_NODE_RDMA_GROUP should be 1 warp.");
  static_assert(INTER_NODE_RDMA_GROUP::size() >= NUM_OF_NODES - 1, "mr_info should be loaded at once.");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % INTER_NODE_RDMA_GROUP::size() == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of 32.");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of sizeof(rdma_to_attn_map_load_t).");
  // The (NUM_OF_NODES - 1) queue pairs of one block were arranged together.
  int block_offset = blockIdx.x * (NUM_OF_NODES - 1);
  // Mr_infos and rdma_mbarrier_buffer in shared memory.
  struct combine_memory_region_info_t *smem_mr_info_ptr = nullptr;
  uint32_t *smem_inter_node_num_of_write_per_node_ptr = nullptr;
  uint64_t (*intra_node_to_rdma_mbarrier_buffer_ptr)[MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK] = nullptr;
  if constexpr(NUM_OF_NODES != 1) {
    smem_mr_info_ptr = smem_buffer_ptr->combine_memory_region_info;
    smem_inter_node_num_of_write_per_node_ptr = smem_buffer_ptr->inter_node_num_of_write_per_node;
    if (INTER_NODE_RDMA_GROUP::thread_rank() < NUM_OF_NODES - 1) {
      smem_mr_info_ptr[INTER_NODE_RDMA_GROUP::thread_rank()] = mr_info[INTER_NODE_RDMA_GROUP::thread_rank() + block_offset];
      smem_inter_node_num_of_write_per_node_ptr[INTER_NODE_RDMA_GROUP::thread_rank()] = 0;
    }
    intra_node_to_rdma_mbarrier_buffer_ptr = smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer;
  }
  __syncwarp();
  // Total number of chunks to produce for RDMA warps to consume.
  int NUM_OF_CHUNKS_PER_RANK = (num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
  int TOTAL_NUM_OF_CHUNKS = (NUM_OF_NODES - 1) * NUM_OF_CHUNKS_PER_RANK;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // INTRA_NODE_RED_GROUP should be 1 warp.
  // The inter_node_N2N_warp should process the same chunk as intra_node_red_warp(They belong to the same block.)
  uint32_t token_consumer_parity = 0;
  // Loop for every chunks.
  for(int i = blockIdx.x; i < TOTAL_NUM_OF_CHUNKS; i += NUM_OF_BLOCKS){
    // Which node this chunk will be sent to.
    int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
    // What is the chunk id of this chunk for the node it will be sent to.
    int chunk_id = i / (NUM_OF_NODES - 1);
    int rdma_remote_node_id = node_id > node_rank ? node_id - 1 : node_id;
    int chunk_base_token_idx = node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    int token_range = NUM_OF_TOKENS_PER_CHUNK;
    if (chunk_id * NUM_OF_TOKENS_PER_CHUNK + token_range > num_of_tokens_per_rank) {
      token_range = num_of_tokens_per_rank - chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    }
    // Queue pair for the current block to the current remote.
    struct doca_gpu_dev_verbs_qp *qp = d_qps_gpu[rdma_remote_node_id + block_offset];
    // Try wait mbarrier.
    while(!cuda::ptx::mbarrier_try_wait_parity(&intra_node_to_rdma_mbarrier_buffer_ptr[rdma_remote_node_id][chunk_id], token_consumer_parity)){}
    // Calculating total num of tokens of the current chunk need to be sent.
    int num_of_tokens_need_write = 0;
    for (int token_idx_in_chunk = INTER_NODE_RDMA_GROUP::thread_rank();
         token_idx_in_chunk < token_range;
         token_idx_in_chunk += INTER_NODE_RDMA_GROUP::size()) {
      num_of_tokens_need_write += rdma_to_attn_map[token_idx_in_chunk + chunk_base_token_idx];
    }
    num_of_tokens_need_write = __reduce_add_sync(0xffffffff, num_of_tokens_need_write);
    int total_write_cnt = num_of_tokens_need_write * WQE_NUM_RATIO + 1;
    // Getting wqe buffer.
    uint64_t base_wqe_idx = 0;
    if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
      base_wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(qp, total_write_cnt);
      smem_inter_node_num_of_write_per_node_ptr[rdma_remote_node_id] += total_write_cnt;
    }
    base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);
    uint64_t curr_wqe_idx = base_wqe_idx;
    // Processing one chunk.
    for (int token_idx_in_chunk = INTER_NODE_RDMA_GROUP::thread_rank();
         token_idx_in_chunk < NUM_OF_TOKENS_PER_CHUNK;
         token_idx_in_chunk += INTER_NODE_RDMA_GROUP::size()) {
      int64_t token_idx = token_idx_in_chunk + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
      bool need_write = false;
      if (token_idx_in_chunk < token_range) {
        need_write = rdma_to_attn_map[token_idx_in_chunk + chunk_base_token_idx];
      }
      uint32_t write_map = __ballot_sync(0xffffffff, need_write);
      uint32_t partial_write_map = ((1 << INTER_NODE_RDMA_GROUP::thread_rank()) - 1) & write_map;
      int write_cnt = __popc(write_map);
      int write_idx = __popc(partial_write_map);
      if (need_write) {
        // Construct wqes for tokens
        uint64_t my_wqe_idx = curr_wqe_idx + write_idx;
        struct doca_gpu_dev_verbs_wqe *token_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, my_wqe_idx);
        doca_gpu_dev_verbs_wqe_prepare_write(qp, token_wqe_ptr, my_wqe_idx,
                                                  DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                                  DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                                                  smem_mr_info_ptr[rdma_remote_node_id].token_raddr + token_idx * HIDDEN_DIM * sizeof(uint16_t),
                                                  smem_mr_info_ptr[rdma_remote_node_id].token_rkey,
                                                  smem_mr_info_ptr[rdma_remote_node_id].token_laddr + (rdma_remote_node_id * num_of_tokens_per_rank + token_idx) * HIDDEN_DIM * sizeof(uint16_t),
                                                  smem_mr_info_ptr[rdma_remote_node_id].token_lkey,
                                                  HIDDEN_DIM * sizeof(uint16_t));
        if constexpr(BACKWARD_COMBINE) {
          my_wqe_idx += write_cnt;
          struct doca_gpu_dev_verbs_wqe *prob_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, my_wqe_idx);
          doca_gpu_dev_verbs_wqe_prepare_write(qp, prob_wqe_ptr, my_wqe_idx,
                                                    DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                                    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                                                    smem_mr_info_ptr[rdma_remote_node_id].prob_raddr + token_idx * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float),
                                                    smem_mr_info_ptr[rdma_remote_node_id].prob_rkey,
                                                    smem_mr_info_ptr[rdma_remote_node_id].prob_laddr + (rdma_remote_node_id * num_of_tokens_per_rank + token_idx) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float),
                                                    smem_mr_info_ptr[rdma_remote_node_id].prob_lkey,
                                                    (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
        }
      }
      curr_wqe_idx += write_cnt * WQE_NUM_RATIO;
      __syncwarp();
    }
    if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
      // Construct wqe for flag.
      struct doca_gpu_dev_verbs_wqe *flag_wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, curr_wqe_idx);
      doca_gpu_dev_verbs_wqe_prepare_atomic(qp, flag_wqe_ptr, curr_wqe_idx,
                                                 DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                                 DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                 smem_mr_info_ptr[rdma_remote_node_id].flag_raddr + chunk_id * sizeof(uint64_t),
                                                 smem_mr_info_ptr[rdma_remote_node_id].flag_rkey,
                                                 smem_mr_info_ptr[rdma_remote_node_id].flag_laddr + chunk_id * sizeof(uint64_t),
                                                 smem_mr_info_ptr[rdma_remote_node_id].flag_lkey,
                                                 sizeof(uint64_t), 1, 0);
      // Post send and poll cqs.
      doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(qp, base_wqe_idx, curr_wqe_idx);
      doca_gpu_dev_verbs_submit_db<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                                        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                                        DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
                                        DOCA_GPUNETIO_VERBS_QP_SQ>(qp, (curr_wqe_idx + 1));
    }
    __syncwarp();
  }
  if (INTER_NODE_RDMA_GROUP::thread_rank() < NUM_OF_NODES - 1) {
    struct doca_gpu_dev_verbs_qp *qp = d_qps_gpu[block_offset + INTER_NODE_RDMA_GROUP::thread_rank()];
    int status = doca_gpu_dev_verbs_poll_cq<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA,
                                                 DOCA_GPUNETIO_VERBS_QP_SQ>(
                                                 doca_gpu_dev_verbs_qp_get_cq_sq(qp),
                                                 smem_inter_node_num_of_write_per_node_ptr[INTER_NODE_RDMA_GROUP::thread_rank()]);
    assert(status >= 0);
  }
  token_consumer_parity ^= 1;
}
#endif

// Device function for inter-node G2S warp for combine kernel. There can be only 1 such warp per CUDA block!
template<typename SMEM_TYPE,
         int NUM_OF_STAGES_G2S, 
         int HIDDEN_DIM, 
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         int NUM_OF_TOKENS_PER_GROUP,
         bool BACKWARD_COMBINE>
inline __device__ void inter_node_G2S_warp_group_device_function(const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const uint64_t* expected_flag_value,
                                                                 const bool* rdma_to_attn_map,
                                                                 const bool* attn_to_rdma_map,
                                                                 const int32_t* sparse_to_dense_map, 
                                                                 uint16_t* const* remote_expert_input_token,
                                                                 float* const* remote_expert_input_prob,
                                                                 const uint16_t* rdma_inter_node_group_token,
                                                                 const float* rdma_inter_node_group_prob,
                                                                 const uint64_t* rdma_inter_node_group_flags,
                                                                 SMEM_TYPE* smem_buffer_ptr)
{
  // All chunks in output buffer(attn buffer) will be divided into token groups and assigned to different CUDA blocks. 
  // This is different than other functions where chunks are assigned to different CUDA blocks.
  static_assert(NUM_OF_TOKENS_PER_CHUNK % NUM_OF_TOKENS_PER_GROUP == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of NUM_OF_TOKENS_PER_GROUP.");
  constexpr int NUM_OF_TOKEN_GROUPS_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / NUM_OF_TOKENS_PER_GROUP;
  
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  // Load rdma_to_attn_map for a token group at once. Each dst token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = Copy_t<NUM_OF_TOKENS_PER_GROUP>;
  static_assert(NUM_OF_TOKENS_PER_GROUP == sizeof(rdma_to_attn_map_load_t), "Current implementation requires NUM_OF_TOKENS_PER_GROUP to be 1/2/4/8/16.");
  
  //constexpr int NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  //constexpr int NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  // Load sparse_to_dense_map according to the NUM_OF_RANKS_PER_NODE.
  using sparse_to_dense_map_load_t = Copy_t<NUM_OF_RANKS_PER_NODE * sizeof(int32_t)>;
  constexpr int NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN = (NUM_OF_RANKS_PER_NODE * sizeof(int32_t)) / sizeof(sparse_to_dense_map_load_t);
  constexpr int NUM_OF_INPUT_TOKENS_PER_LOAD_ITER = sizeof(sparse_to_dense_map_load_t) / sizeof(int32_t);

  // The inter node reduction warp group of each CUDA block produce a token group of a chunk at a time. Token groups of each chunk assigned to each CUDA block in interleave pattern.
  // The chunk order is: i.e. chunk 0, then chunk 1, ... the last chunk of attn output buffer.
  // The RDMA network for current rank will produce the same chunk id from node - 1, node - 2 ... node + 1. 
  // So inter node reduction warp group will consume the src chunk in the same order.

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // Total number of chunks to process in the output buffer(attn buffer). output buffer(attn buffer) will only have 1 rank's tokens.
  const int total_num_of_chunks = num_of_chunks_per_rank;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // Token stage id and phase.
  int token_stage = 0;
  uint32_t token_consumer_parity = 1;

  // Only 1 thread within the intra-node G2S warp will be active, other threads will just exit.
  if(elect_sync(~0)){
    // Iterate through all chunks. All chunks will assign to all CUDA block.
    for(int i = 0; i < total_num_of_chunks; i++){
      // How many rdma_to_attn load iter(a.k.a token group) for this chunk.
      int num_of_token_groups_for_current_chunk;
      // How many token for this chunk.
      int current_chunk_size;
      if(remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1){
        num_of_token_groups_for_current_chunk = ((remainder_chunk_size - 1) / NUM_OF_TOKENS_PER_GROUP) + 1;
        current_chunk_size = remainder_chunk_size;
      }else{
        num_of_token_groups_for_current_chunk = NUM_OF_TOKEN_GROUPS_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }
      const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map + 
                                                                         (node_rank * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK));
      const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (node_rank * num_of_tokens_per_rank + i * NUM_OF_TOKENS_PER_CHUNK) * NUM_OF_RANKS_PER_NODE;

      const bool* attn_to_rdma_map_load_base_addr = attn_to_rdma_map + (i * NUM_OF_TOKENS_PER_CHUNK) * (NUM_OF_NODES - 1); 

      // Padding from NUM_OF_NODES - 1 to NUM_OF_NODES in case NUM_OF_NODES = 1.
      // We still only use first NUM_OF_NODES - 1 flags, the last flag is the padding and not been used.
      bool rdma_flag_clear[NUM_OF_NODES];
      #pragma unroll
      for(int j = 0; j < NUM_OF_NODES; j++){
        rdma_flag_clear[j] = false;
      }

      // Iterate through all token groups within this chunk which assign to this CUDA block.
      for(int j = blockIdx.x; j < num_of_token_groups_for_current_chunk; j += NUM_OF_BLOCKS){
        rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[j];
        // Iterate through all dst(output) tokens within this token group.
        #pragma unroll
        for(int k = 0; k < NUM_OF_TOKENS_PER_GROUP; k++){
          int current_token_id = j * NUM_OF_TOKENS_PER_GROUP + k;
          // If the current token is out-of-bound, then just end this load iter.
          if(current_token_id >= current_chunk_size){
            break;
          }
          // Each dst token need to accumulate src tokens from local node's ranks(this part is the same as intra-node reduction), and src tokens from rdma inter-node buffers.
          // Accumulate local tokens first, then rdma tokens.

          // Check whether this dst token is needed by this(local) node. If not needed, just skip local accumulation.
          bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + k);
          // If this dst token is needed by this node, load the sparse_to_dense map and load the local src token for this dst token.
          if(token_needed_by_this_node){
            const sparse_to_dense_map_load_t* sparse_to_dense_map_load_addr = reinterpret_cast<const sparse_to_dense_map_load_t*>
                                                                              (sparse_to_dense_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * NUM_OF_RANKS_PER_NODE);
            // Load sparse_to_dense map for this dst token(i.e. a row in sparse_to_dense map).
            sparse_to_dense_map_load_t sparse_to_dense_map_data[NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN];
            // First load sparse_to_dense map and decide the last src token within this row.
            int last_src_token_id = 0;
            #pragma unroll
            for(int n = 0; n < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; n++){
              sparse_to_dense_map_data[n] = sparse_to_dense_map_load_addr[n];
              #pragma unroll
              for(int m = 0; m < NUM_OF_INPUT_TOKENS_PER_LOAD_ITER; m++){
                int32_t sparse_to_dense_map_value = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data[n]) + m);
                if(sparse_to_dense_map_value != -1){
                  last_src_token_id = n * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER + m;
                }
              }
            }
            // Then issue all G2S TMA for this row.
            #pragma unroll
            for(int n = 0; n < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; n++){
              #pragma unroll
              for(int m = 0; m < NUM_OF_INPUT_TOKENS_PER_LOAD_ITER; m++){
                int32_t sparse_to_dense_map_value = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data[n]) + m);
                if(sparse_to_dense_map_value != -1){
                  int current_src_token_id = n * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER + m;
                  // Wait until current token entry within the shared memory has been consumed.
                  while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][1], token_consumer_parity)){}

                  uint32_t total_tx_size = 0;
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(&smem_buffer_ptr->inter_node_token_G2S_buffer[token_stage][0]),
                                           reinterpret_cast<const void*>(remote_expert_input_token[current_src_token_id] + (sparse_to_dense_map_value * HIDDEN_DIM)),
                                           (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)),
                                           &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0]);

                  total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));

                  if constexpr(BACKWARD_COMBINE){
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                             cuda::ptx::space_global,
                                             reinterpret_cast<void*>(&smem_buffer_ptr->inter_node_prob_G2S_buffer[token_stage][0]),
                                             reinterpret_cast<const void*>(remote_expert_input_prob[current_src_token_id] + (sparse_to_dense_map_value * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE))),
                                             (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)),
                                             &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0]);

                    total_tx_size += (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
                  }

                  if(current_src_token_id == last_src_token_id){
                    smem_buffer_ptr->inter_node_flag_G2S_buffer[token_stage] = true;
                  }
                  else{
                    smem_buffer_ptr->inter_node_flag_G2S_buffer[token_stage] = false;
                  }

                  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                       cuda::ptx::scope_cta,
                                                       cuda::ptx::space_shared,
                                                       &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0],
                                                       total_tx_size);

                  // Goto next token entry in shared memory.
                  token_stage += 1;
                  if(token_stage == NUM_OF_STAGES_G2S){
                    token_stage = 0;
                    token_consumer_parity ^= 1;
                  }
                }
              }
            }
          }
          // Then accumulate from rdma inter-node buffers. There are total NUM_OF_NODES - 1 (possible) src tokens from rdma buffer to reduce.
          const bool* attn_to_rdma_map_load_addr = attn_to_rdma_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_OF_NODES - 1);
          #pragma unroll
          for(int n = 1; n < NUM_OF_NODES; n++){
            // The current node been processed. For each chunk id, node_id order is 
            // (no local_node itself, which is already been accumulated above) local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
            int node_id = node_rank >= n ? node_rank - n : node_rank + NUM_OF_NODES - n;
            // The tile id within the rdma buffers for the current node id. Because rdma buffers only have NUM_OF_NODES - 1 tile.
            int rdma_buffer_tile_id = node_id > node_rank ? node_id - 1 : node_id;
            // Check wether current dst token need src token from this node.
            if(attn_to_rdma_map_load_addr[rdma_buffer_tile_id]){
              // If the current chunk is not ready yet, wait for related rdma inter-node group buffer chunks ready first.
              if(rdma_flag_clear[n - 1] == false){
                const uint64_t* flag_location = rdma_inter_node_group_flags + (rdma_buffer_tile_id * num_of_chunks_per_rank + i);
                uint64_t rdma_flag = 0;
                do{
                  rdma_flag = 0;
                  // Need a strong system-scope load to observe external RDMA Atomic result.
                  asm volatile("ld.relaxed.sys.global.b64 %0, [%1];"
                              : "=l"(rdma_flag)
                              : "l"(__cvta_generic_to_global(flag_location))
                              : "memory");
                }while(rdma_flag != *expected_flag_value);
            
                // Mark the chunk from this node(tile) is already clear.
                rdma_flag_clear[n - 1] = true;
              }
              // Wait until current token entry within the shared memory has been consumed.
              while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][1], token_consumer_parity)){}
              // Load the src token from this rdma inter-node group buffer chunk to shared memory entry.
              uint32_t total_tx_size = 0;
              const uint16_t* rdma_inter_node_group_token_load_addr = rdma_inter_node_group_token + 
                                                                      (rdma_buffer_tile_id * num_of_tokens_per_rank + 
                                                                      i * NUM_OF_TOKENS_PER_CHUNK + 
                                                                      j * NUM_OF_TOKENS_PER_GROUP + k) * HIDDEN_DIM;
              cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                       cuda::ptx::space_global,
                                       reinterpret_cast<void*>(&smem_buffer_ptr->inter_node_token_G2S_buffer[token_stage][0]),
                                       reinterpret_cast<const void*>(rdma_inter_node_group_token_load_addr),
                                       (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)),
                                       &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0]);

              total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));

              if constexpr(BACKWARD_COMBINE){
                const float* rdma_inter_node_group_prob_load_addr = rdma_inter_node_group_prob + 
                                                                    (rdma_buffer_tile_id * num_of_tokens_per_rank + 
                                                                    i * NUM_OF_TOKENS_PER_CHUNK + 
                                                                    j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);

                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         reinterpret_cast<void*>(&smem_buffer_ptr->inter_node_prob_G2S_buffer[token_stage][0]),
                                         reinterpret_cast<const void*>(rdma_inter_node_group_prob_load_addr),
                                         (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float)),
                                         &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0]);

                total_tx_size += (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float));
              }

              // Inter-node token does not need flag since the red warp group will also read attn_to_rdma_map.

              cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                   cuda::ptx::scope_cta,
                                                   cuda::ptx::space_shared,
                                                   &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0],
                                                   total_tx_size);

              // Goto next token entry in shared memory.
              token_stage += 1;
              if(token_stage == NUM_OF_STAGES_G2S){
                token_stage = 0;
                token_consumer_parity ^= 1;
              }
            }
          }
        }
      }
    }
  }
}

// Device function for inter-node reduction warp group for combine kernel.
template<typename SMEM_TYPE,
         typename INTER_NODE_RED_GROUP,
         int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM, 
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         int NUM_OF_TOKENS_PER_GROUP,
         bool BACKWARD_COMBINE>
inline __device__ void inter_node_red_warp_group_device_function(const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const bool* rdma_to_attn_map,
                                                                 const bool* attn_to_rdma_map, 
                                                                 uint16_t* attn_output_token,
                                                                 float* attn_output_prob,
                                                                 SMEM_TYPE* smem_buffer_ptr)
{
  // All chunks in output buffer(attn buffer) will be divided into token groups and assigned to different CUDA blocks. 
  // This is different than other functions where chunks are assigned to different CUDA blocks.
  static_assert(NUM_OF_TOKENS_PER_CHUNK % NUM_OF_TOKENS_PER_GROUP == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of NUM_OF_TOKENS_PER_GROUP.");
  constexpr int NUM_OF_TOKEN_GROUPS_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / NUM_OF_TOKENS_PER_GROUP;

  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  // Load rdma_to_attn_map for a token group at once. Each dst token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = Copy_t<NUM_OF_TOKENS_PER_GROUP>;
  static_assert(NUM_OF_TOKENS_PER_GROUP == sizeof(rdma_to_attn_map_load_t), "Current implementation requires NUM_OF_TOKENS_PER_GROUP to be 1/2/4/8/16.");

  // Processing token using BF16x2 intruction, HIDDEN_DIM must be multiple of 2.
  static_assert(HIDDEN_DIM % 2 == 0, "HIDDEN_DIM must be multiple of 2.");
  constexpr int NUM_OF_ELEMENT_PER_THREAD = (HIDDEN_DIM / 2) / INTER_NODE_RED_GROUP::size();
  // Processing prob using fp32.
  constexpr int NUM_OF_PROB_VEC_ELEMENT_PER_THREAD = ((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE - 1) / INTER_NODE_RED_GROUP::size()) + 1;
  //static_assert(INTER_NODE_RED_GROUP::size() >= NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE, "The size of inter-node reduction warp group must not be smaller than prob size.");

  // The inter node reduction warp group of each CUDA block produce a token group of a chunk at a time. Token groups of each chunk assigned to each CUDA block in interleave pattern.
  // The chunk order is: i.e. chunk 0, then chunk 1, ... the last chunk of attn output buffer.
  // The RDMA network for current rank will produce the same chunk id from node - 1, node - 2 ... node + 1. 
  // So inter node reduction warp group will consume the src chunk in the same order.

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // Total number of chunks to process in the output buffer(attn buffer). output buffer(attn buffer) will only have 1 rank's tokens.
  const int total_num_of_chunks = num_of_chunks_per_rank;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // Src token stage id and phase.
  int token_stage = 0;
  uint32_t token_producer_parity = 0;

  // Dst token stage id.
  int dst_token_stage = 0;

  // Iterate through all chunks. All chunks will assign to all CUDA block.
  for(int i = 0; i < total_num_of_chunks; i++){
    // How many rdma_to_attn load iter(a.k.a token group) for this chunk.
    int num_of_token_groups_for_current_chunk;
    // How many token for this chunk.
    int current_chunk_size;
    if(remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1){
      num_of_token_groups_for_current_chunk = ((remainder_chunk_size - 1) / NUM_OF_TOKENS_PER_GROUP) + 1;
      current_chunk_size = remainder_chunk_size;
    }else{
      num_of_token_groups_for_current_chunk = NUM_OF_TOKEN_GROUPS_PER_CHUNK;
      current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
    }
    const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map + 
                                                                      (node_rank * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK));
    const bool* attn_to_rdma_map_load_base_addr = attn_to_rdma_map + (i * NUM_OF_TOKENS_PER_CHUNK) * (NUM_OF_NODES - 1);
    uint16_t* attn_output_token_base_ptr = attn_output_token + (i * NUM_OF_TOKENS_PER_CHUNK) * HIDDEN_DIM;
    float* attn_output_prob_base_ptr;
    if constexpr(BACKWARD_COMBINE){
      attn_output_prob_base_ptr = attn_output_prob + (i * NUM_OF_TOKENS_PER_CHUNK) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES);
    }
    // Iterate through all token groups within this chunk which assign to this CUDA block.
    for(int j = blockIdx.x; j < num_of_token_groups_for_current_chunk; j += NUM_OF_BLOCKS){
      rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[j];
      // Iterate through all dst(output) tokens within this token group.
      #pragma unroll
      for(int k = 0; k < NUM_OF_TOKENS_PER_GROUP; k++){
        int current_token_id = j * NUM_OF_TOKENS_PER_GROUP + k;
        // If the current token is out-of-bound, then just end this load iter.
        if(current_token_id >= current_chunk_size){
          break;
        }
        // Each dst token need to accumulate src tokens from local node's ranks(this part is the same as intra-node reduction), and src tokens from rdma inter-node buffers.
        // Accumulate local tokens first, then rdma tokens.
        // Accumulator for this dst token. Token must be accumulated in FP32.
        float2 acc_token_fp32[NUM_OF_ELEMENT_PER_THREAD];
        // Optional Accumulator for this dst token prob.
        // Different node's prob need to be gathered together to output.
        // 0 used for local node's prob, [1, NUM_OF_NODES - 1] used for remote node's prob.
        float acc_prob[NUM_OF_NODES][NUM_OF_PROB_VEC_ELEMENT_PER_THREAD];
        // Init accumulator.
        #pragma unroll
        for(int n = 0; n < NUM_OF_ELEMENT_PER_THREAD; n++){
          acc_token_fp32[n].x = 0.0f;
          acc_token_fp32[n].y = 0.0f;
        }
        #pragma unroll
        for(int n = 0; n < NUM_OF_NODES; n++){
          #pragma unroll
          for(int m = 0; m < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; m++){
            acc_prob[n][m] = 0.0f;
          }
        }

        // Check whether this dst token is needed by this(local) node. If not needed, just skip local accumulation.
        bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + k);
        // If this dst token is needed by this node, load the local src token from shared memory and accumulate them.
        if(token_needed_by_this_node){
          // End reduction group flag.
          bool last_local_node_src_token = false;
          
          // Continue loading local src token for this dst token and reduce them to accumulator until all local src token for this dst token have been accumulated.
          do{
            // Base address for current token and prob(optional) in shared memory.
            __nv_bfloat162* load_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(&smem_buffer_ptr->inter_node_token_G2S_buffer[token_stage][0]);
            float* load_prob_base_ptr;
            if constexpr(BACKWARD_COMBINE){
              load_prob_base_ptr = &smem_buffer_ptr->inter_node_prob_G2S_buffer[token_stage][0];
            }

            // Wait until current src token ready in shared memory.
            if(INTER_NODE_RED_GROUP::warp_rank() == 0){
              if(elect_sync(~0)){
                while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0], token_producer_parity)){}
              }
            }
            arrive_and_wait(INTER_NODE_RED_GROUP::size(), 2);

            // Accumulate token and prob(optional).
            #pragma unroll
            for(int n = 0; n < NUM_OF_ELEMENT_PER_THREAD; n++){
              int element_id = (n * INTER_NODE_RED_GROUP::size()) + INTER_NODE_RED_GROUP::thread_rank();
              __nv_bfloat162 src_data = load_token_base_ptr[element_id];
              float2 src_data_fp32 = __bfloat1622float2(src_data);
              acc_token_fp32[n].x += src_data_fp32.x;
              acc_token_fp32[n].y += src_data_fp32.y;      
            }

            if constexpr(BACKWARD_COMBINE){
              #pragma unroll
              for(int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++){
                int element_id = INTER_NODE_RED_GROUP::thread_rank() + n * INTER_NODE_RED_GROUP::size();
                if(element_id < NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE){
                  float src_data = load_prob_base_ptr[element_id];
                  acc_prob[0][n] += src_data;
                }
              }
            }

            // Check flag for last src token.
            last_local_node_src_token = smem_buffer_ptr->inter_node_flag_G2S_buffer[token_stage];

            // Make sure all warp group have finished loading the token entry and accumulate it to the register accumulator.
            // Then notify the producer warp to load next token entry to the shared memory as the shared memory can be reused.
            arrive_and_wait(INTER_NODE_RED_GROUP::size(), 2);
            if(INTER_NODE_RED_GROUP::warp_rank() == 0){
              if(elect_sync(~0)){
                cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][1]);
              }
            }
            
            // Goto next src token entry.
            token_stage += 1;
            if(token_stage == NUM_OF_STAGES_G2S){
              token_stage = 0;
              token_producer_parity ^= 1;
            }

          }while(!last_local_node_src_token);
        }

        // Then accumulate from rdma inter-node buffers. There are total NUM_OF_NODES - 1 (possible) src tokens from rdma buffer to reduce.
        const bool* attn_to_rdma_map_load_addr = attn_to_rdma_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_OF_NODES - 1);
        #pragma unroll
        for(int n = 1; n < NUM_OF_NODES; n++){
          // The current node been processed. For each chunk id, node_id order is 
          // (no local_node itself, which is already been accumulated above) local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
          int node_id = node_rank >= n ? node_rank - n : node_rank + NUM_OF_NODES - n;
          // The tile id within the rdma buffers(include attn_to_rdma map) for the current node id. Because these rdma buffers only have NUM_OF_NODES - 1 tile or element.
          int rdma_buffer_tile_id = node_id > node_rank ? node_id - 1 : node_id;
          // Check wether current dst token need src token from this (remote) node.
          if(attn_to_rdma_map_load_addr[rdma_buffer_tile_id]){
            // Base address for current token and prob(optional) in shared memory.
            __nv_bfloat162* load_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(&smem_buffer_ptr->inter_node_token_G2S_buffer[token_stage][0]);
            float* load_prob_base_ptr;
            if constexpr(BACKWARD_COMBINE){
              load_prob_base_ptr = &smem_buffer_ptr->inter_node_prob_G2S_buffer[token_stage][0];
            }
            // Wait until current src token ready in shared memory.
            if(INTER_NODE_RED_GROUP::warp_rank() == 0){
              if(elect_sync(~0)){
                while(!cuda::ptx::mbarrier_try_wait_parity(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0], token_producer_parity)){}
              }
            }
            arrive_and_wait(INTER_NODE_RED_GROUP::size(), 2);

            // Accumulate token and prob(optional).
            #pragma unroll
            for(int m = 0; m < NUM_OF_ELEMENT_PER_THREAD; m++){
              int element_id = (m * INTER_NODE_RED_GROUP::size()) + INTER_NODE_RED_GROUP::thread_rank();
              __nv_bfloat162 src_data = load_token_base_ptr[element_id];
              float2 src_data_fp32 = __bfloat1622float2(src_data);
              acc_token_fp32[m].x += src_data_fp32.x;
              acc_token_fp32[m].y += src_data_fp32.y;      
            }

            if constexpr(BACKWARD_COMBINE){
              #pragma unroll
              for(int m = 0; m < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; m++){
                int element_id = INTER_NODE_RED_GROUP::thread_rank() + m * INTER_NODE_RED_GROUP::size();
                if(element_id < NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE){
                  acc_prob[n][m] = load_prob_base_ptr[element_id];
                }
              }
            }

            // Inter-node token does not need flag.

            // Make sure all warp group have finished loading the token entry and accumulate it to the register accumulator.
            // Then notify the producer warp to load next token entry to the shared memory as the shared memory can be reused.
            arrive_and_wait(INTER_NODE_RED_GROUP::size(), 2);
            if(INTER_NODE_RED_GROUP::warp_rank() == 0){
              if(elect_sync(~0)){
                cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][1]);
              }
            }
            
            // Goto next src token entry.
            token_stage += 1;
            if(token_stage == NUM_OF_STAGES_G2S){
              token_stage = 0;
              token_producer_parity ^= 1;
            }
          }
        }

        // Store the dst token back to share memory. 
        // Because each attn token must have go to TOPK rank in dispatch, so it must have been reduced in combine. So each attn dst token must be written back.
        // Base address for current dst token and prob(optional) in shared memory.
        __nv_bfloat162* store_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(&smem_buffer_ptr->inter_node_token_S2G_buffer[dst_token_stage][0]);
        float* store_prob_base_ptr;
        if constexpr(BACKWARD_COMBINE){
          store_prob_base_ptr = &smem_buffer_ptr->inter_node_prob_S2G_buffer[dst_token_stage][0];
        }

        // Let the TMA thread to wait for previously issued TMA S2G operations finish reading this entry.
        if(INTER_NODE_RED_GROUP::warp_rank() == 0){
          if(elect_sync(~0)){
            cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_STAGES_S2G - 1>{});
          }
        }
        // Make sure all threads within the red warp group have wait for previously issued TMA S2G operations finish reading this entry before storing new data to this entry.
        arrive_and_wait(INTER_NODE_RED_GROUP::size(), 2);
          
        // Store the token.
        #pragma unroll
        for(int n = 0; n < NUM_OF_ELEMENT_PER_THREAD; n++){
          int element_id = (n * INTER_NODE_RED_GROUP::size()) + INTER_NODE_RED_GROUP::thread_rank();
          // Convert accumulated token back to BF16 and store the result back to shared memory token entry.
          store_token_base_ptr[element_id] = __float22bfloat162_rn(acc_token_fp32[n]);
        }

        // Store the prob(optional).
        if constexpr(BACKWARD_COMBINE){
          #pragma unroll
          for(int n = 0; n < NUM_OF_NODES; n++){
            int attn_prob_output_node_id = (node_rank - n) >= 0 ? node_rank - n : node_rank + NUM_OF_NODES - n;
            int element_base_id = attn_prob_output_node_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE);
            #pragma unroll
            for(int m = 0; m < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; m++){
              int element_id = INTER_NODE_RED_GROUP::thread_rank() + m * INTER_NODE_RED_GROUP::size();
              if(element_id < NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE){
                store_prob_base_ptr[element_base_id + element_id] = acc_prob[n][m];
              }
            }
          }
        }

        // Make sure the shared memory stored by current thread is visible by async proxy.
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);

        // Make sure all threads within the red warp group have finished storing the current token entry and making it visible to async proxy.
        arrive_and_wait(INTER_NODE_RED_GROUP::size(), 2);

        // Let the TMA thread to issue S2G TMA operations for current token entry.
        if(INTER_NODE_RED_GROUP::warp_rank() == 0){
          if(elect_sync(~0)){
            uint16_t* current_token_addr = attn_output_token_base_ptr + (j * NUM_OF_TOKENS_PER_GROUP + k) * HIDDEN_DIM;
            // Store the token from shared to global output.
            cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                     cuda::ptx::space_shared,
                                     reinterpret_cast<void*>(current_token_addr),
                                     reinterpret_cast<const void*>(&smem_buffer_ptr->inter_node_token_S2G_buffer[dst_token_stage][0]),
                                     (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)));

            // Store the prob from shared to global output.
            if constexpr(BACKWARD_COMBINE){
              float* current_prob_addr = attn_output_prob_base_ptr + (j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES);
              cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                       cuda::ptx::space_shared,
                                       reinterpret_cast<void*>(current_prob_addr),
                                       reinterpret_cast<const void*>(&smem_buffer_ptr->inter_node_prob_S2G_buffer[dst_token_stage][0]),
                                       (uint32_t)((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) * sizeof(float)));

            }
            // Commit S2G TMA operations for this dst token into a bulk async copy group.
            cuda::ptx::cp_async_bulk_commit_group();
          }
        }

        // Goto next dst token entry.
        dst_token_stage += 1;
        if(dst_token_stage == NUM_OF_STAGES_S2G){
          dst_token_stage = 0;
        }
      }
    }
  }
  // Because the attn output buffers will only be produced by local combine kernel, not by the combine kernels on other ranks,
  // so we only need to wait for local combine kernel to finish writing all token data back to output buffer before we can exit.
  // Also, a kernel will be considered completed from CUDA stream's perspective if and only if all the threads are exit and all memory operations(including TMA operations)
  // issued by all threads have been completed and made visible to sys scope.
  // So the CUDA stream's kernel boundary implicit synchronization should be enough to sync with all TMA operations issued in the combine kernel.
  // So we can directly exit w/o any explicit synchronization with TMA operations.
}

__launch_bounds__(1, 1)
static __global__ void device_sync_kernel(uint32_t* intra_node_remote_flags, const uint32_t* expected_flag_value)
{
  // Atomically reduce add 1 to the u32 flag on rank #0 in current NVLink domain. 
  // Need a strong system-scope red to make sure all ranks from current NVLink domain can see the side effect.
  // But no memory fence(i.e. .release) needed since CUDA stream already do that for us.
  // red.relaxed.sys.global.add.u32          [a], 1; 
  asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                :
                : "l"(__cvta_generic_to_global(intra_node_remote_flags)), "n"(1)
                : "memory");

  // Polling flag value from the u32 flag on rank #0 in current NVLink domain.
  // Keep polling until reach the expected value.
  uint32_t flag_data = 0;
  do{
      flag_data = 0;
      // Need a strong system-scope load to observe other ranks' Atomic result.
      // But no no memory fence(i.e. .aquired) needed since no memory operation behind this.
      asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                    : "=r"(flag_data)
                    : "l"(__cvta_generic_to_global(intra_node_remote_flags))
                    : "memory");
    }while(flag_data != *expected_flag_value);
}

// This kernel will update expected_rdma_flag_value and expected_intra_node_flag_value in local device memory
// by increasing the expected_rdma_flag_value by 1 and expected_intra_node_flag_value by NUM_OF_RANKS_PER_NODE.
template<int NUM_OF_NODES,
         int NUM_OF_RANKS_PER_NODE,
         bool DEVICE_SIDE_SYNC>
__launch_bounds__(1, 1)
__global__ void update_expected_value_kernel(uint64_t* expected_rdma_flag_value, uint32_t* expected_intra_node_flag_value)
{
  if constexpr(NUM_OF_NODES != 1){
    (*expected_rdma_flag_value) += 1;
  }
  if constexpr(DEVICE_SIDE_SYNC){
    (*expected_intra_node_flag_value) += NUM_OF_RANKS_PER_NODE;
  }
}

template<typename TOKEN_DATA_TYPE, 
         // This type represent inter-node warp group.
         typename INTER_NODE_GROUP, 
         // This type represent intra-node G2S warp group.
         typename INTRA_NODE_G2S_GROUP,
         // This type represent intra-node S2G warp group.
         typename INTRA_NODE_S2G_GROUP,
         // Number of token entry in the shared memory.
         int NUM_OF_STAGES,
         // Size of each chunk.
         int NUM_OF_TOKENS_PER_CHUNK,
         // Model configuration.
         int HIDDEN_DIM,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         // Number of CUDA block running dispatch kernel.
         int NUM_OF_BLOCKS,
         // Whether the dispatch kernel is used in forward process or backward process.
         bool FORWARD_DISPATCH>
// Each CUDA block of dispatch kernel has 3 warp groups and has the following layout: 
// 1. inter-node warp group(i.e. RDMA N2N warp group, 1 warp, only valid for multinode scenario) 2. intra-node G2S warp group(i.e. NVL G2S warp group, 1 warp). 
// 3. intra-node S2G warp group(i.e. NVL S2G warp group, 1 warp). Total 2 or 3 warps per CUDA block/SM.
__launch_bounds__(INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size(), 1)
__global__ void dispatch_kernel(const __grid_constant__ dispatch_kernel_param_t<TOKEN_DATA_TYPE> param)
{
  // Compile-time check. For now, 1 G2S and 1 S2G warp should be enough.
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  static_assert(INTER_NODE_GROUP::size() == 32, "Dispatch kernel only support 1 N2N warp currently.");
#endif
  static_assert(INTRA_NODE_G2S_GROUP::size() == 32, "Dispatch kernel only support 1 G2S warp currently.");
  static_assert(INTRA_NODE_S2G_GROUP::size() == 32, "Dispatch kernel only support 1 S2G warp currently.");
  // The token and its properties should meet size and alignment requirement.
  // Currently, we use TMA to copy prob data, which need at least 16B size and alignment(which requires expert per node to be multiple of 4).
  // We need to add padding or not using TMA for prob, if we want to support other scenario.
  static_assert((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * sizeof(float)) % 16 == 0, "Currently, expert per node must be multiple of 4(So the prob for each token is multiple of 16B) to make TMA work.");
  // If FP8 token is used, HIDDEN_DIM must be multiple of 512 to make scaling factor multiple of 16B to make TMA work.
  static_assert(((HIDDEN_DIM / 128) * sizeof(float)) % 16 == 0, "Currently, scaling factor per token must be multiple of 16B.");


  // Shared memory used over 48KB, should use dynamic shared memory.
  extern __shared__ uint8_t smem_bytes[];
  using cur_smem_t = dispatch_kernel_dynamic_shared_memory_buffer_t<TOKEN_DATA_TYPE, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, FORWARD_DISPATCH>;
  cur_smem_t* smem_buffer_ptr = reinterpret_cast<cur_smem_t*>(smem_bytes);

  // Let first thread of each CUDA block initialize the mbarrier.
  if(threadIdx.x == 0){
    for(int i = 0; i < NUM_OF_STAGES; i++){
      // Initialize mbarrier
      cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_mbarrier_buffer[i][0], 1);
      cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_mbarrier_buffer[i][1], 1);
    }
    // Make mbarriers initialization visible to async proxy(TMA).
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  }

  // Make sure all the warps wait for mbarriers to be initialized before producing/consuming data.
  __syncthreads();

  // Now warps can become specialized.
  // The input warp group data type must match the warp groups layout.
  // To prevent compiler generate pointless comparison warning.
  int threadIdx_x_int = (int)threadIdx.x;
  if(threadIdx_x_int < INTER_NODE_GROUP::size()){
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    // Inter-node warps groups.
    if constexpr(NUM_OF_NODES != 1){
      N2N_warp_group_device_function
      <INTER_NODE_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_EXPERTS_PER_RANK, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
      (param.node_rank, param.num_of_tokens_per_rank, param.attn_to_rdma_map, param.d_qps_gpu, param.mr_info, smem_buffer_ptr);
    }
#endif
  }else if(threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size()){
    // Intra-node G2S warp groups.
    G2S_warp_group_device_function
    <TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_EXPERTS_PER_RANK, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
    (param.node_rank, param.num_of_tokens_per_rank, param.expected_rdma_flag_value, param.rdma_to_attn_map, param.attn_input_token, param.attn_input_prob, param.attn_input_token_scaling_factor, param.rdma_inter_node_group_token,
    param.rdma_inter_node_group_prob, param.rdma_inter_node_group_scaling_factor, param.rdma_inter_node_group_flags, smem_buffer_ptr);
  }else if(threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size()){
    // Intra-node S2G warp groups.
    S2G_warp_group_device_function
    <TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_EXPERTS_PER_RANK, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
    (param.local_rank, param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, param.sparse_to_dense_map, param.expert_output_token, param.expert_output_prob,
    param.expert_output_scaling_factor, smem_buffer_ptr);
  }else{
    // Too many threads, should not goes here.
  }
}

template<// This type represent intra-node reduction warp group.
         typename INTRA_NODE_RED_GROUP, 
         // This type represent inter-node reduction warp group.
         typename INTER_NODE_RED_GROUP, 
         // This type represent intra-node G2S warp group.
         typename INTRA_NODE_G2S_GROUP,
         // This type represent inter-node G2S warp group.
         typename INTER_NODE_G2S_GROUP,
         // This type represent inter-node rdma warp group.
         typename INTER_NODE_RDMA_GROUP,
         // Number of token entry in the shared memory for G2S operations.
         int NUM_OF_STAGES_G2S,
         // Number of token entry in the shared memory for S2G operations.
         int NUM_OF_STAGES_S2G,
         // Number of token per group in the inter-node reduction/G2S warp group.
         int NUM_OF_TOKENS_PER_GROUP,
         // Size of each chunk.
         int NUM_OF_TOKENS_PER_CHUNK,
         // Model configuration.
         int HIDDEN_DIM,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         // Number of CUDA block running dispatch kernel.
         int NUM_OF_BLOCKS,
         // Number of fully in-flight S2G in intra-node reduction warp group.
         int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, 
         // Whether the combine kernel is used in backward process. If so, need to transfer the prob for each token as well.
         bool BACKWARD_COMBINE>
// Each CUDA block of combine kernel has 5 warp groups and has the following layout: 
// 1. intra-node reduction warp group(4 warps, only valid for multinode scenario). 2. inter-node reduction warp group(4 warps).
// 3. intra-node G2S warp group(1 warp, only valid for multinode scenario). 4. inter-node G2S warp group(1 warp). 5. inter-node N2N rdma warp group(1 warp, only valid for multinode scenario). 
// Total 5 or 11 warps per CUDA block/SM.
__launch_bounds__(INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size(), 1)
__global__ void combine_kernel(const __grid_constant__ combine_kernel_param_t param)
{
  // Compile-time check. For now, 1 G2S and 1 S2G warp should be enough.
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  static_assert(INTRA_NODE_G2S_GROUP::size() == 32, "Combine kernel only support 1 INTRA_NODE_G2S warp currently.");
#endif
  static_assert(INTER_NODE_G2S_GROUP::size() == 32, "Combine kernel only support 1 INTER_NODE_G2S warp currently.");
  // The token and its properties should meet size and alignment requirement.
  // Currently, we use TMA to copy prob data, which need at least 16B size and alignment(which requires expert per node to be multiple of 4).
  // We need to add padding or not using TMA for prob, if we want to support other scenario.
  static_assert((NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * sizeof(float)) % 16 == 0, "Currently, expert per node must be multiple of 4(So the prob for each token is multiple of 16B) to make TMA work.");
  static_assert(MAX_NUM_OF_TOKENS_PER_RANK % NUM_OF_TOKENS_PER_CHUNK == 0, "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of NUM_OF_TOKENS_PER_CHUNK.");
  constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;

  // Shared memory used over 48KB, should use dynamic shared memory.
  extern __shared__ uint8_t smem_bytes[];
  using cur_smem_t = combine_kernel_dynamic_shared_memory_buffer_t
  <NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, BACKWARD_COMBINE>;
  cur_smem_t* smem_buffer_ptr = reinterpret_cast<cur_smem_t*>(smem_bytes);

  // Let first thread of each CUDA block initialize the mbarrier.
  if(threadIdx.x == 0){
    for(int i = 0; i < NUM_OF_STAGES_G2S; i++){
      // Initialize mbarrier
      if constexpr(NUM_OF_NODES != 1){
        cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[i][0], 1);
        cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_mbarrier_G2S_buffer[i][1], 1);
      }
      cuda::ptx::mbarrier_init(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[i][0], 1);
      cuda::ptx::mbarrier_init(&smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[i][1], 1);
    }
    if constexpr(NUM_OF_NODES != 1){
      // Initialize mbarrier
      for(int i = 0; i < NUM_OF_NODES - 1; i++){
        for(int j = 0; j < MAX_NUM_OF_CHUNKS_PER_RANK; j++){
          cuda::ptx::mbarrier_init(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[i][j], 1);
        }
      }
    }
    // Make mbarriers initialization visible to async proxy(TMA).
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  }

  // Make sure all the warps wait for mbarriers to be initialized before producing/consuming data.
  __syncthreads();

  // Now warps can become specialized.
  // The input warp group data type must match the warp groups layout.
  // To prevent compiler generate pointless comparison warning.
  int threadIdx_x_int = (int)threadIdx.x;
  if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size()){
    // Intra-node reduction warp group.
    if constexpr(NUM_OF_NODES != 1){
      intra_node_red_warp_group_device_function
      <INTRA_NODE_RED_GROUP, cur_smem_t, NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, 
      NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, BACKWARD_COMBINE>
      (param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, param.rdma_intra_node_red_token, param.rdma_intra_node_red_prob, smem_buffer_ptr);
    }
  }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size()){
    // Inter-node reduction warp group.
    inter_node_red_warp_group_device_function
    <cur_smem_t, INTER_NODE_RED_GROUP, NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK,
    NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, NUM_OF_TOKENS_PER_GROUP, BACKWARD_COMBINE>
    (param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, param.attn_to_rdma_map, param.attn_output_token, param.attn_output_prob, smem_buffer_ptr);
  }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size()){
    // Intra-node G2S warp group.
    if constexpr(NUM_OF_NODES != 1){
      intra_node_G2S_warp_group_device_function
      <cur_smem_t, NUM_OF_STAGES_G2S, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, BACKWARD_COMBINE>
      (param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, param.sparse_to_dense_map, param.expert_input_token, param.expert_input_prob, smem_buffer_ptr);
    }
  }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size()){
    // Inter-node G2S warp group.
    inter_node_G2S_warp_group_device_function
    <cur_smem_t, NUM_OF_STAGES_G2S, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS,
    NUM_OF_TOKENS_PER_GROUP, BACKWARD_COMBINE>
    (param.node_rank, param.num_of_tokens_per_rank, param.expected_rdma_flag_value, param.rdma_to_attn_map, param.attn_to_rdma_map, param.sparse_to_dense_map, param.expert_input_token, param.expert_input_prob,
    param.rdma_inter_node_group_token, param.rdma_inter_node_group_prob, param.rdma_inter_node_group_flags, smem_buffer_ptr);
  }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size()){
    // Inter-node rdma warp group.
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    if constexpr(NUM_OF_NODES != 1){
      inter_node_N2N_warp_group_device_function
      <INTER_NODE_RDMA_GROUP, cur_smem_t, NUM_OF_STAGES_S2G, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, BACKWARD_COMBINE>
      (param.node_rank, param.num_of_tokens_per_rank, param.rdma_to_attn_map, param.d_qps_gpu, param.mr_info, smem_buffer_ptr);
    }
#endif
  }else{
    // Too many threads, should not goes here.
  }
}

template<int NUM_THREADS_PER_BLOCK,
         int NUM_OF_BLOCKS,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_EXPERTS_PER_RANK>
__launch_bounds__(NUM_THREADS_PER_BLOCK, 1)
__global__ void scan(const bool* input_routing_map, 
                     tmp_state_t* tmp, 
                     int32_t* sparse_to_dense_map, 
                     bool* rdma_to_attn_map,
                     bool* attn_to_rdma_map,
                     int32_t* num_of_tokens_for_experts,
                     bool* local_expert_routing_map,
                     const int node_rank,
                     const int local_rank,
                     const int num_of_tokens_per_rank)
{
  // Calculate the warps per block.
  constexpr int WARP_SIZE = 32;
  constexpr int NUM_OF_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / WARP_SIZE;

  // Calculate total threads count.
  constexpr int NUM_OF_TOTAL_THREADS = NUM_THREADS_PER_BLOCK * NUM_OF_BLOCKS;
  
  // Calculate the number of tokens belong to each CUDA block, warp and thread.
  // We assign 1 token(row in routing map) to 1 thread.
  const int num_of_total_attn_tokens = num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES;
  //static_assert(NUM_OF_TOTAL_ATTN_TOKENS % NUM_OF_TOTAL_THREADS == 0, "NUM_OF_TOTAL_ATTN_TOKENS must be multiple of NUM_OF_TOTAL_THREADS");
  const int num_of_tokens_per_thread = ((num_of_total_attn_tokens - 1) / NUM_OF_TOTAL_THREADS) + 1;
  const int num_of_tokens_per_warp = num_of_tokens_per_thread * WARP_SIZE;
  const int num_of_tokens_per_block = num_of_tokens_per_warp * NUM_OF_WARPS_PER_BLOCK;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;

  // For each token(row in routing map), calculate how many bytes need to be loaded from the routing map and how to load them.
  static_assert(sizeof(bool) == 1, "Bool is not 1 byte???");
  constexpr int NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN = NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE;
  using copy_t = Copy_t<NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN>;
  static_assert(NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN % sizeof(copy_t) == 0, "NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN and copy_t mismatch");
  constexpr int ROUTING_MAP_LOAD_ITER = NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN / sizeof(copy_t);

  // For each token, calculate how many bytes need to be store to sparse_to_dense_map.
  constexpr int NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN = sizeof(int32_t) * NUM_OF_RANKS_PER_NODE;
  using write_t = Copy_t<NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN>;
  static_assert(NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN % sizeof(write_t) == 0, "NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN and write_t mismatch");
  constexpr int S2D_MAP_STORE_ITER = NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN / sizeof(write_t);

  // How to convert per-expert routing info to per-rank routing info. We support any number of expert per rank.
  using expert_to_rank_t = Reduce_t<NUM_OF_EXPERTS_PER_RANK>;
  static_assert(NUM_OF_EXPERTS_PER_RANK % sizeof(expert_to_rank_t) == 0, "NUM_OF_EXPERTS_PER_RANK and expert_to_rank_t mismatch");
  constexpr int EXPERTS_TO_RANK_REDUCE_ITER = NUM_OF_EXPERTS_PER_RANK / sizeof(expert_to_rank_t);

  // How to convert per-rank routing info to per-node routing info. We support any number of ranks per node(nvl domain).
  //using rank_to_node_t = Reduce_t<NUM_OF_RANKS_PER_NODE>;
  //static_assert(NUM_OF_RANKS_PER_NODE % sizeof(rank_to_node_t) == 0, "NUM_OF_RANKS_PER_NODE and rank_to_node_t mismatch");
  //constexpr int RANKS_TO_NODE_REDUCE_ITER = NUM_OF_RANKS_PER_NODE / sizeof(rank_to_node_t);

  // How do a warp save per-rank routing info back to shared memory. What's the max number of elements does each thread save back.
  constexpr int NUM_OF_RANKS_PER_THREAD = ((NUM_OF_RANKS_PER_NODE - 1) / WARP_SIZE) + 1;

  // Sum of per-rank routing info of all warps within the block.
  __shared__ int32_t warp_token_routing_map_sum[NUM_OF_WARPS_PER_BLOCK][NUM_OF_RANKS_PER_NODE];
  // Sum of previous blocks' per-rank routing info.
  __shared__ int32_t previous_block_sum[NUM_OF_RANKS_PER_NODE];

  // We assign contiguous tokens called chunk to each CUDA block, each CUDA block get the same size of chunk.
  int block_starting_token = blockIdx.x * num_of_tokens_per_block;
  // warp id and lane id.
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  // We assign contiguous tokens called sub-chunk to each warp within a CUDA block, each warp within a CUDA block get the same size of sub-chunk.
  int warp_starting_token = block_starting_token + warp_id * num_of_tokens_per_warp;
  // Within a sub-chunk, we assign tokens to thread in a interleave pattern. So each thread process a token each time and each warp sum a tile of 32 tokens each time.
  int thread_starting_token = warp_starting_token + lane_id;
  
  // Step 0: Each warp sum the sub-chunk assigned to them and store the sum back to shared memory.
  // All warps within all CTA attend this step.
  // Also, some tokens need per-node info which store to rdma_to_attn_map, also processed here.

  // Sum of per-rank token routing map within a thread.
  int32_t token_routing_map_sum[NUM_OF_RANKS_PER_NODE];
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_NODE; i++){
    token_routing_map_sum[i] = 0;
  }

  //#pragma unroll
  for(int i = 0; i < num_of_tokens_per_thread; i++){
    // The global token id conditions for current token.
    int current_token_id = thread_starting_token + i * WARP_SIZE;
    // If the current token is out-of-bound, then just end summing tokens assigned to this thread. 
    if(current_token_id >= num_of_total_attn_tokens){
      break;
    }
    int current_token_node_rank = current_token_id / (num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE);
    int current_token_local_rank = (current_token_id % (num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE)) / num_of_tokens_per_rank;
    int current_token_local_id = current_token_id % num_of_tokens_per_rank;
    // If the token belongs to the inter-node group.
    // We need to calculate the per-node routing info and save back to rdma_to_attn_map.
    bool per_node_routing_info = (current_token_local_rank == local_rank);
    int current_token_rdma_to_attn_map_id = current_token_node_rank * rdma_to_attn_map_size_per_node + current_token_local_id;
    // Global routing map load base addr for current token.
    const copy_t* routing_map_load_base_addr = reinterpret_cast<const copy_t*>(input_routing_map + 
                                                                               current_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) + 
                                                                               node_rank * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));

    // Load the routing map for current token.
    bool token_routing_map[NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
    #pragma unroll
    for(int j = 0; j < ROUTING_MAP_LOAD_ITER; j++){
      *(reinterpret_cast<copy_t*>(token_routing_map) + j) = routing_map_load_base_addr[j];
    }

    // Convert the routing map to per rank routing info and accumulate to accumulator.
    // Also convert the per rank routing info to per node routing info.
    bool token_needed_by_this_node = false;
    #pragma unroll
    for(int j = 0; j < NUM_OF_RANKS_PER_NODE; j++){
      bool token_needed_by_this_rank = false;
      #pragma unroll
      for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
        int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
        expert_to_rank_t reduction_data = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
        if(reduction_data != (expert_to_rank_t)0){
          token_needed_by_this_rank = true;
          break;
        }
      }
      if(token_needed_by_this_rank){
        token_routing_map_sum[j] += 1;
        token_needed_by_this_node = true;
      }
    }

    // Save the per node routing info back to rdma_to_attn_map if needed.
    if(per_node_routing_info){
      rdma_to_attn_map[current_token_rdma_to_attn_map_id] = token_needed_by_this_node;
    }
  }

  // Each warp sum the per-rank routing info from all its threads.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_NODE; i++){
    int dst_tid = i % WARP_SIZE;
    int dst_id = i / WARP_SIZE;
    int32_t temp_sum = __reduce_add_sync(~0, token_routing_map_sum[i]);
    if(lane_id == dst_tid){
      token_routing_map_sum[dst_id] = temp_sum;
    }
  }

  // Each warp store the sum of per-rank routing info back to shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_RANKS_PER_NODE){
      warp_token_routing_map_sum[warp_id][element_id] = token_routing_map_sum[i];
    }
  }

  // Sync within a CUDA block to make sure all warps have produced the per-rank sum data to the shared memory before any thread can consume them to produce CUDA block level's sum data.
  __syncthreads();

  // Step 1: Communication between CUDA blocks. Each CUDA block's threads need to produce and store the current block's per-rank sum data to global memory,
  // and load and accumulate previous blocks' per-rank sum data and save the result to shared memory.

  // Each thread within a CUDA block calculate the CUDA block level sum for a single rank at a time.
  for(int i = threadIdx.x; i < NUM_OF_RANKS_PER_NODE; i += NUM_THREADS_PER_BLOCK){
    int32_t rank_acc = 0;
    // Calculate the sum of current rank within this CUDA block.
    #pragma unroll
    for(int j = 0; j < NUM_OF_WARPS_PER_BLOCK; j++){
      rank_acc += warp_token_routing_map_sum[j][i];
    }

    // Store the sum of current rank within this CUDA block to global memory for later scan opeartions.
    // Strong(atomic) store is needed to be visible to strong(atomic) load from other blocks.
    tmp_state_t* tmp_dst = &tmp[blockIdx.x * NUM_OF_RANKS_PER_NODE + i];
    tmp_state_t tmp_data{PRIV_SUM, rank_acc};
    uint64_t data = *reinterpret_cast<uint64_t*>(&tmp_data);
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;"
                  :
                  : "l"(__cvta_generic_to_global(tmp_dst)), "l"(data)
                  : "memory");
  }

  // Each thread within a CUDA block load previous blocks' block level sum for a single rank at a time.
  for(int i = threadIdx.x; i < NUM_OF_RANKS_PER_NODE; i += NUM_THREADS_PER_BLOCK){
    int32_t previous_block_sum_for_current_rank = 0;
    for(int j = 0; j < blockIdx.x; j++){
      tmp_state_t tmp_data{EMPTY, 0};
      tmp_state_t* tmp_src = &tmp[j * NUM_OF_RANKS_PER_NODE + i];
      do{
          // Load previous blocks' per-rank sum from global memory.
          // Strong(atomic) load is needed to view strong(atomic) store from other blocks.
          uint64_t data = 0;
          asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];"
                        : "=l"(data)
                        : "l"(__cvta_generic_to_global(tmp_src))
                        : "memory");
          tmp_data = *reinterpret_cast<tmp_state_t*>(&data);
      }while(tmp_data.state != PRIV_SUM);
      previous_block_sum_for_current_rank += tmp_data.value;
    }
    previous_block_sum[i] = previous_block_sum_for_current_rank;
  }

  // Sync within a CUDA block to make sure all previous blocks' per-rank sum have been produced to the shared memory before any thread can consume them in scan operation.
  __syncthreads();

  // Step 2: Each warp scan the sub-chunk assigned to them(the same sub-chunk as step 0) and produce sparse_to_dense_map, local_expert_routing_map and num_of_tokens_for_experts.
  int32_t previous_token_sum[NUM_OF_RANKS_PER_NODE];

  // Each warp load the previous blocks' per-rank sum from shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_RANKS_PER_NODE){
      previous_token_sum[i] = previous_block_sum[element_id];
    }
  }

  // Each warp accumulate the previous warps' per-rank sum from shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < NUM_OF_RANKS_PER_NODE){
      for(int j = 0; j < warp_id; j++){
        previous_token_sum[i] += warp_token_routing_map_sum[j][element_id];
      }
    }
  }

  // Each warp broadcast the accumulated previous per-rank routing info to all its threads.
  // Exact reverse of warp reduce operation.
  #pragma unroll
  for(int i = NUM_OF_RANKS_PER_NODE - 1; i >= 0 ; i--){
    int src_tid = i % WARP_SIZE;
    int src_id = i / WARP_SIZE;
    previous_token_sum[i] = __shfl_sync(~0, previous_token_sum[src_id], src_tid);
  }

  // Each warp scan all the tiles within its sub-chunk.
  //#pragma unroll
  for(int i = 0; i < num_of_tokens_per_thread; i++){
    // The global token id conditions for current token.
    int current_token_id = thread_starting_token + i * WARP_SIZE;
    // If the current token is out-of-bound, then mark it as out-of-bound. 
    int token_out_of_bound = 0;
    if(current_token_id >= num_of_total_attn_tokens){
      token_out_of_bound = 1;
    }
    // If the whole tiles are out-of-bound, the warp just finish and exit the scan loop together.
    if(__all_sync(~0, token_out_of_bound) != 0){
      break;
    }
    int current_token_node_rank = current_token_id / (num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE);
    int current_token_local_rank = (current_token_id % (num_of_tokens_per_rank * NUM_OF_RANKS_PER_NODE)) / num_of_tokens_per_rank;
    int current_token_local_id = current_token_id % num_of_tokens_per_rank;

    // Global routing map load base addr for current token.
    const copy_t* routing_map_load_base_addr = reinterpret_cast<const copy_t*>(input_routing_map + 
                                                                               current_token_id * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) + 
                                                                               node_rank * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));

    // Load the routing map for current token. Only load when the token is not out-of-bound.
    bool token_routing_map[NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
    if(token_out_of_bound == 0){
      #pragma unroll
      for(int j = 0; j < ROUTING_MAP_LOAD_ITER; j++){
        *(reinterpret_cast<copy_t*>(token_routing_map) + j) = routing_map_load_base_addr[j];
      }
    }
    
    // Convert the routing map to per rank routing info for current token, 
    // then produce the per-rank final exclusive scan within the warp for this tile.
    int32_t final_ex_scan[NUM_OF_RANKS_PER_NODE];
    #pragma unroll
    for(int j = 0; j < NUM_OF_RANKS_PER_NODE; j++){
      int32_t temp_scan = 0;
      bool token_needed_by_this_rank = false;
      // If the token is not out-of-bound, check whether this rank need this token.
      if(token_out_of_bound == 0){
        #pragma unroll
        for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
          int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
          expert_to_rank_t reduction_data = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
          if(reduction_data != (expert_to_rank_t)0){
            token_needed_by_this_rank = true;
            break;
          }
        }
        if(token_needed_by_this_rank){
          temp_scan = 1;
        }else{
          temp_scan = 0;
        }
      }
      
      // Each warp perform a inclusive scan from all threads(lanes).
      #pragma unroll
      for(int k = 1; k < WARP_SIZE; k *= 2){
        int32_t temp = __shfl_up_sync(~0, temp_scan, k);
        if(lane_id >= k){
          temp_scan += temp;
        }
      }

      // The inclusive scan from last lane is the sum of this rank of this tile. Need to accumulate that for later tiles.
      int32_t temp_sum = __shfl_sync(~0, temp_scan, WARP_SIZE - 1);

      // Make scan exclusive.
      int32_t exclusive_scan = __shfl_up_sync(~0, temp_scan, 1);
      temp_scan = (lane_id >= 1) ? exclusive_scan : 0;

      // Calculate the final exclusive scan for current token. -1 represent that the current rank does not need the current token. 
      final_ex_scan[j] = token_needed_by_this_rank ? previous_token_sum[j] + temp_scan : -1;

      // Accumulate the sum to accumulator.
      previous_token_sum[j] += temp_sum;

      // Each thread save local routing map for this token of the local rank to local_expert_routing_map if this token is needed by the local rank.
      if(j == local_rank && token_needed_by_this_rank){
        expert_to_rank_t* local_expert_routing_map_store_base_addr = reinterpret_cast<expert_to_rank_t*>(local_expert_routing_map + (final_ex_scan[j] * NUM_OF_EXPERTS_PER_RANK));
        #pragma unroll
        for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
          int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
          local_expert_routing_map_store_base_addr[k] = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
        }
      }

      // The thread that processing the global last token save the final sum for current rank to num_of_tokens_for_experts.
      if(current_token_id == num_of_total_attn_tokens - 1 && j == local_rank){
        *num_of_tokens_for_experts = previous_token_sum[j];
      }
    }

    // Save final exclusive scan of this token back to sparse_to_dense_map if current token is not out-of-bound and is needed. 
    if(token_out_of_bound == 0 && current_token_local_rank == local_rank){
      // sparse_to_dense_map store base addr for current token.
      write_t* sparse_to_dense_map_store_base_addr = reinterpret_cast<write_t*>(sparse_to_dense_map + 
                                                                                (current_token_node_rank * num_of_tokens_per_rank + current_token_local_id) * NUM_OF_RANKS_PER_NODE);
      #pragma unroll
      for(int j = 0; j < S2D_MAP_STORE_ITER; j++){
        sparse_to_dense_map_store_base_addr[j] = *(reinterpret_cast<write_t*>(final_ex_scan) + j);
      }
    }
  }

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  // Step 3: When NUM_OF_NODES > 1, we need to produce attn_to_rdma_map.
  // Since each token(row) is fully independent, each token(row) is assigned to each threads in a interleave pattern.
  if constexpr(NUM_OF_NODES != 1){
    const int num_of_total_token_rows = (NUM_OF_NODES - 1) * num_of_tokens_per_rank;
    //static_assert(NUM_OF_TOTAL_TOKEN_ROWS % NUM_OF_TOTAL_THREADS == 0, "NUM_OF_TOTAL_TOKEN_ROWS must be multiple of NUM_OF_TOTAL_THREADS.");
    const int num_of_token_rows_per_thread = ((num_of_total_token_rows - 1) / NUM_OF_TOTAL_THREADS) + 1;

    int tid = threadIdx.x + blockIdx.x * NUM_THREADS_PER_BLOCK;

    //#pragma unroll
    for(int i = 0; i < num_of_token_rows_per_thread; i++){
      int current_token_id = i * NUM_OF_TOTAL_THREADS + tid;
      // If the current token is out-of-bound, then just end processing token rows assigned to this thread. 
      if(current_token_id >= num_of_total_token_rows){
        break;
      }
      int current_token_attn_to_rdma_map_node_id = current_token_id % (NUM_OF_NODES - 1);
      int current_token_node_id = current_token_attn_to_rdma_map_node_id < node_rank ? current_token_attn_to_rdma_map_node_id : current_token_attn_to_rdma_map_node_id + 1;
      int current_token_local_id = current_token_id / (NUM_OF_NODES - 1);

      const copy_t* routing_map_load_base_addr = reinterpret_cast<const copy_t*>(input_routing_map + 
                                                                                ((node_rank * NUM_OF_RANKS_PER_NODE + local_rank) * num_of_tokens_per_rank + current_token_local_id) *
                                                                                (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES) + 
                                                                                (current_token_node_id * NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE));

      bool* attn_to_rdma_map_base_addr = attn_to_rdma_map + (current_token_local_id * (NUM_OF_NODES - 1) + current_token_attn_to_rdma_map_node_id);

      // Load the routing map for current token row.
      bool token_routing_map[NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE];
      #pragma unroll
      for(int j = 0; j < ROUTING_MAP_LOAD_ITER; j++){
        *(reinterpret_cast<copy_t*>(token_routing_map) + j) = routing_map_load_base_addr[j];
      }

      // Convert the routing map to per rank routing info and then to per node routing info.
      bool token_needed_by_this_node = false;
      #pragma unroll
      for(int j = 0; j < NUM_OF_RANKS_PER_NODE; j++){
        bool token_needed_by_this_rank = false;
        #pragma unroll
        for(int k = 0; k < EXPERTS_TO_RANK_REDUCE_ITER; k++){
          int current_expert_to_rank_t_id = j * EXPERTS_TO_RANK_REDUCE_ITER + k;
          expert_to_rank_t reduction_data = *(reinterpret_cast<expert_to_rank_t*>(token_routing_map) + current_expert_to_rank_t_id);
          if(reduction_data != (expert_to_rank_t)0){
            token_needed_by_this_rank = true;
            break;
          }
        }
        if(token_needed_by_this_rank){
          token_needed_by_this_node = true;
          break;
        }
      }

      *attn_to_rdma_map_base_addr = token_needed_by_this_node;
    }
  }
#endif
}

template< 
        // Hidden size of a token.
        int HIDDEN_DIM,
        // The max num of attn tokens output by a rank/GPU. Used by combine API.
        int MAX_NUM_OF_TOKENS_PER_RANK,
        // Number of ranks/GPU per NVLink domain.
        int NUM_OF_RANKS_PER_NODE,
        // Number of total NVLink domain, i.e. the size of RDMA domain.
        int NUM_OF_NODES,
        // Number of experts running on each rank/GPU. Hybrid-ep support multiple experts running on a single rank/GPU.
        int NUM_OF_EXPERTS_PER_RANK>
class hybrid_ep{
public:

  // Ctor, don't need for now.
  /*hybrid_ep(int local_rank, int node_rank, MPI_Comm comm):
    local_rank_(local_rank),
    node_rank_(node_rank),
    comm_(comm) {}*/

  // Dtor, don't need for now.
  //~hybrid_ep() {}

  // Processing metadata. Calculate routing info needed by dispatch and combine operations.
  // input_routing_map: IO: input, dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES, NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES]. 
  // Routing map which contain global routing info from all tokens to all expert. Allgather is needed before passing the routing map to this API.
  // preprocessing_tmp: IO: output/input, dtype: tmp_state_t, shape: [NUM_OF_BLOCKS for preprocessing kernel, NUM_OF_RANKS_PER_NODE].
  // The temp buffer needed by the preprocessing kernel.
  // sparse_to_dense_map: IO: output, dtype: int32_t, shape: [NUM_OF_TOKENS_PER_RANK * NUM_OF_NODES, NUM_OF_RANKS_PER_NODE].
  // The routing info needed by NVL warps(i.e. intra-node communication warps) during both dispatch and combine operation. Remains the same in a trainning iteration(FW+BP).
  // rdma_to_attn_map: IO: output, dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK padded to 16 * NUM_OF_NODES]
  // The routing info mainly needed by RDMA warps during the combine operation. Remains the same in a trainning iteration(FW+BP).
  // attn_to_rdma_map: IO: output, dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK, NUM_OF_NODES - 1].
  // The routing info mainly needed by RDMA warps during the dispatch operation. Remains the same in a trainning iteration(FW+BP).
  // num_of_tokens_for_experts: IO: output, dtype: int32_t, shape: [1].
  // The total size of expert buffer on this rank(in number of tokens), according to the global routing map. If there are multiple expert on this rank, each token will only appear once.
  // Remains the same in a trainning iteration(FW+BP).
  // local_expert_routing_map: IO: output, dtype: bool, shape: [at least num_of_tokens_for_experts, NUM_OF_EXPERTS_PER_RANK].
  // The per-expert routing info for all tokens within the expert buffer of this rank. It is used by later layer to routing the tokens to different experts on this rank.
  // Remains the same in a trainning iteration(FW+BP).
  template<// Block size for preprocessing kernel.
           int NUM_THREADS_PER_BLOCK, 
           // Grid size for preprocessing kernel(1:1 block:SM mapping).
           int NUM_OF_BLOCKS>
  static void metadata_preprocessing(const bool* input_routing_map, 
                                     tmp_state_t* preprocessing_tmp,
                                     int32_t* sparse_to_dense_map,
                                     bool* rdma_to_attn_map,
                                     bool* attn_to_rdma_map,
                                     int32_t* num_of_tokens_for_experts,
                                     bool* local_expert_routing_map,
                                     const int node_rank,
                                     const int local_rank,
                                     const int num_of_tokens_per_rank,
                                     cudaStream_t stream)
  {
    // Gather routing map from all ranks to all ranks.
    // All ranks should have the same global routing map after this communication.
    // It is a synchronous communication.
    /*MPI_CHECK(MPI_Allgather(reinterpret_cast<const void *>(input_routing_map),
                            NUM_OF_TOKENS_PER_RANK * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES), 
                            MPI_BYTE,
                            reinterpret_cast<void *>(global_routing_map_), 
                            NUM_OF_TOKENS_PER_RANK * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES),
                            MPI_BYTE, 
                            comm_));*/

    // Init preprocessing_tmp buffers.
    constexpr size_t preprocessing_tmp_sz = NUM_OF_BLOCKS * NUM_OF_RANKS_PER_NODE * sizeof(tmp_state_t);
    CUDA_CHECK(cudaMemsetAsync(preprocessing_tmp, 0, preprocessing_tmp_sz, stream));

    // Launch the preprocessing kernel to process the global routing map.
    scan<NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_EXPERTS_PER_RANK>
    <<<NUM_OF_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream>>>
    (input_routing_map, preprocessing_tmp, sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map, num_of_tokens_for_experts, local_expert_routing_map, node_rank, local_rank, num_of_tokens_per_rank);

    // Check if there is any CUDA error.
    CUDA_CHECK(cudaGetLastError());
  }

  // Dispatch tokens or token gradient to expert MLPs.
  template<// Token data type. Only support uint16_t(represent for BF16) and uint8_t(represent for FP8) for now.
           typename TOKEN_DATA_TYPE,
           // Number of token entry in the shared memory.
           int NUM_OF_STAGES,
           // The size of token chunk used in dispatch kernel.
           int NUM_OF_TOKENS_PER_CHUNK,
           // Grid size for dispatch kernel(1:1 block:SM mapping).
           int NUM_OF_BLOCKS,
           // Whether the dispatch kernel is used in forward process.
           bool FORWARD_DISPATCH,
           // Whether the dispatch kernel need device-side sync before exit. 
           bool DEVICE_SIDE_SYNC>
  static void dispatch(dispatch_kernel_param_t<TOKEN_DATA_TYPE> param, cudaStream_t stream)
  {
    // The warp groups data type for dispatch kernel, must match the warp groups layout required by the dispatch kernel.
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    using INTER_NODE_GROUP = warp_group<1, 0>;
    using INTRA_NODE_G2S_GROUP = warp_group<1, 1>;
    using INTRA_NODE_S2G_GROUP = warp_group<1, 2>;
#else
    using INTER_NODE_GROUP = warp_group<0, 0>;
    using INTRA_NODE_G2S_GROUP = warp_group<1, 0>;
    using INTRA_NODE_S2G_GROUP = warp_group<1, 1>;
#endif
    // The shared memory needed by the dispatch kernel.
    using dispatch_kernel_smem_t = dispatch_kernel_dynamic_shared_memory_buffer_t<TOKEN_DATA_TYPE, NUM_OF_STAGES, HIDDEN_DIM, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, FORWARD_DISPATCH>;
    // The dispatch kernel to be launched.
    const auto dispatch_kernel_ptr = dispatch_kernel<TOKEN_DATA_TYPE, INTER_NODE_GROUP, INTRA_NODE_G2S_GROUP, INTRA_NODE_S2G_GROUP, NUM_OF_STAGES, NUM_OF_TOKENS_PER_CHUNK, HIDDEN_DIM,
                                                     NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>;

    // Configure dynamic shared memory for the dispatch kernel.
    constexpr int SMEM_SIZE = sizeof(dispatch_kernel_smem_t);
    // The dispatch kernel only need to be configured once.
    static bool config_completed = false;
    if(!config_completed){
      // If the dynamic shared memory requested is too large, we may need to modify the carveout.
      //CUDA_CHECK(cudaFuncSetAttribute(dispatch_kernel_ptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      CUDA_CHECK(cudaFuncSetAttribute(dispatch_kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
      config_completed = true;
    }

    // Launch update_expected_value_kernel to update expected flag value.
    update_expected_value_kernel<NUM_OF_NODES, NUM_OF_RANKS_PER_NODE, DEVICE_SIDE_SYNC>
    <<<1, 1, 0, stream>>>(param.expected_rdma_flag_value, param.expected_intra_node_flag_value);

    // Launch dispatch kernel.
    constexpr int BLOCK_DIM = INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size();
    dispatch_kernel_ptr<<<NUM_OF_BLOCKS, BLOCK_DIM, SMEM_SIZE, stream>>>(param);

    // Launch device sync kernel if needed.
    if constexpr(DEVICE_SIDE_SYNC){
      device_sync_kernel<<<1, 1, 0, stream>>>(param.intra_node_write_completion_flags, param.expected_intra_node_flag_value);
    }

    // Check if there is any CUDA error.
    CUDA_CHECK(cudaGetLastError());
  }

  // Combine tokens or token gradient from expert MLPs.
  template<// Number of token entry in the shared memory for G2S TMA.
           int NUM_OF_STAGES_G2S,
           // Number of token entry in the shared memory for S2G TMA.
           int NUM_OF_STAGES_S2G,
           // The size of token chunk used in combine kernel.
           int NUM_OF_TOKENS_PER_CHUNK,
           // Number of token per group in the inter-node reduction/G2S warp group.
           int NUM_OF_TOKENS_PER_GROUP,
           // Grid size for combine kernel(1:1 block:SM mapping).
           int NUM_OF_BLOCKS,
           // Number of fully in-flight S2G in intra-node reduction warp group.
           int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
           // Whether the combine kernel is used in backward process.
           bool BACKWARD_COMBINE,
           // Whether the combine kernel need device-side sync before launch.
           bool DEVICE_SIDE_SYNC>
  static void combine(combine_kernel_param_t param, cudaStream_t stream)
  {
    // The warp groups data type for combine kernel, must match the warp groups layout required by the combine kernel.
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    using INTRA_NODE_RED_GROUP = warp_group<4, 0>;
    using INTER_NODE_RED_GROUP = warp_group<4, 4>;
    using INTRA_NODE_G2S_GROUP = warp_group<1, 8>;
    using INTER_NODE_G2S_GROUP = warp_group<1, 9>;
    using INTER_NODE_RDMA_GROUP = warp_group<1, 10>;
#else
    using INTRA_NODE_RED_GROUP = warp_group<0, 0>;
    using INTER_NODE_RED_GROUP = warp_group<4, 0>;
    using INTRA_NODE_G2S_GROUP = warp_group<0, 4>;
    using INTER_NODE_G2S_GROUP = warp_group<1, 4>;
    using INTER_NODE_RDMA_GROUP = warp_group<0, 5>;
#endif

    // The shared memory needed by the combine kernel.
    using combine_kernel_smem_t = combine_kernel_dynamic_shared_memory_buffer_t<NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_TOKENS_PER_CHUNK,
                                                                                NUM_OF_EXPERTS_PER_RANK, NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, BACKWARD_COMBINE>;
    // The combine kernel to be launched.
    const auto combine_kernel_ptr = combine_kernel<INTRA_NODE_RED_GROUP, INTER_NODE_RED_GROUP, INTRA_NODE_G2S_GROUP, INTER_NODE_G2S_GROUP, INTER_NODE_RDMA_GROUP, NUM_OF_STAGES_G2S,
                                                   NUM_OF_STAGES_S2G, NUM_OF_TOKENS_PER_GROUP, NUM_OF_TOKENS_PER_CHUNK, HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_EXPERTS_PER_RANK,
                                                   NUM_OF_RANKS_PER_NODE, NUM_OF_NODES, NUM_OF_BLOCKS, NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, BACKWARD_COMBINE>;

    // Configure dynamic shared memory for the combine kernel.
    constexpr int SMEM_SIZE = sizeof(combine_kernel_smem_t);
    // The combine kernel only need to be configured once.
    static bool config_completed = false;
    if(!config_completed){
      // If the dynamic shared memory requested is too large, we may need to modify the carveout.
      //CUDA_CHECK(cudaFuncSetAttribute(combine_kernel_ptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      CUDA_CHECK(cudaFuncSetAttribute(combine_kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
      config_completed = true;
    }

    // Launch update_expected_value_kernel to update expected flag value.
    update_expected_value_kernel<NUM_OF_NODES, NUM_OF_RANKS_PER_NODE, DEVICE_SIDE_SYNC>
    <<<1, 1, 0, stream>>>(param.expected_rdma_flag_value, param.expected_intra_node_flag_value);

    // Launch device sync kernel if needed.
    if constexpr(DEVICE_SIDE_SYNC){
      device_sync_kernel<<<1, 1, 0, stream>>>(param.intra_node_write_completion_flags, param.expected_intra_node_flag_value);
    }

    // Launch combine kernel.
    constexpr int BLOCK_DIM = INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size();
    combine_kernel_ptr<<<NUM_OF_BLOCKS, BLOCK_DIM, SMEM_SIZE, stream>>>(param);

    // Check if there is any CUDA error.
    CUDA_CHECK(cudaGetLastError());
  }



  /*private:
  // Rank within the current node/host.
  int local_rank_; 
  // Rank for the current node/host.
  int node_rank_;

  // MPI Communicator for out-of-bond communication.
  // This is used to gather routing map from all other ranks, so the communicator should contains all ranks.
  MPI_Comm comm_;

  // The global routing map which collected from all other ranks, remains the same in a trainning iteration(FW+BP).
  // dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES, NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES].
  bool* global_routing_map_;
  // The temp buffer needed by the preprocessing kernel.
  // dtype: tmp_state_t, shape: [NUM_OF_BLOCKS for preprocessing kernel, NUM_OF_RANKS_PER_NODE].
  tmp_state_t* preprocessing_tmp_;
  // The routing info needed by NVL warps(i.e. intra-node communication warps) during both dispatch and combine operation.
  // Remains the same in a trainning iteration(FW+BP).
  // dtype: int32_t, shape: [NUM_OF_TOKENS_PER_RANK * NUM_OF_NODES, NUM_OF_RANKS_PER_NODE].
  int32_t* sparse_to_dense_map_;
  // The routing info mainly needed by RDMA warps during the combine operation.
  // Remains the same in a trainning iteration(FW+BP).
  // dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK padded to 16 * NUM_OF_NODES].
  bool* rdma_to_attn_map_;
  // The routing info mainly needed by RDMA warps during the dispatch operation.
  // Remains the same in a trainning iteration(FW+BP).
  // dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK, NUM_OF_NODES - 1].
  bool* attn_to_rdma_map_;
  // The total size of expert input/output buffer on this rank(in number of tokens), according to the global routing map.
  // If there are multiple expert on this rank, each token will only appear once.
  // Remains the same in a trainning iteration(FW+BP).
  int32_t* num_of_tokens_for_experts_;
  // The per-expert routing info for all tokens within the expert input/output buffer of this rank.
  // It is used by later layer to routing the tokens to different experts on this rank.
  // Remains the same in a trainning iteration(FW+BP).
  // dtype: bool, shape: [at least num_of_tokens_for_experts_, NUM_OF_EXPERTS_PER_RANK].
  bool* local_expert_routing_map_;*/
};
} // namespace hybrid_ep

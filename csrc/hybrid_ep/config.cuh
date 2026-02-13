// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved

#pragma once
#include <tuple>
#include <cassert>
#include <stdexcept>
#include "utils.cuh"

// Now we support up to 72(GB200) ranks per node.
// This will be used to initialize the template param_t for communication kernel.
#define MAX_NUM_OF_RANKS_PER_NODE 72

// Config used for buffer allocation.
struct BufferConfig {
  int hidden_dim;
  int max_num_of_tokens_per_rank;
  int num_of_experts_per_rank;
  int num_of_ranks_per_node;
  int num_of_nodes;
  APP_TOKEN_DATA_TYPE token_data_type;
  int num_of_blocks_preprocessing_api;
  int num_of_blocks_dispatch_api;
  int num_of_blocks_combine_api;
  int num_of_blocks_permute_api;
  int num_of_tokens_per_chunk_dispatch_api;
  int num_of_tokens_per_chunk_combine_api;
  /** Number of dispatch chunks (derived from max_num_of_tokens_per_rank and num_of_tokens_per_chunk_dispatch_api), used for buffer sizing; grow_to on this triggers reallocate when chunk size shrinks. */
  int num_of_dispatch_chunks;

  /*
   *  Validation check
   */
   bool is_valid(){
    bool valid = true;
    if (token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
      valid &= (hidden_dim % 512 == 0); // Make TMA work in scaling factor.
    } else {
      valid &= (hidden_dim % 16 == 0); // Make TMA work.
    }
    valid &= ((num_of_experts_per_rank * num_of_ranks_per_node) % 4 == 0);
    valid &= (num_of_ranks_per_node % 2 == 0);
    // TMA requires (num_of_tokens_per_chunk * num_of_ranks_per_node * 4) % 16 == 0
    valid &= ((num_of_tokens_per_chunk_dispatch_api * num_of_ranks_per_node) % 4 == 0);
    if(!valid){
      fprintf(stderr, "[Error] Invalid BufferConfig: hidden_dim=%d, num_of_experts_per_rank=%d, num_of_ranks_per_node=%d, num_of_tokens_per_chunk_dispatch_api=%d\n", 
              hidden_dim, num_of_experts_per_rank, num_of_ranks_per_node, num_of_tokens_per_chunk_dispatch_api);
      fflush(stderr);
    }
    return valid;
  }
};

// Config used for hybrid-ep kernel.
struct HybridEpConfigInstance {
  /*
   *  Hybrid-ep Config
   */
  int hidden_dim;
  int max_num_of_tokens_per_rank;
  int num_of_experts_per_rank;
  int num_of_ranks_per_node;
  int num_of_nodes;
  int pad_multiple;

  /*
   *  Metadata-preprocessing API Config
   */
  int num_of_tokens_per_chunk_preprocessing_api;
  int num_of_threads_per_block_preprocessing_api;
  int num_of_blocks_preprocessing_api;

  // In standalone permute kernel. it is the number of CUDA blocks running permute kernel.
  // In fused permute-dispatch kernel. it is the number of CUDA blocks for permute part in the fused kernel.
  int num_of_blocks_permute_api;

  /*
   *  Dispatch API Config
   */
  APP_TOKEN_DATA_TYPE token_data_type;
  int num_of_stages_dispatch_api;
  int num_of_stages_permute_block_dispatch_api;
  int num_of_in_flight_s2g_dispatch_api;
  int num_of_in_flight_s2g_permute_block_dispatch_api;
  int num_of_additional_in_flight_s2g_dispatch_api;
  int num_of_tokens_per_chunk_dispatch_api;
  int num_of_blocks_dispatch_api;
  bool forward_dispatch_api;
  bool device_side_sync_dispatch_api = true;

  /*
   *  Combine API Config
   */
  int num_of_stages_g2s_combine_api;
  int num_of_stages_s2g_combine_api;
  int num_of_tokens_per_chunk_combine_api;
  int num_of_tokens_per_group_combine_api;
  int num_of_blocks_combine_api;
  int num_of_additional_in_flight_s2g_combine_api;
  bool backward_combine_api;
  bool device_side_sync_combine_api = true;

  /*
   *  Validation check
   */
  bool is_valid(bool fuse_permute_dispatch = false){
    bool valid = true;
    if (token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
      valid &= (hidden_dim % 512 == 0); // Make TMA work in scaling factor.
    } else {
      valid &= (hidden_dim % 16 == 0); // Make TMA work.
    }
    valid &= ((num_of_experts_per_rank * num_of_ranks_per_node) % 4 == 0);
    valid &= (num_of_ranks_per_node % 2 == 0);
    // TMA requires (num_of_tokens_per_chunk * num_of_ranks_per_node * 4) % 16 == 0
    valid &= ((num_of_tokens_per_chunk_dispatch_api * num_of_ranks_per_node) % 4 == 0);
    // In fuse mode, all chunk sizes must be the same
    if (fuse_permute_dispatch) {
      bool chunk_match = (num_of_tokens_per_chunk_dispatch_api == num_of_tokens_per_chunk_combine_api)
                      && (num_of_tokens_per_chunk_dispatch_api == num_of_tokens_per_chunk_preprocessing_api);
      if (!chunk_match) {
        fprintf(stderr, "[Error] Fuse mode requires identical chunk sizes: dispatch=%d, combine=%d, preprocessing=%d\n",
                num_of_tokens_per_chunk_dispatch_api, num_of_tokens_per_chunk_combine_api, num_of_tokens_per_chunk_preprocessing_api);
        fflush(stderr);
      }
      valid &= chunk_match;
    }
    if(!valid){
      fprintf(stderr, "[Error] Invalid HybridEpConfigInstance: hidden_dim=%d, num_of_experts_per_rank=%d, num_of_ranks_per_node=%d, num_of_tokens_per_chunk_dispatch_api=%d\n", 
              hidden_dim, num_of_experts_per_rank, num_of_ranks_per_node, num_of_tokens_per_chunk_dispatch_api);
      fflush(stderr);
    }
    return valid;
  }
};

static int get_env_int(const char* name, int default_value) {
    const char* val = getenv(name);
    return val ? atoi(val) : default_value;
}

class Configurer {
public:
    BufferConfig buffer_config;

    Configurer(
        int hidden_dim,
        int max_num_of_tokens_per_rank,
        int num_local_experts,
        int num_of_ranks_per_node,
        int num_of_nodes,
        bool use_fp8 = false,
        int num_sms_dispatch_api = -1,
        int num_sms_combine_api = -1,
        int num_sms_preprocessing_api = -1,
        int num_blocks_permute_api = -1
    ) {
        // Auto-detect SM count
        cudaDeviceProp props;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        int sm_count = props.multiProcessorCount;

        // Apply SM defaults
        if (num_sms_preprocessing_api < 0) num_sms_preprocessing_api = 108;
        if (num_blocks_permute_api < 0)    num_blocks_permute_api = sm_count * 16;
        if (num_sms_dispatch_api < 0)      num_sms_dispatch_api = (num_of_nodes == 1) ? 16 : 8;
        if (num_sms_combine_api < 0)       num_sms_combine_api = (num_of_nodes == 1) ? 32 : 8;

        assert(sm_count >= num_sms_dispatch_api
            && sm_count >= num_sms_combine_api);

        // Fill BufferConfig
        buffer_config.hidden_dim = hidden_dim;
        buffer_config.max_num_of_tokens_per_rank = std::max(max_num_of_tokens_per_rank, 512);
        buffer_config.num_of_experts_per_rank = num_local_experts;
        buffer_config.num_of_ranks_per_node = num_of_ranks_per_node;
        buffer_config.num_of_nodes = num_of_nodes;
        buffer_config.num_of_blocks_dispatch_api = num_sms_dispatch_api;
        buffer_config.num_of_blocks_combine_api = num_sms_combine_api;
        buffer_config.num_of_blocks_preprocessing_api = num_sms_preprocessing_api;
        buffer_config.num_of_blocks_permute_api = num_blocks_permute_api;
        buffer_config.token_data_type = use_fp8 ? APP_TOKEN_DATA_TYPE::UINT8 : APP_TOKEN_DATA_TYPE::UINT16;
        buffer_config.num_of_tokens_per_chunk_dispatch_api = get_env_int("NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API", 32);
        buffer_config.num_of_tokens_per_chunk_combine_api = get_env_int("NUM_OF_TOKENS_PER_CHUNK_COMBINE_API", 32);
        buffer_config.num_of_dispatch_chunks = (buffer_config.max_num_of_tokens_per_rank - 1)
            / buffer_config.num_of_tokens_per_chunk_dispatch_api + 1;

        if (!buffer_config.is_valid()) {
            fprintf(stderr, "[Error] Configurer: invalid buffer config. hidden_dim=%d, max_num_of_tokens_per_rank=%d, "
                    "num_of_experts_per_rank=%d, num_of_ranks_per_node=%d, num_of_nodes=%d\n",
                    hidden_dim, max_num_of_tokens_per_rank, num_local_experts, num_of_ranks_per_node, num_of_nodes);
            fflush(stderr);
            throw std::runtime_error("The buffer config is not valid.");
        }
    }

    HybridEpConfigInstance get_default_config(bool fuse_permute_dispatch = false) {
        HybridEpConfigInstance config;
        // Defaults from buffer_config (can be overridden per-call)
        config.hidden_dim = buffer_config.hidden_dim;
        config.max_num_of_tokens_per_rank = buffer_config.max_num_of_tokens_per_rank;
        config.num_of_experts_per_rank = buffer_config.num_of_experts_per_rank;
        config.num_of_ranks_per_node = buffer_config.num_of_ranks_per_node;
        config.num_of_nodes = buffer_config.num_of_nodes;

        // Semi-static from buffer_config
        config.num_of_blocks_preprocessing_api = buffer_config.num_of_blocks_preprocessing_api;
        config.num_of_blocks_dispatch_api = buffer_config.num_of_blocks_dispatch_api;
        config.num_of_blocks_combine_api = buffer_config.num_of_blocks_combine_api;
        config.num_of_blocks_permute_api = buffer_config.num_of_blocks_permute_api;
        config.token_data_type = buffer_config.token_data_type;

        // Env-var defaults (runtime chunk sizes use 64, different from buffer's 32)
        config.num_of_threads_per_block_preprocessing_api = get_env_int("NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API", 512);
        int default_chunk_size = fuse_permute_dispatch ? 64 : 128;
        config.num_of_tokens_per_chunk_preprocessing_api  = get_env_int("NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API", default_chunk_size);
        config.device_side_sync_dispatch_api = true;
        config.num_of_stages_dispatch_api = get_env_int("NUM_OF_STAGES_DISPATCH_API", 10);
        config.num_of_stages_permute_block_dispatch_api = get_env_int("NUM_OF_STAGES_PERMUTE_BLOCK_DISPATCH_API", 10);
        config.num_of_in_flight_s2g_dispatch_api = get_env_int("NUM_OF_IN_FLIGHT_S2G_DISPATCH_API", 8);
        config.num_of_in_flight_s2g_permute_block_dispatch_api = get_env_int("NUM_OF_IN_FLIGHT_S2G_PERMUTE_BLOCK_DISPATCH_API", 8);
        config.num_of_additional_in_flight_s2g_dispatch_api = get_env_int("NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_DISPATCH_API", 6);
        config.num_of_tokens_per_chunk_dispatch_api = get_env_int("NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API", default_chunk_size);
        config.device_side_sync_combine_api = true;
        config.num_of_stages_g2s_combine_api = get_env_int("NUM_OF_STAGES_G2S_COMBINE_API",
            buffer_config.num_of_nodes > 1 ? 5 : 10);
        config.num_of_stages_s2g_combine_api = get_env_int("NUM_OF_STAGES_S2G_COMBINE_API", 2);
        config.num_of_tokens_per_chunk_combine_api = get_env_int("NUM_OF_TOKENS_PER_CHUNK_COMBINE_API", default_chunk_size);
        config.num_of_tokens_per_group_combine_api = get_env_int("NUM_OF_TOKENS_PER_GROUP_COMBINE_API", 4);
        config.num_of_additional_in_flight_s2g_combine_api = get_env_int("NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_COMBINE_API", 2);
        config.pad_multiple = 1;

        // If we use the fused permute-dispatch kernel, the number of blocks
        // for the permute part is the same as the number of blocks for the dispatch part.
        if (fuse_permute_dispatch) {
            config.num_of_blocks_permute_api = 108;
        }

        return config;
    }
};
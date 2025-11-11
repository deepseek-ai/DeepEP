// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved

#pragma once
#include <tuple>
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
  int num_of_tokens_per_chunk_dispatch_api;
  int num_of_tokens_per_chunk_combine_api;

  /*
   *  Validation check
   */
   bool is_valid(){
    bool valid = true;
    valid &= (hidden_dim % 512 == 0);
    valid &= ((num_of_experts_per_rank * num_of_ranks_per_node) % 4 == 0);
    valid &= (num_of_ranks_per_node % 2 == 0);
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

  /*
   *  Metadata-preprocessing API Config
   */
  int num_of_threads_per_block_preprocessing_api;
  int num_of_blocks_preprocessing_api;

  /*
   *  Dispatch API Config
   */
  APP_TOKEN_DATA_TYPE token_data_type;
  int num_of_stages_dispatch_api;
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
  bool is_valid(){
    bool valid = true;
    valid &= (hidden_dim % 512 == 0);
    valid &= ((num_of_experts_per_rank * num_of_ranks_per_node) % 4 == 0);
    valid &= (num_of_ranks_per_node % 2 == 0);
    return valid;
  }
};

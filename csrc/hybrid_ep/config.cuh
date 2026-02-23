// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved

#pragma once
#include <tuple>
#include <vector>
#include "utils.cuh"
#include <ATen/core/ivalue.h>

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

  /** Convert all attributes to a single IValue holding a tuple of IValues (ints/string). */
  c10::IValue to_ivalue_tuple() const {
    std::vector<c10::IValue> elements;
    elements.reserve(12);
    elements.push_back(c10::IValue(static_cast<int64_t>(hidden_dim)));
    elements.push_back(c10::IValue(static_cast<int64_t>(max_num_of_tokens_per_rank)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_experts_per_rank)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_ranks_per_node)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_nodes)));
    elements.push_back(c10::IValue(type_to_string(token_data_type)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_blocks_preprocessing_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_blocks_dispatch_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_blocks_combine_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_blocks_permute_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_tokens_per_chunk_dispatch_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_tokens_per_chunk_combine_api)));
    return c10::IValue(c10::ivalue::Tuple::create(std::move(elements)));
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
  int num_of_blocks_permute_api;

  /*
   *  Dispatch API Config
   */
  APP_TOKEN_DATA_TYPE token_data_type;
  int num_of_stages_dispatch_api;
  int num_of_in_flight_s2g_dispatch_api;
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
      fprintf(stderr, "[Error] Invalid HybridEpConfigInstance: hidden_dim=%d, num_of_experts_per_rank=%d, num_of_ranks_per_node=%d, num_of_tokens_per_chunk_dispatch_api=%d\n", 
              hidden_dim, num_of_experts_per_rank, num_of_ranks_per_node, num_of_tokens_per_chunk_dispatch_api);
      fflush(stderr);
    }
    return valid;
  }

  /** Convert all attributes to a single IValue holding a tuple of IValues (ints/bools). */
  c10::IValue to_ivalue_tuple() const {
    std::vector<c10::IValue> elements;
    elements.reserve(23);
    // Hybrid-ep Config
    elements.push_back(c10::IValue(static_cast<int64_t>(hidden_dim)));
    elements.push_back(c10::IValue(static_cast<int64_t>(max_num_of_tokens_per_rank)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_experts_per_rank)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_ranks_per_node)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_nodes)));
    // Metadata-preprocessing API Config
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_threads_per_block_preprocessing_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_blocks_preprocessing_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_blocks_permute_api)));
    // Dispatch API Config
    elements.push_back(c10::IValue(type_to_string(token_data_type)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_stages_dispatch_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_in_flight_s2g_dispatch_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_tokens_per_chunk_dispatch_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_blocks_dispatch_api)));
    elements.push_back(c10::IValue(forward_dispatch_api));
    elements.push_back(c10::IValue(device_side_sync_dispatch_api));
    // Combine API Config
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_stages_g2s_combine_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_stages_s2g_combine_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_tokens_per_chunk_combine_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_tokens_per_group_combine_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_blocks_combine_api)));
    elements.push_back(c10::IValue(static_cast<int64_t>(num_of_additional_in_flight_s2g_combine_api)));
    elements.push_back(c10::IValue(backward_combine_api));
    elements.push_back(c10::IValue(device_side_sync_combine_api));
    return c10::IValue(c10::ivalue::Tuple::create(std::move(elements)));
  }
};

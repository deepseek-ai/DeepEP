// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once
#include <tuple>

enum class TOKEN_DATA_TYPE { UINT16, UINT8 };

constexpr int HIDDEN_DIM = 7168; // HIDDEN_DIM = 512xN, N in [0,1,2,....]
constexpr int MAX_NUM_OF_TOKENS_PER_RANK = 4096; // NUM_OF_TOKENS_PER_RANK = NUM_OF_TOKENS_PER_CHUNK_DISPATCH_APIxN, N in [0,1,2,....]
constexpr int NUM_OF_EXPERTS_PER_RANK = 8; // (NUM_OF_EXPERTS_PER_RANKxNUM_OF_RANKS_PER_NODE) = 4xN

constexpr int NUM_OF_NODES = 1;  // Note: this is the number of nvlink domains
constexpr int NUM_OF_RANKS_PER_NODE = 32; // Note: this is the number of ranks in each NVLink domain

// Multi-node NVLink Staff
constexpr bool USE_MNNVLINK = true;

// Metadata-preprocessing API Config
constexpr int NUM_THREADS_PER_BLOCK_PREPROCESSING_API = 128;
constexpr int NUM_OF_BLOCKS_PREPROCESSING_API = 32;      // how much SM will be used for preprocessing

// Dispatch API Config
constexpr int NUM_OF_STAGES_DISPATCH_API = 12;            // fix to 12
constexpr int NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API = 128; // fix to 128
constexpr int NUM_OF_BLOCKS_DISPATCH_API = 32;            // how much SM will be used for dispatch
constexpr bool FORWARD_DISPATCH_API = true;
constexpr bool DEVICE_SIDE_SYNC_DISPATCH_API = true;



// Combine API Config
// Combine API specific configuration.
constexpr int NUM_OF_STAGES_G2S_COMBINE_API = 12;
constexpr int NUM_OF_STAGES_S2G_COMBINE_API = 2;
constexpr int NUM_OF_TOKENS_PER_CHUNK_COMBINE_API = 128;
constexpr int NUM_OF_TOKENS_PER_GROUP_COMBINE_API = 4;
constexpr int NUM_OF_BLOCKS_COMBINE_API = 32;             // how much SM will be used for combine
constexpr int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_COMBINE_API = 2;
constexpr bool BACKWARD_COMBINE_API = false;
constexpr bool DEVICE_SIDE_SYNC_COMBINE_API = true;

struct HybridEpConfigInstance {
  /*
   *  Hybrid-ep Config
   */
  int hidden_dim;
  int num_of_tokens_per_rank;
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
  TOKEN_DATA_TYPE token_data_type;
  int num_of_stages_dispatch_api;
  int num_of_tokens_per_chunk_dispatch_api;
  int num_of_blocks_dispatch_api;
  bool forward_dispatch_api;
  bool device_side_sync_dispatch_api;

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
  bool device_side_sync_combine_api;
};

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved
#include "internode_nixl.cuh"


void NIXLCoordinator::init(
    pybind11::object process_group,
    int node_rank,
    int local_rank,
    BufferConfig config
  ) {
  this->process_group = process_group;
  this->node_rank = node_rank;
  this->local_rank = local_rank;
  this->buffer_config = config;
  assert(buffer_config.num_of_nodes > 1);
}

bool NIXLCoordinator::grow_buffer_config(const HybridEpConfigInstance& config, BufferConfig& buf_config) {
    bool changed = false;
    changed |= grow_to(buf_config.max_num_of_tokens_per_rank, config.max_num_of_tokens_per_rank);
    changed |= grow_to(buf_config.hidden_dim, config.hidden_dim);
    changed |= grow_to(buf_config.num_of_experts_per_rank, config.num_of_experts_per_rank);
    changed |= grow_to(buf_config.num_of_ranks_per_node, config.num_of_ranks_per_node);
    changed |= grow_to(buf_config.num_of_nodes, config.num_of_nodes);
    changed |= grow_to(buf_config.num_of_blocks_dispatch_api, config.num_of_blocks_dispatch_api);
    changed |= grow_to(buf_config.num_of_blocks_combine_api, config.num_of_blocks_combine_api);
    if (buf_config.num_of_tokens_per_chunk_dispatch_api != config.num_of_tokens_per_chunk_dispatch_api) {
      changed = true;
      buf_config.num_of_tokens_per_chunk_dispatch_api = config.num_of_tokens_per_chunk_dispatch_api;
    }
    if (buf_config.num_of_tokens_per_chunk_combine_api != config.num_of_tokens_per_chunk_combine_api) {
      changed = true;
      buf_config.num_of_tokens_per_chunk_combine_api = config.num_of_tokens_per_chunk_combine_api;
    }
    return changed;
}

void NIXLCoordinator::update_config(BufferConfig config) {
    this->buffer_config = config;
}

void NIXLCoordinator::allocate_buffers() {
    return;
}

void NIXLCoordinator::destroy() {
    return;
}

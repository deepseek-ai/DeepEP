// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once

#include <cstdint>
#include <vector>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "utils.cuh"
#include "config.cuh"
#include "allocator/allocator.cuh"
#include "backend/hybrid_ep_backend.cuh"

struct IntraNodeDispatchBuffers {
    APP_TOKEN_DATA_TYPE data_type;
    // Output buffers to experts
    void *        expert_output_token = nullptr;
    void **       expert_output_token_all_ranks = nullptr;
    float *       expert_output_prob = nullptr;
    float **      expert_output_prob_all_ranks = nullptr;
    float *       expert_output_scaling_factor = nullptr;
    float **      expert_output_scaling_factor_all_ranks = nullptr;
    // Misc flags
    uint32_t *    intra_node_write_completion_flags = nullptr;
    uint32_t *    expected_intra_node_flag_value = nullptr;
};

struct IntraNodeCombineBuffers {
    // Input buffers from experts
    uint16_t *    expert_input_token = nullptr;
    uint16_t **   expert_input_token_all_ranks = nullptr;
    float *       expert_input_prob = nullptr;
    float **      expert_input_prob_all_ranks = nullptr;
    // Misc flags
    uint32_t *    intra_node_write_completion_flags = nullptr;
    uint32_t *    expected_intra_node_flag_value = nullptr;
};
  

class NVLCoordinator {
public:
    NVLCoordinator() = default;
    ~NVLCoordinator();

    void init(pybind11::object process_group, int node_rank, int local_rank, int group_size, bool use_shared_buffer, BufferConfig config, ExtendedMemoryAllocator *remote_allocator);
    void update_config(BufferConfig config);
    void destroy();
    void allocate_preprocessing_buffers();
    void allocate_dispatch_buffers();
    void allocate_combine_buffers();
    void exchange_remote_nvl_info();

    IntraNodeDispatchBuffers dispatch_buffers;
    IntraNodeCombineBuffers combine_buffers;
    // Buffer for metadata preprocessing
    hybrid_ep::tmp_state_t *preprocessing_tmp = nullptr;
    // Maximum number of tokens for experts (worst case: all tokens to one expert)
    int64_t max_num_of_tokens = -1;
    // On intra-node communication, dispatch/combine can share same buffers.
    bool use_shared_buffer = false;

private:
    ExtendedMemoryAllocator *remote_allocator;
    pybind11::object process_group;
    BufferConfig buffer_config;
    // Meta data of communication group.
    int local_rank = -1;
    int node_rank = -1;
    int group_size = -1;

    // Remote memory handles
    torch::Tensor dispatch_memory_handles;
    torch::Tensor combine_memory_handles;

    void open_handles_from_other_ranks(std::vector<torch::Tensor> dispatch_handles,
                                     std::vector<torch::Tensor> combine_handles);
};
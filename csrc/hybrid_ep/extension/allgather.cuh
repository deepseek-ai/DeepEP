// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#pragma once

#include "utils.cuh"
#include "config.cuh"
#include "allocator/allocator.cuh"

class CustomAllgather {
public:
    CustomAllgather() = default;
    ~CustomAllgather();
    void init(pybind11::object process_group, int rank_idx, BufferConfig buffer_config, ExtendedMemoryAllocator* allocator);
    void update(BufferConfig buffer_config);
    void allocate_ag_buffer();
    void open_ag_handles();
    void destroy();
    void launch(torch::Tensor src, int ag_sms = 32, cudaStream_t stream = nullptr);
    void * get_output_buffer();
private:
    // Required pre-allocated buffers
    void* dst_buffer = nullptr;
    void** dst_buffers_all_ranks = nullptr;
    void** dst_buffers_all_ranks_gpu = nullptr;
    int64_t* iter_id_ptr = nullptr;
    unsigned long long* flag_nvl_ptr = nullptr;
    unsigned long long* flag_sm_ptr = nullptr;
    torch::Tensor ag_handles;

    // Meta-data
    int rank_idx;
    int num_of_ranks_per_node;
    int num_of_experts_per_rank;
    int num_of_tokens_per_rank;
    int num_of_nodes;
    ExtendedMemoryAllocator *allocator;
    pybind11::object process_group;
};
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once
#include "config.cuh"
#include "backend/hybrid_ep_backend.cuh"
#include "allocator/allocator.cuh"
#include "utils.cuh"
#include "jit/compiler.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

class HybridEpBuffer {
public:
  HybridEpBuffer(HybridEpConfigInstance config, int local_rank, int node_rank, int group_size, int num_of_ranks_per_node, int nvlink_domain_size);
  ~HybridEpBuffer();

  // Exchange IPC addresses using C++ distributed communication
  void exchange_ipc_address(pybind11::object process_group);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor>
  metadata_preprocessing(torch::Tensor routing_map, int64_t num_of_tokens_per_rank);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  dispatch(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
           c10::optional<torch::Tensor> scaling_factor,
           torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
           torch::Tensor attn_to_rdma_map, int64_t num_of_tokens_for_experts,
           int64_t num_of_tokens_per_rank,
           bool with_probs);

  std::tuple<torch::Tensor, torch::Tensor>
  combine(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
          torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
          torch::Tensor attn_to_rdma_map, int64_t num_of_tokens_per_rank,
          bool with_probs);

private:
  ExtendedMemoryAllocator remote_allocator;
  HybridEpConfigInstance config;
  KernelCache kernel_cache;

  void allocate_buffer();
  void allocate_buffer_for_preprocessing();
  void allocate_buffer_for_dispatch();
  void allocate_buffer_for_combine();
  void open_handles_from_other_ranks(std::vector<torch::Tensor> dispatch_handles,
                                     std::vector<torch::Tensor> combine_handles);

  // Meta data of communication group.
  int rank;
  int local_rank;
  int node_rank;
  int num_of_ranks_per_node;
  int group_size;
  int nvlink_domain_size;
  // Maximum number of tokens for experts.
  int64_t max_num_of_tokens_for_experts; 
  bool use_fp8_dispatch;
  // Only valid on intra-node communication. In this case, the dispatch/combine can share same buffers.
  bool use_shared_buffer;

  hybrid_ep::tmp_state_t *preprocessing_tmp;

  struct DispatchBuffers {
    TOKEN_DATA_TYPE data_type;
    // Output buffers to experts
    void *expert_output_token;
    void **expert_output_token_all_ranks;
    float *expert_output_prob;
    float **expert_output_prob_all_ranks;
    float *expert_output_scaling_factor;
    float **expert_output_scaling_factor_all_ranks;
    // Local temp buffer for dispatch kernel.
    void *rdma_inter_node_group_token;
    float *rdma_inter_node_group_prob;
    float *rdma_inter_node_group_scaling_factor;
    uint64_t *rdma_inter_node_group_flags;
    // Misc flags
    uint32_t *intra_node_write_completion_flags;
    uint64_t *expected_rdma_flag_value;
    uint32_t *expected_intra_node_flag_value;
  } dispatch_buffers;

  torch::Tensor
      dispatch_memory_handles; 
      
  struct CombineBuffers {
    // Input buffers from experts
    uint16_t *expert_input_token;
    uint16_t **expert_input_token_all_ranks;
    float *expert_input_prob;
    float **expert_input_prob_all_ranks;
    // Local temp buffer for combine kernel.
    uint16_t *rdma_intra_node_red_token; 
    float *rdma_intra_node_red_prob;
    uint16_t *rdma_inter_node_group_token;
    float *rdma_inter_node_group_prob; 
    uint64_t *rdma_inter_node_group_flags; 
    // Misc flags
    uint32_t *intra_node_write_completion_flags;
    uint64_t *expected_rdma_flag_value;
    uint32_t *expected_intra_node_flag_value; 
  } combine_buffers;

  torch::Tensor
      combine_memory_handles;

};
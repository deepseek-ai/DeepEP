// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once
#include "config.cuh"
#include "backend/hybrid_ep_backend.cuh"
#include "allocator/allocator.cuh"
#include "utils.cuh"
#include "executor/executor.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <vector>
#include <algorithm>

class HybridEpBuffer {
public:
  HybridEpBuffer(HybridEpConfigInstance config, int local_rank, int node_rank, int group_size, int num_of_ranks_per_node, bool use_fp8_dispatch);
  ~HybridEpBuffer();

  // Exchange IPC addresses using C++ distributed communication
  void exchange_ipc_address(pybind11::object process_group);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor>
  metadata_preprocessing(torch::Tensor global_routing_map, int64_t num_of_tokens_per_rank);

  std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
  dispatch(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
           c10::optional<torch::Tensor> scaling_factor,
           torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
           torch::Tensor attn_to_rdma_map, c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
           int64_t num_dispatched_tokens,
           int64_t num_of_tokens_per_rank,
           bool with_probs);

  std::tuple<torch::Tensor, torch::Tensor>
  combine(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
          torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
          torch::Tensor attn_to_rdma_map, int64_t num_of_tokens_per_rank,
          bool with_probs);
  
  std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor>
  dispatch_with_permute(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
            c10::optional<torch::Tensor> scaling_factor,
            torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
            torch::Tensor attn_to_rdma_map, 
            c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
            c10::optional<torch::Tensor> local_expert_routing_map,
            c10::optional<torch::Tensor> row_id_map,
            int64_t num_dispatched_tokens,
            int64_t num_permuted_tokens,
            int64_t num_of_tokens_per_rank,
            int64_t pad_multiple,
            bool use_host_meta,
            bool with_probs);

  std::tuple<torch::Tensor, torch::Tensor>
  combine_with_unpermute(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
          torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
          torch::Tensor attn_to_rdma_map, c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
          c10::optional<torch::Tensor> row_id_map,
          int64_t num_dispatched_tokens,
          int64_t num_of_tokens_per_rank,
          int64_t pad_multiple,
          bool with_probs);       

private:
  ExtendedMemoryAllocator remote_allocator;
  HybridEpConfigInstance config;
  Executor executor;

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

  // Buffers for dispatch
  DispatchBuffers dispatch_buffers;
  torch::Tensor dispatch_memory_handles; 

  // Buffers for combine
  CombineBuffers combine_buffers;
  torch::Tensor combine_memory_handles;

};
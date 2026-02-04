// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once
#include "config.cuh"
#include "hybrid_ep_backend.cuh"
#include "allocator/allocator.cuh"
#include "utils.cuh"
#include "executor/executor.cuh"
#include "extension/allgather.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <string>

#include "buffer/intranode.cuh"
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
#include "buffer/internode.cuh"
#endif

class HybridEPBuffer {
public:
  HybridEPBuffer(pybind11::object process_group, BufferConfig config, int local_rank, int node_rank, int group_size, std::string base_path, bool load_cached_kernels, bool use_shared_buffer, bool enable_custom_allgather);
  bool update_buffer(HybridEpConfigInstance config); // True means the buffer is reallocated.

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor>
  metadata_preprocessing(HybridEpConfigInstance config, torch::Tensor global_routing_map, int64_t num_of_tokens_per_rank, bool non_blocking);

  std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
  dispatch(HybridEpConfigInstance config, 
           torch::Tensor hidden, c10::optional<torch::Tensor> probs,
           c10::optional<torch::Tensor> scaling_factor,
           torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
           torch::Tensor attn_to_rdma_map, c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
           c10::optional<int64_t> num_dispatched_tokens,
           int64_t num_of_tokens_per_rank,
           bool with_probs);

  std::tuple<torch::Tensor, torch::Tensor>
  combine(HybridEpConfigInstance config, torch::Tensor hidden, c10::optional<torch::Tensor> probs,
          torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
          torch::Tensor attn_to_rdma_map, int64_t num_of_tokens_per_rank,
          bool with_probs);
  
  std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor>
  dispatch_with_permute(
            HybridEpConfigInstance config, 
            torch::Tensor hidden, c10::optional<torch::Tensor> probs,
            c10::optional<torch::Tensor> scaling_factor,
            torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
            torch::Tensor attn_to_rdma_map, 
            c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
            c10::optional<torch::Tensor> local_expert_routing_map,
            c10::optional<torch::Tensor> row_id_map,
            c10::optional<int64_t> num_permuted_tokens,
            int64_t num_of_tokens_per_rank,
            c10::optional<int64_t> pad_multiple,
            bool non_blocking,
            bool with_probs);

  std::tuple<torch::Tensor, torch::Tensor>
  combine_with_unpermute(
          HybridEpConfigInstance config, 
          torch::Tensor hidden, c10::optional<torch::Tensor> probs,
          torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
          torch::Tensor attn_to_rdma_map, c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
          c10::optional<torch::Tensor> row_id_map,
          int64_t num_of_tokens_per_rank,
          c10::optional<int64_t> pad_multiple,
          bool with_probs);       

private:
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  RDMACoordinator rdma_coordinator;
#endif
  NVLCoordinator nvl_coordinator;
  ExtendedMemoryAllocator remote_allocator;
  BufferConfig buffer_config;
  Executor executor;
  pybind11::object process_group;
  CustomAllgather allgather_obj;

  void allocate_buffer();
  void release_buffer();
};
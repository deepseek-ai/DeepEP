// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#include "hybrid_ep.cuh"
#include <iostream>
#include <sstream>
#include <vector>
#include <functional>

std::string get_comm_id(pybind11::object process_group) {
  auto torch = pybind11::module_::import("torch");
  auto torch_distributed = torch.attr("distributed");

  // Get the global id of each rank in the process group
  std::vector<int> global_ranks;
  pybind11::object get_global_rank;
  if (pybind11::hasattr(torch_distributed, "get_global_rank")) {
    get_global_rank = torch_distributed.attr("get_global_rank");
  } 
  int group_size = process_group.attr("size")().cast<int>();
  global_ranks.reserve(group_size);
  for (int i = 0; i < group_size; ++i) {
    int g = get_global_rank(process_group, i).cast<int>();
    global_ranks.push_back(g);
  }

  // Concatenate the global ranks into a string
  std::ostringstream ranks_ss;
  for (size_t i = 0; i < global_ranks.size(); ++i) {
    if (i) ranks_ss << ",";
    ranks_ss << global_ranks[i];
  }

  // Hash the string to get the comm id
  auto hashed = std::hash<std::string>{}(ranks_ss.str());
  return std::to_string(hashed);
}

HybridEPBuffer::HybridEPBuffer(
  pybind11::object process_group, 
  BufferConfig config, 
  int local_rank, 
  int node_rank, 
  int group_size, 
  std::string base_path,
  bool load_cached_kernels,
  bool use_shared_buffer,
  bool enable_custom_allgather
) : process_group(process_group),
    buffer_config(config),
    executor(local_rank, node_rank, base_path, get_comm_id(process_group), load_cached_kernels, enable_custom_allgather)
{
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Initialize the allgather object
    allgather_obj.init(process_group, local_rank, buffer_config, &this->remote_allocator);
    // Initialize the nvl coordinator
    nvl_coordinator.init(process_group, node_rank, local_rank, group_size, use_shared_buffer, buffer_config, &this->remote_allocator);
    // Initialize the rdma coordinator
    if(group_size > buffer_config.num_of_ranks_per_node) {
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
      rdma_coordinator.init(process_group, node_rank, local_rank, buffer_config);
#else
      fprintf(stderr, "Inter-node communication is not supported. Please rebuild with HYBRID_EP_MULTINODE flag, group_size=%d, buffer_config.num_of_ranks_per_node=%d.\n", group_size, buffer_config.num_of_ranks_per_node);
      fflush(stderr);
      assert(false); // inter-node communication is not supported.
#endif
    }

    allocate_buffer();
}

void HybridEPBuffer::release_buffer() {
  // Synchronize the device to ensure all operations are completed.
  CUDA_CHECK(cudaDeviceSynchronize());

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  if(buffer_config.num_of_nodes > 1) {
    rdma_coordinator.destroy();
  }
#endif
  nvl_coordinator.destroy();
  allgather_obj.destroy();
}

void HybridEPBuffer::allocate_buffer() {
  // Buffer allocation for intra-node communication
  nvl_coordinator.allocate_preprocessing_buffers();
  nvl_coordinator.allocate_combine_buffers(); // We should allocate the combine buffer first, because the dispatch could have chance to reuse the combine buffer sometimes.
  nvl_coordinator.allocate_dispatch_buffers();
  nvl_coordinator.exchange_remote_nvl_info();

  // Buffer allocation for inter-node communication
  #ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    if(buffer_config.num_of_nodes > 1) {
      rdma_coordinator.allocate_combine_buffers();
      rdma_coordinator.allocate_dispatch_buffers();
    }
  #endif

  // Allocate the allgather buffer
  allgather_obj.allocate_ag_buffer();

  // Update the executor with the buffers
  executor.set_intra_node_buffers(&nvl_coordinator.dispatch_buffers, &nvl_coordinator.combine_buffers);
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  executor.set_inter_node_buffers(&rdma_coordinator.dispatch_buffers, &rdma_coordinator.combine_buffers);
#endif
}

bool HybridEPBuffer::update_buffer(HybridEpConfigInstance config) {
  // If new config requires bigger buffer, we will release the old buffer and allocate a new one.
  bool need_reallocate = false;
  
  need_reallocate |= grow_to(buffer_config.max_num_of_tokens_per_rank, config.max_num_of_tokens_per_rank);
  need_reallocate |= grow_to(buffer_config.hidden_dim,             config.hidden_dim);
  need_reallocate |= grow_to(buffer_config.num_of_experts_per_rank,config.num_of_experts_per_rank);
  need_reallocate |= grow_to(buffer_config.num_of_ranks_per_node,  config.num_of_ranks_per_node);
  need_reallocate |= grow_to(buffer_config.num_of_nodes,           config.num_of_nodes);
  need_reallocate |= grow_to(buffer_config.num_of_blocks_preprocessing_api, config.num_of_blocks_preprocessing_api);
  need_reallocate |= grow_to(buffer_config.num_of_blocks_dispatch_api, config.num_of_blocks_dispatch_api);
  need_reallocate |= grow_to(buffer_config.num_of_tokens_per_chunk_dispatch_api, config.num_of_tokens_per_chunk_dispatch_api);
  need_reallocate |= grow_to(buffer_config.num_of_tokens_per_chunk_combine_api, config.num_of_tokens_per_chunk_combine_api);
  
  // Special case for token data type.
  if(get_token_data_type_size(buffer_config.token_data_type) < get_token_data_type_size(config.token_data_type)
      && !nvl_coordinator.use_shared_buffer) {
    need_reallocate = true;
    buffer_config.token_data_type = config.token_data_type;
  }

  if(buffer_config.num_of_nodes > 1 && need_reallocate) {
    TORCH_WARN("Reallocating HybridEP buffers in multi-node mode is very slow; "
               "adjust buffer_config to pre-allocate sufficient capacity.");
  }

  if(need_reallocate) {
  #ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    rdma_coordinator.update_config(buffer_config);
  #endif
    nvl_coordinator.update_config(buffer_config);
    // Update the allgather object
    allgather_obj.update(buffer_config);
    release_buffer();
    allocate_buffer();
  }
  return need_reallocate;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
HybridEPBuffer::metadata_preprocessing(HybridEpConfigInstance config, torch::Tensor local_routing_map, int64_t num_of_tokens_per_rank, bool non_blocking) {
  // Basic checks
  assert(local_routing_map.device().is_cuda());
  assert(local_routing_map.is_contiguous());

  // Prepare the global routing map
  auto global_routing_map = executor.allgather_routing_map(
    allgather_obj, config, local_routing_map, process_group
  );

  // Run the hybrid-ep metadata preprocessing kernel
  return executor.metadata_preprocess_core(config, nvl_coordinator.preprocessing_tmp, global_routing_map, num_of_tokens_per_rank, non_blocking);
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
HybridEPBuffer::dispatch(HybridEpConfigInstance config, 
                 torch::Tensor hidden, c10::optional<torch::Tensor> probs,
                 c10::optional<torch::Tensor> scaling_factor,
                 torch::Tensor sparse_to_dense_map,
                 torch::Tensor rdma_to_attn_map, torch::Tensor attn_to_rdma_map,
                 c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
                 c10::optional<int64_t> num_dispatched_tokens,
                 int64_t num_of_tokens_per_rank,
                 bool with_probs) {
  // Check the input tensors
  assert(hidden.device().is_cuda());
  assert(hidden.is_contiguous());
  if (with_probs) {
    assert(probs.has_value());
    assert(probs.value().device().is_cuda());
    assert(probs.value().is_contiguous());
    assert(probs.value().dtype() == torch::kFloat32);
  }
  if (config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
    assert(scaling_factor.has_value());
    assert(scaling_factor.value().device().is_cuda());
    assert(scaling_factor.value().is_contiguous());
  }
  
  // Prepare the parameters
  Executor::DispatchArgs args;
  args.hidden = hidden;
  if(with_probs) args.probs = probs.value();
  if(config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) args.scaling_factor = scaling_factor.value();
  args.sparse_to_dense_map = sparse_to_dense_map;
  args.rdma_to_attn_map = rdma_to_attn_map;
  args.attn_to_rdma_map = attn_to_rdma_map;
  args.num_dispatched_tokens_tensor = num_dispatched_tokens_tensor;
  args.num_dispatched_tokens = (num_dispatched_tokens.has_value()) ? 
                                num_dispatched_tokens.value() : -1;
  args.num_of_tokens_per_rank = num_of_tokens_per_rank;
  args.enable_permute = false;
  args.stream = at::cuda::getCurrentCUDAStream();
  
  // Run the full dispatch operation
  config.forward_dispatch_api = with_probs;
  executor.dispatch_preprocess(config, args);
  if(config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
    executor.dispatch_core<uint8_t>(config, args);
  } else if (config.token_data_type == APP_TOKEN_DATA_TYPE::UINT16) {
    executor.dispatch_core<uint16_t>(config, args);
  }else {
    throw std::runtime_error("Invalid token data type:" +  std::to_string(static_cast<int>(config.token_data_type)));
  }
  auto [dispatched_tokens, dispatched_probs, dispatched_scaling_factor] = executor.dispatch_postprocess(config, args);

  return std::make_tuple(dispatched_tokens, dispatched_probs, dispatched_scaling_factor);
}

std::tuple<torch::Tensor, torch::Tensor>
HybridEPBuffer::combine(HybridEpConfigInstance config, 
                torch::Tensor hidden, c10::optional<torch::Tensor> probs,
                torch::Tensor sparse_to_dense_map,
                torch::Tensor rdma_to_attn_map, torch::Tensor attn_to_rdma_map,
                int64_t num_of_tokens_per_rank,
                bool with_probs) {
  // Check the input tensors
  assert(c10::elementSize(hidden.scalar_type()) == 2);
  assert(hidden.device().is_cuda());
  assert(hidden.dtype() != torch::kUInt8);
  assert(hidden.is_contiguous());
  if (with_probs) {
    assert(probs.has_value());
    assert(probs.value().device().is_cuda());
    assert(probs.value().is_contiguous());
    assert(probs.value().dtype() == torch::kFloat32);
    assert(probs.value().numel() == 0 ||
           probs.value().size(1) == config.num_of_experts_per_rank * config.num_of_ranks_per_node);
  }

  // Construct the output tensors
  torch::Tensor combined_tokens, combined_probs;
  combined_tokens =torch::empty({num_of_tokens_per_rank, config.hidden_dim},
                   torch::dtype(hidden.dtype()).device(torch::kCUDA));
  if (with_probs) {
    combined_probs =
        torch::empty({num_of_tokens_per_rank, config.num_of_experts_per_rank *  config.num_of_ranks_per_node * config.num_of_nodes}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }

  // Prepare the parameters
  Executor::CombineArgs args;
  args.hidden = hidden;
  if(with_probs) args.probs = probs.value();
  args.combined_tokens = reinterpret_cast<uint16_t*>(combined_tokens.data_ptr());
  if(with_probs) args.combined_probs = reinterpret_cast<float*>(combined_probs.data_ptr());
  args.sparse_to_dense_map = sparse_to_dense_map;
  args.rdma_to_attn_map = rdma_to_attn_map;
  args.attn_to_rdma_map = attn_to_rdma_map;
  args.num_of_tokens_per_rank = num_of_tokens_per_rank;
  args.enable_unpermute = false;
  args.stream = at::cuda::getCurrentCUDAStream();

  // Run the full combine operation
  config.backward_combine_api = with_probs;
  executor.combine_preprocess(config, args);
  executor.combine_core(config, args);
  executor.combine_postprocess(config, args);
  
  return std::make_tuple(combined_tokens, combined_probs);
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor>
HybridEPBuffer::dispatch_with_permute(HybridEpConfigInstance config, 
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
          bool with_probs)
{
 // Check the input tensors
 assert(hidden.device().is_cuda());
 assert(hidden.is_contiguous());
 if (with_probs) {
   assert(probs.has_value());
   assert(probs.value().device().is_cuda());
   assert(probs.value().is_contiguous());
   assert(probs.value().dtype() == torch::kFloat32);
 }
 if (config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
   assert(scaling_factor.has_value());
   assert(scaling_factor.value().device().is_cuda());
   assert(scaling_factor.value().is_contiguous());
 }
 
 // Prepare the parameters
 Executor::DispatchArgs args;
 args.hidden = hidden;
 if(with_probs) args.probs = probs.value();
 if(config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) args.scaling_factor = scaling_factor.value();
 args.sparse_to_dense_map = sparse_to_dense_map;
 args.rdma_to_attn_map = rdma_to_attn_map;
 args.attn_to_rdma_map = attn_to_rdma_map;
 args.local_expert_routing_map = local_expert_routing_map;
 args.num_dispatched_tokens_tensor = num_dispatched_tokens_tensor;
 args.max_num_dispatched_tokens = nvl_coordinator.max_num_of_tokens;
 args.row_id_map = row_id_map;
 args.num_permuted_tokens = (num_permuted_tokens.has_value()) ? num_permuted_tokens.value() : -1;
 args.pad_multiple = (pad_multiple.has_value()) ? pad_multiple.value() : 0;
 args.non_blocking = non_blocking;
 args.num_of_tokens_per_rank = num_of_tokens_per_rank;
 args.enable_permute = true;
 args.stream = at::cuda::getCurrentCUDAStream();
 
 // Run the full dispatch operation
 config.forward_dispatch_api = with_probs;
 auto [result_row_id_map, result_tokens_per_expert, overflow_flag] = executor.dispatch_preprocess(config, args);
 if(config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) {
   executor.dispatch_core<uint8_t>(config, args);
 } else if (config.token_data_type == APP_TOKEN_DATA_TYPE::UINT16) {
   executor.dispatch_core<uint16_t>(config, args);
 }else {
   throw std::runtime_error("Invalid token data type:" +  std::to_string(static_cast<int>(config.token_data_type)));
 }

 auto [dispatched_tokens, dispatched_probs, dispatched_scaling_factor] = executor.dispatch_postprocess(config, args);

 return std::make_tuple(dispatched_tokens, dispatched_probs, dispatched_scaling_factor, overflow_flag, result_row_id_map, result_tokens_per_expert);
}

std::tuple<torch::Tensor, torch::Tensor>
HybridEPBuffer::combine_with_unpermute(HybridEpConfigInstance config, 
        torch::Tensor hidden, c10::optional<torch::Tensor> probs,
        torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
        torch::Tensor attn_to_rdma_map, c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
        c10::optional<torch::Tensor> row_id_map,
        int64_t num_of_tokens_per_rank,
        c10::optional<int64_t> pad_multiple,
        bool with_probs)
{
  // Check the input tensors
  assert(c10::elementSize(hidden.scalar_type()) == 2);
  assert(hidden.device().is_cuda());
  assert(hidden.dtype() != torch::kUInt8);
  assert(hidden.is_contiguous());
  if (with_probs) {
    assert(probs.has_value());
    assert(probs.value().device().is_cuda());
    assert(probs.value().is_contiguous());
    assert(probs.value().dtype() == torch::kFloat32);
  }

  // Construct the output tensors
  torch::Tensor combined_tokens, combined_probs;
  combined_tokens =torch::empty({num_of_tokens_per_rank, config.hidden_dim},
                   torch::dtype(hidden.dtype()).device(torch::kCUDA));
  if (with_probs) {
    combined_probs =
        torch::empty({num_of_tokens_per_rank, config.num_of_experts_per_rank *  config.num_of_ranks_per_node * config.num_of_nodes}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }

  // Prepare the parameters
  Executor::CombineArgs args;
  args.hidden = hidden;
  if(with_probs) args.probs = probs.value();
  args.combined_tokens = reinterpret_cast<uint16_t*>(combined_tokens.data_ptr());
  if(with_probs) args.combined_probs = reinterpret_cast<float*>(combined_probs.data_ptr());
  args.sparse_to_dense_map = sparse_to_dense_map;
  args.rdma_to_attn_map = rdma_to_attn_map;
  args.attn_to_rdma_map = attn_to_rdma_map;
  args.num_dispatched_tokens_tensor = num_dispatched_tokens_tensor;
  args.row_id_map = row_id_map;
  args.pad_multiple = (pad_multiple.has_value()) ? pad_multiple.value() : 0;
  args.num_of_tokens_per_rank = num_of_tokens_per_rank;
  args.enable_unpermute = true;
  args.stream = at::cuda::getCurrentCUDAStream();

  // Run the full combine operation
  config.backward_combine_api = with_probs;
  executor.combine_preprocess(config, args);
  executor.combine_core(config, args);
  executor.combine_postprocess(config, args);
  
  return std::make_tuple(combined_tokens, combined_probs);
}

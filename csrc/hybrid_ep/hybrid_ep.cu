// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#include "hybrid_ep.cuh"

HybridEPBuffer::HybridEPBuffer(BufferConfig config, int local_rank, int node_rank, int group_size, std::string base_path)
    : buffer_config(config), local_rank(local_rank), node_rank(node_rank), group_size(group_size),
    executor(local_rank, node_rank, base_path) {
    if(group_size <= buffer_config.num_of_ranks_per_node) {
      // If used on only intra-node communication, the dispatch/combine can share same buffers.
      use_shared_buffer = true;
    }else{
      // Currently, inter-node communication is not supported.
      assert(false);
    }
      
    remote_allocator.init(/*enable_fabric = */ true);
    allocate_buffer();
}

HybridEPBuffer::~HybridEPBuffer() {
    release_buffer();
}

void HybridEPBuffer::release_buffer() {
  // Synchronize the device to ensure all operations are completed.
  CUDA_CHECK(cudaDeviceSynchronize());

  auto free_buffer = [this](void *ptr, bool remote_memory) {
    if (ptr != nullptr) {
      if (remote_memory) {
          // If the memory can be accessed by remote devices, free it from remote allocator.
          remote_allocator.free(ptr);
        } else {
          CUDA_CHECK(cudaFree(ptr));
        }
      }
  };
  
  // Clean up preprocessing buffer
  free_buffer(this->preprocessing_tmp, false);

  // Clean up dispatch buffers
  if (!use_shared_buffer) {
    free_buffer(dispatch_buffers.expert_output_token, true);
    free_buffer(dispatch_buffers.expert_output_prob, true);
  }
  free_buffer(dispatch_buffers.expert_output_scaling_factor, true);
  free_buffer(dispatch_buffers.rdma_inter_node_group_scaling_factor, false);
  free_buffer(dispatch_buffers.rdma_inter_node_group_token,false);
  free_buffer(dispatch_buffers.rdma_inter_node_group_prob, false);
  free_buffer(dispatch_buffers.rdma_inter_node_group_flags, false);
  free_buffer(dispatch_buffers.expected_rdma_flag_value, false);
  free_buffer(dispatch_buffers.expected_intra_node_flag_value, false);
  if (local_rank == 0) {
    free_buffer(dispatch_buffers.intra_node_write_completion_flags, true);
  }else{
    remote_allocator.close_handle(dispatch_buffers.intra_node_write_completion_flags);
  }
  for (int i = 0; i < buffer_config.num_of_ranks_per_node; i++) {
    if (i != local_rank) {
      remote_allocator.close_handle(dispatch_buffers.expert_output_token_all_ranks[i]);
      remote_allocator.close_handle(dispatch_buffers.expert_output_prob_all_ranks[i]);
      remote_allocator.close_handle(dispatch_buffers.expert_output_scaling_factor_all_ranks[i]);
    }
  }
  delete[] dispatch_buffers.expert_output_token_all_ranks;
  delete[] dispatch_buffers.expert_output_prob_all_ranks;
  delete[] dispatch_buffers.expert_output_scaling_factor_all_ranks;

  // Clean up combine buffers
  free_buffer(combine_buffers.expert_input_token, true);
  free_buffer(combine_buffers.expert_input_prob, true);
  free_buffer(combine_buffers.rdma_intra_node_red_token, false);
  free_buffer(combine_buffers.rdma_intra_node_red_prob, false);
  free_buffer(combine_buffers.rdma_inter_node_group_token, false);
  free_buffer(combine_buffers.rdma_inter_node_group_prob, false);
  free_buffer(combine_buffers.rdma_inter_node_group_flags, false);
  free_buffer(combine_buffers.expected_rdma_flag_value, false);
  free_buffer(combine_buffers.expected_intra_node_flag_value, false);
  if (local_rank == 0) {
    free_buffer(combine_buffers.intra_node_write_completion_flags, true);
  }else{
    remote_allocator.close_handle(combine_buffers.intra_node_write_completion_flags);
  }
  for (int i = 0; i < buffer_config.num_of_ranks_per_node; i++) {
    if (i != local_rank) {
      remote_allocator.close_handle(combine_buffers.expert_input_token_all_ranks[i]);
      remote_allocator.close_handle(combine_buffers.expert_input_prob_all_ranks[i]);
    }
  }
  delete[] combine_buffers.expert_input_token_all_ranks;
  delete[] combine_buffers.expert_input_prob_all_ranks;
}

void HybridEPBuffer::allocate_buffer_for_preprocessing() {
  auto preprocessing_tmp_elts =
    buffer_config.num_of_blocks_preprocessing_api * buffer_config.num_of_ranks_per_node;
  CUDA_CHECK(
      cudaMalloc((void **)&this->preprocessing_tmp,
                 preprocessing_tmp_elts * sizeof(hybrid_ep::tmp_state_t)));
}

void HybridEPBuffer::allocate_buffer_for_dispatch() {
  dispatch_buffers.data_type = buffer_config.token_data_type;
  size_t sizeof_token_data_type = get_token_data_type_size(dispatch_buffers.data_type);

  // Calculate buffer sizes
  auto expert_output_token_elts = max_num_of_tokens_for_experts * buffer_config.hidden_dim;
  auto expert_output_prob_elts = max_num_of_tokens_for_experts * 
                                 (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
  auto expert_output_scaling_factor_elts = max_num_of_tokens_for_experts * (buffer_config.hidden_dim / 128);
  // Calculate local temp buffer sizes  
  auto rdma_inter_node_group_token_elts = buffer_config.max_num_of_tokens_per_rank * 
                                          (buffer_config.num_of_nodes - 1) * buffer_config.hidden_dim;
  auto rdma_inter_node_group_prob_elts = buffer_config.max_num_of_tokens_per_rank * (buffer_config.num_of_nodes - 1) *
                                         (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
  auto rdma_inter_node_group_scaling_factor_elts = buffer_config.max_num_of_tokens_per_rank * 
                                                    (buffer_config.num_of_nodes - 1) * (buffer_config.hidden_dim / 128);
  auto rdma_inter_node_group_flags_elts = (buffer_config.max_num_of_tokens_per_rank /
                                           buffer_config.num_of_tokens_per_chunk_dispatch_api) *
                                          (buffer_config.num_of_nodes - 1);

  // Allocate main buffers
  if (use_shared_buffer) {
    assert(combine_buffers.expert_input_token != nullptr);
    assert(combine_buffers.expert_input_prob != nullptr);
    dispatch_buffers.expert_output_token = combine_buffers.expert_input_token;
    dispatch_buffers.expert_output_prob = combine_buffers.expert_input_prob;
  } else {
    remote_allocator.allocate((void**)&dispatch_buffers.expert_output_token, expert_output_token_elts * sizeof_token_data_type);
    remote_allocator.allocate((void**)&dispatch_buffers.expert_output_prob, expert_output_prob_elts * sizeof(float));
  }
  remote_allocator.allocate((void**)&dispatch_buffers.expert_output_scaling_factor, expert_output_scaling_factor_elts * sizeof(float));

  // Allocate RDMA buffers
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_token,
                        rdma_inter_node_group_token_elts * sizeof_token_data_type));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_prob,
                        rdma_inter_node_group_prob_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_scaling_factor,
                        rdma_inter_node_group_scaling_factor_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));

  // Allocate and initialize synchronization buffers
  if (local_rank == 0) {
    remote_allocator.allocate((void**)&dispatch_buffers.intra_node_write_completion_flags, sizeof(uint32_t));
    CUDA_CHECK(cudaMemset(dispatch_buffers.intra_node_write_completion_flags, 0, sizeof(uint32_t)));
  }
  
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.expected_intra_node_flag_value, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(dispatch_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(dispatch_buffers.expected_intra_node_flag_value, 0, sizeof(uint32_t)));

  // Create IPC memory handles
  MemHandle handles[4];
  remote_allocator.get_handle(&handles[0], dispatch_buffers.expert_output_token);
  remote_allocator.get_handle(&handles[1], dispatch_buffers.expert_output_prob);
  remote_allocator.get_handle(&handles[2], dispatch_buffers.expert_output_scaling_factor);
  if (local_rank == 0) {
    remote_allocator.get_handle(&handles[3], dispatch_buffers.intra_node_write_completion_flags);
  }
  
  // Pack handles into tensor
  dispatch_memory_handles = torch::empty({static_cast<int64_t>(sizeof(handles))},
                                        torch::dtype(torch::kUInt8).device(torch::kCPU));
  memcpy(dispatch_memory_handles.data_ptr<uint8_t>(), handles, sizeof(handles));

  // Check possible errors
  CUDA_CHECK(cudaGetLastError());
}

void HybridEPBuffer::allocate_buffer_for_combine() {
  // Calculate buffer sizes
  auto expert_input_token_elts = max_num_of_tokens_for_experts * buffer_config.hidden_dim;
  auto expert_input_prob_elts = max_num_of_tokens_for_experts *
                                (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
  // Calculate local temp buffer sizes
  auto rdma_intra_node_red_token_elts = buffer_config.max_num_of_tokens_per_rank *
                                        (buffer_config.num_of_nodes - 1) * buffer_config.hidden_dim;
  auto rdma_intra_node_red_prob_elts = buffer_config.max_num_of_tokens_per_rank * (buffer_config.num_of_nodes - 1) *
                                       (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
  auto rdma_inter_node_group_token_elts = buffer_config.max_num_of_tokens_per_rank *
                                          (buffer_config.num_of_nodes - 1) * buffer_config.hidden_dim;
  auto rdma_inter_node_group_prob_elts = buffer_config.max_num_of_tokens_per_rank * (buffer_config.num_of_nodes - 1) *
                                         (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
  auto rdma_inter_node_group_flags_elts = (buffer_config.max_num_of_tokens_per_rank /
                                           buffer_config.num_of_tokens_per_chunk_combine_api) *
                                          (buffer_config.num_of_nodes - 1);

  // Allocate main buffers
  remote_allocator.allocate((void**)&combine_buffers.expert_input_token, expert_input_token_elts * sizeof(uint16_t));
  remote_allocator.allocate((void**)&combine_buffers.expert_input_prob, expert_input_prob_elts * sizeof(float));

  // Allocate local temp buffer
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_intra_node_red_token,
                        rdma_intra_node_red_token_elts * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_intra_node_red_prob,
                        rdma_intra_node_red_prob_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_token,
                        rdma_inter_node_group_token_elts * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_prob,
                        rdma_inter_node_group_prob_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));

  // Allocate and initialize synchronization buffers
  if (local_rank == 0) {
    remote_allocator.allocate((void**)&combine_buffers.intra_node_write_completion_flags, sizeof(uint32_t));
    CUDA_CHECK(cudaMemset(combine_buffers.intra_node_write_completion_flags, 0, sizeof(uint32_t)));
  }
  
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.expected_intra_node_flag_value, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(combine_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(combine_buffers.expected_intra_node_flag_value, 0, sizeof(uint32_t)));

  // Create IPC memory handles
  MemHandle handles[3];
  remote_allocator.get_handle(&handles[0], combine_buffers.expert_input_token);
  remote_allocator.get_handle(&handles[1], combine_buffers.expert_input_prob);
  if (local_rank == 0) {
    remote_allocator.get_handle(&handles[2], combine_buffers.intra_node_write_completion_flags);
  }

  // Pack handles into tensor
  combine_memory_handles = torch::empty({static_cast<int64_t>(sizeof(handles))},
                                       torch::dtype(torch::kUInt8).device(torch::kCPU));
  memcpy(combine_memory_handles.data_ptr<uint8_t>(), handles, sizeof(handles));

  // Check possible errors
  CUDA_CHECK(cudaGetLastError());
}

void HybridEPBuffer::allocate_buffer() {
  // Token number at the worst case, all tokens are routed to the same expert.
  this->max_num_of_tokens_for_experts = buffer_config.max_num_of_tokens_per_rank *
                                        buffer_config.num_of_ranks_per_node *
                                        buffer_config.num_of_nodes;
  assert(this->max_num_of_tokens_for_experts % 4 ==
         0); // The number of tokens for experts should be divisible by 4, this
             // is required by the permute make_row_id_map kernel
  allocate_buffer_for_preprocessing();
  allocate_buffer_for_combine(); // We should allocate the combine buffer first, because the dispatch could have chance to reuse the combine buffer sometimes.
  allocate_buffer_for_dispatch();
}

void HybridEPBuffer::exchange_ipc_address(py::object process_group) {
  try {
    // Use Python's torch.distributed APIs through py::object
    auto torch_distributed = py::module_::import("torch.distributed");
    
    // Move tensors to CUDA for communication
    auto dispatch_cuda = dispatch_memory_handles.cuda();
    auto combine_cuda = combine_memory_handles.cuda();
    
    // Get world size from process group
    int world_size = process_group.attr("size")().cast<int>();
    
    // Create empty tensors for allgather output
    py::list dispatch_output_list;
    py::list combine_output_list;
    
    for (int i = 0; i < world_size; i++) {
      dispatch_output_list.append(torch::empty_like(dispatch_cuda));
      combine_output_list.append(torch::empty_like(combine_cuda));
    }
    
    // Perform allgather using Python API
    torch_distributed.attr("all_gather")(dispatch_output_list, dispatch_cuda, process_group);
    torch_distributed.attr("all_gather")(combine_output_list, combine_cuda, process_group);
    
    // Convert back to C++ vectors and move to CPU
    std::vector<torch::Tensor> dispatch_cpu_tensors;
    std::vector<torch::Tensor> combine_cpu_tensors;
    
    for (int i = 0; i < world_size; i++) {
      dispatch_cpu_tensors.push_back(dispatch_output_list[i].cast<torch::Tensor>().cpu());
      combine_cpu_tensors.push_back(combine_output_list[i].cast<torch::Tensor>().cpu());
    }
    
    // Open handles from other ranks
    open_handles_from_other_ranks(dispatch_cpu_tensors, combine_cpu_tensors);
    
  } catch (const std::exception& e) {
    throw std::runtime_error(
      "C++ distributed communication failed: " + std::string(e.what())
    );
  }
}


void HybridEPBuffer::open_handles_from_other_ranks(
    std::vector<torch::Tensor> dispatch_handles,
    std::vector<torch::Tensor> combine_handles) {
  // Malloc the pointer arrays used in the dispatch kernel.
  dispatch_buffers.expert_output_token_all_ranks =
      (void **)malloc(buffer_config.num_of_ranks_per_node * sizeof(void *));
  dispatch_buffers.expert_output_prob_all_ranks =
      (float **)malloc(buffer_config.num_of_ranks_per_node * sizeof(float *));
  dispatch_buffers.expert_output_scaling_factor_all_ranks =
      (float **)malloc(buffer_config.num_of_ranks_per_node * sizeof(float *));

  // Global offset means the position in the multi-node case.
  auto global_offset = node_rank * buffer_config.num_of_ranks_per_node;

  // Open the dispatch handles for intra_node_write_completion_flags
  if (local_rank != 0) {
    MemHandle intra_node_write_completion_flags_handle;
    // Only rank 0 will allocate memory for this flag
    memcpy(&intra_node_write_completion_flags_handle,
           dispatch_handles[global_offset].data_ptr<uint8_t>() +
               sizeof(MemHandle) * 3,
           sizeof(MemHandle));
    remote_allocator.open_handle((void**)(&dispatch_buffers.intra_node_write_completion_flags),
                           &intra_node_write_completion_flags_handle);
  }

  // Open the handles for export_output
  for (int i = 0; i < buffer_config.num_of_ranks_per_node; i++) {
    MemHandle expert_output_token_handle, expert_output_prob_handle,
        expert_output_scaling_factor_handle;

    // Extract the handles from the tensor.
    auto base_ptr = dispatch_handles[i + global_offset].data_ptr<uint8_t>();
    memcpy(&expert_output_token_handle, base_ptr, sizeof(MemHandle));
    memcpy(&expert_output_prob_handle, base_ptr + sizeof(MemHandle),
           sizeof(MemHandle));
    memcpy(&expert_output_scaling_factor_handle,
           base_ptr + sizeof(MemHandle) * 2,
           sizeof(MemHandle));

    // Open the handles for export_output
    if (i != local_rank) {
      remote_allocator.open_handle((void**)(&dispatch_buffers.expert_output_token_all_ranks[i]),
                             &expert_output_token_handle);
      remote_allocator.open_handle((void**)(&dispatch_buffers.expert_output_prob_all_ranks[i]),
                             &expert_output_prob_handle);
      remote_allocator.open_handle((void**)(&dispatch_buffers.expert_output_scaling_factor_all_ranks[i]), 
                             &expert_output_scaling_factor_handle);
    } else {
      // For local rank, use direct pointer assignment (more efficient, no IPC overhead)
      dispatch_buffers.expert_output_token_all_ranks[i] =
          dispatch_buffers.expert_output_token;
      dispatch_buffers.expert_output_prob_all_ranks[i] =
          dispatch_buffers.expert_output_prob;
      dispatch_buffers.expert_output_scaling_factor_all_ranks[i] =
          dispatch_buffers.expert_output_scaling_factor;
    }
  }

  // Malloc the pointer arrays used in the combine kernel.
  combine_buffers.expert_input_token_all_ranks =
      (uint16_t **)malloc(buffer_config.num_of_ranks_per_node * sizeof(uint16_t *));
  combine_buffers.expert_input_prob_all_ranks =
      (float **)malloc(buffer_config.num_of_ranks_per_node * sizeof(float *));
  // Open the combine handles for intra_node_write_completion_flags
  if (local_rank != 0) {
    MemHandle intra_node_write_completion_flags_handle;
    // Only rank 0 will allocate memory for this flag
    memcpy(&intra_node_write_completion_flags_handle,
           combine_handles[global_offset].data_ptr<uint8_t>() +
               sizeof(MemHandle) * 2,
           sizeof(MemHandle));
    remote_allocator.open_handle((void**)(&combine_buffers.intra_node_write_completion_flags),
                           &intra_node_write_completion_flags_handle);
  }
  // Open the handles for expert_input
  for (int i = 0; i < buffer_config.num_of_ranks_per_node; i++) {
    MemHandle expert_input_token_handle, expert_input_prob_handle;
    auto base_ptr = combine_handles[i + global_offset].data_ptr<uint8_t>();
    // Extract the handles from the tensor.
    memcpy(&expert_input_token_handle, base_ptr, sizeof(MemHandle));
    memcpy(&expert_input_prob_handle, base_ptr + sizeof(MemHandle),
           sizeof(MemHandle));
    // Open the handles for expert_input
    if (i != local_rank) {
      remote_allocator.open_handle((void**)(&combine_buffers.expert_input_token_all_ranks[i]),
                             &expert_input_token_handle);
      remote_allocator.open_handle((void**)(&combine_buffers.expert_input_prob_all_ranks[i]),
                             &expert_input_prob_handle);
    } else {
      // For local rank, use direct pointer assignment (more efficient, no IPC overhead)
      combine_buffers.expert_input_token_all_ranks[i] =
          combine_buffers.expert_input_token;
      combine_buffers.expert_input_prob_all_ranks[i] =
          combine_buffers.expert_input_prob;
    }
  }
}

bool HybridEPBuffer::update_buffer(HybridEpConfigInstance config) {
  // If new config requires bigger buffer, we will release the old buffer and allocate a new one.
  bool need_reallocate = false;
  
  need_reallocate |= grow_to(buffer_config.hidden_dim,             config.hidden_dim);
  need_reallocate |= grow_to(buffer_config.num_of_experts_per_rank,config.num_of_experts_per_rank);
  need_reallocate |= grow_to(buffer_config.num_of_ranks_per_node,  config.num_of_ranks_per_node);
  need_reallocate |= grow_to(buffer_config.num_of_nodes,           config.num_of_nodes);
  need_reallocate |= grow_to(buffer_config.num_of_blocks_preprocessing_api, config.num_of_blocks_preprocessing_api);
  need_reallocate |= grow_to(buffer_config.num_of_tokens_per_chunk_dispatch_api, config.num_of_tokens_per_chunk_dispatch_api);
  need_reallocate |= grow_to(buffer_config.num_of_tokens_per_chunk_combine_api, config.num_of_tokens_per_chunk_combine_api);
  
  // Special case for token data type.
  if(get_token_data_type_size(buffer_config.token_data_type) < get_token_data_type_size(config.token_data_type)
      && !use_shared_buffer) {
    need_reallocate = true;
    buffer_config.token_data_type = config.token_data_type;
  }

  if(need_reallocate) {
    release_buffer();
    allocate_buffer();
  }
  return need_reallocate;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
HybridEPBuffer::metadata_preprocessing(HybridEpConfigInstance config, torch::Tensor global_routing_map, int64_t num_of_tokens_per_rank) {
  // Basic checks
  assert(global_routing_map.device().is_cuda());
  assert(global_routing_map.is_contiguous());

  // Run the hybrid-ep metadata preprocessing kernel
  return executor.metadata_preprocess_core(config, preprocessing_tmp, global_routing_map, num_of_tokens_per_rank);
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
  if (config.token_data_type == TOKEN_DATA_TYPE::UINT8) {
    assert(scaling_factor.has_value());
    assert(scaling_factor.value().device().is_cuda());
    assert(scaling_factor.value().is_contiguous());
  }
  
  // Prepare the parameters
  Executor::DispatchArgs args;
  args.hidden = hidden;
  if(with_probs) args.probs = probs.value();
  if(config.token_data_type == TOKEN_DATA_TYPE::UINT8) args.scaling_factor = scaling_factor.value();
  args.sparse_to_dense_map = sparse_to_dense_map;
  args.rdma_to_attn_map = rdma_to_attn_map;
  args.attn_to_rdma_map = attn_to_rdma_map;
  args.num_dispatched_tokens_tensor = num_dispatched_tokens_tensor;
  args.num_dispatched_tokens = (num_dispatched_tokens.has_value()) ? 
                                num_dispatched_tokens.value() : 
                                num_dispatched_tokens_tensor.value().item<int64_t>();
  args.num_of_tokens_per_rank = num_of_tokens_per_rank;
  args.enable_permute = false;
  args.stream = at::cuda::getCurrentCUDAStream();
  
  // Run the full dispatch operation
  config.forward_dispatch_api = with_probs;
  executor.dispatch_preprocess(config, dispatch_buffers, args);
  if(config.token_data_type == TOKEN_DATA_TYPE::UINT8) {
    executor.dispatch_core<uint8_t>(config, dispatch_buffers, args);
  } else if (config.token_data_type == TOKEN_DATA_TYPE::UINT16) {
    executor.dispatch_core<uint16_t>(config, dispatch_buffers, args);
  }else {
    throw std::runtime_error("Invalid token data type:" +  std::to_string(static_cast<int>(config.token_data_type)));
  }
  auto [dispatched_tokens, dispatched_probs, dispatched_scaling_factor, row_id_map, tokens_per_expert] = executor.dispatch_postprocess(config, dispatch_buffers, args);

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
  executor.combine_preprocess(config, combine_buffers, args);
  executor.combine_core(config, combine_buffers, args);
  executor.combine_postprocess(config, combine_buffers, args);
  
  return std::make_tuple(combined_tokens, combined_probs);
}


std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor>
HybridEPBuffer::dispatch_with_permute(HybridEpConfigInstance config, 
          torch::Tensor hidden, c10::optional<torch::Tensor> probs,
          c10::optional<torch::Tensor> scaling_factor,
          torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
          torch::Tensor attn_to_rdma_map, 
          c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
          c10::optional<torch::Tensor> local_expert_routing_map,
          c10::optional<torch::Tensor> row_id_map,
          c10::optional<int64_t> num_dispatched_tokens,
          c10::optional<int64_t> num_permuted_tokens,
          int64_t num_of_tokens_per_rank,
          c10::optional<int64_t> pad_multiple,
          bool use_host_meta,
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
 if (config.token_data_type == TOKEN_DATA_TYPE::UINT8) {
   assert(scaling_factor.has_value());
   assert(scaling_factor.value().device().is_cuda());
   assert(scaling_factor.value().is_contiguous());
 }
 
 // Prepare the parameters
 Executor::DispatchArgs args;
 args.hidden = hidden;
 if(with_probs) args.probs = probs.value();
 if(config.token_data_type == TOKEN_DATA_TYPE::UINT8) args.scaling_factor = scaling_factor.value();
 args.sparse_to_dense_map = sparse_to_dense_map;
 args.rdma_to_attn_map = rdma_to_attn_map;
 args.attn_to_rdma_map = attn_to_rdma_map;
 args.local_expert_routing_map = local_expert_routing_map;
 args.num_dispatched_tokens_tensor = num_dispatched_tokens_tensor;
 args.num_dispatched_tokens = (num_dispatched_tokens.has_value()) ? 
                                num_dispatched_tokens.value() : 
                                num_dispatched_tokens_tensor.value().item<int64_t>();
 args.row_id_map = row_id_map;
 args.num_permuted_tokens = (num_permuted_tokens.has_value()) ? num_permuted_tokens.value() : -1;
 args.pad_multiple = (pad_multiple.has_value()) ? pad_multiple.value() : 0;
 args.use_host_meta = use_host_meta;
 args.num_of_tokens_per_rank = num_of_tokens_per_rank;
 args.enable_permute = true;
 args.stream = at::cuda::getCurrentCUDAStream();
 
 // Run the full dispatch operation
 config.forward_dispatch_api = with_probs;
 executor.dispatch_preprocess(config, dispatch_buffers, args);
 if(config.token_data_type == TOKEN_DATA_TYPE::UINT8) {
   executor.dispatch_core<uint8_t>(config, dispatch_buffers, args);
 } else if (config.token_data_type == TOKEN_DATA_TYPE::UINT16) {
   executor.dispatch_core<uint16_t>(config, dispatch_buffers, args);
 }else {
   throw std::runtime_error("Invalid token data type:" +  std::to_string(static_cast<int>(config.token_data_type)));
 }

 return executor.dispatch_postprocess(config, dispatch_buffers, args);
}

std::tuple<torch::Tensor, torch::Tensor>
HybridEPBuffer::combine_with_unpermute(HybridEpConfigInstance config, 
        torch::Tensor hidden, c10::optional<torch::Tensor> probs,
        torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
        torch::Tensor attn_to_rdma_map, c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
        c10::optional<torch::Tensor> row_id_map,
        c10::optional<int64_t> num_dispatched_tokens,
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
  args.num_dispatched_tokens = (num_dispatched_tokens.has_value()) ? 
                                num_dispatched_tokens.value() : 
                                num_dispatched_tokens_tensor.value().item<int64_t>();
  args.row_id_map = row_id_map;
  args.pad_multiple = (pad_multiple.has_value()) ? pad_multiple.value() : 0;
  args.num_of_tokens_per_rank = num_of_tokens_per_rank;
  args.enable_unpermute = true;
  args.stream = at::cuda::getCurrentCUDAStream();

  // Run the full combine operation
  config.backward_combine_api = with_probs;
  executor.combine_preprocess(config, combine_buffers, args);
  executor.combine_core(config, combine_buffers, args);
  executor.combine_postprocess(config, combine_buffers, args);
  
  return std::make_tuple(combined_tokens, combined_probs);
}

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#include "hybrid_ep.cuh"

HybridEpBuffer::HybridEpBuffer(HybridEpConfigInstance config, int local_rank, int node_rank, int group_size, int num_of_ranks_per_node, int nvlink_domain_size)
    : config(config), local_rank(local_rank), node_rank(node_rank), group_size(group_size),
      num_of_ranks_per_node(num_of_ranks_per_node), kernel_cache(local_rank), nvlink_domain_size(nvlink_domain_size) {

    assert(config.num_of_ranks_per_node <= nvlink_domain_size);

    if(group_size <= config.num_of_ranks_per_node) {
      // If used on only intra-node communication, the dispatch/combine can share same buffers.
      use_shared_buffer = true;
    }else{
      // Currently, inter-node communication is not supported.
      assert(false);
    }
      
    remote_allocator.init(/*enable_fabric = */ true);
    allocate_buffer();
}

HybridEpBuffer::~HybridEpBuffer() {
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
  if (use_fp8_dispatch) {
    free_buffer(dispatch_buffers.expert_output_scaling_factor, true);
    free_buffer(dispatch_buffers.rdma_inter_node_group_scaling_factor, false);
  }
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
  for (int i = 0; i < config.num_of_ranks_per_node; i++) {
    if (i != local_rank) {
      remote_allocator.close_handle(dispatch_buffers.expert_output_token_all_ranks[i]);
      remote_allocator.close_handle(dispatch_buffers.expert_output_prob_all_ranks[i]);
      if (use_fp8_dispatch) {
        remote_allocator.close_handle(dispatch_buffers.expert_output_scaling_factor_all_ranks[i]);
      }
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
  for (int i = 0; i < config.num_of_ranks_per_node; i++) {
    if (i != local_rank) {
      remote_allocator.close_handle(combine_buffers.expert_input_token_all_ranks[i]);
      remote_allocator.close_handle(combine_buffers.expert_input_prob_all_ranks[i]);
    }
  }
  delete[] combine_buffers.expert_input_token_all_ranks;
  delete[] combine_buffers.expert_input_prob_all_ranks;
}

void HybridEpBuffer::allocate_buffer_for_preprocessing() {
  auto preprocessing_tmp_elts =
      config.num_of_blocks_preprocessing_api * config.num_of_ranks_per_node;
  CUDA_CHECK(
      cudaMalloc((void **)&this->preprocessing_tmp,
                 preprocessing_tmp_elts * sizeof(hybrid_ep::tmp_state_t)));
}

void HybridEpBuffer::allocate_buffer_for_dispatch() {
  dispatch_buffers.data_type = config.token_data_type;
  size_t sizeof_token_data_type = get_token_data_type_size(dispatch_buffers.data_type);

  // Calculate buffer sizes
  auto expert_output_token_elts = max_num_of_tokens_for_experts * config.hidden_dim;
  auto expert_output_prob_elts = max_num_of_tokens_for_experts * 
                                 (config.num_of_experts_per_rank * config.num_of_ranks_per_node);
  auto expert_output_scaling_factor_elts = max_num_of_tokens_for_experts * (config.hidden_dim / 128);
  // Calculate local temp buffer sizes  
  auto rdma_inter_node_group_token_elts = config.max_num_of_tokens_per_rank * 
                                          (config.num_of_nodes - 1) * config.hidden_dim;
  auto rdma_inter_node_group_prob_elts = config.max_num_of_tokens_per_rank * (config.num_of_nodes - 1) *
                                         (config.num_of_experts_per_rank * config.num_of_ranks_per_node);
  auto rdma_inter_node_group_scaling_factor_elts = config.max_num_of_tokens_per_rank * 
                                                    (config.num_of_nodes - 1) * (config.hidden_dim / 128);
  auto rdma_inter_node_group_flags_elts = (config.max_num_of_tokens_per_rank /
                                           config.num_of_tokens_per_chunk_dispatch_api) *
                                          (config.num_of_nodes - 1);

  // Allocate main buffers
  if (use_shared_buffer) {
    assert(combine_buffers.expert_input_token != nullptr);
    assert(combine_buffers.expert_input_prob != nullptr);
    dispatch_buffers.expert_output_token = combine_buffers.expert_input_token;
    dispatch_buffers.expert_output_prob = combine_buffers.expert_input_prob;
  }
  else {
    remote_allocator.allocate((void**)&dispatch_buffers.expert_output_token, expert_output_token_elts * sizeof_token_data_type);
    remote_allocator.allocate((void**)&dispatch_buffers.expert_output_prob, expert_output_prob_elts * sizeof(float));
  }
  if (use_fp8_dispatch) {
    remote_allocator.allocate((void**)&dispatch_buffers.expert_output_scaling_factor, expert_output_scaling_factor_elts * sizeof(float));
  }

  // Allocate RDMA buffers
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_token,
                        rdma_inter_node_group_token_elts * sizeof_token_data_type));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_prob,
                        rdma_inter_node_group_prob_elts * sizeof(float)));
  if (use_fp8_dispatch) {
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_scaling_factor,
                          rdma_inter_node_group_scaling_factor_elts * sizeof(float)));
  }
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));

  // Allocate and initialize synchronization buffers
  if (local_rank == 0) {
    remote_allocator.allocate((void**)&dispatch_buffers.intra_node_write_completion_flags, sizeof(uint32_t));
  }
  
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.expected_intra_node_flag_value, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(dispatch_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(dispatch_buffers.expected_intra_node_flag_value, 0, sizeof(uint32_t)));

  // Create IPC memory handles
  MemHandle handles[4];
  remote_allocator.get_handle(&handles[0], dispatch_buffers.expert_output_token);
  remote_allocator.get_handle(&handles[1], dispatch_buffers.expert_output_prob);
  if (use_fp8_dispatch) {
    remote_allocator.get_handle(&handles[2], dispatch_buffers.expert_output_scaling_factor);
  }
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

void HybridEpBuffer::allocate_buffer_for_combine() {
  // Calculate buffer sizes
  auto expert_input_token_elts = max_num_of_tokens_for_experts * config.hidden_dim;
  auto expert_input_prob_elts = max_num_of_tokens_for_experts *
                                (config.num_of_experts_per_rank * config.num_of_ranks_per_node);
  // Calculate local temp buffer sizes
  auto rdma_intra_node_red_token_elts = config.max_num_of_tokens_per_rank *
                                        (config.num_of_nodes - 1) * config.hidden_dim;
  auto rdma_intra_node_red_prob_elts = config.max_num_of_tokens_per_rank * (config.num_of_nodes - 1) *
                                       (config.num_of_experts_per_rank * config.num_of_ranks_per_node);
  auto rdma_inter_node_group_token_elts = config.max_num_of_tokens_per_rank *
                                          (config.num_of_nodes - 1) * config.hidden_dim;
  auto rdma_inter_node_group_prob_elts = config.max_num_of_tokens_per_rank * (config.num_of_nodes - 1) *
                                         (config.num_of_experts_per_rank * config.num_of_ranks_per_node);
  auto rdma_inter_node_group_flags_elts = (config.max_num_of_tokens_per_rank /
                                           config.num_of_tokens_per_chunk_combine_api) *
                                          (config.num_of_nodes - 1);

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

void HybridEpBuffer::allocate_buffer() {
  // Token number at the worst case, all tokens are routed to the same expert.
  this->max_num_of_tokens_for_experts = config.max_num_of_tokens_per_rank *
                                        config.num_of_ranks_per_node *
                                        config.num_of_nodes;
  assert(this->max_num_of_tokens_for_experts % 4 ==
         0); // The number of tokens for experts should be divisible by 4, this
             // is required by the permute make_row_id_map kernel
  allocate_buffer_for_preprocessing();
  allocate_buffer_for_combine(); // We should allocate the combine buffer first, because the dispatch could have chance to reuse the combine buffer sometimes.
  allocate_buffer_for_dispatch();
}

void HybridEpBuffer::exchange_ipc_address(py::object process_group) {
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


void HybridEpBuffer::open_handles_from_other_ranks(
    std::vector<torch::Tensor> dispatch_handles,
    std::vector<torch::Tensor> combine_handles) {

  // Malloc the pointer arrays used in the dispatch kernel.
  dispatch_buffers.expert_output_token_all_ranks =
      (void **)malloc(config.num_of_ranks_per_node * sizeof(void *));
  dispatch_buffers.expert_output_prob_all_ranks =
      (float **)malloc(config.num_of_ranks_per_node * sizeof(float *));
  dispatch_buffers.expert_output_scaling_factor_all_ranks =
      (float **)malloc(config.num_of_ranks_per_node * sizeof(float *));

  // Global offset means the position in the multi-node case.
  auto global_offset = node_rank * num_of_ranks_per_node;

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
  for (int i = 0; i < num_of_ranks_per_node; i++) {
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
      (uint16_t **)malloc(config.num_of_ranks_per_node * sizeof(uint16_t *));
  combine_buffers.expert_input_prob_all_ranks =
      (float **)malloc(config.num_of_ranks_per_node * sizeof(float *));
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
  for (int i = 0; i < num_of_ranks_per_node; i++) {
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
HybridEpBuffer::metadata_preprocessing(torch::Tensor routing_map, int64_t num_of_tokens_per_rank) {
  assert(routing_map.device().is_cuda());
  assert(routing_map.is_contiguous());

  // padding for the routing map
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;

  // Construt the output tensor of the metadata preprocessing kernel.
  auto sparse_to_dense_map =
      torch::empty({num_of_tokens_per_rank * config.num_of_nodes,
                    config.num_of_ranks_per_node},
                   torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto rdma_to_attn_map =
      torch::empty({rdma_to_attn_map_size_per_node, config.num_of_nodes},
                   torch::dtype(torch::kBool).device(torch::kCUDA));
  auto attn_to_rdma_map =
      torch::empty({num_of_tokens_per_rank, config.num_of_nodes - 1},
                   torch::dtype(torch::kBool).device(torch::kCUDA));
  auto num_of_tokens_for_experts =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto local_expert_routing_map = torch::empty(
      {num_of_tokens_per_rank * config.num_of_ranks_per_node * config.num_of_nodes, config.num_of_experts_per_rank},
      torch::dtype(torch::kBool).device(torch::kCUDA));
  
  kernel_cache.run_proprecess_kernel(
      config, routing_map.data_ptr<bool>(), this->preprocessing_tmp,
      sparse_to_dense_map.data_ptr<int32_t>(),
      rdma_to_attn_map.data_ptr<bool>(), attn_to_rdma_map.data_ptr<bool>(),
      num_of_tokens_for_experts.data_ptr<int32_t>(),
      local_expert_routing_map.data_ptr<bool>(), static_cast<int>(node_rank),
      static_cast<int>(local_rank), num_of_tokens_per_rank, at::cuda::getCurrentCUDAStream());

  return std::make_tuple(sparse_to_dense_map, rdma_to_attn_map,
                         attn_to_rdma_map, num_of_tokens_for_experts,
                         local_expert_routing_map);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
HybridEpBuffer::dispatch(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
                 c10::optional<torch::Tensor> scaling_factor,
                 torch::Tensor sparse_to_dense_map,
                 torch::Tensor rdma_to_attn_map, torch::Tensor attn_to_rdma_map,
                 int64_t num_of_tokens_for_experts, 
                 int64_t num_of_tokens_per_rank,
                 bool with_probs) {

  // Use exact token count if available, otherwise use maximum bound
  auto token_count = (num_of_tokens_for_experts >= 0) ? num_of_tokens_for_experts : max_num_of_tokens_for_experts;

  auto stream = at::cuda::getCurrentCUDAStream();

  // Create and return output tensors
  size_t sizeof_token_data_type = get_token_data_type_size(dispatch_buffers.data_type);
  auto create_output_tensors = [&](int64_t token_count) -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> {
    torch::Tensor dispatched_tokens, dispatched_probs, dispatched_scaling_factor;
    
    // Create dispatched tokens tensor and copy data
    dispatched_tokens = torch::empty({token_count, config.hidden_dim},
                                   torch::dtype(hidden.dtype()).device(torch::kCUDA));
    auto res_sz = token_count * config.hidden_dim * sizeof_token_data_type;
    CUDA_CHECK(cudaMemcpyAsync(dispatched_tokens.data_ptr(), 
                               dispatch_buffers.expert_output_token,
                               res_sz, cudaMemcpyDeviceToDevice, stream));
    
    // Create and copy probs if needed
    if (with_probs) {
      dispatched_probs = torch::empty({token_count,
                                     config.num_of_experts_per_rank * config.num_of_ranks_per_node},
                                     torch::dtype(torch::kFloat32).device(torch::kCUDA));
      auto probs_sz = token_count * config.num_of_experts_per_rank * 
                      config.num_of_ranks_per_node * sizeof(float);
      CUDA_CHECK(cudaMemcpyAsync(dispatched_probs.data_ptr<float>(),
                                 dispatch_buffers.expert_output_prob,
                                 probs_sz, cudaMemcpyDeviceToDevice, stream));
    }
    
    // Create and copy scaling factor if using UINT8
    if (config.token_data_type == TOKEN_DATA_TYPE::UINT8) {
      dispatched_scaling_factor = torch::empty({token_count, config.hidden_dim / 128},
                                              torch::dtype(torch::kFloat32).device(torch::kCUDA));
      auto scaling_factor_sz = token_count * config.hidden_dim / 128 * sizeof(float);
      CUDA_CHECK(cudaMemcpyAsync(dispatched_scaling_factor.data_ptr<float>(),
                                dispatch_buffers.expert_output_scaling_factor,
                                scaling_factor_sz, cudaMemcpyDeviceToDevice, stream));
    }
    
    return std::make_tuple(dispatched_tokens, dispatched_probs, dispatched_scaling_factor);
  };

  // Fast return if there are no tokens to dispatch
  if (token_count == 0) {
    return create_output_tensors(0);
  }
  
  assert(hidden.device().is_cuda());
  assert(hidden.is_contiguous());
  
  float *probs_fp32 = nullptr;
  float *scaling_factor_fp32 = nullptr;
  if (with_probs) {
    assert(probs.has_value());
    assert(probs.value().device().is_cuda());
    assert(probs.value().is_contiguous());
    assert(probs.value().dtype() == torch::kFloat32);
    auto probs_tensor = probs.value().view(torch::kFloat32);
    probs_fp32 = probs_tensor.data_ptr<float>();
  }
  if (config.token_data_type == TOKEN_DATA_TYPE::UINT8) {
    assert(scaling_factor.has_value());
    assert(scaling_factor.value().device().is_cuda());
    assert(scaling_factor.value().is_contiguous());
    auto scaling_factor_tensor = scaling_factor.value().view(torch::kFloat32);
    scaling_factor_fp32 = scaling_factor_tensor.data_ptr<float>();
  }
  
  // Template function to setup and launch kernel parameters for uint8
  auto launch_uint8_kernel = [&]() {
    auto hidden_uint8 = hidden.view(torch::kUInt8);
    
    hybrid_ep::dispatch_kernel_param_t<uint8_t> param;
    param.attn_input_token = hidden_uint8.data_ptr<uint8_t>();
    param.attn_input_prob = probs_fp32;
    param.attn_input_token_scaling_factor = scaling_factor_fp32;
    
    // Setup output pointers
    for (int i = 0; i < config.num_of_ranks_per_node; i++) {
      param.expert_output_token[i] = reinterpret_cast<uint8_t*>(
          dispatch_buffers.expert_output_token_all_ranks[i]);
      param.expert_output_prob[i] = dispatch_buffers.expert_output_prob_all_ranks[i];
      param.expert_output_scaling_factor[i] = 
          dispatch_buffers.expert_output_scaling_factor_all_ranks[i];
    }
    
    // Setup RDMA parameters
    param.rdma_inter_node_group_token = reinterpret_cast<uint8_t*>(
        dispatch_buffers.rdma_inter_node_group_token);
    param.rdma_inter_node_group_prob = dispatch_buffers.rdma_inter_node_group_prob;
    param.rdma_inter_node_group_scaling_factor = 
        dispatch_buffers.rdma_inter_node_group_scaling_factor;
    param.rdma_inter_node_group_flags = dispatch_buffers.rdma_inter_node_group_flags;
    param.intra_node_write_completion_flags = 
        dispatch_buffers.intra_node_write_completion_flags;
    param.rdma_to_attn_map = rdma_to_attn_map.data_ptr<bool>();
    param.attn_to_rdma_map = attn_to_rdma_map.data_ptr<bool>();
    param.sparse_to_dense_map = sparse_to_dense_map.data_ptr<int32_t>();
    param.local_rank = local_rank;
    param.node_rank = node_rank;
    param.num_of_tokens_per_rank = num_of_tokens_per_rank;
    param.expected_rdma_flag_value = dispatch_buffers.expected_rdma_flag_value;
    param.expected_intra_node_flag_value = dispatch_buffers.expected_intra_node_flag_value;
    
    // Launch kernel
    if (with_probs) {
      config.forward_dispatch_api = true;
      kernel_cache.run_dispatch_kernel<uint8_t>(config, param, stream);
    } else {
      config.forward_dispatch_api = false;
      kernel_cache.run_dispatch_kernel<uint8_t>(config, param, stream);
    }
  };

  // Template function to setup and launch kernel parameters for uint16
  auto launch_uint16_kernel = [&]() {
    auto hidden_uint16 = hidden.view(torch::kUInt16);
    
    hybrid_ep::dispatch_kernel_param_t<uint16_t> param;
    param.attn_input_token = hidden_uint16.data_ptr<uint16_t>();
    param.attn_input_prob = probs_fp32;
    param.attn_input_token_scaling_factor = scaling_factor_fp32;
    
    // Setup output pointers
    for (int i = 0; i < config.num_of_ranks_per_node; i++) {
      param.expert_output_token[i] = reinterpret_cast<uint16_t*>(
          dispatch_buffers.expert_output_token_all_ranks[i]);
      param.expert_output_prob[i] = dispatch_buffers.expert_output_prob_all_ranks[i];
      param.expert_output_scaling_factor[i] = 
          dispatch_buffers.expert_output_scaling_factor_all_ranks[i];
    }
    
    // Setup RDMA parameters
    param.rdma_inter_node_group_token = reinterpret_cast<uint16_t*>(
        dispatch_buffers.rdma_inter_node_group_token);
    param.rdma_inter_node_group_prob = dispatch_buffers.rdma_inter_node_group_prob;
    param.rdma_inter_node_group_scaling_factor = 
        dispatch_buffers.rdma_inter_node_group_scaling_factor;
    param.rdma_inter_node_group_flags = dispatch_buffers.rdma_inter_node_group_flags;
    param.intra_node_write_completion_flags = 
        dispatch_buffers.intra_node_write_completion_flags;
    param.rdma_to_attn_map = rdma_to_attn_map.data_ptr<bool>();
    param.attn_to_rdma_map = attn_to_rdma_map.data_ptr<bool>();
    param.sparse_to_dense_map = sparse_to_dense_map.data_ptr<int32_t>();
    param.local_rank = local_rank;
    param.node_rank = node_rank;
    param.num_of_tokens_per_rank = num_of_tokens_per_rank;
    param.expected_rdma_flag_value = dispatch_buffers.expected_rdma_flag_value;
    param.expected_intra_node_flag_value = dispatch_buffers.expected_intra_node_flag_value;
 
    // Launch kernel
    if (with_probs) {
      config.forward_dispatch_api = true;
      kernel_cache.run_dispatch_kernel<uint16_t>(config, param, stream);
    } else {
      config.forward_dispatch_api = false;
      kernel_cache.run_dispatch_kernel<uint16_t>(config, param, stream);
    }
  };

  // Dispatch based on token data type
  bool kernel_launched = false;
  switch (config.token_data_type) {
    case TOKEN_DATA_TYPE::UINT8:
      launch_uint8_kernel();
      kernel_launched = true;
      break;
    case TOKEN_DATA_TYPE::UINT16:
      launch_uint16_kernel();
      kernel_launched = true;
      break;
    default:
      throw std::runtime_error("Invalid token data type:" + 
                              std::to_string(static_cast<int>(config.token_data_type)));
  }

  if (!kernel_launched) {
    throw std::runtime_error("Failed to launch dispatch kernel for num_of_ranks_per_node: " +
                             std::to_string(config.num_of_ranks_per_node));
  }

  return create_output_tensors(token_count);
}

std::tuple<torch::Tensor, torch::Tensor>
HybridEpBuffer::combine(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
                torch::Tensor sparse_to_dense_map,
                torch::Tensor rdma_to_attn_map, torch::Tensor attn_to_rdma_map,
                int64_t num_of_tokens_per_rank,
                bool with_probs) {

  // The result tensor of the combine kernel
  torch::Tensor combined_tokens, combined_probs;
  combined_tokens =
      torch::empty({num_of_tokens_per_rank, config.hidden_dim},
                   torch::dtype(hidden.dtype()).device(torch::kCUDA));
  if (with_probs) {
    combined_probs =
        torch::empty({num_of_tokens_per_rank,
                      config.num_of_experts_per_rank *
                          config.num_of_ranks_per_node * config.num_of_nodes},
                     torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }

  // Fast return if there are no tokens after combine
  if (num_of_tokens_per_rank == 0) {
    return std::make_tuple(combined_tokens, combined_probs);
  }

  assert(hidden.device().is_cuda());
  assert(hidden.dtype() != torch::kUInt8);
  assert(hidden.is_contiguous());

  float *probs_fp32 = nullptr;
  auto stream = at::cuda::getCurrentCUDAStream();

  if (with_probs) {
    assert(probs.has_value());
    assert(probs.value().device().is_cuda());
    assert(probs.value().is_contiguous());
    assert(probs.value().dtype() == torch::kFloat32);
    assert(probs.value().size(1) ==
           config.num_of_experts_per_rank * config.num_of_ranks_per_node);
    auto probs_tensor = probs.value().view(torch::kFloat32);
    probs_fp32 = probs_tensor.data_ptr<float>();
  }

  // Copy the input tensor to the input buffer
  auto input_sz = hidden.numel() * sizeof(uint16_t);
  CUDA_CHECK(
      cudaMemcpyAsync(combine_buffers.expert_input_token,
                      reinterpret_cast<uint16_t *>(hidden.data_ptr()), input_sz,
                      cudaMemcpyDeviceToDevice, stream));
  if (with_probs) {
    auto probs_sz = probs.value().numel() * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(combine_buffers.expert_input_prob,
                                         probs_fp32, probs_sz,
                                         cudaMemcpyDeviceToDevice, stream));
  }

  bool kernel_launched = false;

  hybrid_ep::combine_kernel_param_t param;
      for (int i = 0; i < config.num_of_ranks_per_node; i++) {
        param.expert_input_token[i] =
            combine_buffers.expert_input_token_all_ranks[i];
        param.expert_input_prob[i] =
            combine_buffers.expert_input_prob_all_ranks[i];
      }
      param.attn_output_token =
          reinterpret_cast<uint16_t *>(combined_tokens.data_ptr());
      param.attn_output_prob =
          with_probs ? combined_probs.data_ptr<float>() : nullptr;
      param.rdma_intra_node_red_token =
          combine_buffers.rdma_intra_node_red_token;
      param.rdma_intra_node_red_prob = combine_buffers.rdma_intra_node_red_prob;
      param.rdma_inter_node_group_token =
          combine_buffers.rdma_inter_node_group_token;
      param.rdma_inter_node_group_prob =
          combine_buffers.rdma_inter_node_group_prob;
      param.rdma_inter_node_group_flags =
          combine_buffers.rdma_inter_node_group_flags;
      param.intra_node_write_completion_flags =
          combine_buffers.intra_node_write_completion_flags;
      param.rdma_to_attn_map = rdma_to_attn_map.data_ptr<bool>();
      param.attn_to_rdma_map = attn_to_rdma_map.data_ptr<bool>();
      param.sparse_to_dense_map = sparse_to_dense_map.data_ptr<int32_t>();
      param.node_rank = this->node_rank;
      param.num_of_tokens_per_rank = num_of_tokens_per_rank;
      param.expected_rdma_flag_value = combine_buffers.expected_rdma_flag_value;
      param.expected_intra_node_flag_value =
          combine_buffers.expected_intra_node_flag_value;

      //   param.dgqps = combine_buffers.dgqps;
      //   param.mr_info = combine_buffers.mr_info;
      // Call the combine kernel directly using template instantiation
      if (with_probs) {
        config.backward_combine_api = true;
        kernel_cache.run_combine_kernel(config, param, stream);
      } else {
        config.backward_combine_api = false;
        kernel_cache.run_combine_kernel(config, param, stream);
      }
      kernel_launched = true;

  if (!kernel_launched) {
    throw std::runtime_error(
        "fail to launch the combine kernel, corresponding num_of_ranks_per_node:" +
        std::to_string(config.num_of_ranks_per_node));
  }

  return std::make_tuple(combined_tokens, combined_probs);
}


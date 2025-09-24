// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#include "hybrid_ep.cuh"

HybridEpBuffer::HybridEpBuffer(HybridEpConfigInstance config, int rank, int group_size,
               int num_of_ranks_per_node)
    : config(config), rank(rank), group_size(group_size),
      num_of_ranks_per_node(num_of_ranks_per_node) {
    this->local_rank = rank % num_of_ranks_per_node;
    this->node_rank = rank / num_of_ranks_per_node;

  allocate_buffer();
}

HybridEpBuffer::~HybridEpBuffer() {
  auto free_buffer = [](void *ptr) {
    if (ptr != nullptr) {
      CUDA_CHECK(cudaFree(ptr));
    }
  };
  
  // Clean up preprocessing buffer
  free_buffer(this->preprocessing_tmp);

  // Clean up dispatch buffers
  std::vector<void*> dispatch_ptrs = {
    dispatch_buffers.rdma_inter_node_group_token,
    dispatch_buffers.rdma_inter_node_group_prob,
    dispatch_buffers.rdma_inter_node_group_scaling_factor,
    dispatch_buffers.rdma_inter_node_group_flags,
    dispatch_buffers.expected_rdma_flag_value,
    dispatch_buffers.expected_intra_node_flag_value
  };
  std::for_each(dispatch_ptrs.begin(), dispatch_ptrs.end(), free_buffer);
  
  device_mem_free(dispatch_buffers.expert_output_token, USE_MNNVLINK);
  device_mem_free(dispatch_buffers.expert_output_prob, USE_MNNVLINK);
  device_mem_free(dispatch_buffers.expert_output_scaling_factor, USE_MNNVLINK);

  if (this->local_rank == 0) {
    device_mem_free(dispatch_buffers.intra_node_write_completion_flags, USE_MNNVLINK);
  } else {
    close_device_mem_handle(dispatch_buffers.intra_node_write_completion_flags, USE_MNNVLINK);
  }

  for (int i = 0; i < config.num_of_ranks_per_node; i++) {
    if (i != this->local_rank) {
      close_device_mem_handle(dispatch_buffers.expert_output_token_all_ranks[i], USE_MNNVLINK);
      close_device_mem_handle(dispatch_buffers.expert_output_prob_all_ranks[i], USE_MNNVLINK);
      close_device_mem_handle(dispatch_buffers.expert_output_scaling_factor_all_ranks[i], USE_MNNVLINK);
    }
  }
  
  // Clean up dispatch pointer arrays
  delete[] dispatch_buffers.expert_output_token_all_ranks;
  delete[] dispatch_buffers.expert_output_prob_all_ranks;
  delete[] dispatch_buffers.expert_output_scaling_factor_all_ranks;

  // Clean up combine buffers
  std::vector<void*> combine_ptrs = {
    combine_buffers.rdma_intra_node_red_token,
    combine_buffers.rdma_intra_node_red_prob,
    combine_buffers.rdma_inter_node_group_token,
    combine_buffers.rdma_inter_node_group_prob,
    combine_buffers.rdma_inter_node_group_flags,
    combine_buffers.expected_rdma_flag_value,
    combine_buffers.expected_intra_node_flag_value
  };
  std::for_each(combine_ptrs.begin(), combine_ptrs.end(), free_buffer);

  device_mem_free(combine_buffers.expert_input_token, USE_MNNVLINK);
  device_mem_free(combine_buffers.expert_input_prob, USE_MNNVLINK);
  
  if (this->local_rank == 0) {
    device_mem_free(combine_buffers.intra_node_write_completion_flags, USE_MNNVLINK);
  } else {
    close_device_mem_handle(combine_buffers.intra_node_write_completion_flags, USE_MNNVLINK);
  }
  
  for (int i = 0; i < config.num_of_ranks_per_node; i++) {
    if (i != this->local_rank) {
      close_device_mem_handle(combine_buffers.expert_input_token_all_ranks[i], USE_MNNVLINK);
      close_device_mem_handle(combine_buffers.expert_input_prob_all_ranks[i], USE_MNNVLINK);
    }
  }
  // Clean up combine pointer arrays
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
  device_mem_malloc((void**)&dispatch_buffers.expert_output_token, expert_output_token_elts * sizeof_token_data_type, USE_MNNVLINK);
  device_mem_malloc((void**)&dispatch_buffers.expert_output_prob, expert_output_prob_elts * sizeof(float), USE_MNNVLINK);
  device_mem_malloc((void**)&dispatch_buffers.expert_output_scaling_factor, expert_output_scaling_factor_elts * sizeof(float), USE_MNNVLINK);
  
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
  if (this->local_rank == 0) {
    device_mem_malloc((void**)&dispatch_buffers.intra_node_write_completion_flags, sizeof(uint32_t), USE_MNNVLINK);
  }
  
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.expected_intra_node_flag_value, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(dispatch_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(dispatch_buffers.expected_intra_node_flag_value, 0, sizeof(uint32_t)));

  // Create IPC memory handles
  MemHandle handles[4];
  get_device_mem_handle(&handles[0], dispatch_buffers.expert_output_token, USE_MNNVLINK);
  get_device_mem_handle(&handles[1], dispatch_buffers.expert_output_prob, USE_MNNVLINK);
  get_device_mem_handle(&handles[2], dispatch_buffers.expert_output_scaling_factor, USE_MNNVLINK);
  if (this->local_rank == 0) {
    get_device_mem_handle(&handles[3], dispatch_buffers.intra_node_write_completion_flags, USE_MNNVLINK);
  }
  
  // Pack handles into tensor
  dispatch_memory_handles = torch::empty({static_cast<int64_t>(sizeof(handles))},
                                        torch::dtype(torch::kUInt8).device(torch::kCPU));
  memcpy(dispatch_memory_handles.data_ptr<uint8_t>(), handles, sizeof(handles));
}

void HybridEpBuffer::allocate_buffer_for_combine() {
  // Calculate buffer sizes
  auto expert_input_token_elts = max_num_of_tokens_for_experts * config.hidden_dim;
  auto expert_input_prob_elts = max_num_of_tokens_for_experts *
                                (config.num_of_experts_per_rank * config.num_of_ranks_per_node);
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
  device_mem_malloc((void**)&combine_buffers.expert_input_token, expert_input_token_elts * sizeof(uint16_t), USE_MNNVLINK);
  device_mem_malloc((void**)&combine_buffers.expert_input_prob, expert_input_prob_elts * sizeof(float), USE_MNNVLINK);

  // Allocate RDMA buffers
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
  if (this->local_rank == 0) {
    device_mem_malloc((void**)&combine_buffers.intra_node_write_completion_flags, sizeof(uint32_t), USE_MNNVLINK);
  }
  
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.expected_intra_node_flag_value, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(combine_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(combine_buffers.expected_intra_node_flag_value, 0, sizeof(uint32_t)));

  // Create IPC memory handles
  MemHandle handles[3];
  get_device_mem_handle(&handles[0], combine_buffers.expert_input_token, USE_MNNVLINK);
  get_device_mem_handle(&handles[1], combine_buffers.expert_input_prob, USE_MNNVLINK);
  if (this->local_rank == 0) {
    get_device_mem_handle(&handles[2], combine_buffers.intra_node_write_completion_flags, USE_MNNVLINK);
  }

  // Pack handles into tensor
  combine_memory_handles = torch::empty({static_cast<int64_t>(sizeof(handles))},
                                       torch::dtype(torch::kUInt8).device(torch::kCPU));
  memcpy(combine_memory_handles.data_ptr<uint8_t>(), handles, sizeof(handles));
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
  allocate_buffer_for_dispatch();
  allocate_buffer_for_combine();
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

void HybridEpBuffer::update_num_of_tokens_per_rank(int num_of_tokens_per_rank) {
  config.num_of_tokens_per_rank = num_of_tokens_per_rank;
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
    open_device_mem_handle((void**)(&dispatch_buffers.intra_node_write_completion_flags),
                           &intra_node_write_completion_flags_handle, USE_MNNVLINK);
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
      open_device_mem_handle((void**)(&dispatch_buffers.expert_output_token_all_ranks[i]),
                             &expert_output_token_handle, USE_MNNVLINK);
      open_device_mem_handle((void**)(&dispatch_buffers.expert_output_prob_all_ranks[i]),
                             &expert_output_prob_handle, USE_MNNVLINK);
      open_device_mem_handle((void**)(&dispatch_buffers.expert_output_scaling_factor_all_ranks[i]),
                             &expert_output_scaling_factor_handle, USE_MNNVLINK);
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
    open_device_mem_handle((void**)(&combine_buffers.intra_node_write_completion_flags),
                           &intra_node_write_completion_flags_handle, USE_MNNVLINK);
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
      open_device_mem_handle((void**)(&combine_buffers.expert_input_token_all_ranks[i]),
                             &expert_input_token_handle, USE_MNNVLINK);
      open_device_mem_handle((void**)(&combine_buffers.expert_input_prob_all_ranks[i]),
                             &expert_input_prob_handle, USE_MNNVLINK);
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
HybridEpBuffer::metadata_preprocessing(torch::Tensor routing_map, int64_t node_rank,
                               int64_t local_rank) {
  assert(routing_map.device().is_cuda());
  assert(routing_map.is_contiguous());

  // padding for the routing map
  const int rdma_to_attn_map_size_per_node = (((config.num_of_tokens_per_rank - 1) / 16) + 1) * 16;

  // Construt the output tensor of the metadata preprocessing kernel.
  auto sparse_to_dense_map =
      torch::empty({config.num_of_tokens_per_rank * config.num_of_nodes,
                    config.num_of_ranks_per_node},
                   torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto rdma_to_attn_map =
      torch::empty({rdma_to_attn_map_size_per_node, config.num_of_nodes},
                   torch::dtype(torch::kBool).device(torch::kCUDA));
  auto attn_to_rdma_map =
      torch::empty({config.num_of_tokens_per_rank, config.num_of_nodes - 1},
                   torch::dtype(torch::kBool).device(torch::kCUDA));
  auto num_of_tokens_for_experts =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto local_expert_routing_map = torch::empty(
      {config.num_of_tokens_per_rank * config.num_of_ranks_per_node * config.num_of_nodes, config.num_of_experts_per_rank},
      torch::dtype(torch::kBool).device(torch::kCUDA));

  hybrid_ep::hybrid_ep<HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_RANKS_PER_NODE,
      NUM_OF_NODES, NUM_OF_EXPERTS_PER_RANK>::metadata_preprocessing<NUM_THREADS_PER_BLOCK_PREPROCESSING_API, NUM_OF_BLOCKS_PREPROCESSING_API>(
      routing_map.data_ptr<bool>(), this->preprocessing_tmp,
      sparse_to_dense_map.data_ptr<int32_t>(),
      rdma_to_attn_map.data_ptr<bool>(), attn_to_rdma_map.data_ptr<bool>(),
      num_of_tokens_for_experts.data_ptr<int32_t>(),
      local_expert_routing_map.data_ptr<bool>(), static_cast<int>(node_rank),
      static_cast<int>(local_rank), static_cast<int>(config.num_of_tokens_per_rank), at::cuda::getCurrentCUDAStream());

  return std::make_tuple(sparse_to_dense_map, rdma_to_attn_map,
                         attn_to_rdma_map, num_of_tokens_for_experts,
                         local_expert_routing_map);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
HybridEpBuffer::dispatch(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
                 c10::optional<torch::Tensor> scaling_factor,
                 torch::Tensor sparse_to_dense_map,
                 torch::Tensor rdma_to_attn_map, torch::Tensor attn_to_rdma_map,
                 int64_t num_of_tokens_for_experts, bool with_probs) {

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
    assert(NUM_OF_RANKS_PER_NODE == config.num_of_ranks_per_node);
    
    hybrid_ep::dispatch_kernel_param_t<uint8_t, NUM_OF_RANKS_PER_NODE> param;
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
    param.num_of_tokens_per_rank = config.num_of_tokens_per_rank;
    param.expected_rdma_flag_value = dispatch_buffers.expected_rdma_flag_value;
    param.expected_intra_node_flag_value = dispatch_buffers.expected_intra_node_flag_value;
    
    // Launch kernel
    if (with_probs) {
      hybrid_ep::hybrid_ep<HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_RANKS_PER_NODE, 
                          NUM_OF_NODES, NUM_OF_EXPERTS_PER_RANK>
      ::dispatch<uint8_t, NUM_OF_STAGES_DISPATCH_API, NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API,
                NUM_OF_BLOCKS_DISPATCH_API, true, DEVICE_SIDE_SYNC_DISPATCH_API>(param, stream);
    } else {
      hybrid_ep::hybrid_ep<HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_RANKS_PER_NODE, 
                          NUM_OF_NODES, NUM_OF_EXPERTS_PER_RANK>
      ::dispatch<uint8_t, NUM_OF_STAGES_DISPATCH_API, NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API,
                NUM_OF_BLOCKS_DISPATCH_API, false, DEVICE_SIDE_SYNC_DISPATCH_API>(param, stream);
    }
  };

  // Template function to setup and launch kernel parameters for uint16
  auto launch_uint16_kernel = [&]() {
    auto hidden_uint16 = hidden.view(torch::kUInt16);
    assert(NUM_OF_RANKS_PER_NODE == config.num_of_ranks_per_node);
    
    hybrid_ep::dispatch_kernel_param_t<uint16_t, NUM_OF_RANKS_PER_NODE> param;
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
    param.num_of_tokens_per_rank = config.num_of_tokens_per_rank;
    param.expected_rdma_flag_value = dispatch_buffers.expected_rdma_flag_value;
    param.expected_intra_node_flag_value = dispatch_buffers.expected_intra_node_flag_value;
 
    // Launch kernel
    if (with_probs) {
      hybrid_ep::hybrid_ep<HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_RANKS_PER_NODE, 
                          NUM_OF_NODES, NUM_OF_EXPERTS_PER_RANK>
      ::dispatch<uint16_t, NUM_OF_STAGES_DISPATCH_API, NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API,
                NUM_OF_BLOCKS_DISPATCH_API, true, DEVICE_SIDE_SYNC_DISPATCH_API>(param, stream);
    } else {
      hybrid_ep::hybrid_ep<HIDDEN_DIM, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_RANKS_PER_NODE, 
                          NUM_OF_NODES, NUM_OF_EXPERTS_PER_RANK>
      ::dispatch<uint16_t, NUM_OF_STAGES_DISPATCH_API, NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API,
                NUM_OF_BLOCKS_DISPATCH_API, false, DEVICE_SIDE_SYNC_DISPATCH_API>(param, stream);
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
                bool with_probs) {

  // The result tensor of the combine kernel
  torch::Tensor combined_tokens, combined_probs;
  combined_tokens =
      torch::empty({config.num_of_tokens_per_rank, config.hidden_dim},
                   torch::dtype(hidden.dtype()).device(torch::kCUDA));
  if (with_probs) {
    combined_probs =
        torch::empty({config.num_of_tokens_per_rank,
                      config.num_of_experts_per_rank *
                          config.num_of_ranks_per_node * config.num_of_nodes},
                     torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }

  // Fast return if there are no tokens after combine
  if (config.num_of_tokens_per_rank == 0) {
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

  assert(NUM_OF_RANKS_PER_NODE == config.num_of_ranks_per_node);
  hybrid_ep::combine_kernel_param_t<NUM_OF_RANKS_PER_NODE> param;
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
      param.num_of_tokens_per_rank = config.num_of_tokens_per_rank;
      param.expected_rdma_flag_value = combine_buffers.expected_rdma_flag_value;
      param.expected_intra_node_flag_value =
          combine_buffers.expected_intra_node_flag_value;

      //   param.dgqps = combine_buffers.dgqps;
      //   param.mr_info = combine_buffers.mr_info;
      // Call the combine kernel directly using template instantiation
      if (with_probs) {
          hybrid_ep::hybrid_ep<
              HIDDEN_DIM, 
              MAX_NUM_OF_TOKENS_PER_RANK, 
              NUM_OF_RANKS_PER_NODE, 
              NUM_OF_NODES, 
              NUM_OF_EXPERTS_PER_RANK
          >::combine<
              NUM_OF_STAGES_G2S_COMBINE_API, 
              NUM_OF_STAGES_S2G_COMBINE_API, 
              NUM_OF_TOKENS_PER_CHUNK_COMBINE_API, 
              NUM_OF_TOKENS_PER_GROUP_COMBINE_API, 
              NUM_OF_BLOCKS_COMBINE_API, 
              NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_COMBINE_API, 
              true, 
              DEVICE_SIDE_SYNC_COMBINE_API
          >(param, stream);
      } else {
          hybrid_ep::hybrid_ep<
              HIDDEN_DIM, 
              MAX_NUM_OF_TOKENS_PER_RANK, 
              NUM_OF_RANKS_PER_NODE, 
              NUM_OF_NODES, 
              NUM_OF_EXPERTS_PER_RANK
          >::combine<
              NUM_OF_STAGES_G2S_COMBINE_API, 
              NUM_OF_STAGES_S2G_COMBINE_API, 
              NUM_OF_TOKENS_PER_CHUNK_COMBINE_API, 
              NUM_OF_TOKENS_PER_GROUP_COMBINE_API, 
              NUM_OF_BLOCKS_COMBINE_API, 
              NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_COMBINE_API, 
              false, 
              DEVICE_SIDE_SYNC_COMBINE_API
          >(param, stream);
      }
      kernel_launched = true;

  if (!kernel_launched) {
    throw std::runtime_error(
        "fail to launch the combine kernel, corresponding num_of_ranks_per_node:" +
        std::to_string(config.num_of_ranks_per_node));
  }

  return std::make_tuple(combined_tokens, combined_probs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "HybridEP, efficiently enable the expert-parallel communication in "
            "the Hopper+ architectures";

  pybind11::enum_<TOKEN_DATA_TYPE>(m, "TokenDataType")
      .value("UINT16", TOKEN_DATA_TYPE::UINT16)
      .value("UINT8", TOKEN_DATA_TYPE::UINT8)
      .export_values() // So we can use hybrid_ep_cpp.TYPE instead of the
                       // hybrid_ep_cpp.TOKEN_DATA_TYPE.TYPE
      .def("__str__",
           [](const TOKEN_DATA_TYPE &type) { return type_to_string(type); });

  pybind11::class_<HybridEpConfigInstance>(m, "HybridEpConfigInstance")
      .def(py::init<>())
      // Hybrid-ep Config
      .def_readwrite("hidden_dim", &HybridEpConfigInstance::hidden_dim)
      .def_readwrite("num_of_tokens_per_rank",
                     &HybridEpConfigInstance::num_of_tokens_per_rank)
      .def_readwrite("max_num_of_tokens_per_rank",
                     &HybridEpConfigInstance::max_num_of_tokens_per_rank)
      .def_readwrite("num_of_experts_per_rank",
                     &HybridEpConfigInstance::num_of_experts_per_rank)
      .def_readwrite("num_of_ranks_per_node",
                     &HybridEpConfigInstance::num_of_ranks_per_node)
      .def_readwrite("num_of_nodes", &HybridEpConfigInstance::num_of_nodes)
      // Metadata-preprocessing API Config
      .def_readwrite(
          "num_of_threads_per_block_preprocessing_api",
          &HybridEpConfigInstance::num_of_threads_per_block_preprocessing_api)
      .def_readwrite("num_of_blocks_preprocessing_api",
                     &HybridEpConfigInstance::num_of_blocks_preprocessing_api)
      // Dispatch API Config
      .def_readwrite("token_data_type", &HybridEpConfigInstance::token_data_type)
      .def_readwrite("num_of_stages_dispatch_api",
                     &HybridEpConfigInstance::num_of_stages_dispatch_api)
      .def_readwrite("num_of_tokens_per_chunk_dispatch_api",
                     &HybridEpConfigInstance::num_of_tokens_per_chunk_dispatch_api)
      .def_readwrite("num_of_blocks_dispatch_api",
                     &HybridEpConfigInstance::num_of_blocks_dispatch_api)
      .def_readwrite("forward_dispatch_api",
                     &HybridEpConfigInstance::forward_dispatch_api)
      .def_readwrite("device_side_sync_dispatch_api",
                     &HybridEpConfigInstance::device_side_sync_dispatch_api)
      // Combine API Config
      .def_readwrite("num_of_stages_g2s_combine_api",
                     &HybridEpConfigInstance::num_of_stages_g2s_combine_api)
      .def_readwrite("num_of_stages_s2g_combine_api",
                     &HybridEpConfigInstance::num_of_stages_s2g_combine_api)
      .def_readwrite("num_of_tokens_per_chunk_combine_api",
                     &HybridEpConfigInstance::num_of_tokens_per_chunk_combine_api)
      .def_readwrite("num_of_tokens_per_group_combine_api",
                     &HybridEpConfigInstance::num_of_tokens_per_group_combine_api)
      .def_readwrite("num_of_blocks_combine_api",
                     &HybridEpConfigInstance::num_of_blocks_combine_api)
      .def_readwrite(
          "num_of_additional_in_flight_s2g_combine_api",
          &HybridEpConfigInstance::num_of_additional_in_flight_s2g_combine_api)
      .def_readwrite("backward_combine_api",
                     &HybridEpConfigInstance::backward_combine_api)
      .def_readwrite("device_side_sync_combine_api",
                     &HybridEpConfigInstance::device_side_sync_combine_api)
      .def("__repr__", [](const HybridEpConfigInstance &config) {
        return "<HybridEpConfigInstance hidden_dim=" +
               std::to_string(config.hidden_dim) + " max_num_of_tokens_per_rank=" +
               std::to_string(config.max_num_of_tokens_per_rank) +
               " token_data_type=" + type_to_string(config.token_data_type) +
               ">";
      });

  pybind11::class_<HybridEpBuffer>(m, "HybridEpBuffer")
      .def(py::init<HybridEpConfigInstance, int, int, int>())
      .def("exchange_ipc_address", &HybridEpBuffer::exchange_ipc_address)
      .def("update_num_of_tokens_per_rank", &HybridEpBuffer::update_num_of_tokens_per_rank,
           py::arg("num_of_tokens_per_rank"))
      .def("metadata_preprocessing", &HybridEpBuffer::metadata_preprocessing,
           py::kw_only(), py::arg("routing_map"), py::arg("node_rank"),
           py::arg("local_rank"))
      .def("dispatch", &HybridEpBuffer::dispatch, py::kw_only(), py::arg("hidden"),
           py::arg("probs") = c10::nullopt,
           py::arg("scaling_factor") = c10::nullopt,
           py::arg("sparse_to_dense_map"), py::arg("rdma_to_attn_map"),
           py::arg("attn_to_rdma_map"),
           py::arg("num_of_tokens_for_experts") = -1, py::arg("with_probs"))
      .def("combine", &HybridEpBuffer::combine, py::kw_only(), py::arg("hidden"),
           py::arg("probs") = c10::nullopt, py::arg("sparse_to_dense_map"),
           py::arg("rdma_to_attn_map"), py::arg("attn_to_rdma_map"),
           py::arg("with_probs"));
}

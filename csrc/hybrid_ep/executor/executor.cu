// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "executor.cuh"

Executor::Executor(int local_rank, int node_rank) : local_rank(local_rank), node_rank(node_rank), kernel_cache(local_rank) {}  

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Executor::metadata_preprocess_core(
    HybridEpConfigInstance config, 
    hybrid_ep::tmp_state_t *preprocessing_tmp,
    torch::Tensor global_routing_map,
    int num_of_tokens_per_rank
) {
  nvtxRangePushA("metadata_preprocess_core in hybrid-ep");
  // padding for the routing map
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;

  auto num_of_expert = global_routing_map.size(-1);
  assert(num_of_expert == config.num_of_experts_per_rank * config.num_of_ranks_per_node * config.num_of_nodes);

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
      config, global_routing_map.data_ptr<bool>(), preprocessing_tmp,
      sparse_to_dense_map.data_ptr<int32_t>(),
      rdma_to_attn_map.data_ptr<bool>(), attn_to_rdma_map.data_ptr<bool>(),
      num_of_tokens_for_experts.data_ptr<int32_t>(),
      local_expert_routing_map.data_ptr<bool>(), static_cast<int>(node_rank),
      static_cast<int>(local_rank), num_of_tokens_per_rank, at::cuda::getCurrentCUDAStream());

  nvtxRangePop();  // End of metadata_preprocess_core nvtx range
  return std::make_tuple(sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map, num_of_tokens_for_experts, local_expert_routing_map);
}

void Executor::dispatch_preprocess(HybridEpConfigInstance config, DispatchBuffers& dispatch_buffers, DispatchArgs& args) {
    // Empty now, will be filled with D2D in the inter-node case
    nvtxRangePushA("dispatch_preprocess in hybrid-ep");
    nvtxRangePop();  // End of dispatch_preprocess nvtx range
}

template void Executor::dispatch_core<uint8_t>(HybridEpConfigInstance config, DispatchBuffers& dispatch_buffers, DispatchArgs& args);
template void Executor::dispatch_core<uint16_t>(HybridEpConfigInstance config, DispatchBuffers& dispatch_buffers, DispatchArgs& args);

template<typename DType>
void Executor::dispatch_core(HybridEpConfigInstance config, DispatchBuffers& dispatch_buffers, DispatchArgs& args) {
    nvtxRangePushA("dispatch_core in hybrid-ep");

    hybrid_ep::dispatch_kernel_param_t<DType> param;
    // Setup input pointers
    param.attn_input_token = reinterpret_cast<DType*>(args.hidden.data_ptr());
    param.attn_input_prob = (config.forward_dispatch_api) ? reinterpret_cast<float*>(args.probs.data_ptr()) : nullptr;
    param.attn_input_token_scaling_factor = (config.token_data_type == TOKEN_DATA_TYPE::UINT8) ? reinterpret_cast<float*>(args.scaling_factor.data_ptr()) : nullptr;
    
    // Setup output pointers
    for (int i = 0; i < config.num_of_ranks_per_node; i++) {
      param.expert_output_token[i] = reinterpret_cast<DType*>(
          dispatch_buffers.expert_output_token_all_ranks[i]);
      param.expert_output_prob[i] = dispatch_buffers.expert_output_prob_all_ranks[i];
      param.expert_output_scaling_factor[i] = 
          dispatch_buffers.expert_output_scaling_factor_all_ranks[i];
    }
    
    // Setup local buffer pointers
    param.rdma_inter_node_group_token = reinterpret_cast<DType*>(
        dispatch_buffers.rdma_inter_node_group_token);
    param.rdma_inter_node_group_prob = dispatch_buffers.rdma_inter_node_group_prob;
    param.rdma_inter_node_group_scaling_factor = 
        dispatch_buffers.rdma_inter_node_group_scaling_factor;
    param.rdma_inter_node_group_flags = dispatch_buffers.rdma_inter_node_group_flags;
    param.intra_node_write_completion_flags = 
        dispatch_buffers.intra_node_write_completion_flags;
    param.rdma_to_attn_map = args.rdma_to_attn_map.data_ptr<bool>();
    param.attn_to_rdma_map = args.attn_to_rdma_map.data_ptr<bool>();
    param.sparse_to_dense_map = args.sparse_to_dense_map.data_ptr<int32_t>();

    // Misc
    param.local_rank = local_rank;
    param.node_rank = node_rank;
    param.num_of_tokens_per_rank = args.num_of_tokens_per_rank;
    param.expected_rdma_flag_value = dispatch_buffers.expected_rdma_flag_value;
    param.expected_intra_node_flag_value = dispatch_buffers.expected_intra_node_flag_value;
    
    // Launch kernel
    kernel_cache.run_dispatch_kernel<DType>(config, param, args.stream);

    nvtxRangePop();  // End of dispatch_core nvtx range
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor>
Executor::dispatch_postprocess(HybridEpConfigInstance config, DispatchBuffers& dispatch_buffers, DispatchArgs& args) {
    nvtxRangePushA("dispatch_postprocess in hybrid-ep");

    // Create and return output tensors
    // The output tensor of the dispatch kernel.
    torch::Tensor dispatched_tokens;
    c10::optional<torch::Tensor> dispatched_probs;
    c10::optional<torch::Tensor> dispatched_scaling_factor;
    // Possible ouput from the permute part
    torch::Tensor row_id_map, tokens_per_expert;

    if(args.num_dispatched_tokens == 0 ) {
        // Fast return empty tensors if there are no tokens to dispatch
        dispatched_tokens = torch::empty({0, config.hidden_dim}, torch::dtype(args.hidden.dtype()).device(torch::kCUDA));
        if(config.forward_dispatch_api) {
            dispatched_probs = torch::empty({0}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        }
        if(config.token_data_type == TOKEN_DATA_TYPE::UINT8) {
            dispatched_scaling_factor = torch::empty({0, config.hidden_dim / 128}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        }
        row_id_map = torch::empty({0, config.num_of_experts_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        tokens_per_expert = torch::full({config.num_of_experts_per_rank}, 0, torch::dtype(torch::kInt32).device(torch::kCUDA));
        return std::make_tuple(dispatched_tokens, dispatched_probs, dispatched_scaling_factor, row_id_map, tokens_per_expert);
    }

    if(args.enable_permute) {
        // Use permute kernel to avoid standalone D2D memory copy
        assert(args.num_dispatched_tokens_tensor.has_value());

        int num_dispatched_tokens = args.num_dispatched_tokens;
        int num_permuted_tokens = args.num_permuted_tokens;
        torch::Tensor num_dispatched_tokens_tensor = args.num_dispatched_tokens_tensor.value();

        if (args.row_id_map.has_value()) {
          // The row_id_map is valid, which means that the cached model is used.
          // Then we will use the values in args directly.
          assert(args.num_permuted_tokens >= 0);
          row_id_map = args.row_id_map.value();
        } else {
          // Otherwise, we will compute the row_id_map/tokens_per_expert by preprocessing kernel.
          assert(args.local_expert_routing_map.has_value());
    
          std::tie(row_id_map, tokens_per_expert) = permute_processing(
              args.local_expert_routing_map.value().data_ptr<bool>(), num_dispatched_tokens_tensor,
              num_dispatched_tokens, config.num_of_experts_per_rank, args.pad_multiple, args.stream);
    
          // If use pre-allocated sync-free mode, we use the value in args directly.
          // otherwise, we will compute the num_permuted_tokens by summing the tokens_per_expert.
          if (num_permuted_tokens < 0) {
            if (args.use_host_meta) {
              tokens_per_expert = tokens_per_expert.cpu();
            }
            num_permuted_tokens = tokens_per_expert.sum().item<int64_t>();
          }
        }
    
        if (config.token_data_type == TOKEN_DATA_TYPE::UINT16) {
          std::tie(dispatched_tokens, dispatched_scaling_factor, dispatched_probs) = permute_launcher(
              reinterpret_cast<uint16_t*>(dispatch_buffers.expert_output_token),
              reinterpret_cast<float*>(dispatch_buffers.expert_output_prob),
              reinterpret_cast<float*>(dispatch_buffers.expert_output_scaling_factor), row_id_map,
              config.hidden_dim, config.hidden_dim / 128, local_rank, config.num_of_ranks_per_node,
              config.num_of_experts_per_rank, num_dispatched_tokens_tensor, num_dispatched_tokens,
              num_permuted_tokens, args.pad_multiple,
              false,  // use_fp8
              config.forward_dispatch_api, args.hidden.options(), args.stream);
    
        } else {
          std::tie(dispatched_tokens, dispatched_scaling_factor, dispatched_probs) = permute_launcher(
              reinterpret_cast<uint8_t*>(dispatch_buffers.expert_output_token),
              reinterpret_cast<float*>(dispatch_buffers.expert_output_prob),
              reinterpret_cast<float*>(dispatch_buffers.expert_output_scaling_factor), row_id_map,
              config.hidden_dim, config.hidden_dim / 128, local_rank, config.num_of_ranks_per_node,
              config.num_of_experts_per_rank, num_dispatched_tokens_tensor, num_dispatched_tokens,
              num_permuted_tokens, args.pad_multiple,
              true,  // use_fp8
              config.forward_dispatch_api, args.hidden.options(), args.stream);
        }    
    }else {
        // D2D copy the result to the pytorch tensor
        size_t sizeof_token_data_type = get_token_data_type_size(dispatch_buffers.data_type);
        dispatched_tokens = torch::empty({args.num_dispatched_tokens, config.hidden_dim}, torch::dtype(args.hidden.dtype()).device(torch::kCUDA));
        auto res_sz = args.num_dispatched_tokens * config.hidden_dim * sizeof_token_data_type;
        CUDA_CHECK(cudaMemcpyAsync(dispatched_tokens.data_ptr(), dispatch_buffers.expert_output_token, res_sz, cudaMemcpyDeviceToDevice, args.stream));

        if(config.forward_dispatch_api) {
            dispatched_probs = torch::empty({args.num_dispatched_tokens,
                config.num_of_experts_per_rank * config.num_of_ranks_per_node},
                            torch::dtype(torch::kFloat32).device(torch::kCUDA));
            auto probs_sz = args.num_dispatched_tokens * config.num_of_experts_per_rank * config.num_of_ranks_per_node * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(dispatched_probs.value().data_ptr<float>(),
                                        dispatch_buffers.expert_output_prob,
                                        probs_sz, cudaMemcpyDeviceToDevice, args.stream));
        }

        if(config.token_data_type == TOKEN_DATA_TYPE::UINT8) {
            dispatched_scaling_factor = torch::empty({
                    args.num_dispatched_tokens, 
                    config.hidden_dim / 128}, 
                    torch::dtype(torch::kFloat32).device(torch::kCUDA));
            auto scaling_factor_sz = args.num_dispatched_tokens * config.hidden_dim / 128 * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(dispatched_scaling_factor.value().data_ptr<float>(),
                                        dispatch_buffers.expert_output_scaling_factor,
                                        scaling_factor_sz, cudaMemcpyDeviceToDevice, args.stream));
        }
    }

    nvtxRangePop();  // End of dispatch_postprocess nvtx range
    return std::make_tuple(dispatched_tokens, dispatched_probs, dispatched_scaling_factor, row_id_map, tokens_per_expert);
}

void Executor::combine_preprocess(HybridEpConfigInstance config, CombineBuffers& combine_buffers, CombineArgs& args) {
    nvtxRangePushA("combine_preprocess in hybrid-ep");

    if(args.enable_unpermute) {
        // If enable_unpermute is true, unpermute the token/probs according to the
        // routing map.
        assert(args.row_id_map.has_value());
        assert(args.num_dispatched_tokens_tensor.has_value());
    
        auto num_dispatched_tokens = args.num_dispatched_tokens;
        auto num_dispatched_tokens_tensor = args.num_dispatched_tokens_tensor.value();
        // If args.num_dispatched_tokens >= 0, which means that the sync-free model is used.
        // Otherwise, we will use the values in args.num_dispatched_tokens_tensor.
        if (num_dispatched_tokens < 0) {
          num_dispatched_tokens = num_dispatched_tokens_tensor.item<int>();
        }
    
        unpermute_launcher(
            args.hidden, args.probs, reinterpret_cast<uint16_t*>(combine_buffers.expert_input_token),
            reinterpret_cast<float*>(combine_buffers.expert_input_prob), args.row_id_map.value(),
            config.num_of_experts_per_rank, num_dispatched_tokens_tensor, num_dispatched_tokens,
            args.pad_multiple, config.hidden_dim, local_rank, config.num_of_ranks_per_node, 
            config.backward_combine_api, args.stream);
    
    }else{
        // Copy the input tensor to the input buffer
        auto input_sz = args.hidden.numel() * sizeof(uint16_t);
        CUDA_CHECK(
            cudaMemcpyAsync(combine_buffers.expert_input_token,
                            reinterpret_cast<uint16_t *>(args.hidden.data_ptr()), input_sz,
                            cudaMemcpyDeviceToDevice, args.stream));
        if (config.backward_combine_api) {
            auto probs_sz = args.probs.numel() * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(combine_buffers.expert_input_prob,
                                                reinterpret_cast<float*>(args.probs.data_ptr()), probs_sz,
                                                cudaMemcpyDeviceToDevice, args.stream));
        }
    }
    nvtxRangePop();  // End of combine_preprocess nvtx range
}

void Executor::combine_core(HybridEpConfigInstance config, CombineBuffers& combine_buffers, CombineArgs& args) {
    nvtxRangePushA("combine_core in hybrid-ep");
    hybrid_ep::combine_kernel_param_t param;
    
    // Setup input pointers
    for (int i = 0; i < config.num_of_ranks_per_node; i++) {
        param.expert_input_token[i] =
            combine_buffers.expert_input_token_all_ranks[i];
        param.expert_input_prob[i] =
            combine_buffers.expert_input_prob_all_ranks[i];
    }

    // Setup output pointers
    param.attn_output_token = args.combined_tokens;
    param.attn_output_prob = (config.backward_combine_api) ? args.combined_probs : nullptr;

    // Setup local buffer pointers
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
    param.rdma_to_attn_map = args.rdma_to_attn_map.data_ptr<bool>();
    param.attn_to_rdma_map = args.attn_to_rdma_map.data_ptr<bool>();
    param.sparse_to_dense_map = args.sparse_to_dense_map.data_ptr<int32_t>();

    // Misc
    param.node_rank = this->node_rank;
    param.num_of_tokens_per_rank = args.num_of_tokens_per_rank;
    param.expected_rdma_flag_value = combine_buffers.expected_rdma_flag_value;
    param.expected_intra_node_flag_value =
        combine_buffers.expected_intra_node_flag_value;

    // Launch kernel
    kernel_cache.run_combine_kernel(config, param, args.stream);
    nvtxRangePop();  // End of combine_core nvtx range
}

void Executor::combine_postprocess(HybridEpConfigInstance config, CombineBuffers& combine_buffers, CombineArgs& args) {
    nvtxRangePushA("combine_postprocess in hybrid-ep");
    // TODO: Implement the combine postprocessing
    nvtxRangePop();  // End of combine_postprocess nvtx range
}

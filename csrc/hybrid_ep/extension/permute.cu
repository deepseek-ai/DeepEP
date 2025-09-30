// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "permute.cuh"

 template std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
 permute_launcher<uint16_t, float, float>(uint16_t* tokens_ptr,
                                          float* probs_ptr,
                                          float* scaling_factor_ptr,
                                          torch::Tensor row_id_map,
                                          int hidden_size,
                                          int scales_per_token,
                                          int local_rank,
                                          int num_ranks_per_node,
                                          int num_of_local_experts,
                                          torch::Tensor num_dispatched_token_tensor,
                                          int num_dispatched_tokens,
                                          int num_permuted_token,
                                          int pad_multiple,
                                          bool use_fp8,
                                          bool with_probs,
                                          torch::TensorOptions token_options,
                                          cudaStream_t stream);
 
 template std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
 permute_launcher<uint8_t, float, float>(uint8_t* tokens_ptr,
                                         float* probs_ptr,
                                         float* scaling_factor_ptr,
                                         torch::Tensor row_id_map,
                                         int hidden_size,
                                         int scales_per_token,
                                         int local_rank,
                                         int num_ranks_per_node,
                                         int num_of_local_experts,
                                         torch::Tensor num_dispatched_token_tensor,
                                         int num_dispatched_tokens,
                                         int num_permuted_token,
                                         int pad_multiple,
                                         bool use_fp8,
                                         bool with_probs,
                                         torch::TensorOptions token_options,
                                         cudaStream_t stream);
 
 template void unpermute_launcher<uint16_t, float>(torch::Tensor permuted_tokens,
                                                   c10::optional<torch::Tensor> permuted_probs,
                                                   uint16_t* tokens_ptr,
                                                   float* probs_ptr,
                                                   torch::Tensor row_id_map,
                                                   int num_of_local_experts,
                                                   torch::Tensor num_dispatched_tokens_tensor,
                                                   int num_dispatched_tokens,
                                                   int pad_multiple,
                                                   int hidden_size,
                                                   int local_rank,
                                                   int num_ranks_per_node,
                                                   bool with_probs,
                                                   cudaStream_t stream);
 /**
  * @brief Permute the tokens to the experts
  * @param routing_map[in] shape: [num_dispatched_tokens, num_of_local_experts],
  * type: bool
  * @param num_dispatched_tokens_ptr[in]
  * @param num_of_local_experts[in]
  * @param workspace_1[in] shape: [rows_workspace_1, num_of_local_experts], type:
  * int
  * @param rows_workspace_1[in] rows_workspace_1 = num_dispatched_tokens /
  * block_size
  * @param workspace_2[in] shape: [rows_workspace_2, num_of_local_experts], type:
  * int
  * @param rows_workspace_2[in] rows_workspace_2 = rows_workspace_1 / block_size
  * @param pad_multiple[in]
  * @param tokens_per_expert[out] shape: [num_of_local_experts], type: int
  * @param row_id_map[out] shape: [num_dispatched_tokens, num_of_local_experts],
  * type: int, 0 means the token shoule be dispatched, < 0 mean it is a padded
  * token, >0 values means the offset in the permuted tokens buffer
  */
 template <const int block_size = 512, const int warp_size = 32>
 __global__ void permute_processing_kernel(bool* routing_map,
                                           int* num_dispatched_tokens_ptr,
                                           int num_of_local_experts,
                                           int* workspace_1,
                                           int rows_workspace_1,
                                           int* workspace_2,
                                           int rows_workspace_2,
                                           int pad_multiple,
                                           int* tokens_per_expert,
                                           int* row_id_map) {
   /**
    * Common variables
    */
   auto grid = cooperative_groups::this_grid();
   using BlockScan = cub::BlockScan<int32_t, block_size>;
   __shared__ typename BlockScan::TempStorage temp_storage;
   extern __shared__ int shmem_in_permute_processing_kernel[];
   int num_dispatched_tokens = *num_dispatched_tokens_ptr;
 
   /**
    * Pass 1: Compute the cumsum for each block, then store the result in the
    * workspace_1 memset the workspace_2 and the tokens_per_expert with 0
    */
   // Memeset part
   for (int i = grid.thread_rank(); i < rows_workspace_2 * num_of_local_experts; i += grid.size())
     workspace_2[i] = 0;
   for (int i = grid.thread_rank(); i < num_of_local_experts; i += grid.size())
     tokens_per_expert[i] = 0;
 
   // tile size [block_size, num_of_local_experts]
   int* tile_pass_1 = reinterpret_cast<int*>(shmem_in_permute_processing_kernel);
   for (int tile_idx = blockIdx.x; tile_idx < rows_workspace_1; tile_idx += gridDim.x) {
     int tile_offset = tile_idx * block_size;
     // Load the routing map to the tile
     for (int i = threadIdx.x; i < block_size * num_of_local_experts; i += block_size) {
       tile_pass_1[i] = (tile_offset + i / num_of_local_experts < num_dispatched_tokens)
                            ? static_cast<int>(routing_map[tile_offset * num_of_local_experts + i])
                            : 0;
     }
     __syncthreads();
 
     // Example for each column: 1,0,1,0,1,1,0 => 1,0,2,0,3,4,0
     for (int i = 0; i < num_of_local_experts; i++) {
       // TO SOLVE: many bank conflicts here
       int32_t in = tile_pass_1[threadIdx.x * num_of_local_experts + i];
       int32_t out, sum;
       BlockScan(temp_storage).InclusiveSum(in, out, sum);
       tile_pass_1[threadIdx.x * num_of_local_experts + i] = in == 1 ? out : 0;
       if (threadIdx.x == 0) {
         workspace_1[tile_idx * num_of_local_experts + i] = sum;
       }
     }
     __syncthreads();
 
     // Update the row_id_map in the local tile
     for (int64_t i = threadIdx.x; i < block_size * num_of_local_experts; i += block_size) {
       if ((tile_offset + i / num_of_local_experts < num_dispatched_tokens)) {
         row_id_map[tile_offset * num_of_local_experts + i] = static_cast<int>(tile_pass_1[i]);
       }
     }
   }
 
   grid.sync();
 
   /**
    * Pass 2: Compute the cumsum for each block in workspace_1
    * Use atomic to compute the prefix sum of the all block-sum, store the result
    * in the workspace_2, update the tokens_per_expert
    */
   // tile size [block_size, num_of_local_experts]
   int* tile_pass_2 = reinterpret_cast<int*>(shmem_in_permute_processing_kernel);
   for (int tile_idx = blockIdx.x; tile_idx < rows_workspace_2; tile_idx += gridDim.x) {
     int tile_offset = tile_idx * block_size;
     // Load the workspace_1 to the tile
     for (int i = threadIdx.x; i < block_size * num_of_local_experts; i += block_size) {
       tile_pass_2[i] = (tile_offset + i / num_of_local_experts < rows_workspace_1)
                            ? workspace_1[tile_offset * num_of_local_experts + i]
                            : 0;
     }
     __syncthreads();
 
     for (int i = 0; i < num_of_local_experts; i++) {
       // TO SOLVE: many bank conflicts here
       int32_t in = tile_pass_2[threadIdx.x * num_of_local_experts + i];
       int32_t out, sum;
       BlockScan(temp_storage).ExclusiveSum(in, out, sum);
       tile_pass_2[threadIdx.x * num_of_local_experts + i] = out;
       // Loop form [tile_idx + 1, num_rows_workspace_2]
       for (int pos = threadIdx.x + tile_idx + 1; pos < rows_workspace_2; pos += block_size) {
         atomicAdd(&workspace_2[pos * num_of_local_experts + i], sum);
       }
       if (threadIdx.x == 0) {
         atomicAdd(&tokens_per_expert[i], static_cast<int>(sum));
       }
     }
     __syncthreads();
 
     // Update the workspace_1 in the local tile
     for (int64_t i = threadIdx.x; i < block_size * num_of_local_experts; i += block_size) {
       if ((tile_offset + i / num_of_local_experts < rows_workspace_1)) {
         workspace_1[tile_offset * num_of_local_experts + i] = tile_pass_2[i];
       }
     }
     __syncthreads();
   }
 
   grid.sync();
 
   // These 2 buffers will be used in both pass 3 and pass 4
   int* tokens_per_expert_shmem = reinterpret_cast<int*>(shmem_in_permute_processing_kernel);
   int* tokens_per_expert_prefix_sum =
       reinterpret_cast<int*>(tokens_per_expert_shmem + num_of_local_experts);
 
   /**
    * Pass 3: compute the prefix sum of the token_per_expert, use
    * token_per_expert. workspace_1, workspace_2 to update the row_id_map
    */
   for (int i = threadIdx.x; i < num_of_local_experts; i += block_size) {
     tokens_per_expert_shmem[i] = tokens_per_expert[i];
     tokens_per_expert_prefix_sum[i] =
         (tokens_per_expert_shmem[i] + pad_multiple - 1) / pad_multiple * pad_multiple;
   }
   __syncthreads();
   int value = threadIdx.x < num_of_local_experts ? tokens_per_expert_prefix_sum[threadIdx.x] : 0;
   BlockScan(temp_storage).ExclusiveSum(value, value);
   if (threadIdx.x < num_of_local_experts) {
     tokens_per_expert_prefix_sum[threadIdx.x] = value;
   }
   __syncthreads();
 
   for (int tile_idx = blockIdx.x; tile_idx < rows_workspace_1; tile_idx += gridDim.x) {
     int tile_offset = tile_idx * block_size;
     for (int64_t i = threadIdx.x; i < block_size * num_of_local_experts; i += block_size) {
       if (tile_offset + i / num_of_local_experts < num_dispatched_tokens) {
         int64_t offset = (tile_offset * num_of_local_experts + i);
         int expert_id = i % num_of_local_experts;
         auto old_value = row_id_map[offset];
         if (old_value != 0) {
           row_id_map[offset] =
               old_value + workspace_1[tile_idx * num_of_local_experts + expert_id] +
               workspace_2[(tile_idx / block_size) * num_of_local_experts + expert_id] +
               tokens_per_expert_prefix_sum[expert_id];
         }
       }
     }
   }
 
   if (pad_multiple <= 0)
     return;
   grid.sync();
 
   /**
    * Pass 4: compute the padding for the tokens_per_expert
    */
   int* num_padded_tokens =
       reinterpret_cast<int*>(tokens_per_expert_shmem + 2 * num_of_local_experts);
   for (int i = threadIdx.x; i < num_of_local_experts; i += block_size) {
     int padded_value =
         (tokens_per_expert_shmem[i] + pad_multiple - 1) / pad_multiple * pad_multiple;
     num_padded_tokens[i] = padded_value - tokens_per_expert_shmem[i];
   }
   __syncthreads();
 
   // each warp handle 1 token here
   for (int i = blockIdx.x; i < pad_multiple; i += gridDim.x) {
     int64_t offset = (i + num_dispatched_tokens) * num_of_local_experts;
     for (int j = 0; j < num_of_local_experts; j++) {
       if (i < num_padded_tokens[j]) {
         row_id_map[offset + j] =
             -(tokens_per_expert_shmem[j] + tokens_per_expert_prefix_sum[j] + i + 1);
       } else {
         row_id_map[offset + j] = 0;
       }
     }
   }
 
   if (blockIdx.x == 0) {
     for (int i = threadIdx.x; i < num_of_local_experts; i += block_size) {
       tokens_per_expert[i] = tokens_per_expert_shmem[i] + num_padded_tokens[i];
     }
   }
 }
 
 std::tuple<torch::Tensor, torch::Tensor> permute_processing(
     bool* routing_map,
     torch::Tensor num_dispatched_token_tensor,
     int num_dispatched_tokens,
     int num_of_local_experts,
     int pad_multiple,
     cudaStream_t stream) {
   constexpr int block_size = 256;
   const int warp_size = 32;
 
   // Get the number of SMs for the current device
   int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
   // Leave 20 SMs for other kernels
   int grid_size = num_sms - 20;
 
   assert(num_of_local_experts <= block_size);
   assert(num_of_local_experts > 0);
 
   auto row_id_map = torch::empty({num_dispatched_tokens + pad_multiple, num_of_local_experts},
                                  torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
   auto tokens_per_expert = torch::empty(
       {num_of_local_experts}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
 
   // If the size of the allocated dispatched tokens is 0, return the empty
   // tensors
   if (num_dispatched_tokens == 0) {
     // Fill the tokens_per_expert with 0 if no tokens need to permute in the
     // current rank
     tokens_per_expert.zero_();
     row_id_map.zero_();
     return std::make_tuple(row_id_map, tokens_per_expert);
   }
 
   // Construct the template buffers
   int rows_workspace_1 = (num_dispatched_tokens + block_size - 1) / block_size;
   int rows_workspace_2 = (rows_workspace_1 + block_size - 1) / block_size;
   auto workspace1 = torch::empty({rows_workspace_1, num_of_local_experts},
                                  torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
   auto workspace2 = torch::empty({rows_workspace_2, num_of_local_experts},
                                  torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
 
   // Compute the size of the shared memory
   int shared_mem_size_pass_1_2 = block_size * num_of_local_experts * sizeof(int);
   int shared_mem_size_pass_3_4 = 3 * num_of_local_experts * sizeof(int);
   int shared_mem_size = max(shared_mem_size_pass_1_2, shared_mem_size_pass_3_4);
 
   // Construct the parameters for the cooperative kernel
   auto workspace1_ptr = workspace1.data_ptr<int>();
   auto workspace2_ptr = workspace2.data_ptr<int>();
   auto tokens_per_expert_ptr = tokens_per_expert.data_ptr<int>();
   auto row_id_map_ptr = row_id_map.data_ptr<int>();
   auto num_dispatched_token_ptr = num_dispatched_token_tensor.data_ptr<int>();
   void* args[] = {
       &routing_map,           &num_dispatched_token_ptr, &num_of_local_experts, &workspace1_ptr,
       &rows_workspace_1,      &workspace2_ptr,           &rows_workspace_2,     &pad_multiple,
       &tokens_per_expert_ptr, &row_id_map_ptr,
   };
 
   cudaFuncSetAttribute(permute_processing_kernel<block_size, warp_size>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
   cudaLaunchCooperativeKernel(permute_processing_kernel<block_size, warp_size>, grid_size,
                               block_size, args, shared_mem_size, stream);
 
   return std::make_tuple(row_id_map, tokens_per_expert);
 }
 

 template <const int block_size = 512, typename DType, typename ProbType, typename ScalarType>
 __global__ void permute_kernel(DType* tokens,
                                DType* permuted_tokens,
                                ScalarType* scaling_factor,
                                ScalarType* permuted_scaling_factor,
                                ProbType* probs,
                                ProbType* permuted_probs,
                                int* row_id_map,
                                int* num_dispatched_tokens_ptr,
                                int pad_multiple,
                                int num_of_local_experts,
                                int hidden_size,
                                int scales_per_token,
                                int local_rank,
                                int num_ranks_per_node) {
   // Index of the current token
   // Each extended warp contains 4 warps, and will dispatch 1 tokens to
   // multi-experts
   int64_t tokens_per_block = blockDim.x / 128;
   int64_t extended_laned_id = threadIdx.x % 128;
   int64_t extended_warp_id = threadIdx.x / 128;
   int64_t block_start = blockIdx.x * tokens_per_block;
   int64_t token_id = block_start + extended_warp_id;
   int num_dispatched_tokens = *num_dispatched_tokens_ptr + pad_multiple;
 
   // Compute the offset for each expert, means the prefix sum of tokens per
   // expert
   extern __shared__ int shmem_in_permute_kernel[];
   int* expert_routing_map = shmem_in_permute_kernel;
   // Load the current routing map
   for (int i = threadIdx.x; i < num_of_local_experts * tokens_per_block; i += block_size) {
     expert_routing_map[i] = (block_start + i / num_of_local_experts < num_dispatched_tokens)
                                 ? row_id_map[block_start * num_of_local_experts + i]
                                 : 0;
   }
   __syncthreads();
 
   if (token_id >= num_dispatched_tokens) {  // If the token is out of range, return
     return;
   }
 
   // Permute the tokens
   int num_eles_per_float4 = sizeof(float4) / sizeof(DType);
   int64_t hidden_size_fp4 = hidden_size / num_eles_per_float4;
   float4* tokens_fp4 = reinterpret_cast<float4*>(tokens);
   float4* permuted_tokens_fp4 = reinterpret_cast<float4*>(permuted_tokens);
   for (int64_t i = 0; i < num_of_local_experts; i++) {
     int64_t dest_token_id = expert_routing_map[extended_warp_id * num_of_local_experts + i];
     if (dest_token_id > 0) {
       for (int64_t j = extended_laned_id; j < hidden_size_fp4; j += 128) {
         permuted_tokens_fp4[(dest_token_id - 1) * hidden_size_fp4 + j] =
             tokens_fp4[token_id * hidden_size_fp4 + j];
       }
     } else if (dest_token_id < 0) {
       for (int64_t j = extended_laned_id; j < hidden_size_fp4; j += 128) {
         permuted_tokens_fp4[(-dest_token_id - 1) * hidden_size_fp4 + j] = {0.0f, 0.0f, 0.0f, 0.0f};
       }
     }
   }
 
   // If use fp8, permute the scaling factor
   if (scaling_factor != nullptr) {
     for (int64_t i = 0; i < num_of_local_experts; i++) {
       int64_t dest_token_id = expert_routing_map[extended_warp_id * num_of_local_experts + i];
       if (dest_token_id > 0) {
         for (int64_t j = extended_laned_id; j < scales_per_token; j += 128) {
           permuted_scaling_factor[(dest_token_id - 1) * scales_per_token + j] =
               scaling_factor[token_id * scales_per_token + j];
         }
       } else if (dest_token_id < 0) {
         for (int64_t j = extended_laned_id; j < scales_per_token; j += 128) {
           permuted_scaling_factor[(-dest_token_id - 1) * scales_per_token + j] = 0;
         }
       }
     }
   }
 
   // If use probs, permute the probs
   if (probs != nullptr) {
     for (int64_t i = 0; i < num_of_local_experts; i++) {
       int64_t dest_token_id = expert_routing_map[extended_warp_id * num_of_local_experts + i];
       if (dest_token_id > 0) {
         permuted_probs[dest_token_id - 1] =
             probs[token_id * num_of_local_experts * num_ranks_per_node +
                   local_rank * num_of_local_experts + i];
       } else if (dest_token_id < 0) {
         permuted_probs[(-dest_token_id - 1)] = 0;
       }
     }
   }
 }
 
 template <typename DType, typename ProbType = float, typename ScalarType = float>
 std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
 permute_launcher(DType* tokens_ptr,
                  ProbType* probs_ptr,
                  ScalarType* scaling_factor_ptr,
                  torch::Tensor row_id_map,
                  int hidden_size,
                  int scales_per_token,
                  int local_rank,
                  int num_ranks_per_node,
                  int num_of_local_experts,
                  torch::Tensor num_dispatched_token_tensor,
                  int num_dispatched_tokens,
                  int num_permuted_token,
                  int pad_multiple,
                  bool use_fp8,
                  bool with_probs,
                  torch::TensorOptions token_options,
                  cudaStream_t stream) {
   // Current only support 8-bits and 16-bits tokens
   assert((std::is_same<DType, uint8_t>::value || std::is_same<DType, uint16_t>::value));
   // Current only support float probs
   assert((std::is_same<ProbType, float>::value));
   // Current only support 4 bytes for 128 elements
   assert((std::is_same<ScalarType, float>::value));
   // For alignment of float4 vectorizatized load
   if(std::is_same<DType, uint8_t>::value) {
      assert(hidden_size % 16 == 0);
   }else if(std::is_same<DType, uint16_t>::value) {
      assert(hidden_size % 8 == 0);
   }
   assert(num_permuted_token >= 0);
 
   // Construct the output tensors
   auto permuted_tokens =
       torch::empty({num_permuted_token, hidden_size}, token_options.device(torch::kCUDA));
 
   int padded_num_dispatched_tokens = num_dispatched_tokens + pad_multiple;
 
   torch::Tensor permuted_scaling_factor, permuted_probs;
   if (use_fp8) {
     permuted_scaling_factor =
         torch::empty({num_permuted_token, scales_per_token},
                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
   }
   if (with_probs) {
     permuted_probs = torch::empty(
         {num_permuted_token}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
   }
 
   // If the size of the allocated dispatched tokens is 0, return the empty
   // tensors
   if (padded_num_dispatched_tokens == 0) {
     return std::make_tuple(permuted_tokens, permuted_scaling_factor, permuted_probs);
   }
 
   // Launch the kernel
   constexpr int block_size = 512;
   constexpr int tokens_per_block = block_size / 128;
   int grid_size = (padded_num_dispatched_tokens + tokens_per_block - 1) / tokens_per_block;
   int shared_mem_size = num_of_local_experts * tokens_per_block * sizeof(int);
   permute_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
       reinterpret_cast<DType*>(tokens_ptr),
       reinterpret_cast<DType*>(permuted_tokens.data_ptr()),
       use_fp8 ? reinterpret_cast<float*>(scaling_factor_ptr) : nullptr,
       use_fp8 ? permuted_scaling_factor.data_ptr<float>() : nullptr,
       with_probs ? reinterpret_cast<float*>(probs_ptr) : nullptr,
       with_probs ? permuted_probs.data_ptr<float>() : nullptr, row_id_map.data_ptr<int>(),
       num_dispatched_token_tensor.data_ptr<int>(), pad_multiple, num_of_local_experts, hidden_size,
       scales_per_token, local_rank, num_ranks_per_node);
   CUDA_CHECK(cudaGetLastError());
 
   return std::make_tuple(permuted_tokens, permuted_scaling_factor, permuted_probs);
 }
 
 template <const int block_size = 512, typename DType, typename ProbType>
 __global__ void unpermute_kernel(DType* permuted_tokens,
                                  DType* tokens,
                                  ProbType* permuted_probs,
                                  ProbType* probs,
                                  int* row_id_map,
                                  int* num_dispatched_tokens_ptr,
                                  int num_of_local_experts,
                                  int hidden_size,
                                  int local_rank,
                                  int num_ranks_per_node) {
   // Index of the current token
   // Each extended warp contains 4 warps, and will reduce multi-experts tokens
   // to 1 token
   int64_t tokens_per_block = blockDim.x / 128;
   int64_t extended_laned_id = threadIdx.x % 128;
   int64_t extended_warp_id = threadIdx.x / 128;
   int64_t block_start = blockIdx.x * tokens_per_block;
   int64_t token_id = block_start + extended_warp_id;
   int num_dispatched_tokens = *num_dispatched_tokens_ptr;
 
   // Compute the offset for each expert, means the prefix sum of tokens per
   // expert
   extern __shared__ int shmem_in_permute_kernel[];
   int* expert_routing_map = shmem_in_permute_kernel;
   // Load the current routing map
   for (int i = threadIdx.x; i < num_of_local_experts * tokens_per_block; i += block_size) {
     expert_routing_map[i] = (block_start + i / num_of_local_experts < num_dispatched_tokens)
                                 ? row_id_map[block_start * num_of_local_experts + i]
                                 : 0;
   }
   __syncthreads();
 
   if (token_id >= num_dispatched_tokens) {  // If the token is out of range, return
     return;
   }
 
   // Unpermute the tokens
   constexpr int num_eles_per_float4 = sizeof(float4) / sizeof(DType);
   int64_t hidden_size_fp4 = hidden_size / num_eles_per_float4;
   float4* tokens_fp4 = reinterpret_cast<float4*>(tokens);
   float4* permuted_tokens_fp4 = reinterpret_cast<float4*>(permuted_tokens);
   // Use float4 buffer to accumulate the tokens
   float4 buffer_fp4;
   float accumulator_fp4[num_eles_per_float4];
   DType* buffer_ptr = reinterpret_cast<DType*>(&buffer_fp4);
   // Accumulate the tokens from multi-experts
   for (int64_t j = extended_laned_id; j < hidden_size_fp4; j += 128) {
 // Initialize the accumulator
 #pragma unroll
     for (int k = 0; k < num_eles_per_float4; k++)
       accumulator_fp4[k] = 0.0f;
     for (int i = 0; i < num_of_local_experts; i++) {
       int64_t source_token_id = expert_routing_map[extended_warp_id * num_of_local_experts + i];
       if (source_token_id > 0) {
         buffer_fp4 = permuted_tokens_fp4[(source_token_id - 1) * hidden_size_fp4 + j];
 #pragma unroll
         for (int k = 0; k < num_eles_per_float4; k++) {
           accumulator_fp4[k] += DType2Float<DType>(buffer_ptr[k]);
         }
       }
     }
 #pragma unroll
     for (int k = 0; k < num_eles_per_float4; k++) {
       buffer_ptr[k] = Float2DType<DType>(accumulator_fp4[k]);
     }
     // Store the accumulated tokens to the output tensor
     tokens_fp4[token_id * hidden_size_fp4 + j] = buffer_fp4;
   }
 
   // If use probs, unpermute the probs
   if (permuted_probs != nullptr) {
     for (int64_t j = extended_laned_id; j < num_of_local_experts * num_ranks_per_node; j += 128) {
       float value = 0.0f;
       if (j / num_of_local_experts == local_rank) {
         int64_t source_token_id =
             expert_routing_map[extended_warp_id * num_of_local_experts + j % num_of_local_experts];
         if (source_token_id > 0) {
           value = static_cast<float>(permuted_probs[source_token_id - 1]);
         }
       }
       probs[token_id * num_of_local_experts * num_ranks_per_node + j] =
           static_cast<ProbType>(value);
     }
   }
 }
 
 template <typename DType, typename ProbType>
 void unpermute_launcher(torch::Tensor permuted_tokens,
                         c10::optional<torch::Tensor> permuted_probs,
                         DType* tokens_ptr,
                         ProbType* probs_ptr,
                         torch::Tensor row_id_map,
                         int num_of_local_experts,
                         torch::Tensor num_dispatched_tokens_tensor,
                         int num_dispatched_tokens,
                         int pad_multiple,
                         int hidden_size,
                         int local_rank,
                         int num_ranks_per_node,
                         bool with_probs,
                         cudaStream_t stream) {
   assert(permuted_tokens.dtype() == torch::kBFloat16);
   if (with_probs) {
     assert(permuted_probs.has_value());
     assert(permuted_probs.value().dtype() == torch::kFloat32);
   }
   assert((std::is_same<DType, uint16_t>::value));
   assert((std::is_same<ProbType, float>::value));
   assert(hidden_size % 2 == 0);
 
   constexpr int block_size = 512;
   constexpr int tokens_per_block = block_size / 128;
   int grid_size = (num_dispatched_tokens + tokens_per_block - 1) / tokens_per_block;
   int shared_mem_size = num_of_local_experts * tokens_per_block * sizeof(int);
 
   // If the size of the dispatched tokens is 0, return
   if (num_dispatched_tokens == 0) {
     return;
   }
 
   unpermute_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
       reinterpret_cast<__nv_bfloat16*>(permuted_tokens.data_ptr()),
       reinterpret_cast<__nv_bfloat16*>(tokens_ptr),
       with_probs ? reinterpret_cast<float*>(permuted_probs.value().data_ptr()) : nullptr,
       with_probs ? reinterpret_cast<float*>(probs_ptr) : nullptr, row_id_map.data_ptr<int>(),
       num_dispatched_tokens_tensor.data_ptr<int>(), num_of_local_experts, hidden_size, local_rank,
       num_ranks_per_node);
 
   CUDA_CHECK(cudaGetLastError());
 }
 
// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include <type_traits>
#include "utils.cuh"
 
struct PermuteArgs {
  // The address of the input 
  void* tokens_ptr;
  float* probs_ptr;
  float* scaling_factor_ptr;
  torch::Tensor row_id_map;

  // The shape message of the input
  int hidden_size;
  int scales_per_token; // Now is hidden_size/128
  torch::Tensor num_dispatched_token_tensor; // We assume it is only valid on GPU
  int num_permuted_token;
  int num_ranks_per_node; // Probs dimension 0 = num_ranks_per_node * num_of_local_experts
  int num_of_local_experts;
  int pad_multiple;

  // Misc
  int local_rank;
  bool use_fp8;
  bool with_probs;
  int num_of_blocks_permute_api;
  torch::TensorOptions token_options; // To record the Dtype of the input tokens from the expert mlp, maybe bf16/fp16/fp8...
  cudaStream_t stream;
};

struct UnpermuteArgs {
  // Input tensors
  torch::Tensor permuted_tokens;
  c10::optional<torch::Tensor> permuted_probs;
  torch::Tensor row_id_map;

  // The address of the output
  uint16_t* tokens_ptr; 
  float* probs_ptr;

  // The shape message of the output
  int num_of_local_experts;
  torch::Tensor num_dispatched_tokens_tensor; // We assume it is only valid on GPU
  int pad_multiple;
  int hidden_size;

  // Misc
  int local_rank;
  int num_ranks_per_node;
  bool with_probs;
  int num_of_blocks_permute_api;
  cudaStream_t stream;
};

 /**
  * @brief Make the row id map for the permute kernel, padding at the num of
  * tokens dimension
  * @param routing_map[in] shape: [num_dispatched_tokens, num_of_local_experts],
  * type: bool
  * @param max_num_dispatched_tokens[in]
  * @param num_of_local_experts[in]
  * @param pad_multiple[in]
  * @param non_blocking[in]
  * @param stream[in]
  * @return row_id_map[out] shape: [num_dispatched_tokens, num_of_local_experts],
  * type: int
  */
 std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
 permute_processing(
     bool* routing_map,
     torch::Tensor num_dispatched_token_tensor,
     int max_num_dispatched_tokens,
     int num_of_local_experts,
     int pad_multiple,
     int num_of_blocks,
     int num_permuted_tokens,
     bool non_blocking,
     cudaStream_t stream);
 
 /**
  * @brief Permute the tokens to the experts
  * @param tokens_ptr[in] shape: [num_dispatched_tokens, hidden_size], type:
  * DType
  * @param probs_ptr[in] shape: [num_dispatched_tokens, num_of_local_experts],
  * type: ProbType, now only support float
  * @param scaling_factor_ptr[in] shape: [num_dispatched_tokens,
  * scales_per_token], type: ScalarType
  * @param row_id_map[in] shape: [num_dispatched_tokens, num_of_local_experts],
  * type: int
  * @return permuted_tokens[out] shape: [num_dispatched_tokens, hidden_size],
  * type: DType
  * @return permuted_scaling_factor[out] shape: [num_dispatched_tokens,
  * scales_per_token], type: ScalarType
  * @return permuted_probs[out] shape: [num_dispatched_tokens,
  * num_of_local_experts], type: ProbType, now only support float
  */
 template <typename DType, typename ProbType, typename ScalarType>
 std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
 permute_launcher(PermuteArgs args);
 
 /**
  * @brief Unpermute the tokens to the original order
  * @param permuted_tokens[in] shape: [num_permuted_token_from_permute,
  * hidden_size], type: DType
  * @param permuted_probs[in] shape: [num_permuted_token_from_permute], type:
  * ProbType, now only support float
  * @param tokens_ptr[out] shape: [num_dispatched_tokens, hidden_size], type:
  * DType
  * @param probs_ptr[out] shape: [num_dispatched_tokens, num_of_local_experts],
  * type: ProbType, now only support float
  * @param row_id_map[in] shape: [num_dispatched_tokens, num_of_local_experts],
  * type: int
  */
 template <typename DType, typename ProbType>
 void unpermute_launcher(UnpermuteArgs args);
 
 template <typename DType>
 inline __device__ float DType2Float(DType value) {
   if constexpr (std::is_same<DType, __nv_bfloat16>::value) {
     return __bfloat162float(value);
   } else {
     return static_cast<float>(value);
   }
 }
 
 template <typename DType>
 inline __device__ DType Float2DType(float value) {
   if constexpr (std::is_same<DType, __nv_bfloat16>::value) {
     return __float2bfloat16(value);
   } else {
     return static_cast<DType>(value);
   }
 }


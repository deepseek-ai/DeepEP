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
 
 /**
  * @brief Make the row id map for the permute kernel, padding at the num of
  * tokens dimension
  * @param routing_map[in] shape: [num_dispatched_tokens, num_of_local_experts],
  * type: bool
  * @param num_dispatched_tokens[in]
  * @param num_of_local_experts[in]
  * @param pad_multiple[in]
  * @param stream[in]
  * @return row_id_map[out] shape: [num_dispatched_tokens, num_of_local_experts],
  * type: int
  */
 std::tuple<torch::Tensor, torch::Tensor> permute_processing(
     bool* routing_map,
     torch::Tensor num_dispatched_token_tensor,
     int num_dispatched_tokens,
     int num_of_local_experts,
     int pad_multiple,
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
                  cudaStream_t stream);
 
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
                         cudaStream_t stream);
 
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
 
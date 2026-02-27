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

  // The address of the output (pre-allocated by caller)
  void* output_tokens_ptr = nullptr;
  float* output_probs_ptr = nullptr;
  float* output_scaling_factor_ptr = nullptr;

  // The shape message of the input
  int hidden_size;
  int scales_per_token; // Now is hidden_size/128
  torch::Tensor num_dispatched_token_tensor; // We assume it is only valid on GPU
  int64_t num_permuted_token;
  int num_ranks_per_node; // Probs dimension 0 = num_ranks_per_node * num_of_local_experts
  int num_of_local_experts;
  int pad_multiple;

  // Misc
  int local_rank;
  bool use_fp8;
  bool with_probs;
  int num_of_blocks_permute;
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
  int num_of_blocks_unpermute;
  cudaStream_t stream;
};

 /**
  * @brief Preprocess routing map to produce row_id_map, tokens_per_expert, and overflow_flag for permute.
  */
 std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
 permute_preprocessing(
     bool* routing_map,
     torch::Tensor num_dispatched_token_tensor,
     int64_t max_num_dispatched_tokens,
     int num_of_local_experts,
     int pad_multiple,
     int num_of_blocks,
     int64_t num_permuted_tokens,
     bool non_blocking,
     cudaStream_t stream);
 
 /**
  * @brief Pad each element of tokens_per_expert to the nearest multiple of pad_multiple, writing int64 result to dst.
  */
 void pad_tokens_per_expert(
     const int32_t* src,   // GPU raw counts [num_experts]
     int64_t* dst,         // pinned or device [num_experts]
     int num_experts,
     int pad_multiple,
     cudaStream_t stream);

 /**
  * @brief Permute tokens according to row_id_map. Input/output via PermuteArgs (output buffers pre-allocated by caller).
  */
 template <typename DType, typename ProbType, typename ScalarType>
 void permute_launcher(PermuteArgs args);
 
 /**
  * @brief Unpermute tokens back to original order. Input/output via UnpermuteArgs.
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


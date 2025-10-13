// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>

#include "utils.cuh"
#include "backend/hybrid_ep_backend.cuh"
#include "jit/compiler.cuh"
#include "extension/permute.cuh"

class Executor {
public:
    Executor(int local_rank, int node_rank);

    struct DispatchArgs {
        // Input tensors
        torch::Tensor hidden;
        torch::Tensor probs;
        torch::Tensor scaling_factor;
        // Output of Metadata Preprocessing
        torch::Tensor sparse_to_dense_map;
        torch::Tensor rdma_to_attn_map;
        torch::Tensor attn_to_rdma_map;
        c10::optional<torch::Tensor> num_dispatched_tokens_tensor;  // Used in the permute
        c10::optional<torch::Tensor> local_expert_routing_map;      // Used in the permute
        // Used in the sync-free permute
        int64_t num_dispatched_tokens = -1;
        // Cached permute
        c10::optional<torch::Tensor> row_id_map;
        int64_t num_permuted_tokens = -1;
        // Misc
        int pad_multiple;  // Used in the padding case of permute
        bool enable_permute = false;
        bool use_host_meta = false;  // If enable this, the produced num_dispatched_tokens will be put
                                        // on the CPU pinned memory, and the tokens_per_expert will be put
                                        // on the CPU, which may reduce the times of the sync
        int64_t num_of_tokens_per_rank;  // Dynamic sequence length
        cudaStream_t stream;
    };

    struct CombineArgs {
        // Input tensors
        torch::Tensor hidden;
        torch::Tensor probs;
        // Combine output tensors
        uint16_t *combined_tokens;
        float *combined_probs;
        // Output of Metadata Preprocessing
        torch::Tensor sparse_to_dense_map;
        torch::Tensor rdma_to_attn_map;
        torch::Tensor attn_to_rdma_map;
        c10::optional<torch::Tensor> num_dispatched_tokens_tensor;
        // Output of Permute-preprocess
        c10::optional<torch::Tensor> row_id_map;  // Used in the unpermute
        // Used in the sync-free Unpermute
        int64_t num_dispatched_tokens = -1;
        // Misc
        int pad_multiple;  // Used in the padding case of unpermute
        bool enable_unpermute = false;
        int64_t num_of_tokens_per_rank;  // Dynamic sequence length
        cudaStream_t stream;
    };

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    metadata_preprocess_core(
        HybridEpConfigInstance config,
        hybrid_ep::tmp_state_t *preprocessing_tmp,
        torch::Tensor global_routing_map,
        int num_of_tokens_per_rank
    );

    void dispatch_preprocess(
        HybridEpConfigInstance config, DispatchBuffers& dispatch_buffers, DispatchArgs& args); // Now is empty op, will be filled with D2D in the inter-node case
    template<typename DType> 
    void dispatch_core(
        HybridEpConfigInstance config, DispatchBuffers& dispatch_buffers, DispatchArgs& args);
    std::tuple<torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor, torch::Tensor> 
    dispatch_postprocess(
        HybridEpConfigInstance config, DispatchBuffers& dispatch_buffers, DispatchArgs& args); 

    void combine_preprocess(
        HybridEpConfigInstance config, CombineBuffers& combine_buffers, CombineArgs& args);
    void combine_core(
        HybridEpConfigInstance config, CombineBuffers& combine_buffers, CombineArgs& args);
    void combine_postprocess(
        HybridEpConfigInstance config, CombineBuffers& combine_buffers, CombineArgs& args); // Now is empty op, will be filled with D2D in the inter-node case

private:
    KernelCache kernel_cache;
    HybridEpConfigInstance config;
    int local_rank;
    int node_rank;
};
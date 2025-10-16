// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#pragma once

#include <any>
#include <string>
#include <unordered_map>
#include <dlfcn.h>
#include <any>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>
#include <chrono>
#include <iostream>

#include "config.cuh"
#include "hybrid_ep_backend.cuh"
#include "utils.cuh"

class NVCCCompiler{
public:
    // Init the flags required by nvcc compiler
    NVCCCompiler(std::string base_path);

    // Generate the code for jit compile
    std::string get_metadata_preprocessing_code(HybridEpConfigInstance config);
    std::string get_dispatch_code(HybridEpConfigInstance config);
    std::string get_combine_code(HybridEpConfigInstance config);

    /**
    * @brief Build the code to a .so file in the runtime
    *
    * @param code The code to be compiled
    * @param signature The signature of the code, which is used to name the .so
    * file
    * @param local_rank The local rank of the current process
    * @return std::string The path of the compiled .so file
    */
    std::string build(std::string code, std::string signature, int local_rank);

    /**
    * @brief Get the compiled function pointer from the compiled .so file
    *
    * @param library_path The path of the compiled .so file
    * @param kernel_key The key of the kernel, used to cache the compiled function pointer
    * @return std::any The function pointer
    */
    std::any get_instance(std::string library_path, std::string kernel_key);


private:
    std::string base_path;  // The path of the installed package
    std::string flags;      // The flags required by nvcc compiler, which contains the
    // base flags(-O3, -arch...), include files, library files
    std::string nvcc_path;  // The path of the nvcc compiler
    std::string include;
    std::string library;
};

class KernelCache{
public:
    KernelCache(int local_rank, std::string base_path);

    void run_proprecess_kernel(
        HybridEpConfigInstance config, 
        const bool* input_routing_map,
        hybrid_ep::tmp_state_t* preprocessing_tmp,
        int32_t* sparse_to_dense_map,
        bool* rdma_to_attn_map,
        bool* attn_to_rdma_map,
        int32_t* num_of_tokens_for_experts,
        bool* local_expert_routing_map,
        const int node_rank,
        const int local_rank,
        int num_of_tokens_per_rank,
        cudaStream_t stream
    );

    template <typename DATA_TYPE>
    void run_dispatch_kernel(
        HybridEpConfigInstance config, 
        hybrid_ep::dispatch_kernel_param_t<DATA_TYPE> param,
        cudaStream_t stream
    );

    void run_combine_kernel(
        HybridEpConfigInstance config, 
        hybrid_ep::combine_kernel_param_t param,
        cudaStream_t stream
    );

private:
    NVCCCompiler nvcc_compiler;
    std::unordered_map<std::string, std::any> kernel_cache;
    std::string base_path;  // The path of the installed package
    int local_rank;   // Used to generate the unique signature for each rank
};
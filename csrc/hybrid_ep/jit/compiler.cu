// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "compiler.cuh"

inline std::string get_env(std::string name) {
    const char* env = std::getenv(name.c_str());
    if (env == nullptr) {
        return std::string("");
    }
    return std::string(env);
}

NVCCCompiler::NVCCCompiler(std::string base_path): base_path(base_path) {
    nvcc_path = get_env("CUDA_HOME") + "/bin/nvcc";

    // Init the flags to compiler
    std::string sm_arch_flags = convert_to_nvcc_arch_flags(SM_ARCH);
    flags = "-std=c++17 " + sm_arch_flags +
            " -O3 --expt-relaxed-constexpr "
            " -Xcompiler -fPIC -shared ";
    // Add the include path of the hybrid-ep library
    include = " -I" + base_path + "/backend" 
            + " -I" + get_env("CUDA_HOME") + "/include ";
    // Add the library path of the hybrid-ep library
    library = "-L" + get_env("CUDA_HOME") + "/lib64 -lcudart ";

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    // Add the dependency of the inter-node jit
    flags += " -DHYBRID_EP_BUILD_MULTINODE_ENABLE";
    std::string rdma_core_home = RDMA_CORE_HOME;
    if (!rdma_core_home.empty()) {
        include += " -I" + rdma_core_home + "/include ";
        library += " -L" + rdma_core_home + "/lib ";
    }
    include += " -I" + base_path + "/backend/nccl/include ";
    library += " -lmlx5 -libverbs ";
    std::string doca_obj_path = base_path + "/backend/nccl/obj";
    objs = doca_obj_path + "/doca_gpunetio.o "
        + doca_obj_path + "/doca_gpunetio_high_level.o "
        + doca_obj_path + "/doca_verbs_cuda_wrapper.o "
        + doca_obj_path + "/doca_verbs_device_attr.o "
        + doca_obj_path + "/doca_verbs_ibv_wrapper.o "
        + doca_obj_path + "/doca_verbs_mlx5dv_wrapper.o "
        + doca_obj_path + "/doca_verbs_qp.o "
        + doca_obj_path + "/doca_verbs_cq.o "
        + doca_obj_path + "/doca_verbs_srq.o "
        + doca_obj_path + "/doca_verbs_uar.o "
        + doca_obj_path + "/doca_verbs_umem.o "
        + doca_obj_path + "/doca_gpunetio_gdrcopy.o "
        + doca_obj_path + "/doca_gpunetio_log.o ";
#endif

    flags = flags + " " + include + " " + library;
}
  

std::string NVCCCompiler::build(std::string code, std::string signature, int local_rank, int node_rank) {
    // Create the source directory
    std::string jit_dir = base_path + "/build/jit";
    std::filesystem::create_directories(jit_dir);

    // Get a unique signature for each run
    auto now = std::chrono::steady_clock::now();
    auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    std::string timestamp_str = std::to_string(ns);
    std::string extended_signature = signature + "-rank-" + std::to_string(local_rank) + "-node-" + std::to_string(node_rank) + "-" + timestamp_str;

    // Write the code to the source file
    std::string source_path =
        jit_dir + "/" + extended_signature + ".cu";
    // Remove the source file if it exists
    remove(source_path.c_str());
    std::ofstream out(source_path, std::ios::binary);
    out.write(code.data(), code.size());
    out.close();

    // Compile the code
    std::string output_path =
        jit_dir + "/" + extended_signature + ".so";
    // Remove the output .so file if it exists
    remove(output_path.c_str());
    std::string compile_command = 
        nvcc_path + " " + flags + " " + source_path + " " + objs + " -o " + output_path;

    auto ret = std::system(compile_command.c_str());
    if (ret != 0) {
        throw std::runtime_error("Failed to compile the code, compile command: " + compile_command);
    }

    // Remove the source file after compilation
    remove(source_path.c_str());

    return output_path;
}

std::any NVCCCompiler::get_instance(std::string library_path, std::string kernel_key) {
    // Open the compiled library with RTLD_GLOBAL for symbol visibility
    void* handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        std::string error_msg = "Failed to open library: " + library_path + "\n";
        error_msg += "dlopen error: " + std::string(error ? error : "unknown") + "\n";
        error_msg += "Dependencies (ldd " + library_path + ")";
        throw std::runtime_error(error_msg);
    }

    // Get the pointer of the get_function_ptr
    std::any (*get_ptr)() = (std::any (*)())dlsym(handle, "get_function_ptr");
    if (get_ptr == nullptr) {
        throw std::runtime_error("Failed to get the function pointer from the library: " +
                                library_path);
    }

    // Unique the compiled lib from different rank
    std::string unique_library_path = base_path + "/build/jit/" + kernel_key + ".so";
    std::string unique_command = "mv " + library_path + " " + unique_library_path;
    if(library_path != unique_library_path) {
        auto ret = std::system(unique_command.c_str());
        if (ret != 0) {
            throw std::runtime_error("Failed to unique the library: " + unique_command);
        }
    }

    // Run the get_function_ptr, then we get the compiled template
    std::any func_ptr = get_ptr();
    return func_ptr;
}


std::string NVCCCompiler::get_metadata_preprocessing_code(HybridEpConfigInstance config) {
  return R"(
        #include "hybrid_ep_backend.cuh"
        #include <any>
        
        extern "C" {
          std::any get_function_ptr() {
            std::any func_ptr = &hybrid_ep::hybrid_ep<)" +
         std::to_string(config.hidden_dim) + ", " + std::to_string(config.max_num_of_tokens_per_rank) + ", " +
         std::to_string(config.num_of_ranks_per_node) + ", " + std::to_string(config.num_of_nodes) + ", " +
         std::to_string(config.num_of_experts_per_rank) + ">::metadata_preprocessing<" +
         std::to_string(config.num_of_threads_per_block_preprocessing_api) + ", " + std::to_string(config.num_of_blocks_preprocessing_api) + R"(>;
            return func_ptr;
          }
        }
      )";
}

std::string NVCCCompiler::get_dispatch_code(HybridEpConfigInstance config) {
  std::string token_type =
      (config.token_data_type == APP_TOKEN_DATA_TYPE::UINT8) ? "uint8_t" : "uint16_t";

  return R"(
        #include "hybrid_ep_backend.cuh"
        #include <any>
        
        extern "C" {
          std::any get_function_ptr() {
            std::any func_ptr = &hybrid_ep::hybrid_ep<)" +
         std::to_string(config.hidden_dim) + ", " + std::to_string(config.max_num_of_tokens_per_rank) + ", " +
         std::to_string(config.num_of_ranks_per_node) + ", " + std::to_string(config.num_of_nodes) + ", " +
         std::to_string(config.num_of_experts_per_rank) + ">::dispatch<" + token_type + ", " +
         std::to_string(config.num_of_stages_dispatch_api) + ", " + std::to_string(config.num_of_tokens_per_chunk_dispatch_api) + ", " +
         std::to_string(config.num_of_blocks_dispatch_api) + ", " + (config.forward_dispatch_api ? "true" : "false") + ", " +
         (config.device_side_sync_dispatch_api ? "true" : "false") + R"(>;
            return func_ptr;
          }
        }
      )";
}

std::string NVCCCompiler::get_combine_code(HybridEpConfigInstance config) {
  return R"(
        #include "hybrid_ep_backend.cuh"
        #include <any>

        extern "C" {
          std::any get_function_ptr() {
            std::any func_ptr = &hybrid_ep::hybrid_ep<)" +
         std::to_string(config.hidden_dim) + ", " + std::to_string(config.max_num_of_tokens_per_rank) + ", " +
         std::to_string(config.num_of_ranks_per_node) + ", " + std::to_string(config.num_of_nodes) + ", " +
         std::to_string(config.num_of_experts_per_rank) + ">::combine<" +
         std::to_string(config.num_of_stages_g2s_combine_api) + ", " + std::to_string(config.num_of_stages_s2g_combine_api) + ", " +
         std::to_string(config.num_of_tokens_per_chunk_combine_api) + ", " + std::to_string(config.num_of_tokens_per_group_combine_api) +
         ", " + std::to_string(config.num_of_blocks_combine_api) + ", " +
         std::to_string(config.num_of_additional_in_flight_s2g_combine_api) + ", " +
         (config.backward_combine_api ? "true" : "false") + ", " +
         (config.device_side_sync_combine_api ? "true" : "false") + R"(>;
            return func_ptr;
          }
        }
      )";
}

KernelCache::KernelCache(int node_rank, int local_rank, std::string base_path, bool load_cached_kernels): 
node_rank(node_rank), local_rank(local_rank), base_path(base_path), nvcc_compiler(base_path) {
    // Load all cached kernels from the cache directory
    std::string cache_dir = base_path + "/build/jit";
    std::filesystem::create_directories(cache_dir);
    if(load_cached_kernels) {
        for (const auto& entry : std::filesystem::directory_iterator(cache_dir)) {
            if (entry.path().extension() == ".so") {
                std::string kernel_key = entry.path().stem().string();
                kernel_cache[kernel_key] = nvcc_compiler.get_instance(entry.path().string(), kernel_key);
            }
        }
    }
}

void KernelCache::run_proprecess_kernel(
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
){
    // Generate the unique key to search the kernel in the cache
    std::string preprocess_kernel_key = get_key(
        config.hidden_dim,
        config.max_num_of_tokens_per_rank,
        config.num_of_experts_per_rank,
        config.num_of_ranks_per_node,
        config.num_of_nodes,
        config.num_of_threads_per_block_preprocessing_api,
        config.num_of_blocks_preprocessing_api
    );
    
    auto it = kernel_cache.find(preprocess_kernel_key);
    if (it == kernel_cache.end()) {
        auto preprocessing_code = nvcc_compiler.get_metadata_preprocessing_code(config);
        auto preprocessing_path = nvcc_compiler.build(preprocessing_code, preprocess_kernel_key, local_rank, node_rank);
        kernel_cache[preprocess_kernel_key] = nvcc_compiler.get_instance(preprocessing_path, preprocess_kernel_key);
    }
    auto preprocessing_instance = kernel_cache[preprocess_kernel_key];

    // Cast the function pointer to the correct type
    using PreprocessingFuncPtr = void (*)(const bool*, hybrid_ep::tmp_state_t*, int32_t*, bool*, bool*, int32_t*, bool*, const int, const int, int, cudaStream_t);
    auto func_ptr = std::any_cast<PreprocessingFuncPtr>(preprocessing_instance);

    // Run the kernel
    func_ptr(input_routing_map, preprocessing_tmp, sparse_to_dense_map, rdma_to_attn_map,
        attn_to_rdma_map, num_of_tokens_for_experts, local_expert_routing_map, node_rank,
        local_rank, num_of_tokens_per_rank, stream);

}

template void KernelCache::run_dispatch_kernel<uint8_t>(
    HybridEpConfigInstance config, 
    hybrid_ep::dispatch_kernel_param_t<uint8_t> param,
    cudaStream_t stream
);

template void KernelCache::run_dispatch_kernel<uint16_t>(
    HybridEpConfigInstance config, 
    hybrid_ep::dispatch_kernel_param_t<uint16_t> param,
    cudaStream_t stream
);

template<typename DATA_TYPE>
void KernelCache::run_dispatch_kernel(
    HybridEpConfigInstance config, 
    hybrid_ep::dispatch_kernel_param_t<DATA_TYPE> param,
    cudaStream_t stream
){
    // Generate the unique key to search the kernel in the cache
    std::string dispatch_kernel_key = get_key(
        config.hidden_dim,
        config.max_num_of_tokens_per_rank,
        config.num_of_experts_per_rank,
        config.num_of_ranks_per_node,
        config.num_of_nodes,
        type_to_string(config.token_data_type),
        config.num_of_stages_dispatch_api,
        config.num_of_tokens_per_chunk_dispatch_api,
        config.num_of_blocks_dispatch_api,
        config.forward_dispatch_api,
        config.device_side_sync_dispatch_api
    );

    auto it = kernel_cache.find(dispatch_kernel_key);
    if (it == kernel_cache.end()) {
        // JIT Compile the kernel
        auto dispatch_code = nvcc_compiler.get_dispatch_code(config);
        auto dispatch_path = nvcc_compiler.build(dispatch_code, dispatch_kernel_key, local_rank, node_rank);
        kernel_cache[dispatch_kernel_key] = nvcc_compiler.get_instance(dispatch_path, dispatch_kernel_key);
    }
    auto dispatch_instance = kernel_cache[dispatch_kernel_key];

    // Cast the function pointer to the correct type
    using DispatchFuncPtr = void (*)(
        hybrid_ep::dispatch_kernel_param_t<DATA_TYPE>, cudaStream_t);
    auto func_ptr = std::any_cast<DispatchFuncPtr>(dispatch_instance);

    // Run the kernel
    func_ptr(param, stream);
}

void KernelCache::run_combine_kernel(
    HybridEpConfigInstance config, 
    hybrid_ep::combine_kernel_param_t param,
    cudaStream_t stream
){
    // Generate the unique key to search the kernel in the cache
    std::string combine_kernel_key = get_key(
        config.hidden_dim,
        config.max_num_of_tokens_per_rank,
        config.num_of_experts_per_rank,
        config.num_of_ranks_per_node,
        config.num_of_nodes,
        config.num_of_stages_g2s_combine_api,
        config.num_of_stages_s2g_combine_api,
        config.num_of_tokens_per_chunk_combine_api,
        config.num_of_tokens_per_group_combine_api,
        config.num_of_blocks_combine_api,
        config.num_of_additional_in_flight_s2g_combine_api,
        config.backward_combine_api,
        config.device_side_sync_combine_api
    );

    auto it = kernel_cache.find(combine_kernel_key);
    if (it == kernel_cache.end()) {
        // JIT Compile the kernel
        auto combine_code = nvcc_compiler.get_combine_code(config);
        auto combine_path = nvcc_compiler.build(combine_code, combine_kernel_key, local_rank, node_rank);
        kernel_cache[combine_kernel_key] = nvcc_compiler.get_instance(combine_path, combine_kernel_key);
    }
    auto combine_instance = kernel_cache[combine_kernel_key];
    
    // Cast the function pointer to the correct type
    using CombineFuncPtr = void (*)(hybrid_ep::combine_kernel_param_t, cudaStream_t);
    auto func_ptr = std::any_cast<CombineFuncPtr>(combine_instance);

    // Run the kernel
    func_ptr(param, stream);
}


  
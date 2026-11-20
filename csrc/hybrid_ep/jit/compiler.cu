// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "compiler.cuh"
#include <any>
#include <cstdio>
#include <unistd.h>
#include <pwd.h>
#include <sys/types.h>
#include <stdexcept>
#include <cstdint>
#include <fstream>
#include <set>
#include <vector>
#include <iterator>

inline std::string get_env(std::string name) {
    const char* env = std::getenv(name.c_str());
    if (env == nullptr) {
        return std::string("");
    }
    return std::string(env);
}

// FNV-1a 64-bit content hash, used to salt kernel cache keys so a shared
// cache cannot serve kernels compiled from different source code or flags.
inline std::string content_hash(const std::string& s) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : s) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
    return std::string(buf);
}

std::string get_jit_dir() {
    std::string cache_dir = get_env("HYBRID_EP_CACHE_DIR");
    if (cache_dir.empty()) {
        struct passwd* pw = getpwuid(getuid());
        if (pw != nullptr && pw->pw_dir != nullptr) {
            cache_dir = pw->pw_dir;
        }
        if (cache_dir.empty()) {
            cache_dir = "/tmp";  // Fallback 
        }
    }
    // Share one JIT directory across processes so that load_cached_kernels can
    // actually hit: build() compiles into a unique temp name (timestamp + comm id)
    // and get_instance() installs it with an atomic rename, so concurrent writers
    // cannot corrupt each other or expose a half-written .so. Cross-version
    // staleness is handled by salting the kernel key with a content hash of the
    // generated code and compile flags (see NVCCCompiler::content_salt).
    std::string base_jit = cache_dir + "/.deepep/hybrid_ep/jit";
    return base_jit;
}

NVCCCompiler::NVCCCompiler(std::string base_path, std::string comm_id): 
    base_path(base_path), comm_id(comm_id) {
    jit_dir = get_jit_dir();

    nvcc_path = get_env("CUDA_HOME") + "/bin/nvcc";

    // Init the flags to compiler
    std::string sm_arch_flags = convert_to_nvcc_arch_flags(SM_ARCH);
    std::string flags = "-std=c++17 " + sm_arch_flags +
            " -O3 --expt-relaxed-constexpr "
            " -Xcompiler -fPIC -shared ";
    // Add the include path of the hybrid-ep library
    std::string include = " -I" + base_path + "/backend" 
            + " -I" + get_env("CUDA_HOME") + "/include ";
    // Add the library path of the hybrid-ep library
    std::string library = "-L" + get_env("CUDA_HOME") + "/lib64 -lcudart ";

    intra_node_flags = flags + " " + include + " " + library;

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
    // Add the dependency of the inter-node jit
    flags += " -DHYBRID_EP_BUILD_MULTINODE_ENABLE";
#ifdef USE_NIXL
    flags += " -DUSE_NIXL";
    std::string nixl_home = get_env("NIXL_HOME");
    if (nixl_home.empty()) nixl_home = "/usr/local/nixl";
    std::string ucx_home = get_env("UCX_HOME");
    if (ucx_home.empty()) ucx_home = "/usr";
    include += " -I" + nixl_home + "/include ";
    include += " -I" + nixl_home + "/include/gpu/ucx ";
    include += " -I" + ucx_home + "/include ";
    // Must match the directory setup.py links against, otherwise the inter-node
    // JIT cannot find libnixl. The default assumes a Debian multiarch layout;
    // NIXL_LIB_DIR overrides it for other distros and architectures.
    std::string nixl_lib = get_env("NIXL_LIB_DIR");
    if (nixl_lib.empty()) {
#if defined(__aarch64__)
        nixl_lib = nixl_home + "/lib/aarch64-linux-gnu";
#else
        nixl_lib = nixl_home + "/lib/x86_64-linux-gnu";
#endif
    }
    library += " -L" + nixl_lib + " -lnixl -lnixl_build -lnixl_common ";
    library += " -Xlinker -rpath -Xlinker " + nixl_lib + " ";
#else
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
#endif

    inter_node_flags = flags + " " + include + " " + library;
}
  

std::string NVCCCompiler::content_salt(const std::string& code) const {
    // Hash the generated code, every transitively `#include "..."`-ed source
    // file, and the compile flags (which embed the target arch and include/lib
    // paths), so kernels from a different DeepEP revision or build
    // configuration get a different key instead of being silently reused from
    // a shared cache.
    std::string content = code + "\n" + intra_node_flags + "\n" + inter_node_flags;
    std::vector<std::string> pending{code};
    std::set<std::string> hashed;
    while (!pending.empty()) {
        const std::string text = pending.back();
        pending.pop_back();
        for (size_t pos = 0; pos < text.size();) {
            const size_t h = text.find("#include \"", pos);
            if (h == std::string::npos) break;
            const size_t q1 = h + 10;
            const size_t q2 = text.find('"', q1);
            if (q2 == std::string::npos) break;
            const std::string name = text.substr(q1, q2 - q1);
            pos = q2 + 1;
            if (!hashed.insert(name).second) continue;
            for (const auto& dir : {base_path + "/backend", base_path}) {
                std::ifstream f(dir + "/" + name, std::ios::binary);
                if (!f) continue;
                const std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                content += "\n// " + name + "\n" + body;
                pending.push_back(body);
                break;
            }
        }
    }
    return content_hash(content);
}

std::string NVCCCompiler::build(std::string code, std::string signature, int local_rank, int node_rank, int num_of_nodes, bool enable_permute_fusion, bool enable_token_drop) {
    // Create the source directory
    std::filesystem::create_directories(jit_dir);

    // Get a unique signature for each run
    auto now = std::chrono::steady_clock::now();
    auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    std::string timestamp_str = std::to_string(ns);
    std::string extended_signature = signature + "-rank-" + std::to_string(local_rank) + "-node-" + std::to_string(node_rank) + "-" + timestamp_str + "-" + comm_id;

    // Write the code to the source file
    std::string source_path =
        jit_dir + "/" + extended_signature + ".cu";
    // Remove the source file if it exists
    remove(source_path.c_str());
    std::ofstream out(source_path, std::ios::binary);
    out.write(code.data(), code.size());
    out.close();

    std::string output_path =
        jit_dir + "/" + extended_signature + ".so";
    // Remove the output .so file if it exists
    remove(output_path.c_str());

    // Build extra define flags
    std::string extra_flags;
    if (enable_permute_fusion) {
        extra_flags += " -DHYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE";
    }
    if (enable_token_drop) {
        extra_flags += " -DHYBRID_EP_BUILD_TOKEN_DROP_ENABLE";
    }

    // Choose the flags based on the number of nodes
    std::string compile_command;
    if(num_of_nodes > 1) {
#ifdef USE_NIXL
        compile_command = nvcc_path + " " + inter_node_flags + extra_flags + " " + source_path + " -o " + output_path;
#else
        compile_command = nvcc_path + " " + inter_node_flags + extra_flags + " " + source_path + " " + objs + " -o " + output_path;
#endif
    }else {
        compile_command = nvcc_path + " " + intra_node_flags + extra_flags + " " + source_path + " -o " + output_path;
    }
    
    // Run the compile command
    auto ret = std::system(compile_command.c_str());
    if (ret != 0) {
        throw std::runtime_error("Failed to compile the code, compile command: " + compile_command);
    }

    // Remove the source file after compilation
    remove(source_path.c_str());

    return output_path;
}

std::any NVCCCompiler::get_instance(std::string library_path, std::string kernel_key) {
    // Open the compiled library with RTLD_LOCAL to avoid symbol conflicts
    // between JIT-compiled templates (e.g. with HYBRID_EP_BUILD_PERMUTE_FUSION_ENABLE)
    // and the same template instantiated in the main module (without the macro).
    void* handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
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
    std::string unique_library_path = jit_dir + "/" + kernel_key + ".so";
    if (library_path != unique_library_path) {
        std::error_code ec;
        std::filesystem::rename(library_path, unique_library_path, ec); 
        if (ec) {
            // If rename failed, the comm should not be blocked, so we just print a warning message
            printf("[Warning] Failed to unique the library: %s -> %s, err: %s\n",
                   library_path.c_str(), unique_library_path.c_str(), ec.message().c_str());
            std::filesystem::remove(library_path, ec); // best-effort cleanup
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
         std::to_string(config.pad_multiple) + ", " + std::to_string(config.num_of_tokens_per_chunk_preprocessing_api) + ", " +
         std::to_string(config.num_of_threads_per_block_preprocessing_api) + ", " + std::to_string(config.num_of_blocks_preprocessing_api) + ", " +
         std::to_string(config.topk) + R"(>;
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
         std::to_string(config.num_of_stages_dispatch_api) + ", " + std::to_string(config.num_of_stages_permute_block_dispatch_api) + ", " + std::to_string(config.num_of_in_flight_s2g_dispatch_api) + ", " + std::to_string(config.num_of_in_flight_s2g_permute_block_dispatch_api) + ", " + std::to_string(config.pad_multiple) + ", " + std::to_string(config.num_of_additional_in_flight_s2g_dispatch_api) + ", " + std::to_string(config.num_of_tokens_per_chunk_dispatch_api) + ", " +
         std::to_string(config.num_of_blocks_dispatch_api) + ", " + std::to_string(config.num_of_blocks_permute) + ", " + (config.forward_dispatch_api ? "true" : "false") + ", " +
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
         std::to_string(config.num_of_stages_g2s_unpermute_block) + ", " + std::to_string(config.num_of_stages_s2g_unpermute_block) + ", " +
         std::to_string(config.num_of_tokens_per_chunk_combine_api) + ", " + std::to_string(config.num_of_tokens_per_group_combine_api) + ", " +
         std::to_string(config.num_of_blocks_combine_api) + ", " + std::to_string(config.num_of_blocks_unpermute) + ", " +
         std::to_string(config.num_of_additional_in_flight_s2g_combine_api) + ", " +
         std::to_string(config.num_of_additional_in_flight_s2g_unpermute_block_combine_api) + ", " +
         (config.backward_combine_api ? "true" : "false") + ", " +
         (config.device_side_sync_combine_api ? "true" : "false") + R"(>;
            return func_ptr;
          }
        }
      )";
}

KernelCache::KernelCache(int node_rank, int local_rank, std::string base_path, std::string comm_id, bool load_cached_kernels):
node_rank(node_rank), local_rank(local_rank), nvcc_compiler(base_path, comm_id), load_cached_kernels(load_cached_kernels) {
    // Cached kernels are loaded lazily, keyed, in try_load_cached_kernel()
    jit_dir = get_jit_dir();
    std::filesystem::create_directories(jit_dir);
}

bool KernelCache::try_load_cached_kernel(const std::string& kernel_key) {
    if (!load_cached_kernels)
        return false;
    if (kernel_cache.find(kernel_key) != kernel_cache.end())
        return true;
    auto library_path = jit_dir + "/" + kernel_key + ".so";
    std::error_code ec;
    if (!std::filesystem::exists(library_path, ec))
        return false;
    try {
        kernel_cache[kernel_key] = nvcc_compiler.get_instance(library_path, kernel_key);
    } catch (const std::exception& e) {
        printf("[Warning] Ignoring cached kernel `%s`: %s\n",
               kernel_key.c_str(), e.what());
        return false;
    }
    return true;
}

void KernelCache::run_preprocess_kernel(
    HybridEpConfigInstance config, 
    const void* input_routing_map,
    hybrid_ep::tmp_state_t* preprocessing_tmp,
    hybrid_ep::tmp_state_t* preprocessing_local_experts_tmp,
    int32_t* sparse_to_dense_map,
    bool* rdma_to_attn_map,
    bool* attn_to_rdma_map,
    int32_t* num_of_tokens_for_experts,
    bool* local_expert_routing_map,
    int32_t* dense_chunk_layout,
    int32_t* dense_to_expert_map,
    int32_t* num_of_local_experts_tokens,
    int* token_drop_triggered,
    const int node_rank,
    const int local_rank,
    const int local_experts_tokens_limit,
    const int num_of_tokens_per_rank,
    bool fuse_permute_dispatch,
    bool non_blocking,
    cudaStream_t stream
){
    // Generate the unique key to search the kernel in the cache
    bool enable_token_drop = fuse_permute_dispatch && non_blocking;
    std::string preprocess_kernel_key = get_key(
        config.hidden_dim,
        config.max_num_of_tokens_per_rank,
        config.num_of_experts_per_rank,
        config.num_of_ranks_per_node,
        config.num_of_nodes,
        config.pad_multiple,
        config.num_of_tokens_per_chunk_preprocessing_api,
        config.num_of_threads_per_block_preprocessing_api,
        config.num_of_blocks_preprocessing_api,
        config.topk,
        fuse_permute_dispatch,
        non_blocking
    );
    
    auto preprocessing_code = nvcc_compiler.get_metadata_preprocessing_code(config);
    preprocess_kernel_key += "-" + nvcc_compiler.content_salt(preprocessing_code);

    auto it = kernel_cache.find(preprocess_kernel_key);
    if (it == kernel_cache.end() && !try_load_cached_kernel(preprocess_kernel_key)) {
        auto preprocessing_path = nvcc_compiler.build(preprocessing_code, preprocess_kernel_key, local_rank, node_rank, config.num_of_nodes, fuse_permute_dispatch, enable_token_drop);
        kernel_cache[preprocess_kernel_key] = nvcc_compiler.get_instance(preprocessing_path, preprocess_kernel_key);
    }
    auto preprocessing_instance = kernel_cache[preprocess_kernel_key];

    // Cast the function pointer to the correct type
    using PreprocessingFuncPtr = void (*)(const void*, hybrid_ep::tmp_state_t*, hybrid_ep::tmp_state_t*, int32_t*, bool*, bool*, int32_t*, bool*, int32_t*, int32_t*, int32_t*, int*, const int, const int, const int, const int, cudaStream_t);
    auto func_ptr = std::any_cast<PreprocessingFuncPtr>(preprocessing_instance);

    // Run the kernel
    func_ptr(input_routing_map, preprocessing_tmp, preprocessing_local_experts_tmp, sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map, num_of_tokens_for_experts, local_expert_routing_map, dense_chunk_layout, dense_to_expert_map, num_of_local_experts_tokens, token_drop_triggered, node_rank, local_rank, local_experts_tokens_limit, num_of_tokens_per_rank, stream);

}

template void KernelCache::run_dispatch_kernel<uint8_t>(
    HybridEpConfigInstance config, 
    hybrid_ep::dispatch_kernel_param_t<uint8_t> param,
    bool fuse_permute_dispatch,
    bool non_blocking,
    cudaStream_t stream
);

template void KernelCache::run_dispatch_kernel<uint16_t>(
    HybridEpConfigInstance config, 
    hybrid_ep::dispatch_kernel_param_t<uint16_t> param,
    bool fuse_permute_dispatch,
    bool non_blocking,
    cudaStream_t stream
);

template<typename DATA_TYPE>
void KernelCache::run_dispatch_kernel(
    HybridEpConfigInstance config, 
    hybrid_ep::dispatch_kernel_param_t<DATA_TYPE> param,
    bool fuse_permute_dispatch,
    bool non_blocking,
    cudaStream_t stream
){
    // Generate the unique key to search the kernel in the cache
    bool enable_token_drop = fuse_permute_dispatch && non_blocking;
    std::string dispatch_kernel_key = get_key(
        config.hidden_dim,
        config.max_num_of_tokens_per_rank,
        config.num_of_experts_per_rank,
        config.num_of_ranks_per_node,
        config.num_of_nodes,
        type_to_string(config.token_data_type),
        config.num_of_stages_dispatch_api,
        config.num_of_stages_permute_block_dispatch_api,
        config.num_of_in_flight_s2g_dispatch_api,
        config.num_of_in_flight_s2g_permute_block_dispatch_api,
        config.pad_multiple,
        config.num_of_additional_in_flight_s2g_dispatch_api,
        config.num_of_tokens_per_chunk_dispatch_api,
        config.num_of_blocks_dispatch_api,
        config.num_of_blocks_permute,
        config.forward_dispatch_api,
        config.device_side_sync_dispatch_api,
        fuse_permute_dispatch,
        non_blocking
    );

    auto dispatch_code = nvcc_compiler.get_dispatch_code(config);
    dispatch_kernel_key += "-" + nvcc_compiler.content_salt(dispatch_code);

    auto it = kernel_cache.find(dispatch_kernel_key);
    if (it == kernel_cache.end() && !try_load_cached_kernel(dispatch_kernel_key)) {
        // JIT Compile the kernel
        auto dispatch_path = nvcc_compiler.build(dispatch_code, dispatch_kernel_key, local_rank, node_rank, config.num_of_nodes, fuse_permute_dispatch, enable_token_drop);
        kernel_cache[dispatch_kernel_key] = nvcc_compiler.get_instance(dispatch_path, dispatch_kernel_key);
    }
    auto dispatch_instance = kernel_cache[dispatch_kernel_key];

    // Cast the function pointer to the correct type
    using DispatchFuncPtr = void (*)(
        hybrid_ep::dispatch_kernel_param_t<DATA_TYPE>, cudaStream_t);
    DispatchFuncPtr func_ptr;
    try {
        func_ptr = std::any_cast<DispatchFuncPtr>(dispatch_instance);
    } catch (const std::bad_any_cast& e) {
        throw std::runtime_error(
            "Kernel cache type mismatch for dispatch (key=" + dispatch_kernel_key +
            "): expected " + (sizeof(DATA_TYPE) == 1 ? "uint8_t" : "uint16_t") +
            " kernel. Original error: " + std::string(e.what()));
    }

    // Run the kernel
    func_ptr(param, stream);
}

void KernelCache::run_combine_kernel(
    HybridEpConfigInstance config, 
    hybrid_ep::combine_kernel_param_t param,
    bool fuse_unpermute_combine,
    bool non_blocking,
    cudaStream_t stream
){
    // Generate the unique key to search the kernel in the cache
    bool enable_token_drop = fuse_unpermute_combine && non_blocking;
    std::string combine_kernel_key = get_key(
        config.hidden_dim,
        config.max_num_of_tokens_per_rank,
        config.num_of_experts_per_rank,
        config.num_of_ranks_per_node,
        config.num_of_nodes,
        config.num_of_stages_g2s_combine_api,
        config.num_of_stages_s2g_combine_api,
        config.num_of_stages_g2s_unpermute_block,
        config.num_of_stages_s2g_unpermute_block,
        config.num_of_tokens_per_chunk_combine_api,
        config.num_of_tokens_per_group_combine_api,
        config.num_of_blocks_combine_api,
        config.num_of_blocks_unpermute,
        config.num_of_additional_in_flight_s2g_combine_api,
        config.num_of_additional_in_flight_s2g_unpermute_block_combine_api,
        config.backward_combine_api,
        config.device_side_sync_combine_api,
        fuse_unpermute_combine,
        non_blocking
    );

    auto combine_code = nvcc_compiler.get_combine_code(config);
    combine_kernel_key += "-" + nvcc_compiler.content_salt(combine_code);

    auto it = kernel_cache.find(combine_kernel_key);
    if (it == kernel_cache.end() && !try_load_cached_kernel(combine_kernel_key)) {
        // JIT Compile the kernel
        auto combine_path = nvcc_compiler.build(combine_code, combine_kernel_key, local_rank, node_rank, config.num_of_nodes, fuse_unpermute_combine, enable_token_drop);
        kernel_cache[combine_kernel_key] = nvcc_compiler.get_instance(combine_path, combine_kernel_key);
    }
    auto combine_instance = kernel_cache[combine_kernel_key];
    
    // Cast the function pointer to the correct type
    using CombineFuncPtr = void (*)(hybrid_ep::combine_kernel_param_t, cudaStream_t);
    auto func_ptr = std::any_cast<CombineFuncPtr>(combine_instance);

    // Run the kernel
    func_ptr(param, stream);
}


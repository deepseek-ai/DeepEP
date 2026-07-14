#pragma once

#include <cuda.h>
#include <filesystem>
#include <type_traits>
#include <deep_ep/common/exception.cuh>

#include "../utils/lazy_driver.hpp"
#include "no_ref.hpp"

namespace deep_ep::jit {

template <typename arg_t>
static void* get_kernel_arg_ptr(arg_t& arg) {
    if constexpr (std::is_base_of_v<NoRefPtr, std::decay_t<arg_t>>) {
        return arg.ptr;
    } else {
        return const_cast<void*>(static_cast<const void*>(&arg));
    }
}

#if CUDART_VERSION >= 12080 and defined(EP_JIT_USE_RUNTIME_API)

// Use CUDA runtime API
using LibraryHandle = cudaLibrary_t;
using KernelHandle = cudaKernel_t;
using LaunchConfigHandle = cudaLaunchConfig_t;
using LaunchAttrHandle = cudaLaunchAttribute;

#define EP_CUDA_UNIFIED_CHECK CUDA_RUNTIME_CHECK

static KernelHandle load_kernel(const std::filesystem::path& cubin_path, const std::string& func_name,
                                LibraryHandle *library_opt = nullptr) {
    LibraryHandle library;
    KernelHandle kernel{};
    CUDA_RUNTIME_CHECK(cudaLibraryLoadFromFile(&library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    nvshmemx_culibrary_init(library);
    CUDA_RUNTIME_CHECK(cudaLibraryGetKernel(&kernel, library, func_name.c_str()));

    if (library_opt != nullptr)
        *library_opt = library;
    return kernel;
}

static void unload_library(const LibraryHandle& library) {
    nvshmemx_culibrary_finalize(library);
    const auto error = cudaLibraryUnload(library);
    EP_HOST_ASSERT(error == cudaSuccess or error == cudaErrorCudartUnloading);
}

static LaunchConfigHandle construct_launch_config(const KernelHandle& kernel,
                                                  const cudaStream_t& stream, const int& smem_size,
                                                  const dim3& grid_dim, const dim3& block_dim, const int& cluster_dim) {
    if (smem_size > 0)
        CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    LaunchConfigHandle config;
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    config.numAttrs = 0;
    config.attrs = nullptr;

    // TODO: support cooperative and dependent kernel launch
    // NOTES: must use `static` or the `attr` will be deconstructed
    static LaunchAttrHandle attr;
    if (cluster_dim > 1) {
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {static_cast<unsigned>(cluster_dim), 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;
    }
    return config;
}

template <typename... arg_ts>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, arg_ts&&... args) {
    void *ptr_args[] = {get_kernel_arg_ptr(args)...};
    return cudaLaunchKernelExC(&config, kernel, ptr_args);
}

#else

// Use CUDA driver API
using LibraryHandle = CUmodule;
using KernelHandle = CUfunction;
using LaunchConfigHandle = CUlaunchConfig;
using LaunchAttrHandle = CUlaunchAttribute;

#define EP_CUDA_UNIFIED_CHECK CUDA_DRIVER_CHECK

static KernelHandle load_kernel(const std::filesystem::path& cubin_path, const std::string& func_name,
                               LibraryHandle *library_opt = nullptr) {
    LibraryHandle library;
    KernelHandle kernel;
    CUDA_DRIVER_CHECK(lazy_cuModuleLoad(&library, cubin_path.c_str()));
    CUDA_DRIVER_CHECK(lazy_cuModuleGetFunction(&kernel, library, func_name.c_str()));

    if (library_opt != nullptr)
        *library_opt = library;
    return kernel;
}

static void unload_library(const LibraryHandle& library) {
    const auto error = lazy_cuModuleUnload(library);
    EP_HOST_ASSERT(error == CUDA_SUCCESS or error == CUDA_ERROR_DEINITIALIZED);
}

static LaunchConfigHandle construct_launch_config(const KernelHandle& kernel,
                                                  const cudaStream_t& stream, const int& smem_size,
                                                  const dim3& grid_dim, const dim3& block_dim, const int& cluster_dim,
                                                  const bool& cooperative, const bool& enable_pdl) {
    if (smem_size > 0)
        CUDA_DRIVER_CHECK(lazy_cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size));

    LaunchConfigHandle config;
    config.gridDimX = grid_dim.x;
    config.gridDimY = grid_dim.y;
    config.gridDimZ = grid_dim.z;
    config.blockDimX = block_dim.x;
    config.blockDimY = block_dim.y;
    config.blockDimZ = block_dim.z;
    config.sharedMemBytes = smem_size;
    config.hStream = stream;

    // Create attributes
    static LaunchAttrHandle attrs[3];
    config.attrs = attrs;
    config.numAttrs = 0;

    // Cooperative launch
    if (cooperative) {
        auto& attr = attrs[config.numAttrs ++];
        attr.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
        attr.value.cooperative = 1;
    }

    // Cluster size
    // NOTES: must use `static` or the `attr` will be deconstructed
    if (cluster_dim > 1) {
        auto& attr = attrs[config.numAttrs ++];
        attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attr.value.clusterDim.x = cluster_dim;
        attr.value.clusterDim.y = 1;
        attr.value.clusterDim.z = 1;
    }

    // Dependent kernel launch
    if (enable_pdl) {
        auto& attr = attrs[config.numAttrs ++];
        attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attr.value.programmaticStreamSerializationAllowed = 1;
    }
    return config;
}

template <typename... arg_ts>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, arg_ts&&... args) {
    void *ptr_args[] = {get_kernel_arg_ptr(args)...};
    return lazy_cuLaunchKernelEx(&config, kernel, ptr_args, nullptr);
}

#endif

} // namespace deep_ep::jit

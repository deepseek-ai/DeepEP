#pragma once

#include <cuda.h>
#include <dlfcn.h>

#include <deep_ep/common/exception.cuh>

namespace deep_ep {

// Lazy loading all driver symbols
static void* get_driver_handle() {
    static void* handle = nullptr;
    if (handle == nullptr) {
        handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        EP_HOST_ASSERT(handle != nullptr and "Failed to load CUDA driver `libcuda.so.1`");
    }
    return handle;
}

// Macro to define wrapper functions named `lazy_cu{API name}`
#define DECL_LAZY_CUDA_DRIVER_FUNCTION(name) \
template <typename... Args> \
static auto lazy_##name(Args&&... args) -> decltype(name(args...)) { \
    using FuncType = decltype(&name); \
    static FuncType func = nullptr; \
    if (func == nullptr) { \
        func = reinterpret_cast<FuncType>(dlsym(get_driver_handle(), #name)); \
        EP_HOST_ASSERT(func != nullptr and "Failed to load CUDA driver API"); \
    } \
    return func(std::forward<decltype(args)>(args)...); \
}

DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorString);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuFuncSetAttribute);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleLoad);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleUnload);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleGetFunction);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLaunchKernelEx);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemSetAccess);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemRetainAllocationHandle);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemGetAddressRange_v2);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemAddressReserve);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemAddressFree);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemMap);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemUnmap);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemCreate);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemRelease);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuCtxGetDevice);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemImportFromShareableHandle);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemExportToShareableHandle);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuMemGetAllocationGranularity);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuDeviceGet);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuDeviceGetAttribute);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuStreamBatchMemOp);

#if (defined(CUDART_VERSION) and CUDART_VERSION >= 13030) or (defined(CUDA_VERSION) and CUDA_VERSION >= 13030)
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLogicalEndpointIdReserve);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLogicalEndpointIdRelease);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLogicalEndpointGetLimits);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLogicalEndpointCreate);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLogicalEndpointDestroy);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLogicalEndpointBindAddr);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLogicalEndpointBindMem);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLogicalEndpointUnbind);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLogicalEndpointExport);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLogicalEndpointImport);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLogicalEndpointQuery);
#endif

}  // namespace deep_ep

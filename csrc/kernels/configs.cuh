#pragma once

#define NUM_MAX_NVL_PEERS 8
#define NUM_MAX_RDMA_PEERS 20
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define FINISHED_SUM_TAG 1024
#define NUM_WAIT_NANOSECONDS 500

#ifndef ENABLE_FAST_DEBUG
#define NUM_CPU_TIMEOUT_SECS 100
#define NUM_TIMEOUT_CYCLES 200000000000ull  // 200G cycles ~= 100s
#else
#define NUM_CPU_TIMEOUT_SECS 10
#define NUM_TIMEOUT_CYCLES 20000000000ull  // 20G cycles ~= 10s
#endif

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

// Make CLion CUDA indexing work
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900  // NOLINT(*-reserved-identifier)
#define __CUDACC_RDC__     // NOLINT(*-reserved-identifier)
#endif

// Remove Torch restrictions
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#ifndef DISABLE_SM90_FEATURES
#include <cuda_fp8.h>
#else
// Define minimal FP8 types to avoid compilation errors in CUDA standard library
typedef uint8_t __nv_fp8_storage_t;
typedef uint32_t __nv_fp8x2_storage_t;

struct __nv_fp8_e4m3 {
    __nv_fp8_storage_t __x;
    __nv_fp8_e4m3() = default;
    __device__ __nv_fp8_e4m3(__nv_fp8_storage_t x) : __x(x) {}
};

struct __nv_fp8_e5m2 {
    __nv_fp8_storage_t __x;
    __nv_fp8_e5m2() = default;
    __device__ __nv_fp8_e5m2(__nv_fp8_storage_t x) : __x(x) {}
};

// Define minimal FP8 constants and functions as stubs
#define __NV_E4M3 0
#define __NV_E5M2 1
#define __NV_SATFINITE 0

// Stub FP8 conversion function
__device__ static inline __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(float2, int, int) {
    return 0;  // Return dummy value
}

// Prevent cuda_fp8.h from being included when SM90 features are disabled
#define CUDA_FP8_H
#define __CUDA_FP8_H__
#endif

namespace deep_ep {

#ifndef TOPK_IDX_BITS
#define TOPK_IDX_BITS 64
#endif

#define INT_BITS_T2(bits) int##bits##_t
#define INT_BITS_T(bits) INT_BITS_T2(bits)
typedef INT_BITS_T(TOPK_IDX_BITS) topk_idx_t;  // int32_t or int64_t
#undef INT_BITS_T
#undef INT_BITS_T2

}  // namespace deep_ep

#ifndef DISABLE_NVSHMEM
#include <device_host_transport/nvshmem_common_ibgda.h>
#include <infiniband/mlx5dv.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh>
#endif

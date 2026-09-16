#pragma once

// MIT License
//
// Copyright (c) 2025 DeepSeek
// Changes and additions copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Make CLion CUDA indexing work
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900
#define __CUDACC_RDC__
#define __CUDACC__
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

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifndef DISABLE_SM90_FEATURES
#include <cuda_fp8.h>
#else
// Ampere does not support FP8 features
#define __NV_E4M3 0
#define __NV_E5M2 1
typedef int __nv_fp8_interpretation_t;
typedef int __nv_fp8x4_e4m3;
typedef uint8_t __nv_fp8_storage_t;
#endif

// Compatibility: 256 bits LD/ST instructions
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000
using longlong4_t = longlong4_32a;
#define make_longlong4_t make_longlong4_32a
#else
struct alignas(32) longlong4_t { long long x, y, z, w; };
__device__ __forceinline__ longlong4_t make_longlong4_t(
    const long long& x, const long long& y, const long long& z, const long long& w) {
    return {x, y, z, w};
}
#endif

#ifndef EP_NUM_TOPK_IDX_BITS
#define EP_NUM_TOPK_IDX_BITS 64
#endif

namespace deep_ep {

#ifndef DISABLE_SM90_FEATURES
constexpr bool kEnableSM90Features = true;
#else
constexpr bool kEnableSM90Features = false;
#endif

template <int kNumBits> struct int_with_bits;
template <> struct int_with_bits<8>  { using type = int8_t;  };
template <> struct int_with_bits<16> { using type = int16_t; };
template <> struct int_with_bits<32> { using type = int32_t; };
template <> struct int_with_bits<64> { using type = int64_t; };

using topk_idx_t = int_with_bits<EP_NUM_TOPK_IDX_BITS>::type;

union sf_pack_t {
    float fp32;
    int ue8m0x4;
};

constexpr int kNumTMAAlignedBytes = 16;
constexpr int kNumAlignedSFPacks = 16 / sizeof(sf_pack_t);

// Some communication channel settings
constexpr int kNumMaxChannels = 1024;
constexpr int kGinQPDepth = 1024;
constexpr int kGinQPFlushDepth = 768;

// Per-channel shared-memory TMA buffer count for the hybrid dispatch kernel:
// 1 for the scale-out send warp + 2 for the forward warp (double-buffered
// loads). The host channel budget and the kernel pool must both use this.
constexpr int kNumDispatchSendBuffers = 1;
constexpr int kNumDispatchFwdBuffers = 2;
constexpr int kNumDispatchBuffersPerChannel = kNumDispatchSendBuffers + kNumDispatchFwdBuffers;

} // namespace deep_ep

#pragma once

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

#include <cuda_fp8.h>

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

constexpr int kNumAlignedSFPacks = 16 / sizeof(sf_pack_t);

// Alignment requirements
constexpr int kNumTMAAlignmentBytes = 32;
constexpr int kNumRDMAAlignmentBytes = 64;
constexpr int kNumAllocationAlignmentBytes = 2 * 1024 * 1024;
static_assert(kNumRDMAAlignmentBytes % kNumTMAAlignmentBytes == 0);
static_assert(kNumAllocationAlignmentBytes % kNumRDMAAlignmentBytes == 0);

// Communication limits
constexpr int kNumMaxRanks = 1024;
constexpr int kNumMaxRDMARanks = kNumMaxRanks;
constexpr int kNumMaxNVLRanks = kNumMaxRanks;
constexpr int kNumMaxScaleoutRanks = kNumMaxRanks;
constexpr int kNumMaxScaleupRanks = kNumMaxRanks;
constexpr int kNumMaxContexts = 8;
constexpr int kNumMaxQPs = 1024;
constexpr int kDefaultQPDepth = 1024;
constexpr int kNumMaxSignalBytes = 16 * 1024 * 1024;

} // namespace deep_ep

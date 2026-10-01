#pragma once

#include <cuda/barrier>
#include <cuda_bf16.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

namespace deep_ep::ptx {

/// Declarations
using mbarrier = cuda::barrier<cuda::thread_scope_block>;
using arrival_phase = uint32_t;

#ifdef __CUDACC__

/// Exceptions
__forceinline__ __device__ void trap() {
    asm volatile("trap;");
}

/// Thread layout
__forceinline__ __device__ int get_warp_idx() {
    return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
}

__forceinline__ __device__ int get_lane_idx() {
    int lane_idx;
    asm volatile("mov.s32 %0, %laneid;" : "=r"(lane_idx));
    return lane_idx;
}

/// Election
__forceinline__ __device__ int elect_one_sync() {
    int pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffff));
    return pred;
}

/// TMA and `cp.async`
__forceinline__ __device__ void mbarrier_init_with_fence(mbarrier* ptr, const int& arrive_count = 1) {
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::
                 "r"(arrive_count), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
    asm volatile("fence.mbarrier_init.release.cluster;" ::);
}

__forceinline__ __device__ void mbarrier_invalidate(mbarrier* ptr) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

__forceinline__ __device__ void mbarrier_arrive(mbarrier* ptr) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

__forceinline__ __device__ void mbarrier_arrive_and_set_tx(mbarrier* ptr, const int& num_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::
                 "r"(num_bytes), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

__forceinline__ __device__ void mbarrier_wait_and_flip_phase(mbarrier* ptr, arrival_phase& phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),
        "r"(phase), "r"(0x989680));
    phase ^= 1;
}

template <int kNumBytes>
__forceinline__ __device__ void st_bulk(void* smem_ptr) {
    EP_STATIC_ASSERT(kNumBytes % 8 == 0, "`st.bulk` requires size to be a multiple of 8");
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    if (elect_one_sync()) {
        asm volatile("st.bulk.weak.shared::cta [%0], %1, 0;\n" ::
                     "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr))),
                     "r"(kNumBytes)
                     : "memory");
    }
#else
    #pragma unroll
    for (int i = get_lane_idx(); i < kNumBytes / 8; i += 32)
        static_cast<uint64_t*>(smem_ptr)[i] = 0;
#endif
    __syncwarp();
}

__forceinline__ __device__ void tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;");
}

template <int kNumRemainingWaits = 0>
__forceinline__ __device__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(kNumRemainingWaits) : "memory");
}

template <int kNumRemainingWaits = 0>
__forceinline__ __device__ void tma_store_wait_read() {
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(kNumRemainingWaits) : "memory");
}

enum L2CacheHint: int64_t {
    kEvictFirst = 0x12f0000000000000ll,
    kEvictNormal = 0x1000000000000000ll
};

__forceinline__ __device__ void tma_load_1d(
    const void* dst_ptr, const void* src_ptr, mbarrier* ptr, const int& num_bytes,
    const L2CacheHint& hint = L2CacheHint::kEvictFirst) {
    // NOTES: normally, the loaded part will be evicted soon
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr))),
        "l"(src_ptr),
        "r"(num_bytes),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),
        "l"(hint)
        : "memory");
}

__forceinline__ __device__ void tma_store_1d(
    const void* dst_ptr, const void* src_ptr, const int& num_bytes,
    const L2CacheHint& hint = L2CacheHint::kEvictNormal) {
    // NOTES: normally, the stored part will be used soon
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::
                 "l"(dst_ptr),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),
                 "r"(num_bytes),
                 "l"(hint)
                 : "memory");
}

__forceinline__ __device__ void tma_store_reduce_add_f32(
    float* dst_ptr, const void* src_ptr, const int& num_bytes) {
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n" ::
        "l"(dst_ptr),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),
        "r"(num_bytes)
        : "memory");
}

__forceinline__ __device__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;");
}

__forceinline__ __device__ float4 multimem_ld_reduce_add_f32_with_gt_pred(const float4* ptr, const int& lhs = 1, const int& rhs = 0) {
    float4 value = make_float4(0, 0, 0, 0);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.gt.s32 p, %5, %6;\n\t"
        "  @p multimem.ld_reduce.relaxed.gpu.global.add.v4.f32 {%0, %1, %2, %3}, [%4];\n\t"
        "}"
        : "+f"(value.x), "+f"(value.y), "+f"(value.z), "+f"(value.w)
        : "l"(ptr), "r"(lhs), "r"(rhs)
        : "memory");
    return value;
}

// Multicast store: writes `value` to the same address on every rank in the NVLink domain.
__forceinline__ __device__ void multimem_st_f32_with_gt_pred(float4* ptr, const float4& value, const int& lhs = 1, const int& rhs = 0) {
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.gt.s32 p, %5, %6;\n\t"
        "  @p multimem.st.relaxed.gpu.global.v4.f32 [%0], {%1, %2, %3, %4};\n\t"
        "}"
        :: "l"(ptr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w), "r"(lhs), "r"(rhs)
        : "memory");
}

__forceinline__ __device__ void multimem_cp_async_bulk(
    const void* dst_ptr, const void* src_ptr, const int& num_bytes) {
    asm volatile("multimem.cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::
                 "l"(dst_ptr),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),
                 "r"(num_bytes)
                 : "memory");
}

template <class dtype_t>
__forceinline__ __device__ void cp_async_ca(const dtype_t* gmem_src, const dtype_t* smem_dst) {
    EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8 or sizeof(dtype_t) == 16, "Invalid dtype bytes");
    asm volatile("cp.async.ca.shared::cta.global.L2::128B [%0], [%1], %2;\n" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst))),
                 "l"(gmem_src),
                 "n"(sizeof(dtype_t)));
}

__forceinline__ __device__ void cp_async_mbarrier_arrive(mbarrier* ptr) {
    asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n" ::
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

/// Barriers
template <int kNumThreads>
__forceinline__ __device__ void named_barrier(const int& idx) {
    // Equivalent to `barrier.sync.aligned`, which requires all threads run the same location of code
    asm volatile("bar.sync %0, %1;" ::"r"(idx), "r"(kNumThreads));
}

template <int kNumThreads>
__forceinline__ __device__ void named_barrier_unaligned(const int& idx) {
    asm volatile("barrier.sync %0, %1;" ::"r"(idx), "r"(kNumThreads) : "memory");
}

/// LD/ST instructions
__forceinline__ __device__ int4 ldg_with_ge_pred(const int4* ptr, const int& lhs = 1, const int& rhs = 0, const L2CacheHint& cache_hint = L2CacheHint::kEvictFirst) {
    int4 ret = make_int4(0, 0, 0, 0);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ge.s32 p, %5, %6;\n\t"
        "  @p ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s32 {%0, %1, %2, %3}, [%4], %7;\n\t"
        "}"
        : "+r"(ret.x), "+r"(ret.y), "+r"(ret.z), "+r"(ret.w)
        : "l"(ptr), "r"(lhs), "r"(rhs), "l"(cache_hint)
        : "memory");
    return ret;
}

__forceinline__ __device__ int4 ldg_with_gt_pred(const int4* ptr, const int& lhs = 1, const int& rhs = 0, const L2CacheHint& cache_hint = L2CacheHint::kEvictFirst) {
    int4 ret = make_int4(0, 0, 0, 0);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.gt.s32 p, %5, %6;\n\t"
        "  @p ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s32 {%0, %1, %2, %3}, [%4], %7;\n\t"
        "}"
        : "+r"(ret.x), "+r"(ret.y), "+r"(ret.z), "+r"(ret.w)
        : "l"(ptr), "r"(lhs), "r"(rhs), "l"(cache_hint)
        : "memory");
    return ret;
}

__forceinline__ __device__ int4 ld_with_ge_pred(const int4* ptr, const int& lhs = 1, const int& rhs = 0, const L2CacheHint& cache_hint = L2CacheHint::kEvictFirst) {
    int4 ret = make_int4(0, 0, 0, 0);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ge.s32 p, %5, %6;\n\t"
        "  @p ld.L1::no_allocate.L2::cache_hint.global.v4.s32 {%0, %1, %2, %3}, [%4], %7;\n\t"
        "}"
        : "+r"(ret.x), "+r"(ret.y), "+r"(ret.z), "+r"(ret.w)
        : "l"(ptr), "r"(lhs), "r"(rhs), "l"(cache_hint)
        : "memory");
    return ret;
}

__forceinline__ __device__ float4 ld_with_gt_pred(const float4* ptr, const int& lhs = 1, const int& rhs = 0, const L2CacheHint& cache_hint = L2CacheHint::kEvictFirst) {
    float4 ret = make_float4(0, 0, 0, 0);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.gt.s32 p, %5, %6;\n\t"
        "  @p ld.L1::no_allocate.L2::cache_hint.global.v4.f32 {%0, %1, %2, %3}, [%4], %7;\n\t"
        "}"
        : "+f"(ret.x), "+f"(ret.y), "+f"(ret.z), "+f"(ret.w)
        : "l"(ptr), "r"(lhs), "r"(rhs), "l"(cache_hint)
        : "memory");
    return ret;
}

#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
__forceinline__ __device__ longlong4_t ld_with_gt_pred(const longlong4_t* ptr, const int& lhs = 1, const int& rhs = 0, const L2CacheHint& cache_hint = L2CacheHint::kEvictFirst) {
    longlong4_t ret = make_longlong4_t(0, 0, 0, 0);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.gt.s32 p, %5, %6;\n\t"
        "  @p ld.L1::no_allocate.L2::cache_hint.global.v4.s64 {%0, %1, %2, %3}, [%4], %7;\n\t"
        "}"
        : "+l"(ret.x), "+l"(ret.y), "+l"(ret.z), "+l"(ret.w)
        : "l"(ptr), "r"(lhs), "r"(rhs), "l"(cache_hint)
        : "memory");
    return ret;
}

__forceinline__ __device__ longlong4_t ldg_with_ge_pred(const longlong4_t* ptr, const int& lhs = 1, const int& rhs = 0, const L2CacheHint& cache_hint = L2CacheHint::kEvictFirst) {
    longlong4_t ret = make_longlong4_t(0, 0, 0, 0);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ge.s32 p, %5, %6;\n\t"
        "  @p ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s64 {%0, %1, %2, %3}, [%4], %7;\n\t"
        "}"
        : "+l"(ret.x), "+l"(ret.y), "+l"(ret.z), "+l"(ret.w)
        : "l"(ptr), "r"(lhs), "r"(rhs), "l"(cache_hint)
        : "memory");
    return ret;
}

__forceinline__ __device__ longlong4_t ldg(const longlong4_t* ptr) {
    longlong4_t ret;
    asm volatile(
        "ld.L1::no_allocate.global.nc.v4.s64 {%0, %1, %2, %3}, [%4];\n\t"
        : "=l"(ret.x), "=l"(ret.y), "=l"(ret.z), "=l"(ret.w)
        : "l"(ptr)
        : "memory");
    return ret;
}
#endif

__forceinline__ __device__ int4 ldg(const int4* ptr) {
    return __ldg(ptr);
}

template <typename dtype_t>
__forceinline__ __device__ void st_with_ge_pred(dtype_t* ptr, dtype_t value, const int& lhs = 1, const int& rhs = 0) {
    EP_STATIC_ASSERT(sizeof(dtype_t) == 4, "Invalid data type");
    auto view = *reinterpret_cast<int*>(&value);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ge.s32 p, %2, %3;\n\t"
        "  @p st.global.s32 [%0], %1;\n\t"
        "}"
        :: "l"(ptr), "r"(view), "r"(lhs), "r"(rhs)
        : "memory");
}


__forceinline__ __device__ void st_with_gt_pred(float4* ptr, const float4& value, const int& lhs = 1, const int& rhs = 0) {
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.gt.s32 p, %5, %6;\n\t"
        "  @p st.global.v4.f32 [%0], {%1, %2, %3, %4};\n\t"
        "}"
        :: "l"(ptr), "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w), "r"(lhs), "r"(rhs)
        : "memory");
}

#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)

__forceinline__ __device__ void st_with_gt_pred(longlong4_t* ptr, const longlong4_t& value, const int& lhs = 1, const int& rhs = 0) {
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.gt.s32 p, %5, %6;\n\t"
        "  @p st.global.v4.s64 [%0], {%1, %2, %3, %4};\n\t"
        "}"
        :: "l"(ptr), "l"(value.x), "l"(value.y), "l"(value.z), "l"(value.w), "r"(lhs), "r"(rhs)
        : "memory");
}
#endif

template <typename dtype_t>
__forceinline__ __device__ dtype_t ld_volatile(const void* ptr) {
    if constexpr (sizeof(dtype_t) == 4) {
        uint32_t value;
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
        return reinterpret_cast<const dtype_t&>(value);
    } else if constexpr (sizeof(dtype_t) == 8) {
        uint64_t value;
        asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(value) : "l"(ptr));
        return reinterpret_cast<const dtype_t&>(value);
    } else {
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");
    }
}

__forceinline__ __device__ void red_add(const int64_t* ptr, const int64_t& value) {
    // TODO(NVCC): why don't NVCC support `s64`?
    asm volatile("red.gpu.global.add.u64 [%0], %1;" :: "l"(ptr), "l"(value));
}

__forceinline__ __device__ void red_add_rel_sys(const int* ptr, const int& value) {
    asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));
}

__forceinline__ __device__ void red_add_rel_sys(const int64_t* ptr, const int64_t& value) {
    asm volatile("red.release.sys.global.add.u64 [%0], %1;" :: "l"(ptr), "l"(value));
}

__forceinline__ __device__ void red_add_rel_gpu(const int* ptr, const int& value) {
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(ptr), "r"(value));
}

__forceinline__ __device__ void red_add_rel_gpu(const int64_t* ptr, const int64_t& value) {
    asm volatile("red.release.gpu.global.add.u64 [%0], %1;" :: "l"(ptr), "l"(value));
}

template <typename dtype_t>
__forceinline__ __device__ dtype_t ld_acquire_sys(const dtype_t* ptr) {
    if constexpr (sizeof(dtype_t) == 4) {
        uint32_t value;
        asm volatile("ld.acquire.sys.L1::no_allocate.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
        return reinterpret_cast<const dtype_t&>(value);
    } else if constexpr (sizeof(dtype_t) == 8) {
        uint64_t value;
        asm volatile("ld.acquire.sys.L1::no_allocate.global.u64 %0, [%1];" : "=l"(value) : "l"(ptr));
        return reinterpret_cast<const dtype_t&>(value);
    } else {
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");
    }
}

template <typename dtype_t>
__forceinline__ __device__ void st_relaxed_gpu(void* ptr, dtype_t value) {
    if constexpr (sizeof(dtype_t) == 4) {
        uint32_t int_value = reinterpret_cast<const uint32_t&>(value);
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(ptr), "r"(int_value));
    } else if constexpr (sizeof(dtype_t) == 8) {
        uint64_t int_value = reinterpret_cast<const uint64_t&>(value);
        asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" :: "l"(ptr), "l"(int_value));
    } else {
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");
    }
}

template <typename dtype_t>
__forceinline__ __device__ void st_relaxed_sys(void* ptr, dtype_t value) {
    if constexpr (sizeof(dtype_t) == 4) {
        uint32_t int_value = reinterpret_cast<const uint32_t&>(value);
        asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(ptr), "r"(int_value));
    } else if constexpr (sizeof(dtype_t) == 8) {
        uint64_t int_value = reinterpret_cast<const uint64_t&>(value);
        asm volatile("st.relaxed.sys.global.u64 [%0], %1;" :: "l"(ptr), "l"(int_value));
    } else {
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");
    }
}

template <typename dtype_t>
__forceinline__ __device__ void st_release_sys(void* ptr, dtype_t value) {
    if constexpr (sizeof(dtype_t) == 4) {
        uint32_t int_value = reinterpret_cast<const uint32_t&>(value);
        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(ptr), "r"(int_value));
    } else if constexpr (sizeof(dtype_t) == 8) {
        uint64_t int_value = reinterpret_cast<const uint64_t&>(value);
        asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(ptr), "l"(int_value));
    } else {
        EP_STATIC_ASSERT(sizeof(dtype_t) == 4 or sizeof(dtype_t) == 8, "Invalid data type length");
    }
}

// Adjust registers
// Initial register adjustment assuming the budget for __launch_bounds__(kNumThreads, 1).
template <int kNumRegs, int kNumThreads>
__device__ __forceinline__ void warpgroup_reg_realloc() {
    EP_STATIC_ASSERT(kNumRegs % 8 == 0 and kNumRegs >= 24 and kNumRegs <= 256, "Invalid register target");
    constexpr int kNumInitialRegisters = (65536 / kNumThreads / 8) * 8;

    if constexpr (kNumInitialRegisters > 255) {
        return;
    }

    if constexpr (kNumRegs >= kNumInitialRegisters)
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));
    else
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(kNumRegs));
}

/// General fences
__device__ __forceinline__ void fence_acquire_gpu() {
    asm volatile("fence.acquire.gpu;" ::: "memory");
}

__device__ __forceinline__ void fence_acq_rel_sys() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

/// Intrinsics
template <typename dtype_t>
__device__ __forceinline__ dtype_t exchange(dtype_t ptr, const int& src_lane_idx) {
    EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");
    const auto send_int_values = reinterpret_cast<int*>(&ptr);
    dtype_t recv_dtype;
    auto recv_int_values = reinterpret_cast<int*>(&recv_dtype);
    #pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], src_lane_idx);
    return recv_dtype;
}

__device__ __forceinline__ unsigned gather(const bool& value) {
    return __ballot_sync(0xffffffff, value);
}

__device__ __forceinline__ bool all(const bool& value) {
    return __all_sync(0xffffffff, value);
}

__device__ __forceinline__ bool any(const bool& value) {
    return __any_sync(0xffffffff, value);
}

__device__ __forceinline__ unsigned reduce_or(const unsigned& value) {
    return __reduce_or_sync(0xffffffff, value);
}

__device__ __forceinline__ unsigned long long reduce_or(const unsigned long long& value) {
    const auto low = __reduce_or_sync(0xffffffff, static_cast<unsigned>(value));
    const auto high = __reduce_or_sync(0xffffffff, static_cast<unsigned>(value >> 32));
    return (static_cast<unsigned long long>(high) << 32) | low;
}

__device__ __forceinline__ int reduce_add(const int& value) {
    return __reduce_add_sync(0xffffffff, value);
}

__device__ __forceinline__ unsigned match(const int& value) {
    return __match_any_sync(0xffffffff, value);
}

__device__ __forceinline__ int fns(const unsigned& value, const int& offset) {
    return __fns(value, 0, offset);
}

template <typename dtype_t>
__device__ __forceinline__ auto ffs(const dtype_t& value) {
    if constexpr (sizeof(dtype_t) == 4) {
        return __ffs(static_cast<int>(value)) - 1;
    } else {
        EP_STATIC_ASSERT(sizeof(dtype_t) == 8, "Invalid data type");
        return __ffsll(static_cast<long long>(value)) - 1;
    }
}

__device__ __forceinline__ int get_master_lane_idx(const unsigned& mask) {
    // Equivalent to `31 - __clz(mask)`
    int highest_idx;
    asm volatile("bfind.u32 %0, %1;" : "=r"(highest_idx) : "r"(mask));
    return highest_idx;
}

__device__ __forceinline__ bool deduplicate(const int& value, const int& lane_idx) {
    return get_master_lane_idx(match(value)) == lane_idx;
}

__device__ __forceinline__ int warp_inclusive_sum(int value, const int& lane_idx) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const auto synced = __shfl_up_sync(0xffffffff, value, offset);
        if (lane_idx >= offset)
            value += synced;
    }
    return value;
}

__device__ __forceinline__ int warp_exclusive_sum(const int& value, const int& lane_idx) {
    return warp_inclusive_sum(value, lane_idx) - value;
}

__device__ __forceinline__ float4 fadd4(const float4& a, const float4& b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    const auto xy = __fadd2_rn(make_float2(a.x, a.y), make_float2(b.x, b.y));
    const auto zw = __fadd2_rn(make_float2(a.z, a.w), make_float2(b.z, b.w));
    return make_float4(xy.x, xy.y, zw.x, zw.y);
#else
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
#endif
}

__device__ __forceinline__ float4 fmul4(const float4& a, const float4& b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    const auto xy = __fmul2_rn(make_float2(a.x, a.y), make_float2(b.x, b.y));
    const auto zw = __fmul2_rn(make_float2(a.z, a.w), make_float2(b.z, b.w));
    return make_float4(xy.x, xy.y, zw.x, zw.y);
#else
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
#endif
}

__device__ __forceinline__ float4 fmul4(const float4& a, const float& b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    const auto b2 = make_float2(b, b);
    const auto xy = __fmul2_rn(make_float2(a.x, a.y), b2);
    const auto zw = __fmul2_rn(make_float2(a.z, a.w), b2);
    return make_float4(xy.x, xy.y, zw.x, zw.y);
#else
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
#endif
}

__device__ __forceinline__ void accumulate(float2& a, nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    // Use `add.rn.f32.bf16` instruction to perform fused (cast + add) operation on SM100
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.x) : "h"(*reinterpret_cast<uint16_t*>(&b.x)));
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.y) : "h"(*reinterpret_cast<uint16_t*>(&b.y)));
#else
    const auto [x, y] = __bfloat1622float2(b);
    a.x += x, a.y += y;
#endif
}

__device__ __forceinline__ nv_bfloat162 cvt_rs_bf16x2(const float2& value, const uint32_t& bits) {
    uint32_t result;
    asm("cvt.rs.bf16x2.f32 %0, %2, %1, %3;" : "=r"(result) : "f"(value.x), "f"(value.y), "r"(bits));
    return reinterpret_cast<const nv_bfloat162&>(result);
}

#endif

} // namespace deep_ep::ptx

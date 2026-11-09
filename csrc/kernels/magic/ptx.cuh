#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace deep_ep::magic::ptx {

__device__ __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t* pointer) {
    uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(pointer) : "memory");
    return value;
}

__device__ __forceinline__ void st_release_sys_global(uint64_t value, uint64_t* pointer) {
    asm volatile("st.release.sys.global.b64 [%1], %0;" :: "l"(value), "l"(pointer) : "memory");
}

__device__ __forceinline__ void st_volatile_global(uint4 value, uint4* pointer) {
    asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};"
                 :: "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w), "l"(pointer)
                 : "memory");
}

__device__ __forceinline__ uint4 ld_volatile_global(const uint4* pointer) {
    uint4 value;
    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(pointer)
                 : "memory");
    return value;
}

__device__ __forceinline__ uint2 ld_volatile_global(const uint2* pointer) {
    uint2 value;
    asm volatile("ld.volatile.global.v2.b32 {%0, %1}, [%2];" : "=r"(value.x), "=r"(value.y) : "l"(pointer) : "memory");
    return value;
}

}

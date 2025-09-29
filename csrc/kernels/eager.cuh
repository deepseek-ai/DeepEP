
#ifndef __EAGER_SUPPORT_CU_H__
#define __EAGER_SUPPORT_CU_H__

#include "eager.h"

namespace deep_ep {

#include "configs.cuh"
#include "utils.cuh"
__device__ __forceinline__ uint8_t ld_acquire_sys_global(const uint8_t *ptr) {
    uint32_t ret;
    asm volatile("ld.acquire.sys.global.u8 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret & 0xff;
}

__device__ __forceinline__ int ld_acquire_sys_global(int *ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int ld_relaxed_sys_global(int *ptr) {
    int ret;
    asm volatile("ld.relaxed.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int ld_acquire_shared(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.shared.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_release_shared(const int* ptr, int value) {
    asm volatile("st.release.shared.cta.s32 [%0], %1;" :: "l"(ptr), "r"(value) : "memory");
}

__device__  __forceinline__ void st_release_cta(const uint8_t *ptr, uint8_t val) {
    asm volatile("st.release.cta.u8 [%0], %1;"::"l"(ptr), "h"(static_cast<uint16_t>(val)) : "memory");
}

__device__  __forceinline__ void st_release_cta(const uint64_t *ptr, uint64_t val) {
    asm volatile("st.release.cta.u64 [%0], %1;"::"l"(ptr), "l"(val) : "memory");
}

__device__  __forceinline__ void st_release_cta(const int4 *ptr, int4 val) {
    asm volatile("st.release.cta.v4.s32 [%0], {%1, %2, %3, %4};"
            : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__device__  __forceinline__ void st_release_sys_global(const uint8_t *ptr, uint8_t val) {
    asm volatile("st.release.sys.global.u8 [%0], %1;"::"l"(ptr), "h"(static_cast<uint16_t>(val)) : "memory");
}

__device__  __forceinline__ void st_release_sys_global(const int4 *ptr, int4 val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
            : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");
}

__device__ __forceinline__ void st_na_release(const int4 *ptr, int4 val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
            : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__forceinline__ __device__ int warp_reduce_max(int value) {
    value = max(value, __shfl_xor_sync(0xffffffff, value, 16));
    value = max(value, __shfl_xor_sync(0xffffffff, value, 8));
    value = max(value, __shfl_xor_sync(0xffffffff, value, 4));
    value = max(value, __shfl_xor_sync(0xffffffff, value, 2));
    value = max(value, __shfl_xor_sync(0xffffffff, value, 1));
    return value;
}

__forceinline__ __device__ int warp_reduce_min(int value) {
    value = __reduce_min_sync(0xffffffff, value);
    return value;
}


// x < 1048576 approximation
#define DIV255(x) (((x) * 0x8081U) >> 23)
#define DIV4080(x) DIV255((x) >> 4)


#define SHIFTED_ADDR(a) ((a) + (DIV4080(a) * PCIE_TAIL_SZ))
#define PTR_DIFF(a, b) (reinterpret_cast<const uint8_t*>(a) - reinterpret_cast<const uint8_t*>(b))
#define SHIFTED_ADDR_P(ptr, bptr) ((ptr) + DIV4080(PTR_DIFF(ptr, bptr)) * PCIE_TAIL_SZ / sizeof(decltype(*(ptr))))
#define IS_PAGE_SUB_HEAD(ptr, bptr, size) (((PTR_DIFF(ptr, bptr) == (size)) || (PTR_DIFF(ptr, bptr) % (PCIE_SEG_LEN - PCIE_TAIL_SZ)) == 0))
#define CHK_POSITION(bptr, ext_size, pn, ptotal) (reinterpret_cast<uint8_t*>(bptr) + ((pn == ptotal - 1) ? (ext_size - PCIE_TAIL_SZ) : (((pn) << PCIE_SEG_LEN_LOG) + (PCIE_SEG_LEN - PCIE_TAIL_SZ))))
#define EXT_PAGE_N(size) ((size >> PCIE_SEG_LEN_LOG) + ((size & PCIE_SEG_LEN_MASK) != 0))

#define TAG_V_OFFSET 0
#define ZTAG(tag) (tag + TAG_V_OFFSET)
#define BF16NAN(v) (((v % 0x7f) + 1) | 0x7f80)
#define NANTAG(tag) (BF16NAN(tag) | (BF16NAN(tag) << 16))
//#define NANTAG(tag) ZTAG(tag)
#define TAG_CNT_MASK 0x3fffffff
#define TAG_TYPE(tag) ((tag >> 31) & 1)
#define SHORT_TAG(tag) (((TAG_TYPE(tag) << 15) | ((((tag) & TAG_CNT_MASK) % 0x7fff) + 1)) << 16)
#define CHECK_TIME_MASK 0xffffff
#define FINAL_TIME_MASK 0x1000000


#define PARALLEL_SET_TAG(send_buf, ext_len, tagv, exec_id, exec_total, st_func) {\
    const int __pages = EXT_PAGE_N(ext_len);\
    for (int __pn = exec_id; __pn < __pages; __pn += exec_total) {\
        int *__check_ptr = reinterpret_cast<int*>(CHK_POSITION(send_buf, ext_len, __pn, __pages));\
        st_func(__check_ptr, __pn == __pages - 1 ? NANTAG(tagv) : ZTAG(tagv));\
    }\
}

#define NORMAL_ST(PTR, VALUE) *(PTR) = VALUE
#define NORMAL_LD(PTR) *(PTR)

#define LD_SHIFTED(LD_FUNC, SRC_PTR, SRC_BASE) LD_FUNC(SHIFTED_ADDR_P(SRC_PTR, SRC_BASE))

#define ST_SHIFTED(ST_FUNC, DST_PTR, DST_BASE, VALUE) ST_FUNC(SHIFTED_ADDR_P(DST_PTR, DST_BASE), VALUE)

#define N_LD_SHIFTED(SRC_PTR, SRC_BASE) LD_SHIFTED(NORMAL_LD, SRC_PTR, SRC_BASE)

#define N_ST_SHIFTED(DST_PTR, VALUE, DST_BASE) ST_SHIFTED(NORMAL_ST, DST_PTR, DST_BASE, VALUE)

#define TOKEN_OUT_OF_RANGE(idx, count) ((count) != 0 && ((idx) >= (-(count)-1)))
#define MAX_PAGES_DIV4 1
#define MAX_PAGES (MAX_PAGES_DIV4 << 2)

template <typename T>
__device__ __forceinline__ void Normal_ST(T *ptr, T& value) {
    *ptr = value;
}

template <typename T>
__device__ __forceinline__ T Normal_LD(const T *ptr) {
    return *ptr;
}

__device__ __forceinline__ void Default_Eager_Timeout_Func(const int* ptr, int value) {
    printf("[EAGER TAG CHECK TIMEOUT] ptr: %p, value: 0x%08x\n", ptr, value);
}

class EagerRDMASendBuffer {
    void *buf;
    size_t original_len;
    int tag_value;
    bool true_rdma;
public:
    void (*tag_st_func)(int *, int);
    __device__ EagerRDMASendBuffer(void *send_buf, size_t original_len, int tag_value, bool true_rdma, void (*int_st_func)(int*, int)): buf(send_buf), original_len(original_len), tag_value(tag_value), true_rdma(true_rdma), tag_st_func(int_st_func) {}
    __device__ EagerRDMASendBuffer(void *send_buf, size_t original_len, int tag_value, bool true_rdma): buf(send_buf), original_len(original_len), tag_value(tag_value), true_rdma(true_rdma), tag_st_func(nullptr) {}
    __device__ EagerRDMASendBuffer(void *send_buf, size_t original_len, int tag_value): buf(send_buf), original_len(original_len), tag_value(tag_value), true_rdma(true), tag_st_func(nullptr) {}
    template <bool kEager, typename T, typename Func, bool kNormalLDST = false>
    __device__ __forceinline__ void store(Func&& func, T* ptr, T value) {
        if constexpr (kEager) {
            EP_STATIC_ASSERT(sizeof(T) >= sizeof(int), "can not support <4 byte element read/write");
            auto ptr_diff = PTR_DIFF(ptr, buf);
            if ((ptr_diff & PCIE_SEG_LEN_MASK) == (PCIE_SEG_LEN - PCIE_TAIL_SZ)) {
                auto st_ptr = &reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buf) + original_len)[ptr_diff >> PCIE_SEG_LEN_LOG];
                auto *bp = reinterpret_cast<int*>(&value);
                if constexpr (kNormalLDST) {
                    *st_ptr = *bp;
                } else {
                    tag_st_func(st_ptr, *bp);
                }
                *bp = tag_value;
            }
            if (ptr_diff + sizeof(T) == original_len) {
                auto st_ptr = &reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buf) + original_len)[original_len >> PCIE_SEG_LEN_LOG];
                if constexpr (kNormalLDST) {
                    *st_ptr = tag_value;
                } else {
                    tag_st_func(st_ptr, tag_value);
                }
            }
        }
        if constexpr (kNormalLDST) {
            *ptr = value;
        } else {
            func(ptr, value);
        }
    }
    template <bool kEager, typename T>
    __device__ __forceinline__ void store(T* ptr, T& value) {
        this->store<kEager, T, decltype(&Normal_ST<T>), true>(&Normal_ST<T>, ptr, value);
    }
    template <bool kEager, typename T, typename Func, bool kNormalLDST = false>
    __device__ __forceinline__ void shift_store(Func&& func, T* ptr, T value) {
        if constexpr (kEager) {
            EP_STATIC_ASSERT(sizeof(T) >= sizeof(int), "can not support <4 byte element read/write");
            if constexpr (kNormalLDST) {
                N_ST_SHIFTED(ptr, value, buf);
            } else {
                ST_SHIFTED(func, ptr, buf, value);
            }
        } else {
            if constexpr (kNormalLDST) {
                *ptr = value;
            } else {
                func(ptr, value);
            }
        }
    }
    template <bool kEager, typename T>
    __device__ __forceinline__ void shift_store(T* ptr, T& value) {
        this->shift_store<kEager, T, decltype(&Normal_ST<T>), true>(&Normal_ST<T>, ptr, value);
    }
    template <bool kEager, bool kNormalLDST = false>
    __device__ __forceinline__ void shift_tag(int exec_id, int exec_total) {
        if constexpr (kEager) {
        auto ext_len = EXTEND_FOR_TAG_AND_ALIGN(original_len, AR_MSG_ALIGNMENT);
            if (!kNormalLDST) {
                PARALLEL_SET_TAG(buf, ext_len, tag_value, exec_id, exec_total, tag_st_func);
            } else {
                PARALLEL_SET_TAG(buf, ext_len, tag_value, exec_id, exec_total, NORMAL_ST);
            }
        }
    }

    template <bool kEager>
    __device__ __forceinline__ void inplace_tag(int exec_id, int exec_total, int& num_send_bytes) {
        if constexpr (kEager) {
            const int ext_len = num_send_bytes + gmem_extend_delta<kEager>();
            const int pages = EXT_PAGE_N(ext_len);
            uint8_t *rdma_send_buf = reinterpret_cast<uint8_t*>(buf);
            for (int pn = exec_id; pn < pages; pn += exec_total) {
                int st_value = pn < (pages - 1) ? *reinterpret_cast<int*>(rdma_send_buf + (pn << PCIE_SEG_LEN_LOG) + PCIE_SEG_LEN - PCIE_TAIL_SZ) : NANTAG(tag_value);
                int *st_pos = reinterpret_cast<int*>(rdma_send_buf + num_send_bytes + (pn < (pages - 1) ? pn : (MAX_PAGES)) * sizeof(int));
                NORMAL_ST(st_pos, st_value);
                int tag_v = ZTAG(tag_value);
                pn < (pages - 1) ? NORMAL_ST(reinterpret_cast<int*>(rdma_send_buf + (pn << PCIE_SEG_LEN_LOG) + PCIE_SEG_LEN - PCIE_TAIL_SZ), tag_v) : 0;
            }
            num_send_bytes = ext_len;
        }
    }

    template <bool kElasticBuffer = false>
    __device__ __forceinline__ void* buffer() {
        return kElasticBuffer ? (reinterpret_cast<int4*>(buf) + 1) : buf;
    }

    __device__ __forceinline__ size_t len() {
        return original_len;
    }

    __device__ __forceinline__ int tag_v() {
        return tag_value;
    }

    // Elastic Buffer Functions

    __device__ __forceinline__ void update_len(int new_len) {
        original_len = new_len;
    }
    template <bool kEager, bool kElasticBuffer>
    __device__ __forceinline__ void eb_tma_replace_by_lane(void *smem_ptr, void* gmem_ptr, size_t bytes, int *tag_save) {
        if constexpr (kEager) {
            if (true_rdma) {
                auto mapped_gmem_ptr = gmem_ptr_view<kEager, kElasticBuffer>(gmem_ptr);
                const auto diff = PTR_DIFF(mapped_gmem_ptr, buf);
                int __BASE_PN__ = diff >> PCIE_SEG_LEN_LOG;
                int __TAIL_PN__ = (diff + bytes) >> PCIE_SEG_LEN_LOG;
                if (__BASE_PN__ != __TAIL_PN__) {
                    int tag_tma_offset = (PCIE_SEG_LEN - PCIE_TAIL_SZ) - (diff & PCIE_SEG_LEN_MASK);
                    int* tag_tma_ptr = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(smem_ptr) + tag_tma_offset);
                    tag_save[__BASE_PN__] = *tag_tma_ptr;
                    *tag_tma_ptr = ZTAG(tag_value);
                    tma_store_fence();
                }
            }
        }
    }
    template <bool kEager, bool kElasticBuffer, typename T>
    __device__ __forceinline__ T* gmem_ptr_view(T* gmem_ptr) {
        if constexpr (kEager) {
            if constexpr (kElasticBuffer) {
                return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(gmem_ptr) + sizeof(int4));
            } else {
                return gmem_ptr;
            }
        } else {
            return gmem_ptr;
        }
    }

    template <bool kEager, bool kElasticBuffer = false>
    __device__ __forceinline__ size_t gmem_tail_extend_delta() {
        if constexpr (kEager) {
            return (MAX_PAGES_DIV4 + (kElasticBuffer ? 0 : 1)) * sizeof(int4);
        } else {
            return 0;
        }
    }

    template <bool kEager, bool kElasticBuffer = false>
    __device__ __forceinline__ size_t gmem_head_extend_delta() {
        if constexpr ((kEager) && kElasticBuffer) {
            return sizeof(int4);
        } else {
            return 0;
        }
    }

    template <bool kEager>
    __device__ __forceinline__ size_t gmem_extend_delta() {
        if constexpr (kEager) {
            return sizeof(int4) * (MAX_PAGES_DIV4 + 1);
        } else {
            return 0;
        }
    }

    template <bool kEager, bool kElasticBuffer>
    __device__ __forceinline__ void eb_tma_store_tail_tags(int exec_id, int exec_total, int* tag_save, int& num_send_bytes) {
        if constexpr (kEager) {
            if (true_rdma) {
                #pragma unroll
                for (int __pn = 0; __pn < MAX_PAGES; ++__pn) {\
                    reinterpret_cast<int*>(tag_save)[__pn] = warp_reduce_or(reinterpret_cast<int*>(tag_save)[__pn]);
                }
                if (exec_id <= MAX_PAGES) {\
                    int st_value = exec_id < MAX_PAGES ? tag_save[exec_id] : NANTAG(tag_value);
                    int *target_ptr = nullptr;
                    if constexpr (kElasticBuffer) {
                        target_ptr = exec_id < 2 ? (reinterpret_cast<int*>(buf) + exec_id + 2) : reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buf) + num_send_bytes + sizeof(int4)) + (exec_id == MAX_PAGES ? 0 : exec_id);
                        *target_ptr = st_value;
                    } else {
                        auto target_ptr = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buf) + num_send_bytes) + exec_id;
                        *gmem_ptr_view<kEager, kElasticBuffer>(target_ptr) = st_value;
                    }
                    //printf("write 0x%08x to gmem offset %lu\n", st_value, PTR_DIFF(target_ptr, buf));
                }
                if constexpr (kElasticBuffer) {
                    if (exec_id == 0) {
                        *reinterpret_cast<int2*>(buf) = make_int2(ZTAG(tag_value), num_send_bytes);
                    }
                }
                num_send_bytes += (MAX_PAGES_DIV4 + 1) * sizeof(int4);
            }
        }
    }
};

class EagerRDMARecvBuffer {
    void *buf;
    size_t original_len;
    int tag_value;
    bool true_rdma;
    int (*int_load_func)(const int*);
public:
    __device__ EagerRDMARecvBuffer(void *recv_buf, size_t original_len, int tag_value, bool true_rdma, int (*int_load_func)(const int*) = nullptr): buf(recv_buf), original_len(original_len), tag_value(tag_value), true_rdma(true_rdma), int_load_func(int_load_func) {}
    template <bool kEager, typename T, typename Func, bool kNormalLDST = false>
    __device__ __forceinline__ T inplace_load(T* ptr, Func&& func) {
        std::remove_const_t<T> ld_value = func(ptr);
        if constexpr (kEager) {
            auto ptr_diff = PTR_DIFF(ptr, buf);
            if ((ptr_diff & PCIE_SEG_LEN_MASK) >= (PCIE_SEG_LEN - PCIE_TAIL_SZ) && (ptr_diff & PCIE_SEG_LEN_MASK) < (PCIE_SEG_LEN - PCIE_TAIL_SZ + sizeof(T))) {
                auto ld_ptr = &reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buf) + original_len)[ptr_diff >> PCIE_SEG_LEN_LOG];
                int got;
                if constexpr (kNormalLDST) {
                    got = *ld_ptr;
                } else {
                    got = int_load_func(ld_ptr);
                }
                if constexpr (sizeof(T) >= sizeof(int)) {
                    *reinterpret_cast<int*>(&ld_value) = got;
                } else {
                    ld_value = static_cast<T>(got >> ((ptr_diff & PCIE_SEG_LEN_MASK) - (PCIE_SEG_LEN - PCIE_TAIL_SZ)));
                }
            }
        }
        return ld_value;
    }
    template <bool kEager, typename T>
    __device__ __forceinline__ T inplace_load(T *ptr) {
        return this->inplace_load<kEager, T, decltype(&Normal_LD<T>), true>(&Normal_LD<T>, ptr);
    }
    template <bool kEager, typename T, typename Func, bool kNormalLDST = false>
    __device__ __forceinline__ T shift_load(T* ptr, Func&& func) {
        if constexpr (kEager) {
            if constexpr (kNormalLDST) {
                return N_LD_SHIFTED(ptr, buf);
            } else {
                return LD_SHIFTED(func, ptr, buf);
            }
        } else {
            if constexpr (kNormalLDST) {
                return *ptr;
            } else {
                return func(ptr);
            }
        }
    }
    template <bool kEager, typename T>
    __device__ __forceinline__ T shift_load(T *ptr) {
        return this->shift_load<kEager, T, decltype(&Normal_LD<T>), true>(&Normal_LD<T>, ptr);
    }
    template <bool kEager, bool kCountCheck = false, typename PrintFunc>
    __device__ __forceinline__ bool shift_wait(int exec_id, int exec_total, PrintFunc func, int &count, int token_idx = 0, int *count_ptr = nullptr, int *count_cache_ptr = nullptr, bool count_use_cache = false) {
        if constexpr (kEager) {
            size_t ext_len = EXTEND_FOR_TAG_AND_ALIGN(original_len, AR_MSG_ALIGNMENT);
            int __page_n = EXT_PAGE_N(ext_len);
            for (int target = exec_id; target < __page_n; target += exec_total) {
                int ld_value;
                auto start_time = clock64();
                int* __check_ptr = reinterpret_cast<int*>(CHK_POSITION(buf, ext_len, target, __page_n));
                int expect_tag = (target == __page_n - 1) ? NANTAG(tag_value) : ZTAG(tag_value);
                while (true) {
                    ld_value = ld_acquire_sys_global(__check_ptr);
                    if (ld_value == expect_tag) break;
                    if constexpr (kCountCheck) {
                        if (count == 0) {
                            if (!count_use_cache) {
                                count = ld_relaxed_sys_global(count_ptr);
                                count = ((count & 0xffff0000) == SHORT_TAG(tag_value)) ? (count | 0xffff0000) : 0;
                                if (count != 0) {
                                    st_release_cta(count_cache_ptr, count);
                                }
                            } else {
                                count = ld_acquire_cta(count_cache_ptr);
                            }
                            if (TOKEN_OUT_OF_RANGE(token_idx, count)) {
                                break;
                            }
                        }
                    }
                    if ((clock64() - start_time) > NUM_TIMEOUT_CYCLES) {
                        func(__check_ptr, ld_value, expect_tag);
                        return false;
                    }
                }
            }
        }
        return true;
    }

    template <bool kEager, bool kElasticBuffer = false, typename PrintFunc>
    __device__ __forceinline__ bool inplace_wait(int exec_id, int exec_total, int *len_save, PrintFunc func) {
        if constexpr (kEager) {
            if (!true_rdma) return true;
            if (kElasticBuffer) {
                if (exec_id == 0) {
                    uint64_t head;
                    const auto start_time = clock64();
                    while ((int)(head = ld_acquire_sys_global(reinterpret_cast<uint64_t*>(buf))) != ZTAG(tag_value)) {
                        if ((clock64() - start_time) > NUM_TIMEOUT_CYCLES) {
                            func(reinterpret_cast<int*>(buf), static_cast<int>(head), ZTAG(tag_value));
                            return false;
                        }
                    }
                    original_len = head >> 32;
                    *len_save = original_len;
                    st_release_cta(reinterpret_cast<uint64_t*>(len_save + 2), ld_nc_global(reinterpret_cast<uint64_t*>(buf) + 1));
                }
                original_len = __shfl_sync(0xffffffff, original_len, 0);
            }
            size_t ext_len = original_len + gmem_extend_delta<kEager>();
            int __page_n = EXT_PAGE_N(ext_len);
            for (int target = exec_id; target < __page_n; target += exec_total) {
                int ld_value;
                const auto start_time = clock64();
                int* __check_ptr = reinterpret_cast<int*>(CHK_POSITION(buf, ext_len, target, __page_n));
                int expect_tag = (target == __page_n - 1) ? NANTAG(tag_value) : ZTAG(tag_value);
                while (true) {
                    ld_value = ld_acquire_sys_global(__check_ptr);
                    if (ld_value == expect_tag) break;
                    if ((clock64() - start_time) > NUM_TIMEOUT_CYCLES) {
                        func(__check_ptr, ld_value, expect_tag);
                        return false;
                    }
                }
            }
        }
        return true;
    }

    template <bool kEager, bool kElasticBuffer = false>
    __device__ __forceinline__ void inplace_restore(int exec_id, int exec_total, int* len_save) {
        if constexpr (kEager) {
            if (!true_rdma) return;
            if (kElasticBuffer) {
                original_len = *len_save;
            }
            size_t ext_len = original_len + gmem_extend_delta<kEager>();
            int __page_n = EXT_PAGE_N(ext_len);
            if (exec_id < __page_n - 1) {
                size_t ld_offset = 0;
                int save_value = 0;
                if constexpr (kElasticBuffer) {
                    if (exec_id < 2) {
                        save_value = len_save[exec_id + 2];
                    } else {
                        ld_offset = original_len + gmem_head_extend_delta<kEager, kElasticBuffer>() + sizeof(int) * exec_id;
                        save_value = *(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buf) + ld_offset));
                    }
                } else {
                    ld_offset = original_len + gmem_head_extend_delta<kEager, kElasticBuffer>() + sizeof(int) * exec_id;
                    save_value = *(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buf) + ld_offset));
                }
                const auto st_offset = ((exec_id << PCIE_SEG_LEN_LOG) + (PCIE_SEG_LEN - PCIE_TAIL_SZ));
                st_release_cta(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buf) + st_offset), save_value);
            }
        }
    }

    template <bool kEager, bool kElasticBuffer, typename T>
    __device__ __forceinline__ T* gmem_ptr_view(T* gmem_ptr) {
        if constexpr (kEager) {
            if constexpr (kElasticBuffer) {
                return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(gmem_ptr) + sizeof(int4));
            } else {
                return gmem_ptr;
            }
        } else {
            return gmem_ptr;
        }
    }

    template <bool kEager, bool kElasticBuffer = false>
    __device__ __forceinline__ size_t gmem_tail_extend_delta() {
        if constexpr (kEager) {
            return (MAX_PAGES_DIV4 + 1) * sizeof(int4);
        } else {
            return 0;
        }
    }

    template <bool kEager, bool kElasticBuffer = false>
    __device__ __forceinline__ size_t gmem_head_extend_delta() {
        if constexpr ((kEager) && kElasticBuffer) {
            return sizeof(int4);
        } else {
            return 0;
        }
    }

    template <bool kEager>
    __device__ __forceinline__ size_t gmem_extend_delta() {
        if constexpr (kEager) {
            return sizeof(int4) * (MAX_PAGES_DIV4 + 1);
        } else {
            return 0;
        }
    }

    __device__ __forceinline__ void* buffer() {
        return buf;
    }
};

#define EagerAutoAMO(ptr, value, dst_pe, qp_id) {\
    if constexpr (!kEager) {\
        nvshmemi_ibgda_amo_nonfetch_add(ptr, value, dst_pe, qp_id);\
    } else {\
        nvshmemi_ibgda_rma_p(ptr, value, dst_pe, qp_id);\
    }\
}

#define E_WRAPPER_SHIFT_LOAD(kEager, wrapper, func, ptr) (kEager ? LD_SHIFTED(func, ptr, wrapper.buffer()) : func(ptr))
#define E_WRAPPER_SHIFT_STORE(kEager, wrapper, func, ptr, value) {\
    if constexpr (kEager) {\
        ST_SHIFTED(func, ptr, wrapper.buffer(), value);\
    } else {\
        func(ptr, value);\
    }\
}
#define E_WRAPPER_SHIFT_TAG(wrapper, exec_id, exec_total) if constexpr (kEager) PARALLEL_SET_TAG(wrapper.buffer(), EXTEND_FOR_TAG_AND_ALIGN(wrapper.len(), AR_MSG_ALIGNMENT), wrapper.tag_v(), exec_id, exec_total, wrapper.tag_st_func)

#define WARP_DECIDE_INTRANODE(cache_ptr, target_rank, intra_node) {\
    if (kEager) {\
        if (lane_id == 0) {\
            intra_node = ld_acquire_cta(cache_ptr + target_rank);\
            if (intra_node == -1) {\
                intra_node = nvshmemi_is_p2p_connected(rank, target_rank);\
                st_release_cta(cache_ptr + target_rank, intra_node);\
            }\
        }\
        intra_node = __shfl_sync(0xffffffff, intra_node, 0);\
    }\
}

};

#endif // __EAGER_SUPPORT_CU_H__
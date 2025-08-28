#pragma once

#include "configs.cuh"
#include "exception.cuh"

namespace deep_ep {

template <typename dtype_t>
struct Buffer {
private:
    uint8_t* ptr;

public:
    uint32_t total_bytes;

    __device__ __forceinline__ Buffer() : ptr(nullptr), total_bytes(0) {}

    __device__ __forceinline__ Buffer(void* &gbl_ptr, uint32_t num_elems, uint32_t offset = 0) {
        total_bytes = num_elems * sizeof(dtype_t);
        ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + offset * sizeof(dtype_t);
        gbl_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ Buffer advance_also(void* &gbl_ptr) {
        gbl_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    __device__ __forceinline__ dtype_t* buffer() {
        return reinterpret_cast<dtype_t*>(ptr);
    }

    __device__ __forceinline__ dtype_t& operator[](uint32_t idx) {
        return buffer()[idx];
    }
};

template <typename dtype_t, uint32_t kNumRanks = 1>
struct AsymBuffer {
private:
    uint8_t* ptrs[kNumRanks];
    uint32_t num_bytes;

public:
    uint32_t total_bytes;

    __device__ __forceinline__ AsymBuffer(void* &gbl_ptr, uint32_t num_elems, uint32_t num_ranks,
                                          uint32_t sm_id = 0, uint32_t num_sms = 1, uint32_t offset = 0) {
        EP_STATIC_ASSERT(kNumRanks == 1, "");
        num_bytes = num_elems * sizeof(dtype_t);

        uint32_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms;
        ptrs[0] = reinterpret_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id + num_bytes * offset;
        gbl_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ AsymBuffer(void** gbl_ptrs, uint32_t num_elems, uint32_t num_ranks,
                                          uint32_t sm_id = 0, uint32_t num_sms = 1, uint32_t offset = 0) {
        EP_STATIC_ASSERT(kNumRanks > 1, "");
        num_bytes = num_elems * sizeof(dtype_t);

        uint32_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms;
        for (uint32_t i = 0; i < kNumRanks; ++ i) {
            ptrs[i] = reinterpret_cast<uint8_t*>(gbl_ptrs[i]) + per_channel_bytes * sm_id + num_bytes * offset;
            gbl_ptrs[i] = reinterpret_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;
        }
    }

    __device__ __forceinline__ void advance(uint32_t shift) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumRanks; ++ i)
            ptrs[i] = ptrs[i] + shift * sizeof(dtype_t);
    }

    __device__ __forceinline__ AsymBuffer advance_also(void* &gbl_ptr) {
        gbl_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    template<uint32_t kNumAlsoRanks>
    __device__ __forceinline__ AsymBuffer advance_also(void** gbl_ptrs) {
        for (uint32_t i = 0; i < kNumAlsoRanks; ++ i)
            gbl_ptrs[i] = reinterpret_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;
        return *this;
    }

    __device__ __forceinline__ dtype_t* buffer(uint32_t idx = 0) {
        EP_STATIC_ASSERT(kNumRanks == 1, "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t*>(ptrs[0] + num_bytes * idx);
    }

    __device__ __forceinline__ dtype_t* buffer_by(uint32_t rank_idx, uint32_t idx = 0) {
        EP_STATIC_ASSERT(kNumRanks > 1, "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t*>(ptrs[rank_idx] + num_bytes * idx);
    }
};

template <typename dtype_t, bool kDecoupled = true>
struct SymBuffer {
private:
    // NOTES: for non-decoupled case, `recv_ptr` is not used
    uint8_t* send_ptr;
    uint8_t* recv_ptr;
    uint32_t num_bytes;

public:
    uint32_t total_bytes;

    __device__ __forceinline__ SymBuffer(void* &gbl_ptr, uint32_t num_elems, uint32_t num_ranks,
                                         uint32_t sm_id = 0, uint32_t num_sms = 1) {
        num_bytes = num_elems * sizeof(dtype_t);

        uint32_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms * (static_cast<int>(kDecoupled) + 1);
        send_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id;
        recv_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * (sm_id + num_sms);
        gbl_ptr = reinterpret_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ dtype_t* send_buffer(uint32_t idx = 0) {
        EP_STATIC_ASSERT(kDecoupled, "`send_buffer` is only available for non-decoupled case");
        return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx);
    }

    __device__ __forceinline__ dtype_t* recv_buffer(uint32_t idx = 0) {
        EP_STATIC_ASSERT(kDecoupled, "`recv_buffer` is only available for non-decoupled case");
        return reinterpret_cast<dtype_t*>(recv_ptr + num_bytes * idx);
    }

    __device__ __forceinline__ dtype_t* buffer(uint32_t idx = 0) {
        EP_STATIC_ASSERT(not kDecoupled, "`buffer` is only available for decoupled case");
        return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx);
    }
};

} // namespace deep_ep

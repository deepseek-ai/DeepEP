#pragma once

// MIT License
//
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cstdint>

namespace deep_ep::elastic {

struct ProxyPutDesc {
    void* send_ptr;
    void* recv_ptr;
    int   num_bytes;
    int   dst;
};
static_assert(sizeof(ProxyPutDesc) == 24, "ProxyPutDesc must be tightly packed (2 ptrs + 2 ints)");

inline constexpr int kProxyRingDepthDefault = 20;
inline constexpr int kProxyControlWordsPerForwardWarp = 3;   // head, tail, done

// Shared-memory layout for the proxy warp hand-off rings
// (n = num_forward_warps):
//   memory layout = [ring[n*depth], head[n], tail[n], done[n]]
struct ProxyRingLayout {
    int num_forward_warps;
    int ring_depth;
    void* base;

    __forceinline__ __device__ __host__
    ProxyRingLayout(const int& num_forward_warps, const int& ring_depth, void* base = nullptr) :
        num_forward_warps(num_forward_warps), ring_depth(ring_depth), base(base) {}

    // Region starts (warp 0). Each region begins where the previous one ends.
    __forceinline__ __device__ __host__
    ProxyPutDesc* get_ring_base() const {
        return reinterpret_cast<ProxyPutDesc*>(base);
    }
    __forceinline__ __device__ __host__
    unsigned* get_head_base() const {
        return reinterpret_cast<unsigned*>(get_ring_base() + num_forward_warps * ring_depth);
    }
    __forceinline__ __device__ __host__
    unsigned* get_tail_base() const {
        return get_head_base() + num_forward_warps;
    }
    __forceinline__ __device__ __host__
    int* get_done_base() const {
        return reinterpret_cast<int*>(get_tail_base() + num_forward_warps);
    }

    // Per-forward-warp entities.
    __forceinline__ __device__ __host__
    ProxyPutDesc* get_ring(const int& forward_warp_idx) const {
        return get_ring_base() + forward_warp_idx * ring_depth;
    }
    __forceinline__ __device__ __host__
    unsigned* get_head(const int& forward_warp_idx) const {
        return get_head_base() + forward_warp_idx;
    }
    __forceinline__ __device__ __host__
    unsigned* get_tail(const int& forward_warp_idx) const {
        return get_tail_base() + forward_warp_idx;
    }
    __forceinline__ __device__ __host__
    int* get_done(const int& forward_warp_idx) const {
        return get_done_base() + forward_warp_idx;
    }

    __forceinline__ __device__ __host__
    void* get_end_ptr() const {
        return get_done_base() + num_forward_warps;
    }

    __forceinline__ __device__ __host__
    static int64_t get_num_bytes(const int& num_forward_warps, const int& ring_depth) {
        return static_cast<int64_t>(num_forward_warps) *
               (static_cast<int64_t>(ring_depth) * static_cast<int64_t>(sizeof(ProxyPutDesc)) +
                static_cast<int64_t>(kProxyControlWordsPerForwardWarp) * static_cast<int64_t>(sizeof(int)));
    }
};

}  // namespace deep_ep::elastic

#pragma once

namespace deep_ep::math {

template <typename T>
__forceinline__ __device__ __host__ T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__forceinline__ __device__ __host__ constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T, bool kDoCeilAlignment = true>
__forceinline__ __device__ __host__ T align(T a, T b) {
    return (kDoCeilAlignment ? ceil_div(a, b) : (a / b)) * b;
}

template <typename T, bool kDoCeilAlignment = true>
__forceinline__ __device__ __host__ constexpr T constexpr_align(T a, T b) {
    return (kDoCeilAlignment ? constexpr_ceil_div(a, b) : (a / b)) * b;
}

template <typename dtype_t>
__forceinline__ __device__ __host__ bool is_decoded_positive_ready(const dtype_t& value) {
    return value >= 0;
}

template <typename dtype_t>
__forceinline__ __device__ __host__ dtype_t encode_decode_positive(const dtype_t& value) {
    return -value - static_cast<dtype_t>(1);
}

template <typename dtype_t = void>
__forceinline__ __device__ __host__ dtype_t* advance_ptr(void* ptr, const int64_t num_bytes) {
    return reinterpret_cast<dtype_t*>(static_cast<int8_t*>(ptr) + num_bytes);
}

__forceinline__ __device__ __host__ ptrdiff_t ptr_diff(const void* ptr, const void* base) {
    return static_cast<const int8_t*>(ptr) - static_cast<const int8_t*>(base);
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ std::tuple<dtype_a_t, dtype_a_t> unpack2(const dtype_b_t& packed) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
    dtype_a_t x = unpacked_ptr[0], y = unpacked_ptr[1];
    return {x, y};
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}

}  // namespace deep_ep::math

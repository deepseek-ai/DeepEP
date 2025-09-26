// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <type_traits>

enum class TOKEN_DATA_TYPE { UINT16, UINT8 };

inline std::string type_to_string(TOKEN_DATA_TYPE token_data_type) {
  switch (token_data_type) {
  case TOKEN_DATA_TYPE::UINT16:
    return "uint16_t";
  case TOKEN_DATA_TYPE::UINT8:
    return "uint8_t";
  default:
    return "unknown";
  }
}

inline int get_token_data_type_size(TOKEN_DATA_TYPE token_data_type) {
  switch (token_data_type) {
  case TOKEN_DATA_TYPE::UINT16:
    return sizeof(uint16_t);
  case TOKEN_DATA_TYPE::UINT8:
    return sizeof(uint8_t);
  }
  return 0;
}


__device__ __forceinline__ bool elect_sync(uint32_t membermask) {
    uint32_t is_elected;
    asm volatile("{\n\t"
                    "  .reg .pred p;\n\t"
                    "  elect.sync _|p, %1;\n\t"
                    "  selp.u32 %0, 1, 0, p;\n\t"
                    "}\n\t"
                    : "=r"(is_elected)
                    : "r"(membermask));
    return is_elected != 0;
}

// We combine all the input config parameters to a string key
template <typename... Args>
inline std::string get_key(Args&&... args) {
  std::string result;
  std::size_t count = 0;

  // Convert the arguments to string.
  auto to_string_helper = [](auto&& t) -> std::string {
    if constexpr (std::is_arithmetic_v<std::decay_t<decltype(t)>>) {
      return std::to_string(t);
    } else {
      std::ostringstream oss;
      oss << t;
      return oss.str();
    }
  };

  ((result += to_string_helper(args) + (++count < sizeof...(args) ? "-" : "")), ...);
  return result;
}


#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t const status = call;                                           \
    if (status != cudaSuccess) {                                               \
      cudaGetLastError();                                                      \
      fprintf(stderr,                                                          \
              "CUDA error encountered at: "                                    \
              "file=%s, line=%d, "                                             \
              "call='%s', Reason=%s:%s",                                       \
              __FILE__, __LINE__, #call, cudaGetErrorName(status),             \
              cudaGetErrorString(status));                                     \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    auto result = call;                                                        \
    if (result != CUDA_SUCCESS) {                                              \
      const char *p_err_str = nullptr;                                         \
      if (cuGetErrorString(result, &p_err_str) == CUDA_ERROR_INVALID_VALUE) {  \
        p_err_str = "Unrecoginzed CU error num";                               \
      }                                                                        \
      fprintf(stderr, "CU error encountered at: "                              \
              "file=%s line=%d, call='%s' Reason=%s.\n",                       \
              __FILE__, __LINE__,                                              \
              #call, p_err_str);                                               \
      abort();                                                                 \
    }                                                                          \
  } while (0)

inline std::string convert_to_nvcc_arch_flags(std::string torch_arch_list) {
  std::stringstream ss(torch_arch_list);
  std::string item;
  std::string nvcc_arch_flags;

  while (std::getline(ss, item, ';')) {
    // Remove the dot from the item
    item.erase(std::remove(item.begin(), item.end(), '.'), item.end());
    // Generate the nvcc flags
    nvcc_arch_flags += "-gencode=arch=compute_" + item + ",code=sm_" + item + " ";
  }

  // If the nvcc_arch_flags is empty, get the cuda version from the device
  if (nvcc_arch_flags.empty()) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int cc_major, cc_minor;
    CUDA_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device));
    nvcc_arch_flags = "-arch=sm_" + std::to_string(cc_major) + std::to_string(cc_minor);
  }

  return nvcc_arch_flags;
}

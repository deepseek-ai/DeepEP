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
#include <cctype>
#include <unordered_set>
#include <type_traits>
#include <linux/types.h>
#define MAX_NUM_OF_RANKS_PER_NODE 72

enum class APP_TOKEN_DATA_TYPE { UINT16, UINT8 };

inline std::string type_to_string(APP_TOKEN_DATA_TYPE token_data_type) {
  switch (token_data_type) {
  case APP_TOKEN_DATA_TYPE::UINT16:
    return "uint16_t";
  case APP_TOKEN_DATA_TYPE::UINT8:
    return "uint8_t";
  default:
    return "unknown";
  }
}

inline int get_token_data_type_size(APP_TOKEN_DATA_TYPE token_data_type) {
  switch (token_data_type) {
  case APP_TOKEN_DATA_TYPE::UINT16:
    return sizeof(uint16_t);
  case APP_TOKEN_DATA_TYPE::UINT8:
    return sizeof(uint8_t);
  }
  return 0;
}

#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
struct dispatch_memory_region_info_t {
  __be32 token_lkey;
  __be32 token_rkey;
  __be32 prob_lkey;
  __be32 prob_rkey;
  __be32 scaling_factor_lkey;
  __be32 scaling_factor_rkey;
  __be32 flag_lkey;
  __be32 flag_rkey;
  uint64_t token_laddr;
  uint64_t token_raddr;
  uint64_t prob_laddr;
  uint64_t prob_raddr;
  uint64_t scaling_factor_laddr;
  uint64_t scaling_factor_raddr;
  uint64_t flag_laddr;
  uint64_t flag_raddr;
} __attribute__((__aligned__(8)));

struct combine_memory_region_info_t {
  __be32 token_lkey;
  __be32 token_rkey;
  __be32 prob_lkey;
  __be32 prob_rkey;
  __be32 flag_lkey;
  __be32 flag_rkey;
  uint64_t token_laddr;
  uint64_t token_raddr;
  uint64_t prob_laddr;
  uint64_t prob_raddr;
  uint64_t flag_laddr;
  uint64_t flag_raddr;
} __attribute__((__aligned__(8)));
#endif

struct DispatchBuffers {
  APP_TOKEN_DATA_TYPE data_type;
  // Input buffers from attn, only used in inter-node case
  void *        attn_input_token = nullptr;
  void *        attn_input_prob = nullptr;
  void *        attn_input_flags = nullptr;
  void *        attn_input_scaling_factor = nullptr;
  // Output buffers to experts
  void *        expert_output_token = nullptr;
  void **       expert_output_token_all_ranks = nullptr;
  float *       expert_output_prob = nullptr;
  float **      expert_output_prob_all_ranks = nullptr;
  float *       expert_output_scaling_factor = nullptr;
  float **      expert_output_scaling_factor_all_ranks = nullptr;
  // RDMA buffers for dispatch kernel.
  void *        rdma_inter_node_group_token = nullptr;
  float *       rdma_inter_node_group_prob = nullptr;
  float *       rdma_inter_node_group_scaling_factor = nullptr;
  uint64_t *    rdma_inter_node_group_flags = nullptr;
  // Misc flags
  uint32_t *    intra_node_write_completion_flags = nullptr;
  uint64_t *    expected_rdma_flag_value = nullptr;
  uint32_t *    expected_intra_node_flag_value = nullptr;
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  // qp info and mr info
  struct doca_gpu_dev_verbs_qp ** d_qps_gpu = nullptr;
  struct dispatch_memory_region_info_t * mr_info = nullptr;
#endif
};

struct CombineBuffers {
  // Input buffers from experts
  uint16_t *    expert_input_token = nullptr;
  uint16_t **   expert_input_token_all_ranks = nullptr;
  float *       expert_input_prob = nullptr;
  float **      expert_input_prob_all_ranks = nullptr;
  // Output buffers to attn, only used in inter-node case
  void *        attn_output_flags = nullptr;
  // RDMA buffers for combine kernel.
  uint16_t *    rdma_intra_node_red_token = nullptr;
  float *       rdma_intra_node_red_prob = nullptr;
  uint16_t *    rdma_inter_node_group_token = nullptr;
  float *       rdma_inter_node_group_prob = nullptr;
  uint64_t *    rdma_inter_node_group_flags = nullptr;
  // Misc flags
  uint32_t *    intra_node_write_completion_flags = nullptr;
  uint64_t *    expected_rdma_flag_value = nullptr;
  uint32_t *    expected_intra_node_flag_value = nullptr;
#ifdef HYBRID_EP_BUILD_MULTINODE_ENABLE
  // qp info and mr info
  struct doca_gpu_dev_verbs_qp ** d_qps_gpu = nullptr;
  struct combine_memory_region_info_t * mr_info = nullptr;
#endif
};

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
  // ; , => space
  for (char &c : torch_arch_list) {
    if (c == ';' || c == ',')
      c = ' ';
  }

  std::stringstream ss(torch_arch_list);
  std::string item;
  std::string nvcc_arch_flags;
  std::unordered_set<std::string> seen_arch;

  while (ss >> item) {
    if (item.empty()) {
      continue;
    }

    bool emit_ptx = false;
    auto plus_pos = item.find('+');
    // Handle the case like 80+PTX
    if (plus_pos != std::string::npos && plus_pos + 1 < item.size()) {
      std::string suffix = item.substr(plus_pos + 1);
      // ptx => PTX
      std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
      });
      if (suffix == "PTX") {
        emit_ptx = true;
      }
      item = item.substr(0, plus_pos);
    }

    item.erase(std::remove(item.begin(), item.end(), '.'), item.end());
    if (item.empty()) {
      continue;
    }

    if (seen_arch.insert(item).second) {
      nvcc_arch_flags += "-gencode=arch=compute_" + item + ",code=sm_" + item + " ";
    }
    if (emit_ptx) {
      nvcc_arch_flags += "-gencode=arch=compute_" + item + ",code=compute_" + item + " ";
    }
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

template <typename T>
inline bool grow_to(T& dst, const T& src) {
  if (dst < src) { dst = src; return true; }
  return false;
}

inline void print_ptr_info(void* p) {
  cudaPointerAttributes attr{};
  cudaError_t err = cudaPointerGetAttributes(&attr, p);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaPointerGetAttributes failed: %s\n", cudaGetErrorString(err));
    return;
  }
  cudaMemoryType memory_type;
#if CUDART_VERSION >= 10000
  memory_type = attr.type;
#else
  memory_type = attr.memoryType;
#endif
  std::string memory_type_str;
  switch (memory_type) {
    case cudaMemoryTypeHost: memory_type_str = "Host"; break;
    case cudaMemoryTypeDevice: memory_type_str = "Device"; break;
    case cudaMemoryTypeManaged: memory_type_str = "Managed"; break;
    default: memory_type_str = "Unregistered/Unknown"; break;
  }
  fprintf(stderr, "type=%s, device=%d\n", memory_type_str.c_str(), attr.device);

  // If this is a device/managed pointer, try to query its allocation range (base + size)
  if (memory_type == cudaMemoryTypeDevice || memory_type == cudaMemoryTypeManaged) {
    cuInit(0);
    CUdeviceptr base = 0;
    size_t size = 0;
    CUresult r = cuMemGetAddressRange(&base, &size, reinterpret_cast<CUdeviceptr>(p));
    fprintf(stderr, "alloc_base=%p, alloc_size=%zu bytes\n", reinterpret_cast<void*>(base), size);
  }
}
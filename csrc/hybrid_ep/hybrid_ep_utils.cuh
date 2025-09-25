#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

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



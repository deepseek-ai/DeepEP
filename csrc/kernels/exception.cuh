#pragma once

#include <exception>
#include <string>

#include "configs.cuh"

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

class EPException : public std::exception {
private:
    std::string message = {};

public:
    explicit EPException(const char* name, const char* file, const int line, const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
    }

    const char* what() const noexcept override { return message.c_str(); }
};

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd)                                                           \
    do {                                                                          \
        cudaError_t e = (cmd);                                                    \
        if (e != cudaSuccess) {                                                   \
            throw EPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
        }                                                                         \
    } while (0)
#endif

#ifndef CU_CHECK
#define CU_CHECK(cmd)                                                            \
    do {                                                                         \
        CUresult e = (cmd);                                                      \
        if (e != CUDA_SUCCESS) {                                                 \
            const char* error_str = NULL;                                        \
            cuGetErrorString(e, &error_str);                                     \
            throw EPException("CU", __FILE__, __LINE__, std::string(error_str)); \
        }                                                                        \
    } while (0)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                           \
    do {                                                               \
        if (not(cond)) {                                               \
            throw EPException("Assertion", __FILE__, __LINE__, #cond); \
        }                                                              \
    } while (0)
#endif

#ifndef EP_DEVICE_ASSERT
#define EP_DEVICE_ASSERT(cond)                                                             \
    do {                                                                                   \
        if (not(cond)) {                                                                   \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
            asm("trap;");                                                                  \
        }                                                                                  \
    } while (0)
#endif

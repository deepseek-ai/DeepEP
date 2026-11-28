#pragma once

#include <exception>
#include <string>
#include <sstream>

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

class EPException : public std::exception {
private:
    std::string message = {};

public:
    explicit EPException(const char* name, const char* file, const int line, const std::string& error) {
        std::stringstream ss;
        ss << name << " exception (" << file << ":" << line << "): " << error;
        message = ss.str();
    }

    const char* what() const noexcept override { return message.c_str(); }
};

#define EPExceptionWithLineInfo(name, message) EPException(name, __FILE__, __LINE__, message)

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                           \
    do {                                                               \
        if (not(cond)) {                                               \
            throw EPException("Assertion", __FILE__, __LINE__, #cond); \
        }                                                              \
    } while (0)
#endif

#ifndef EP_HOST_UNREACHABLE
#define EP_HOST_UNREACHABLE(reason) (throw EPException("Assertion", __FILE__, __LINE__, reason))
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

#ifndef EP_UNIFIED_ASSERT
#ifdef __CUDA_ARCH__
#define EP_UNIFIED_ASSERT(cond) EP_DEVICE_ASSERT(cond)
#else
#define EP_UNIFIED_ASSERT(cond) EP_HOST_ASSERT(cond)
#endif
#endif

#ifndef CUDA_RUNTIME_CHECK
#define CUDA_RUNTIME_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != cudaSuccess) { \
        std::stringstream ss; \
        ss << static_cast<int>(e) << " (" << cudaGetErrorName(e) << ", " << cudaGetErrorString(e) << ")"; \
        throw EPException("CUDA runtime", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

#ifndef CUDA_DRIVER_CHECK
#define CUDA_DRIVER_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != CUDA_SUCCESS) { \
        std::stringstream ss; \
        const char *name, *info; \
        ::deep_jit::cuda::driver::lazy_cuGetErrorName(e, &name); \
        ::deep_jit::cuda::driver::lazy_cuGetErrorString(e, &info); \
        ss << static_cast<int>(e) << " (" << name << ", " << info << ")"; \
        throw EPException("CUDA driver", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

#ifndef NCCL_CHECK 
#define NCCL_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != ncclSuccess) { \
        std::stringstream ss; \
        ss << static_cast<int>(e) << " (" << ncclGetLastError(nullptr) << ")"; \
        throw EPException("NCCL", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

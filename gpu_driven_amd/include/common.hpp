#ifndef COMMON_HPP
#define COMMON_HPP

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <unistd.h>

// #define DEBUG_PRINT
// HIP error checking macro
#define CHECK_HIP(call)                                            \
  do {                                                              \
    hipError_t _e = (call);                                        \
    if (_e != hipSuccess) {                                        \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, \
              hipGetErrorString(_e));                              \
      std::exit(EXIT_FAILURE);                                      \
    }                                                               \
  } while (0)

#define hipCheckErrors(msg)                                  \
  do {                                                        \
    hipError_t __err = hipGetLastError();                   \
    if (__err != hipSuccess) {                               \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, \
              hipGetErrorString(__err), __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n");             \
      exit(1);                                                \
    }                                                         \
  } while (0)

// Keep CUDA compatibility macros for easier transition
#define CHECK_CUDA CHECK_HIP
#define cudaCheckErrors hipCheckErrors

// #define REMOTE_PERSISTENT_KERNEL
#define USE_GRACE_HOPPER
#define MEASURE_PER_OP_LATENCY
#define ENABLE_WRITE_WITH_IMMEDIATE
#define ASSUME_WR_IN_ORDER
#define ENABLE_PROXY_HIP_MEMCPY
#define SYNCHRONOUS_COMPLETION
#define RDMA_BATCH_TOKENS
#define kQueueSize 1024
#define kQueueMask (kQueueSize - 1)
#define kMaxInflight 64
#define kBatchSize 32
#define kIterations 1000000
#define kNumThBlocks 6
#define kNumThPerBlock 1
#ifdef SYNCHRONOUS_COMPLETION
#define kRemoteNVLinkBatchSize \
  16  // Immediately synchronize stream for latency.
#else
#define kRemoteNVLinkBatchSize 512
#endif
#define kObjectSize 10752  // 10.5 KB
#define kMaxOutstandingSends 2048
#define kMaxOutstandingRecvs 2048
#define kSignalledEvery 1
#define kSenderAckQueueDepth 1024
#define kNumPollingThreads 0  // Rely on CPU proxy to poll.
#define kPollingThreadStartPort kNumThBlocks * 2
#define kWarmupOps 10000
#define kRemoteBufferSize kBatchSize* kNumThBlocks* kObjectSize * 100
#define MAIN_THREAD_CPU_IDX 31
#define NUM_GPUS 1
#define RECEIVER_BATCH_SIZE 16
#ifdef SYNCHRONOUS_COMPLETION
#define NVLINK_SM_PER_PROCESS \
  1  // Total number of SMs used is NVLINK_SM_PER_PROCESS * kNumThBlocks
#else
#define NVLINK_SM_PER_PROCESS 2
#endif
// #define SEPARATE_POLLING

bool pin_thread_to_cpu(int cpu);

void cpu_relax();

#endif  // COMMON_HPP
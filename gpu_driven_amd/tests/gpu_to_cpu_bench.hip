#include <chrono>
#include <thread>
#include <tuple>
#include <vector>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <hip/hip_runtime.h>

static constexpr int kNumThBlocks = 8;
static constexpr int kNumThPerBlock = 512;

#define hipCheckErrors(msg)                                   \
  do {                                                        \
    hipError_t __err = hipGetLastError();                     \
    if (__err != hipSuccess) {                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, \
              hipGetErrorString(__err), __FILE__, __LINE__);  \
      fprintf(stderr, "*** FAILED - ABORTING\n");             \
      exit(1);                                                \
    }                                                         \
  } while (0)

struct alignas(128) GPUSignal {
  volatile uint64_t cmd;
  volatile uint64_t ack;
  char _pad[128 - 16];
};

__device__ unsigned long long cycle_accum[kNumThBlocks] = {0};
__device__ unsigned int op_count[kNumThBlocks] = {0};

// GPU kernel — Each block has its own GPUSignal
__global__ void gpu_issue_command(GPUSignal* signals, int iterations) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  GPUSignal* signal = &signals[bid];

  if (tid == 0) {
    for (int i = 0; i < iterations; i++) {
      unsigned long long start = wall_clock64();
      signal->cmd = i + 1;

      while (signal->ack != (i + 1)) {
        // __nanosleep(10);
        ;
      }
      unsigned long long end = wall_clock64();
      // printf("Block %d: Command %d issued, acked in %llu cycles\n", bid, i +
      // 1, end - start);
      cycle_accum[bid] += (end - start);
      op_count[bid]++;
    }
  }
}

// CPU polling thread for each GPUSignal
void cpu_polling(GPUSignal* signal, int iterations, int block_id) {
  for (int i = 0; i < iterations; ++i) {
    while (signal->cmd != (i + 1)) {
      // std::this_thread::yield();
      ;
    }
    signal->ack = i + 1;
  }
}

// make -j
// ./gpu_to_cpu_bench
// Block 0: Average latency = 3.82 us over 1000 iterations, avg_cycles: 5391.95
int main() {
  hipStream_t stream1;
  hipStreamCreate(&stream1);
  hipCheckErrors("hipStreamCreate failed");

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  printf("clock rate: %d kHz\n", prop.clockRate);

  GPUSignal* signals;
  hipHostMalloc(&signals, sizeof(GPUSignal) * kNumThBlocks,
                hipHostMallocMapped);

  for (int i = 0; i < kNumThBlocks; ++i) {
    signals[i].cmd = 0;
    signals[i].ack = 0;
  }

  int iterations = 1000;

  // Launch one CPU polling thread per block
  std::vector<std::thread> cpu_threads;
  for (int i = 0; i < kNumThBlocks; ++i) {
    cpu_threads.emplace_back(cpu_polling, &signals[i], iterations, i);
  }

  hipLaunchKernelGGL(gpu_issue_command, kNumThBlocks, kNumThPerBlock, 0, stream1,
                     signals, iterations);
  hipCheckErrors("gpu_issue_command kernel failed");

  hipStreamSynchronize(stream1);
  hipCheckErrors("hipStreamSynchronize failed");

  for (auto& t : cpu_threads) {
    t.join();
  }

  unsigned long long host_cycle_accum[kNumThBlocks];
  unsigned int host_op_count[kNumThBlocks];
  hipMemcpyFromSymbol(host_cycle_accum, HIP_SYMBOL(cycle_accum), sizeof(host_cycle_accum));
  hipMemcpyFromSymbol(host_op_count, HIP_SYMBOL(op_count), sizeof(host_op_count));
  hipCheckErrors("hipMemcpyFromSymbol failed");

  for (int i = 0; i < kNumThBlocks; ++i) {
    double avg_cycles =
        static_cast<double>(host_cycle_accum[i]) / host_op_count[i];
    double avg_latency_us = avg_cycles * 1000 / prop.clockRate;
    printf(
        "Block %d: Average latency = %.2f us over %u iterations, avg_cycles: "
        "%.2f\n",
        i, avg_latency_us, host_op_count[i], avg_cycles);
  }

  hipHostFree(signals);
  hipCheckErrors("hipHostFree failed");
  hipStreamDestroy(stream1);
  hipCheckErrors("hipStreamDestroy failed");
}
#include "common.hpp"
#include "gpu_kernel.hip.h"
#include "proxy.hpp"
#include "rdma.hpp"
#include "ring_buffer.hip.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  if (argc > 1) {
    std::cerr << "Usage: ./benchmark_local\n";
    return 1;
  }

  GdrSupportInitOnce();
  if (!GdrSupportInitOnce()) {
    printf(
        "Error: GPUDirect RDMA module is not loaded. Please load "
        "amdgpu or rocm_smi modules!\n");
    exit(1);
  }

  hipStream_t stream1;
  hipStreamCreate(&stream1);
  hipCheckErrors("hipStreamCreate failed");

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  printf("clock rate: %d kHz\n", prop.clockRate);

  DeviceToHostCmdBuffer* rbs;
  hipHostMalloc(&rbs, sizeof(DeviceToHostCmdBuffer) * kNumThBlocks);

  for (int i = 0; i < kNumThBlocks; ++i) {
    rbs[i].head = 0;
    rbs[i].tail = 0;
    for (uint32_t j = 0; j < kQueueSize; ++j) {
      rbs[i].buf[j].cmd = 0;  // Initialize the buffer
    }
  }

  // Launch one CPU polling thread per block
  std::vector<std::thread> cpu_threads;
  for (int i = 0; i < kNumThBlocks; ++i) {
    cpu_threads.emplace_back(cpu_proxy_local, &rbs[i], i);
  }

  printf("Launching kernel with %d blocks, %d threads per block\n", kNumThBlocks,
         kNumThPerBlock);
  auto start = std::chrono::high_resolution_clock::now();

  // Launch GPU kernel
  size_t smem_size = kQueueSize * sizeof(unsigned long long);
  hipLaunchKernelGGL(gpu_issue_batched_commands, dim3(kNumThBlocks), dim3(kNumThPerBlock), 
                     smem_size, stream1, rbs);
  
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "Kernel launch failed: %s\n", hipGetErrorString(err));
    exit(1);
  }

  hipStreamSynchronize(stream1);
  hipCheckErrors("Kernel execution failed");

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  // Wait for CPU threads to finish
  for (auto& t : cpu_threads) {
    t.join();
  }

  printf("Total execution time: %ld microseconds\n", duration.count());
  
  // Calculate performance metrics
  double total_ops = kIterations * kNumThBlocks;
  double ops_per_sec = total_ops / (duration.count() / 1e6);
  printf("Operations per second: %.2f\n", ops_per_sec);

  // Cleanup
  hipStreamDestroy(stream1);
  hipHostFree(rbs);

  return 0;
}
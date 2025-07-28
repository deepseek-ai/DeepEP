#include "common.hpp"
#include "gpu_kernel.hip.h"
#include "peer_copy_worker.hpp"
#include "proxy.hpp"
#include "rdma.hpp"
#include "ring_buffer.hip.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: ./benchmark_remote <rank> <peer_ip>\n";
    return 1;
  }
  int rank = std::atoi(argv[1]);
  char const* peer_ip = argv[2];

  pin_thread_to_cpu(MAIN_THREAD_CPU_IDX);
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
      rbs[i].buf[j].cmd = 0;
    }
  }

  size_t total_size = kRemoteBufferSize;
  void* gpu_buffer = nullptr;
#ifdef USE_GRACE_HOPPER
  // GH200: use CPU-visible memory
  hipHostMalloc(&gpu_buffer, total_size);
#else
  // x86 + discrete GPU: use device memory + peermem
  hipMalloc(&gpu_buffer, total_size);
#endif

  printf("Allocated %zu bytes of GPU buffer at %p\n", total_size, gpu_buffer);

  // Initialize global RDMA resources
  RDMAConnectionInfo local_info;
  global_rdma_init(gpu_buffer, total_size, &local_info, rank);

  // Launch CPU proxy threads
  std::vector<std::thread> cpu_threads;
  CopyRingBuffer g_ring;
  g_ring.head = 0;
  g_ring.tail = 0;

  if (rank == 1) {
    // Remote node - launch peer copy workers
    g_run.store(true);
    std::vector<std::thread> peer_copy_threads;
    for (int i = 0; i < 1; ++i) {  // Single peer copy worker for now
      peer_copy_threads.emplace_back(peer_copy_worker, std::ref(g_ring), i);
    }

    // Launch remote CPU proxy threads
    for (int i = 0; i < kNumThBlocks; ++i) {
      cpu_threads.emplace_back(remote_cpu_proxy, &rbs[i], i, gpu_buffer,
                               total_size, rank, peer_ip, std::ref(g_ring));
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

    // Signal peer copy workers to stop
    g_run.store(false);
    
    // Wait for threads to finish
    for (auto& t : cpu_threads) {
      t.join();
    }
    for (auto& t : peer_copy_threads) {
      t.join();
    }

    printf("Total execution time: %ld microseconds\n", duration.count());
    
    // Calculate performance metrics
    double total_ops = kIterations * kNumThBlocks;
    double ops_per_sec = total_ops / (duration.count() / 1e6);
    printf("Operations per second: %.2f\n", ops_per_sec);

  } else {
    // Sender node (rank 0) - launch regular proxy threads
    for (int i = 0; i < kNumThBlocks; ++i) {
      cpu_threads.emplace_back(cpu_proxy, &rbs[i], i, gpu_buffer,
                               total_size, rank, peer_ip);
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
  }

  // Cleanup
  hipStreamDestroy(stream1);
  hipHostFree(rbs);
#ifdef USE_GRACE_HOPPER
  hipHostFree(gpu_buffer);
#else
  hipFree(gpu_buffer);
#endif

  printf("Benchmark completed successfully\n");
  return 0;
}
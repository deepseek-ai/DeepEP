/* batched_gpu_to_cpu_bench.hip
 *
 * make
 * ./batched_gpu_to_cpu_bench
 *
 * clock rate: 1410000 kHz
 *
 * Per-block avg latency:
 * Block 0 : 6.976 µs over 10000000 ops
 * Block 1 : 6.633 µs over 10000000 ops
 * Block 2 : 6.809 µs over 10000000 ops
 * Block 3 : 6.975 µs over 10000000 ops
 * Block 4 : 6.642 µs over 10000000 ops
 * Block 5 : 6.936 µs over 10000000 ops
 * Block 6 : 6.912 µs over 10000000 ops
 * Block 7 : 6.906 µs over 10000000 ops
 *
 * Overall avg GPU-measured latency  : 6.849 µs
 * Total cycles                       : 772522054232
 * Total ops                          : 80000000
 * End-to-end Wall-clock time        : 11481.282 ms
 * Throughput                        : 6.97 Mops/s
 *
 */

#include <atomic>
#include <chrono>
#include <thread>
#include <tuple>
#include <vector>
#include <assert.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <hip/hip_runtime.h>

static constexpr int kNumThBlocks = 8;
static constexpr int kNumThPerBlock = 1;
static constexpr int kIterations = 10000000;
static constexpr int kBatchSize = 8;  // Higher throughput but higher latency.

static constexpr uint32_t kQueueSize = 128;
static constexpr uint32_t kQueueMask = kQueueSize - 1;

#define MEASURE_PER_OP_LATENCY
// #define DEBUG_PRINT

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

struct alignas(128) Fifo {
  // Using volatile and avoiding atomics.
  uint64_t head;                      // Next slot to produce
  uint64_t tail;                      // Next slot to consume
  uint64_t volatile buf[kQueueSize];  // Payload buffer (8 bytes).
};

__device__ __forceinline__ uint64_t ld_volatile(uint64_t* ptr) {
  uint64_t ans;
  asm volatile("global_load_dwordx2 %0, %1 glc slc"
               : "=v"(ans)
               : "v"(ptr)
               : "memory");
  return ans;
}

#ifdef MEASURE_PER_OP_LATENCY
__device__ unsigned long long cycle_accum[kNumThBlocks] = {0};
__device__ unsigned int op_count[kNumThBlocks] = {0};
#endif

// Manual pipeline implementation for HIP since __pipeline_* functions are CUDA-specific
__device__ __forceinline__ void pipeline_stage_data(unsigned long long* smem_storage, 
                                                    unsigned long long idx,
                                                    unsigned long long start_time) {
  smem_storage[idx] = start_time;
  __syncthreads();
}

__global__ void gpu_issue_batched_commands(Fifo* fifos) {
  int const bid = blockIdx.x;
  int const tid = threadIdx.x;
  Fifo* my_fifo = &fifos[bid];
  if (tid != 0) {
    return;
  }

#ifdef MEASURE_PER_OP_LATENCY
  uint32_t complete = 0;
  __shared__ unsigned long long cycle_accum_smem;
  __shared__ unsigned int op_count_smem;
  cycle_accum_smem = 0ull;
  op_count_smem = 0u;
#endif

  extern __shared__ unsigned long long start_cycle_smem[];

  for (int it = 0; it < kIterations; it += kBatchSize) {
    uint64_t my_hdr;
    uint64_t cur_tail;

    unsigned int const todo =
        (it + kBatchSize <= kIterations) ? kBatchSize : (kIterations - it);

    while (true) {
      // CPU does not modify the head.
      uint64_t cur_head = my_fifo->head;
      cur_tail = ld_volatile(&my_fifo->tail);
      if (cur_head - cur_tail + todo <= kQueueSize) {
        my_fifo->head = cur_head + todo;
        my_hdr = cur_head;
        break;
        // if (atomicAdd_system(&(my_fifo->head), todo) == cur_head) {
        // my_hdr = cur_head;
        // break; // Successfully reserved a slot
        // }
      }
    }

#pragma unroll
    for (int i = 0; i < todo; ++i) {
      unsigned long long idx = (my_hdr + i) & kQueueMask;
      unsigned long long t0 = wall_clock64();
      unsigned long long cmd =
          (static_cast<uint64_t>(bid) << 32) | (it + i + 1);

      start_cycle_smem[idx] = t0;
      my_fifo->buf[idx] = cmd;
    }
    __threadfence_system();
#ifdef MEASURE_PER_OP_LATENCY
    while (complete < my_hdr + todo) {
      uint32_t cidx = complete & kQueueMask;
      if (complete < my_fifo->tail) {
        unsigned long long t1 = wall_clock64();
        unsigned long long cycles = t1 - start_cycle_smem[cidx];
        cycle_accum_smem += cycles;
        op_count_smem++;
        complete++;
      } else {
        break;
      }
    }
#endif
  }
#ifdef MEASURE_PER_OP_LATENCY
  while (complete < kIterations) {
    uint32_t cidx = complete & kQueueMask;
    while (complete >= my_fifo->tail) { /* spin */
    }

    unsigned long long t1 = wall_clock64();
    cycle_accum_smem += (t1 - start_cycle_smem[cidx]);
    ++op_count_smem;
    ++complete;
  }

  cycle_accum[bid] = cycle_accum_smem;
  op_count[bid] = op_count_smem;
#endif
}

static inline bool pin_thread_to_cpu(int cpu) {
  int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (cpu < 0 || cpu >= num_cpus) return false;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);

  pthread_t current_thread = pthread_self();

  return !pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

void cpu_proxy(Fifo* fifo, int block_idx) {
  // printf("CPU thread for block %d started\n", block_idx);
  pin_thread_to_cpu(block_idx);

  uint64_t my_tail = 0;
  for (int seen = 0; seen < kIterations; ++seen) {
    // TODO: here, if CPU caches fifo->head, it may not see the updates from
    // GPU.
    while (fifo->head == my_tail) {
#ifdef DEBUG_PRINT
      if (block_idx == 0) {
        printf(
            "CPU thread for block %d, waiting for head to advance: my_tail: "
            "%lu, head: %llu\n",
            block_idx, my_tail, fifo->head);
      }
#endif
      /* spin */
    }
    uint64_t idx = my_tail & kQueueMask;
    uint64_t cmd;
    do {
      cmd = fifo->buf[idx];
      _mm_pause();  // Avoid hammering the cacheline.
    } while (cmd == 0);

#ifdef DEBUG_PRINT
    printf(
        "CPU thread for block %d, seen: %d, my_head: %llu, my_tail: %lu, "
        "consuming cmd %llu\n",
        block_idx, seen, fifo->head, my_tail,
        static_cast<unsigned long long>(cmd));
#endif
    uint64_t expected_cmd =
        (static_cast<uint64_t>(block_idx) << 32) | (seen + 1);
    if (cmd != expected_cmd) {
      fprintf(stderr, "Error: block %d, expected cmd %llu, got %llu\n",
              block_idx, static_cast<unsigned long long>(expected_cmd),
              static_cast<unsigned long long>(cmd));
      exit(1);
    }
    fifo->buf[idx] = 0;
    // std::atomic_thread_fence(std::memory_order_release);
    my_tail++;
    fifo->tail = my_tail;
    // _mm_clflush(&(fifo->tail));
  }
}

int main() {
  hipStream_t stream1;
  hipStreamCreate(&stream1);
  hipCheckErrors("hipStreamCreate failed");

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  printf("clock rate: %d kHz\n", prop.clockRate);

  Fifo* fifos;
  hipHostMalloc(&fifos, sizeof(Fifo) * kNumThBlocks, hipHostMallocMapped);

  for (int i = 0; i < kNumThBlocks; ++i) {
    fifos[i].head = 0;
    fifos[i].tail = 0;
    for (uint32_t j = 0; j < kQueueSize; ++j) {
      fifos[i].buf[j] = 0;  // Initialize the buffer
    }
  }

  // Launch one CPU polling thread per block
  std::vector<std::thread> cpu_threads;
  for (int i = 0; i < kNumThBlocks; ++i) {
    cpu_threads.emplace_back(cpu_proxy, &fifos[i], i);
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  size_t shmem_bytes = kQueueSize * sizeof(unsigned long long);
  hipLaunchKernelGGL(gpu_issue_batched_commands, kNumThBlocks, kNumThPerBlock, shmem_bytes,
                     stream1, fifos);
  hipCheckErrors("gpu_issue_command kernel failed");

  hipStreamSynchronize(stream1);
  hipCheckErrors("hipStreamSynchronize failed");
  auto t1 = std::chrono::high_resolution_clock::now();

  for (auto& t : cpu_threads) {
    t.join();
  }

#ifdef MEASURE_PER_OP_LATENCY
  unsigned long long h_cycles[kNumThBlocks];
  unsigned int h_ops[kNumThBlocks];
  hipMemcpyFromSymbol(h_cycles, HIP_SYMBOL(cycle_accum), sizeof(h_cycles));
  hipMemcpyFromSymbol(h_ops, HIP_SYMBOL(op_count), sizeof(h_ops));
#endif

  unsigned int tot_ops = 0;
#ifdef MEASURE_PER_OP_LATENCY
  double total_us = 0;
  unsigned long long tot_cycles = 0;
  printf("\nPer-block avg latency:\n");
  for (int b = 0; b < kNumThBlocks; ++b) {
    double us = (double)h_cycles[b] * 1000.0 / prop.clockRate / h_ops[b];
    printf("  Block %d : %.3f µs over %u ops\n", b, us, h_ops[b]);
    total_us += us;
    tot_cycles += h_cycles[b];
    tot_ops += h_ops[b];
  }
#else
  tot_ops = kNumThBlocks * kIterations;
#endif
  double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double throughput = (double)(kNumThBlocks * kIterations) / (wall_ms * 1000.0);

#ifdef MEASURE_PER_OP_LATENCY
  printf("\nOverall avg GPU-measured latency  : %.3f µs\n",
         (double)tot_cycles * 1000.0 / prop.clockRate / tot_ops);
  printf("Total cycles                       : %llu\n", tot_cycles);
#endif
  printf("Total ops                          : %u\n", tot_ops);
  printf("End-to-end Wall-clock time        : %.3f ms\n", wall_ms);
  printf("Throughput                        : %.2f Mops/s\n", throughput);

  hipHostFree(fifos);
  hipCheckErrors("hipHostFree failed");
  hipStreamDestroy(stream1);
  hipCheckErrors("hipStreamDestroy failed");
}
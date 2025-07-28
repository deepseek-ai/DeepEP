#include <chrono>
#include <tuple>
#include <vector>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <hip/hip_runtime.h>

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

__device__ __forceinline__ uint64_t ld_volatile(uint64_t* ptr) {
  uint64_t ans;
  asm volatile("global_load_dwordx2 %0, %1 glc slc"
               : "=v"(ans)
               : "v"(ptr)
               : "memory");
  return ans;
}

__device__ __forceinline__ void fence_acq_rel_sys() {
  // Use AMD equivalent fence
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
  __threadfence_system();
}

__device__ __forceinline__ void st_relaxed_sys(uint64_t* ptr, uint64_t val) {
  // Use AMD equivalent relaxed store
  asm volatile("global_store_dwordx2 %0, %1"
               :
               : "v"(ptr), "v"(val)
               : "memory");
}

template <typename X, typename Y, typename Z = decltype(X() + Y())>
__host__ __device__ constexpr Z divUp(X x, Y y) {
  return (x + y - 1) / y;
}

static constexpr int kNumThBlocks = 4;
static constexpr int kNumThPerBlock = 512;

// Yang: 256 max scattered IOVs
static constexpr int kMaxIovs = 128;
// proxy.h: NCCL_PROXY_MAX_SUBS = 32, NCCL_STEPS = 8; double provisioning
static constexpr int kFifoCap = 32 * 8 / kNumThBlocks * 2;
static constexpr uint64_t kAbortTailValue = (uint64_t)-2;
static constexpr int kCopySize = 8888;
static constexpr int kTestIters = 1024;
static constexpr int kTestIovs = 128;
static constexpr int kTestSteps = kNumThBlocks * 8;

static_assert(kTestSteps <= kFifoCap * kNumThBlocks,
              "kTestSteps should be less than kFifoCap * kNumThBlocks");
static_assert(kTestIovs <= kMaxIovs, "kTestIovs should be less than kMaxIovs");

struct alignas(8) Iov {
  void* srcs[kMaxIovs];
  void* dsts[kMaxIovs];
  int lens[kMaxIovs];
  int iov_n;
  int gpu_idx;  // for debugging
  int step;     // for debugging
};

struct IovFifo {
  uint64_t head;  // GPU writes finished index
  uint64_t tail;  // CPU posts working index
  struct Iov iovs[kFifoCap];
};

__device__ __forceinline__ void copyGlobalMemory(void* dst, void* src,
                                                 int len) {
  uintptr_t src_addr = (uintptr_t)src;
  uintptr_t dst_addr = (uintptr_t)dst;
  int i = 0;

  for (; i + 8 <= len; i += 8) {
    *(uint64_t*)(dst_addr + i) = *(uint64_t*)(src_addr + i);
  }

  // Handle the remaining tail bytes (if any)
  if (i + 8 > len) {
    i -= 8;
    for (; i < len; i++) {
      *(uint8_t*)(dst_addr + i) = *(uint8_t*)(src_addr + i);
    }
  }
}

// Manual pipeline implementation for HIP
__device__ __forceinline__ void manual_pipeline_memcpy_async(void* smem_ptr, 
                                                            const void* gmem_ptr, 
                                                            size_t size) {
  // Stage data through shared memory since HIP doesn't have __pipeline_memcpy_async
  // This is a simplified staging approach
  memcpy(smem_ptr, gmem_ptr, size);
}

__device__ __forceinline__ void manual_pipeline_commit() {
  // Synchronize threads to simulate pipeline commit
  __syncthreads();
}

__device__ __forceinline__ void manual_pipeline_wait_prior(int n) {
  // Wait for all prior operations - simplified with syncthreads
  __syncthreads();
}

__device__ void kernelScatteredMemcpy(struct Iov* iov) {
  typedef float2 T;
  static constexpr int kCpAsycDepth = 8;
  __shared__ T smem[kNumThPerBlock * kCpAsycDepth];

  // Each SM is an independent worker.
  int nthreads = blockDim.x;
  int tid = threadIdx.x;
  int iov_n = iov->iov_n;

  // Speedup tricks for 1 iov copy; could be deleted for generality.
  if (iov_n == 1) {
    void** src_addrs = iov->srcs;
    void** dst_addrs = iov->dsts;
    int* iov_lens = iov->lens;

    // Yang: Doing the scattered memcpy here? directly copy to dst ptrs.
    char* src = (char*)src_addrs[0];
    char* dst = (char*)dst_addrs[0];
    int iov_len = iov_lens[0];

    // Make it t-byte aligned to avoid GPU SEGV.
    int num_packs = iov_len / 8;
    int len_per_th = divUp(num_packs, nthreads) * 8;
    int start = len_per_th * tid;
    int end = min(start + len_per_th, iov_len);
    int len = end - start;
    if (len > 0) copyGlobalMemory(dst + start, src + start, len);
    return;
  }

  // Number of threads per copy: A100 has 8 * 128bit mem transactions.
  // https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
  int nthreads_per_iov = max(8, nthreads / iov_n);
  // Ignoring some non-rounded threads
  if (tid > nthreads_per_iov * iov_n) return;

  int iov_n_per_iter = nthreads / nthreads_per_iov;
  int start_iov = tid / nthreads_per_iov;

  for (int i = start_iov; i < iov_n; i += iov_n_per_iter) {
    // Map each thread to a iov copy.
    int iov_idx = i;
    // Compute local tid within the th group assigned to this iov copy.
    int local_tid = tid % nthreads_per_iov;

    // Retrieve parameters for this copy.
    char* src_ptr = (char*)iov->srcs[iov_idx];
    char* dst_ptr = (char*)iov->dsts[iov_idx];
    int iov_len = iov->lens[iov_idx];
    if (iov_len == 0) return;

    // Copy t-byte chunks first (if possible)
    int num_full = iov_len / sizeof(T);
    T* src_T = (T*)src_ptr;
    T* dst_T = (T*)dst_ptr;

    int depth = 0;
    // Each thread in the group copies its portion of data.
    for (int j = local_tid; j < num_full; j += nthreads_per_iov) {
      // dst_T[j] = src_T[j];

      void* smemBytePtr = (void*)&smem[tid + nthreads * depth++];
      void const* gmemBytePtr = (void const*)&src_T[j];
      manual_pipeline_memcpy_async(smemBytePtr, gmemBytePtr, sizeof(T));

      if (depth == kCpAsycDepth || j + nthreads_per_iov >= num_full) {
        manual_pipeline_commit();
        manual_pipeline_wait_prior(0);
        // Copy the data from shared memory to global memory
        for (int k = 0; k < depth; k++) {
          dst_T[j - (depth - 1 - k) * nthreads_per_iov] =
              smem[tid + nthreads * k];
        }
        depth = 0;
      }
    }

    // Let only one thread in the copy group (e.g. local_tid == 0) copy
    // the tail.
    if (local_tid == 0) {
      // Handle the remaining tail bytes (if any)
      int tail_start = num_full * 8;
      for (int j = tail_start; j < iov_len; j++) {
        dst_ptr[j] = src_ptr[j];
      }
    }
  }
}

__global__ void persistKernel(struct IovFifo** fifo_vec) {
  __shared__ uint64_t cached_tail;
  __shared__ uint64_t abort_flag;
  __shared__ struct Iov* cur_iov;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int global_tid = bid * blockDim.x + tid;

  struct IovFifo* fifo = fifo_vec[bid];

  // This impossible print is necessary to make sure other kernels run.
  if (global_tid == -1) {
    printf("Persist kernel: block %d, thread %d\n", bid, tid);
  }

  // Initing per-threadblock variables
  if (tid == 0) {
    abort_flag = 0;
    cached_tail = (uint64_t)-1;
  }
  __syncthreads();

  // We should avoid all thread loading the global memory at the same, as
  // this will cause severe performance drop. while
  // (ld_volatile(&fifo->abort) == 0) {
  while (true) {
    // Each thread block loads new work from CPU.
    if (tid == 0) {
      uint64_t cur_tail;
      do {
        cur_tail = ld_volatile(&fifo->tail);

        if (cur_tail == kAbortTailValue) {
          // The CPU has posted a abort signal.
          abort_flag = 1;
          break;
        }
      } while ((int64_t)cur_tail < (int64_t)(cached_tail + 1));

      // Processing one iov at a time.
      cur_tail = cached_tail + 1;

      cached_tail = cur_tail;
      cur_iov = fifo->iovs + cached_tail % kFifoCap;
    }
    __syncthreads();
    if (abort_flag) return;

#ifdef MEASURE_KERNEL_TIME
    unsigned long long start = 0;
    unsigned long long end = 0;
    if (tid == 0) {
      start = wall_clock64();
    }
#endif
    kernelScatteredMemcpy(cur_iov);

#ifdef MEASURE_KERNEL_TIME
    if (tid == 0) {
      end = wall_clock64();
    }
#endif
    __syncthreads();

    // Post the finished work to the GPU
    if (tid == 0) {
#ifdef MEASURE_KERNEL_TIME
      printf("Block %d, iov_n %d, cycles = %llu\n", bid, cur_iov->iov_n,
             end - start);
#endif
      fence_acq_rel_sys();
      st_relaxed_sys(&fifo->head, cached_tail);
    }
  }
}

class iovMultiFifo {
  static constexpr int kMaxFifos = 32;

  int num_fifo_;
  volatile struct IovFifo* fifo_vec_[kMaxFifos];
  uint64_t fifo_slot_idx_[kMaxFifos];  // monotonic increasing index

 public:
  iovMultiFifo(int num_fifo) : num_fifo_(num_fifo) {
    for (int i = 0; i < num_fifo; i++) {
      hipHostMalloc(&fifo_vec_[i], sizeof(struct IovFifo), hipHostMallocMapped);
      hipCheckErrors("hipHostMalloc failed");
    }
    init();
  }

  // Init all fifo head, tail, iov_n, and slot_idx.
  void init() {
    for (int i = 0; i < num_fifo_; i++) {
      // Initialize the fifo
      volatile struct IovFifo* fifo = fifo_vec_[i];
      fifo->head = (uint64_t)-1;
      fifo->tail = (uint64_t)-1;
      for (int j = 0; j < kFifoCap; j++) {
        fifo->iovs[j].iov_n = -1;
      }
      __sync_synchronize();

      fifo_slot_idx_[i] = 0;
    }
  }

  ~iovMultiFifo() {
    for (int i = 0; i < num_fifo_; i++) {
      hipHostFree((void*)fifo_vec_[i]);
      hipCheckErrors("hipHostFree failed");
    }
  }

  struct IovFifo** get_fifo_vec() {
    return (struct IovFifo**)fifo_vec_;
  }

  std::tuple<uint64_t, volatile struct Iov*> reserve_fifo_slot(int fifo_idx) {
    auto slot_idx = fifo_slot_idx_[fifo_idx]++;
    auto reserved_iov = fifo_vec_[fifo_idx]->iovs + slot_idx % kFifoCap;
    return {slot_idx, reserved_iov};
  }

  void dispatch_task(int fifo_idx) {
    auto slot_idx = fifo_slot_idx_[fifo_idx] - 1;
    fifo_vec_[fifo_idx]->tail = slot_idx;
    __sync_synchronize();
  }

  bool check_completion(int fifo_idx, uint64_t slot_idx) {
    assert(slot_idx < fifo_slot_idx_[fifo_idx]);
    // The intial -1 will always be less than 0.
    return (int64_t)fifo_vec_[fifo_idx]->head >= (int64_t)slot_idx;
  }

  void abort(int fifo_idx) {
    fifo_vec_[fifo_idx]->tail = kAbortTailValue;
    __sync_synchronize();
  }
};

void fill_data(void** srcs_gpu, int iov_n, int* lens, uint8_t value,
               hipStream_t stream) {
  // make a CPU buffer, then copy to GPU
  uint8_t* cpu_buf = (uint8_t*)malloc(kCopySize);
  for (int i = 0; i < kCopySize / sizeof(uint8_t); i++) {
    cpu_buf[i] = value;
  }
  for (int i = 0; i < iov_n; i++) {
    hipMemcpyAsync(srcs_gpu[i], cpu_buf, lens[i], hipMemcpyHostToDevice,
                    stream);
    hipStreamSynchronize(stream);
    hipCheckErrors("hipMemcpy failed");
  }
  free(cpu_buf);
}

void check_data(void** dsts_gpu, int iov_n, int* lens, uint8_t value,
                hipStream_t stream) {
  // check the data
  uint8_t* cpu_buf = (uint8_t*)malloc(kCopySize);
  for (int i = 0; i < iov_n; i++) {
    hipMemcpyAsync(cpu_buf, dsts_gpu[i], lens[i], hipMemcpyDeviceToHost,
                    stream);
    hipStreamSynchronize(stream);
    hipCheckErrors("hipMemcpy failed");
    for (int j = 0; j < lens[i] / sizeof(uint8_t); j++) {
      assert(cpu_buf[j] == value);
    }
  }
  free(cpu_buf);
}

// make -j
// ./pcie_bench
// 11 us for 1 test iov, 82 us for 128 test iovs
int main() {
  hipStream_t stream1, stream2;
  hipStreamCreate(&stream1);
  hipCheckErrors("hipStreamCreate failed");
  hipStreamCreate(&stream2);
  hipCheckErrors("hipStreamCreate failed");

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  // Use printf for clock rate
  printf("clock rate: %d\n", prop.clockRate);

  // Preallocate a iov work item.
  struct Iov* cpu_iov = (struct Iov*)malloc(sizeof(struct Iov));
  int copy_size_once = 0;
  for (int i = 0; i < kTestIovs; i++) {
    hipMalloc(&cpu_iov->srcs[i], kCopySize);
    hipMalloc(&cpu_iov->dsts[i], kCopySize);
    cpu_iov->lens[i] = kCopySize - rand() % 2048;
    copy_size_once += cpu_iov->lens[i];
  }
  fill_data((void**)cpu_iov->srcs, kTestIovs, cpu_iov->lens, 0, stream2);

  // Create a iovMultiFifo object.
  iovMultiFifo* fifo = new iovMultiFifo(kNumThBlocks);

  // Launch the persist kernel.
  hipLaunchKernelGGL(persistKernel, kNumThBlocks, kNumThPerBlock, 0, stream1,
                     fifo->get_fifo_vec());
  hipCheckErrors("persistKernel failed");

  for (int i = 0; i < kTestIters; i += kTestSteps) {
    std::vector<std::tuple<uint64_t, uint64_t>> poll_handlers;

    for (int k = 0; k < kTestSteps; k++) {
      auto fifo_idx = (i + k) % kNumThBlocks;
      fill_data((void**)cpu_iov->srcs, kTestIovs, cpu_iov->lens, i, stream2);

      // Reserve a slot in the FIFO.
      auto [slot_idx, gpu_iov] = fifo->reserve_fifo_slot(fifo_idx);
      for (int j = 0; j < kTestIovs; j++) {
        gpu_iov->srcs[j] = cpu_iov->srcs[j];
        gpu_iov->dsts[j] = cpu_iov->dsts[j];
        gpu_iov->lens[j] = cpu_iov->lens[j];
      }
      gpu_iov->iov_n = kTestIovs;

      // CPU dispatches work to GPU.
      fifo->dispatch_task(fifo_idx);

      // for (int j = 0; j < kTestIovs; j++) {
      //     hipMemcpyAsync(gpu_iov->dsts[j], gpu_iov->srcs[j],
      //                     gpu_iov->lens[j],
      //                     hipMemcpyDeviceToDevice, stream2);
      //     hipCheckErrors("hipMemcpyAsync failed");
      // }
      // hipStreamSynchronize(stream2);

      poll_handlers.push_back(std::make_tuple(fifo_idx, slot_idx));
    }

    // CPU side work.
    auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < kTestSteps; k++) {
      auto [fifo_idx, slot_idx] = poll_handlers[k];
      // Wait for the GPU to finish the work.
      while (fifo->check_completion(fifo_idx, slot_idx) == false) {
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    check_data((void**)cpu_iov->dsts, kTestIovs, cpu_iov->lens, i, stream2);

    if (i % 128 == 0) {
      auto elapsed_us =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      auto bw_GBps =
          copy_size_once * kTestSteps * 1.0 / elapsed_us.count() / 1000;
      printf("CPU wait time: %ld us, bw: %lf GBps\n", elapsed_us.count(),
             bw_GBps);
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  // Tell the kernel to abort.
  for (int i = 0; i < kNumThBlocks; i++) {
    fifo->abort(i);
  }
  hipStreamSynchronize(stream1);
  hipCheckErrors("hipStreamSynchronize failed");
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  printf("Kernel abort time: %ld us\n", elapsed_us.count());

  delete fifo;
  return 0;
}
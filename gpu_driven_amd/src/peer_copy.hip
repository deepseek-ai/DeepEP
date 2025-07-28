// peer_copy.hip
#include "common.hpp"
#include "peer_copy.hip.h"
#include "ring_buffer.hip.h"
#include <cstdio>
#include <hip/hip_runtime.h>

__global__ void peer_copy_kernel(char const* __restrict__ src,
                                 char* __restrict__ dst, size_t num_bytes) {
  size_t idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total_threads = (gridDim.x * gridDim.y) * blockDim.x;

  for (size_t i = idx; i < num_bytes; i += total_threads) {
    dst[i] = src[i];
  }
}

hipError_t launch_peer_bulk_copy(void* dst_ptr, int dst_dev, void* src_ptr,
                                  int src_dev, size_t bytes,
                                  hipStream_t stream) {
  constexpr int threads_per_block = 256;
  size_t total_threads = (bytes + threads_per_block - 1) / threads_per_block;
  dim3 blocks;
  blocks.x = (total_threads > 65535) ? 65535
                                     : static_cast<unsigned int>(total_threads);
  blocks.y = (total_threads + 65534) / 65535;

  hipLaunchKernelGGL(peer_copy_kernel, blocks, threads_per_block, 0, stream,
      static_cast<char const*>(src_ptr), static_cast<char*>(dst_ptr), bytes);

  return hipGetLastError();
}

__device__ inline void copy128(char const* __restrict__ src,
                               char* __restrict__ dst) {
  // HIP equivalent of CUDA's uint4 vectorized copy
#if defined(__AMDGCN__)
  // Use AMD's vector load/store instructions
  typedef struct { uint32_t x, y, z, w; } uint4_hip;
  *reinterpret_cast<uint4_hip*>(dst) = *reinterpret_cast<uint4_hip const*>(src);
#else
  // Fallback implementation
  for (int i = 0; i < 16; ++i) {
    dst[i] = src[i];
  }
#endif
}

__global__ void peer_copy_kernel_vec(CopyTask const* __restrict__ tasks,
                                     int num_tasks) {
  int const task_id = blockIdx.x;
  if (task_id >= num_tasks) return;

  const CopyTask t = tasks[task_id];
  char const* __restrict__ src = static_cast<char const*>(t.src_ptr);
  char* __restrict__ dst = static_cast<char*>(t.dst_ptr);
  size_t nbytes = t.bytes;

  size_t i = threadIdx.x * 16;
  for (; i + 15 < nbytes; i += blockDim.x * 16) {
    copy128(src + i, dst + i);
  }

  if (threadIdx.x == 0) {
    for (size_t j = (nbytes & ~size_t(15)); j < nbytes; ++j) dst[j] = src[j];
  }
}

__global__ void peer_copy_kernel_vec_batched(CopyTask const* __restrict__ tasks,
                                             int num_tasks,
                                             int tasks_per_block) {
  int block_task_start = blockIdx.x * tasks_per_block;
  int tid = threadIdx.x;

  for (int i = 0; i < tasks_per_block; ++i) {
    int task_id = block_task_start + i;
    if (task_id >= num_tasks) return;

    const CopyTask t = tasks[task_id];
    char const* __restrict__ src = static_cast<char const*>(t.src_ptr);
    char* __restrict__ dst = static_cast<char*>(t.dst_ptr);
    size_t nbytes = t.bytes;

    size_t offset = tid * 16;
    for (; offset + 15 < nbytes; offset += blockDim.x * 16) {
      copy128(src + offset, dst + offset);
    }

    if (tid == 0) {
      for (size_t j = (nbytes & ~size_t(15)); j < nbytes; ++j) {
        dst[j] = src[j];
      }
    }

    __syncthreads();  // avoid interleaved accesses when multiple tasks per
                      // block
  }
}

hipError_t launch_peer_bulk_copy2(CopyTask const* host_tasks, int num_tasks,
                                   hipStream_t stream, int src_device,
                                   CopyTask*& d_tasks) {
  hipMemcpyAsync(d_tasks, host_tasks, num_tasks * sizeof(CopyTask),
                  hipMemcpyHostToDevice, stream);
  constexpr int threads_per_block = 256;
  dim3 blocks(NVLINK_SM_PER_PROCESS);
  if (false) {
    hipLaunchKernelGGL(peer_copy_kernel_vec, blocks, threads_per_block, 0, stream,
                       d_tasks, num_tasks);
  } else if (true) {
    int tasks_per_block = num_tasks / NVLINK_SM_PER_PROCESS;
    hipLaunchKernelGGL(peer_copy_kernel_vec_batched, blocks, threads_per_block, 0, stream,
        d_tasks, num_tasks, tasks_per_block);
  } else {
    int tasks_per_block = num_tasks / NVLINK_SM_PER_PROCESS;
    size_t shmem = threads_per_block * 2 /*PIPE_DEPTH*/ * sizeof(int4);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(peer_copy_kernel_vec_pipelined<2, int4>),
        blocks, threads_per_block, shmem, stream, d_tasks, num_tasks, tasks_per_block);
  }
  return hipGetLastError();
}

// Manual implementation of pipeline functionality for HIP
// Since HIP doesn't have __pipeline_* functions, we implement a simplified version
template <int PIPE_DEPTH = 2, typename VecT = int4>
__global__ void peer_copy_kernel_vec_pipelined(
    CopyTask const* __restrict__ tasks, int num_tasks, int tasks_per_block) {
  extern __shared__ uint8_t shmem_raw[];
  VecT* __restrict__ ring = reinterpret_cast<VecT*>(shmem_raw);

  int const nThreads = blockDim.x;
  int const tid = threadIdx.x;
  int const blockTask0 = blockIdx.x * tasks_per_block;

  for (int local = 0; local < tasks_per_block; ++local) {
    int const task_id = blockTask0 + local;
    if (task_id >= num_tasks) return;

    CopyTask t = tasks[task_id];
    char const* __restrict__ src = static_cast<char const*>(t.src_ptr);
    char* __restrict__ dst = static_cast<char*>(t.dst_ptr);
    const size_t nbytes = t.bytes;

    const size_t nVec = nbytes / sizeof(VecT);
    const size_t vecPerThread = divUp(nVec, nThreads);
    const size_t myFirst = tid * vecPerThread;
    const size_t myLast = min(myFirst + vecPerThread, nVec);

    // Manual pipelining implementation for HIP
    // Since we don't have __pipeline_* functions, we use a simple staging approach
    int wr = 0, rd = 0, issued = 0;

    for (size_t v = myFirst; v < myLast; ++v) {
      // Stage 1: Manual copy from global memory to shared memory
      VecT const* gptr = reinterpret_cast<VecT const*>(src + v * sizeof(VecT));
      VecT* sptr = &ring[wr * nThreads + tid];
      
      // Manual copy instead of __pipeline_memcpy_async
      *sptr = *gptr;
      __threadfence_block(); // Ensure shared memory write is visible

      ++issued;
      wr = (wr + 1) % PIPE_DEPTH;

      // Stage 2: retire oldest when PIPE_DEPTH requests in flight
      if (issued == PIPE_DEPTH) {
        __syncthreads(); // Manual synchronization instead of __pipeline_wait_prior
        size_t dstIdx = v - (PIPE_DEPTH - 1);
        *reinterpret_cast<VecT*>(dst + dstIdx * sizeof(VecT)) =
            ring[rd * nThreads + tid];
        rd = (rd + 1) % PIPE_DEPTH;
        --issued;
      }
    }

    // drain remaining inflight transactions
    while (issued) {
      --issued;
      __syncthreads(); // Manual synchronization
      size_t dstIdx = myLast - issued;
      *reinterpret_cast<VecT*>(dst + dstIdx * sizeof(VecT)) =
          ring[rd * nThreads + tid];
      rd = (rd + 1) % PIPE_DEPTH;
    }

    if (tid == 0) {
      for (size_t j = nVec * sizeof(VecT); j < nbytes; ++j) dst[j] = src[j];
    }

    __syncthreads();
  }
}

__device__ __forceinline__ unsigned long long atomicSubULL(
    unsigned long long* addr, unsigned long long val) {
  return atomicAdd(
      reinterpret_cast<unsigned long long*>(addr),
      static_cast<unsigned long long>(-static_cast<long long>(val)));
}

__device__ __forceinline__ bool pop_global(HostToDeviceNVlinkBuffer* rb,
                                           CopyTask& out) {
  // Reserve a slot atomically
  const uint64_t my_tail =
      atomicAdd(reinterpret_cast<unsigned long long*>(&rb->tail), 1ULL);

  __threadfence_system();

  // Check if we raced past the current head
  if (my_tail >= rb->head) {
    // undo reservation
    atomicSubULL(reinterpret_cast<unsigned long long*>(&rb->tail), 1ULL);
    return false;
  }

  out = rb->get_entry(my_tail);
  return true;
}

__global__ void peer_copy_kernel_vec_many(HostToDeviceNVlinkBuffer* rb) {
  unsigned const lane = threadIdx.x & 0x1F;  // 0–31

  while (true) {
    CopyTask task;
    bool have = false;
    if (lane == 0) have = pop_global(rb, task);
    have = __shfl(have, 0);
    if (!have) continue;

    unsigned long long src_ll = 0, dst_ll = 0;
    if (lane == 0) {
      src_ll = (unsigned long long)task.src_ptr;
      dst_ll = (unsigned long long)task.dst_ptr;
    }
    src_ll = __shfl(src_ll, 0);
    dst_ll = __shfl(dst_ll, 0);
    size_t nbytes = __shfl(task.bytes, 0);

    char const* __restrict__ src = (char const*)src_ll;
    char* __restrict__ dst = (char*)dst_ll;

#if defined(DEBUG) || !defined(NDEBUG)
    if (((uintptr_t)src & 0xF) || ((uintptr_t)dst & 0xF)) {
      // rare – but don't crash the whole grid
      if (lane == 0)
        for (size_t i = 0; i < nbytes; ++i) dst[i] = src[i];
      continue;
    }
#endif

    size_t offset = lane * 16;
    for (; offset + 127 < nbytes; offset += 32 * 16)
      copy128(src + offset, dst + offset);

    if (lane == 0) {
      for (size_t i = (nbytes & ~size_t(127)); i < nbytes; ++i) dst[i] = src[i];
    }
  }
}

__global__ void peer_copy_kernel_vec_persistent(HostToDeviceNVlinkBuffer* rb)
// Only one thread polls task, doesn't work.
{
  __shared__ CopyTask sm_task;

  while (true) {
    if (threadIdx.x == 0) {
      if (!rb->pop(sm_task)) sm_task.bytes = 0;
    }
    __syncthreads();
    if (sm_task.bytes == 0) {
      continue;
    }

    char const* __restrict__ src = static_cast<char const*>(sm_task.src_ptr);
    char* __restrict__ dst = static_cast<char*>(sm_task.dst_ptr);
    size_t nbytes = sm_task.bytes;

    size_t offset = threadIdx.x * 16;
    for (; offset + 127 < nbytes; offset += blockDim.x * 16)
      copy128(src + offset, dst + offset);

    if (threadIdx.x == 0) {
      for (size_t i = (nbytes & ~size_t(127)); i < nbytes; ++i) dst[i] = src[i];
    }
    __syncthreads();
  }
}

HostToDeviceNVlinkBuffer* initialize_ring_buffer_for_nvlink_forwarding(
    hipStream_t stream) {
  HostToDeviceNVlinkBuffer* rb;
  hipError_t err =
      hipHostMalloc(reinterpret_cast<void**>(&rb),
                    sizeof(HostToDeviceNVlinkBuffer));
  if (err != hipSuccess) {
    fprintf(stderr, "Error allocating ring buffer for NVLink forwarding: %s\n",
            hipGetErrorString(err));
    std::abort();
  }

  new (rb) HostToDeviceNVlinkBuffer{};
  constexpr int threads_per_block = 256;
  dim3 blocks(NVLINK_SM_PER_PROCESS);
  hipLaunchKernelGGL(peer_copy_kernel_vec_many, blocks, threads_per_block, 0, stream, rb);
  err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "Error launching kernel for NVLink forwarding: %s\n",
            hipGetErrorString(err));
    std::abort();
  }
  return rb;
}

bool post_copy_task(HostToDeviceNVlinkBuffer* rb, CopyTask const* host_tasks,
                    int num_tasks, hipStream_t stream, int src_device,
                    CopyTask*& d_tasks) {
  uint64_t cur_head = rb->head;
  uint64_t cur_tail = rb->volatile_tail();

  int free_slots = rb->capacity - (cur_head - cur_tail);
  if (free_slots < num_tasks) {
    // printf(
    //     "Not enough free slots in ring buffer: %d available, %d requested, "
    //     "rb->capacity: %d, cur_head: %lu, cur_tail: %lu\n",
    //     free_slots, num_tasks, rb->capacity, cur_head, cur_tail);
    return false;
  }
  for (int i = 0; i < num_tasks; ++i) {
    rb->set_buffer(cur_head + i, host_tasks[i]);
  }
  rb->commit_with_head(cur_head + num_tasks);
  return true;
}
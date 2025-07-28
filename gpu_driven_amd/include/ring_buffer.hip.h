#ifndef RING_BUFFER_HIP
#define RING_BUFFER_HIP

#include "common.hpp"
#include <infiniband/verbs.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>

#ifndef COPY_RING_CAP
#define COPY_RING_CAP 4096
#endif

// Command structure for each transfer
struct TransferCmd {
  uint64_t cmd;
  uint32_t dst_rank;  // remote node id (MPI-style)
  uint32_t dst_gpu;   // GPU id on remote node
  void* src_ptr;      // device pointer to data
  uint64_t bytes;     // transfer size
};

struct CopyTask {
  uint64_t wr_id;
  int dst_dev;
  void* src_ptr;
  void* dst_ptr;
  size_t bytes;
};

enum class FlowDirection { HostToDevice, DeviceToHost, HostToHost };

#if !defined(__HIP_DEVICE_COMPILE__)
#include <atomic>
#define HOST_ACQUIRE() std::atomic_thread_fence(std::memory_order_acquire)
#define HOST_RELEASE() std::atomic_thread_fence(std::memory_order_release)
#else
#define HOST_ACQUIRE()
#define HOST_RELEASE()
#endif

__device__ __forceinline__ uint64_t ld_volatile(uint64_t* ptr) {
  // Use standard volatile load for HIP compatibility
  // This provides the memory ordering guarantees we need
  volatile uint64_t* vptr = reinterpret_cast<volatile uint64_t*>(ptr);
  uint64_t ans = *vptr;
#if defined(__HIP_DEVICE_COMPILE__)
  __threadfence_system(); // Ensure memory ordering
#endif
  return ans;
}

template <typename T, FlowDirection Dir, uint32_t Capacity>
struct alignas(128) RingBuffer {
  uint64_t head;
  uint64_t tail;
  T buf[Capacity];
  uint64_t cycle_accum;
  uint64_t op_count;
  uint64_t cycle_start;
  uint64_t cycle_end;
  uint32_t capacity = Capacity;

  /* TODO(MaoZiming) to refactor */
  struct ibv_qp* ack_qp;
  ibv_mr* ack_mr = nullptr;
  uint64_t ack_buf[RECEIVER_BATCH_SIZE];

  __host__ __device__ static constexpr uint32_t mask() { return Capacity - 1; }

  __host__ __device__ __forceinline__ bool full() const {
    return head - tail == Capacity;
  }

  __host__ __device__ __forceinline__ bool empty() const {
    return head == tail;
  }

  __host__ __device__ __forceinline__ void set_buffer(int idx, T entry) {
    buf[idx & mask()] = entry;
  }

  __host__ __device__ __forceinline__ bool push(T const& item) {
    if (full()) return false;
    buf[head & mask()] = item;
    commit_with_head(head + 1);
    return true;
  }

  __host__ __forceinline__ bool pushN(T const* items, int n) {
    if (n <= 0) return true;
    uint64_t h = head;
    uint64_t t = tail;
    uint64_t free_slots = capacity - (h - t);
    if (n > static_cast<int>(free_slots)) return false;

    for (int i = 0; i < n; ++i) buf[(h + i) & mask()] = items[i];

    commit_with_head(h + n);
    return true;
  }

  __host__ __device__ __forceinline__ T get_entry(int idx) const {
    return buf[idx & mask()];
  }

  __host__ __device__ __forceinline__ void commit_with_head(int new_head) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (Dir == FlowDirection::DeviceToHost) __threadfence_system();
#else
    if constexpr (Dir == FlowDirection::DeviceToHost)
      std::atomic_thread_fence(std::memory_order_release);
    if constexpr (Dir == FlowDirection::HostToHost) HOST_RELEASE();
#endif
    head = new_head;
  }

  __host__ __device__ __forceinline__ bool pop(T& out) {
    if (empty()) return false;

#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (Dir == FlowDirection::HostToDevice) __threadfence();
#else
    if constexpr (Dir == FlowDirection::HostToHost) HOST_ACQUIRE();
#endif
    out = buf[tail & mask()];
    tail++;
    return true;
  }

  __host__ __device__ __forceinline__ int popN(T* out, int n) {
    if (n <= 0) return 0;
    uint64_t t = tail;
    uint64_t h = head;
    uint64_t avail = h - t;
    if (avail == 0) return 0;
    int cnt = (n < static_cast<int>(avail)) ? n : static_cast<int>(avail);
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (Dir == FlowDirection::HostToDevice) __threadfence();
#else
    if constexpr (Dir == FlowDirection::HostToHost) HOST_ACQUIRE();
#endif
    for (int i = 0; i < cnt; ++i) out[i] = buf[(t + i) & mask()];
    tail = t + cnt;
    return cnt;
  }

  __host__ __device__ __forceinline__ uint64_t volatile_tail() {
#if defined(__HIP_DEVICE_COMPILE__)
    return ld_volatile(&tail);
#else
    return *reinterpret_cast<volatile uint64_t const*>(&tail);
#endif
  }

  __host__ __device__ __forceinline__ uint64_t volatile_head() {
    uint64_t val;
#if defined(__HIP_DEVICE_COMPILE__)
    return ld_volatile(&head);
#elif defined(__x86_64__)
    asm volatile("movq %1, %0" : "=r"(val) : "m"(head) : "memory");
#elif defined(__aarch64__)
    asm volatile("ldr %0, [%1]" : "=r"(val) : "r"(&head) : "memory");
#else
#error "Unsupported architecture"
#endif
    return val;
  }
};

typedef RingBuffer<TransferCmd, FlowDirection::DeviceToHost, kQueueSize>
    DeviceToHostCmdBuffer;
typedef RingBuffer<CopyTask, FlowDirection::HostToDevice, COPY_RING_CAP>
    HostToDeviceNVlinkBuffer;
typedef RingBuffer<CopyTask, FlowDirection::HostToHost, COPY_RING_CAP>
    CopyRingBuffer;

#endif  // RING_BUFFER_HIP
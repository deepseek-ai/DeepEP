#include "peer_copy_worker.hpp"
#include "common.hpp"
#include "peer_copy.hip.h"
#include "proxy.hpp"
#include "rdma.hpp"
#include <mutex>

std::atomic<bool> g_run;
thread_local uint64_t async_memcpy_count = 0;
thread_local uint64_t prev_completed_async_memcpy_count = 0;
thread_local uint64_t async_memcpy_total_time = 0;
thread_local uint64_t highest_issued_wr_id = 0;
int src_device = 0;
std::once_flag peer_ok_flag[NUM_GPUS][NUM_GPUS];
thread_local CopyTask tasks[RECEIVER_BATCH_SIZE];
thread_local uint64_t task_wrs[RECEIVER_BATCH_SIZE];

void maybe_enable_peer_access(int src_dev, int dst_dev) {
  if (src_dev == dst_dev) return;
  std::call_once(peer_ok_flag[src_dev][dst_dev], [&]() {
    hipSetDevice(dst_dev);
    hipError_t err = hipDeviceEnablePeerAccess(src_dev, 0);
    if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
      fprintf(stderr, "Peer access from dst_dev=%d to src_dev=%d failed: %s\n",
              dst_dev, src_dev, hipGetErrorString(err));
    }

    hipSetDevice(src_dev);
    err = hipDeviceEnablePeerAccess(dst_dev, 0);
    if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
      fprintf(stderr, "Peer access from src_dev=%d to dst_dev=%d failed: %s\n",
              src_dev, dst_dev, hipGetErrorString(err));
    }
  });
}

void sync_and_post(CopyRingBuffer& g_ring, hipStream_t& stream, int idx) {
  // printf("async_memcpy_count: %lu, prev_completed_async_memcpy_count: %lu,
  // highest_issued_wr_id: %lu\n",
  //        async_memcpy_count, prev_completed_async_memcpy_count,
  //        highest_issued_wr_id);
  if (async_memcpy_count > prev_completed_async_memcpy_count) {
    hipError_t err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
      fprintf(stderr, "Kernel execution failed: %s\n", hipGetErrorString(err));
      std::abort();
    }
    remote_notify_sender_that_wr_id_has_completed(
        g_ring.ack_qp, highest_issued_wr_id, g_ring.ack_mr, g_ring.ack_buf,
        idx);
    prev_completed_async_memcpy_count = async_memcpy_count;
  }
}

void peer_copy_worker(CopyRingBuffer& g_ring, int idx) {
  pin_thread_to_cpu(idx + 1 + MAIN_THREAD_CPU_IDX);
  printf("Peer copy worker %d started on CPU core %d\n", idx + 1,
         sched_getcpu());

  hipStream_t stream;
  hipSetDevice(src_device);
  hipStreamCreate(&stream);
  CopyTask* d_tasks;
  hipMallocAsync(&d_tasks, RECEIVER_BATCH_SIZE * sizeof(CopyTask), stream);

#ifdef REMOTE_PERSISTENT_KERNEL
  hipStream_t persistent_stream;
  hipStreamCreate(&persistent_stream);
  HostToDeviceNVlinkBuffer* rb =
      initialize_ring_buffer_for_nvlink_forwarding(persistent_stream);
#endif

  while (g_run.load(std::memory_order_acquire)) {
    CopyTask t;
    int copy_batch_size = 0;
    if (RECEIVER_BATCH_SIZE == 1) {
      if (!g_ring.pop(t)) {
        sync_and_post(g_ring, stream, idx);
        continue;
      }
      copy_batch_size = 1;
      tasks[0] = t;
    } else {
      size_t n = g_ring.popN(tasks, RECEIVER_BATCH_SIZE);
      if (n == 0) {
        sync_and_post(g_ring, stream, idx);
        continue;
      }
      t = tasks[0];
      copy_batch_size = n;
    }

    if (copy_batch_size == 0) {
      fprintf(stderr, "Error: copy_batch_size is zero\n");
      std::abort();
    }

    for (int i = 0; i < copy_batch_size; ++i) {
      maybe_enable_peer_access(src_device, tasks[i].dst_dev);
      task_wrs[i] = tasks[i].wr_id;
    }

    highest_issued_wr_id =
        std::max(highest_issued_wr_id, task_wrs[copy_batch_size - 1]);

    auto st = std::chrono::high_resolution_clock::now();
    hipError_t err;
    std::string func_name;

    if (false) {
      err = hipMemcpyPeerAsync(t.dst_ptr, t.dst_dev, t.src_ptr, src_device,
                                t.bytes * copy_batch_size, stream);
      func_name = "hipMemcpyPeerAsync";
    } else if (false) {
      err = launch_peer_bulk_copy(t.dst_ptr, t.dst_dev, t.src_ptr, src_device,
                                  t.bytes * copy_batch_size, stream);
      func_name = "launch_peer_bulk_copy";
#ifdef REMOTE_PERSISTENT_KERNEL
    } else if (false) {
#else
    } else {
#endif
      /* The fastest among the three. */
      err = launch_peer_bulk_copy2(tasks, copy_batch_size, stream, src_device,
                                   d_tasks);
      func_name = "launch_peer_bulk_copy2";
    }
#ifdef REMOTE_PERSISTENT_KERNEL
    else {
      bool post_success = true;
      while (!post_success)
        post_success = post_copy_task(rb, tasks, copy_batch_size, stream,
                                      src_device, d_tasks);
    }
#endif
    if (err != hipSuccess) {
      fprintf(stderr, "%s failed (%s) wr_id=%llu\n", func_name.c_str(),
              hipGetErrorString(err),
              static_cast<unsigned long long>(t.wr_id));
      std::abort();
    }

    if (async_memcpy_count % kRemoteNVLinkBatchSize == 0 ||
        async_memcpy_count - prev_completed_async_memcpy_count >=
            kRemoteNVLinkBatchSize) {
      err = hipStreamSynchronize(stream);
      if (err != hipSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n",
                hipGetErrorString(err));
        std::abort();
      }

      if (copy_batch_size > 0) {
        // Post the last wr is enough.
        remote_notify_sender_that_wr_id_has_completed(
            g_ring.ack_qp, highest_issued_wr_id, g_ring.ack_mr, g_ring.ack_buf,
            idx);
      }
      prev_completed_async_memcpy_count = async_memcpy_count;
    }

    async_memcpy_count += copy_batch_size;
    async_memcpy_total_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - st)
            .count();
  }
  hipFreeAsync(d_tasks, stream);
  hipStreamSynchronize(stream);
  hipStreamDestroy(stream);
}
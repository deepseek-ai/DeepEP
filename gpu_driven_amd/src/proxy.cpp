#include "proxy.hpp"
#include "common.hpp"
#include "rdma.hpp"
#include "ring_buffer.hip.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>
#include <assert.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <unistd.h>

thread_local std::unordered_map<int,
                                std::chrono::high_resolution_clock::time_point>
    wr_id_to_start_time;
thread_local uint64_t completion_count;
thread_local uint64_t wr_time_total;

void remote_cpu_proxy(DeviceToHostCmdBuffer* rb, int block_idx,
                      void* gpu_buffer, size_t total_size, int rank,
                      char const* peer_ip, CopyRingBuffer& g_ring) {
  printf("Remote CPU thread for block %d started\n", block_idx + 1);

  per_thread_rdma_init(gpu_buffer, total_size, rank, block_idx);
  pin_thread_to_cpu(block_idx + 1);
  int cpu = sched_getcpu();
  if (cpu == -1) {
    perror("sched_getcpu");
  } else {
    printf("Thread pinned to CPU core %d\n", cpu);
  }
  ibv_cq* cq = create_per_thread_cq();
  printf("Created CQ for block %d: %p\n", block_idx + 1, cq);
  RDMAConnectionInfo local_info, remote_info;
  create_per_thread_qp(gpu_buffer, total_size, &local_info, rank, cq);

  modify_qp_to_init();
  printf("Local RDMA info: addr=0x%lx, rkey=0x%x\n", local_info.addr,
         local_info.rkey);
  exchange_connection_info(rank, peer_ip, block_idx, &local_info, &remote_info);
  printf("Exchanged remote_addr %d: 0x%lx, remote_rkey: 0x%x\n", block_idx + 1,
         remote_info.addr, remote_info.rkey);

  modify_qp_to_rtr(&remote_info);
  modify_qp_to_rts(&local_info);

  remote_addr = remote_info.addr;
  remote_rkey = remote_info.rkey;

  remote_ensure_ack_sender_resources(pd, g_ring.ack_buf, g_ring.ack_mr);
  g_ring.ack_qp = ack_qp;
  printf("Remote CPU thread for block %d: ack_qp=%p,ack_mr=%p\n", block_idx + 1,
         g_ring.ack_qp, g_ring.ack_mr);
#ifdef ENABLE_WRITE_WITH_IMMEDIATE
  post_receive_buffer_for_imm();
  remote_cpu_proxy_poll_write_with_immediate(block_idx, cq, g_ring);
#endif
}

void notify_gpu_completion(std::unordered_set<uint64_t>& finished_wrs,
                           std::mutex& finished_wrs_mutex,
                           DeviceToHostCmdBuffer* rb, int block_idx,
                           uint64_t& my_tail) {
// This assumes we don't have EFA NICs.
#ifdef ASSUME_WR_IN_ORDER
  if (finished_wrs.size() > 0) {
    {
      std::lock_guard<std::mutex> lock(finished_wrs_mutex);
      int check_i = 0;
      int actually_completed = 0;

      /* To optimize (MaoZiming.)*/
      // std::vector<uint64_t> sorted_wrs(finished_wrs.begin(),
      // finished_wrs.end());
      std::unordered_set finished_wrs_copy(finished_wrs);
      // std::sort(sorted_wrs.begin(), sorted_wrs.end());
      for (auto i : finished_wrs_copy) {
#ifdef SYNCHRONOUS_COMPLETION
        if (has_received_ack && largest_completed_wr >= i) {
          finished_wrs.erase(i);
#ifdef DEBGU_PRINT
          printf(
              "WR ID %lu is smaller than largest_completed_wr %lu, "
              "completed\n",
              i, largest_completed_wr);
#endif
        } else {
#ifdef DEBGU_PRINT
          printf(
              "WR ID %lu is larger than largest_completed_wr %lu, "
              "completed, largest_finished_wr: %lu, smallest_finished_wr:"
              "% lu\n ",
              i, largest_completed_wr,
              *std::max_element(sorted_wrs.begin(), sorted_wrs.end()),
              *std::min_element(sorted_wrs.begin(), sorted_wrs.end()));
#endif
          continue;
        }
#endif
        actually_completed++;
        rb->buf[(my_tail + check_i) & kQueueMask].cmd = 0;
        check_i++;
        auto it = wr_id_to_start_time.find(i);
        if (it == wr_id_to_start_time.end()) {
          fprintf(stderr, "Error: WR ID %lu not found in wr_id_to_start_time\n",
                  i);
          exit(1);
        }
        auto start_time = it->second;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        // printf("WR ID %lu took %ld us\n", i, duration.cou());
        wr_time_total += duration.count();
        completion_count++;
      }
#ifdef DEBUG_PRINT
      if (!finished_wrs.empty()) {
        size_t min =
            *std::min_element(finished_wrs.begin(), finished_wrs.end());
        size_t max =
            *std::max_element(finished_wrs.begin(), finished_wrs.end());
        if (min != my_tail) {
          fprintf(stderr, "WRs not contiguous at block %d: tail=%lu min=%lu\n",
                  block_idx, my_tail, min);
          exit(1);
        }
        if (max != my_tail + finished_wrs.size() - 1) {
          fprintf(stderr, "WRs not contiguous at block %d: tail=%lu max=%lu\n",
                  block_idx, my_tail, max);
          exit(1);
        }
      }
#endif
      my_tail += actually_completed;
#ifndef SYNCHRONOUS_COMPLETION
      finished_wrs.clear();
#endif
    }
#else
  for (size_t i = my_tail; i < cur_head; ++i) {
    // uint64_t cmd = rb->buf[i & kQueueMask].cmd;
    {
      std::lock_guard<std::mutex> lock(finished_wrs_mutex);
      if (finished_wrs.count(i)) {
        rb->buf[i & kQueueMask].cmd = 0;
        my_tail++;
        finished_wrs.erase(i);
      } else {
        break;
      }
    }
  }
#endif
    rb->tail = my_tail;
    // _mm_clwb(&(rb->tail));
    // _mm_sfence();
  }
}

void post_gpu_command(
    DeviceToHostCmdBuffer* rb, uint64_t& my_tail, size_t& seen, int block_idx,
    void* gpu_buffer, ibv_cq* cq, std::unordered_set<uint64_t>& finished_wrs,
    std::mutex& finished_wrs_mutex,
    std::chrono::duration<double, std::micro>& total_rdma_write_durations) {
  // Force loading rb->head from DRAM.
  uint64_t cur_head = rb->volatile_head();

  if (cur_head == my_tail) {
#ifdef DEBUG_PRINT
    if (block_idx == 0) {
      printf(
          "CPU thread for block %d, waiting for head to advance: my_tail: "
          "%lu, head: %lu\n",
          block_idx, my_tail, rb->head);
    }
#endif
    /* spin */
    cpu_relax();
    return;
  }

#ifdef DEBUG_PRINT
  printf(
      "CPU thread for block %d, seen: %d, my_head: %lu, my_tail: %lu, "
      "consuming cmd %llu\n",
      block_idx, seen, rb->head, my_tail, static_cast<unsigned long long>(cmd));
#endif

  size_t batch_size = cur_head - seen;

  if (batch_size > kMaxInflight) {
    fprintf(stderr, "Error: batch_size %zu exceeds kMaxInflight %d\n",
            batch_size, kMaxInflight);
    exit(1);
  }

  if (batch_size < 0) {
    fprintf(stderr, "Error: batch_size %zu is negative\n", batch_size);
    exit(1);
  }

  std::vector<uint64_t> wrs_to_post;
  for (size_t i = seen; i < cur_head; ++i) {
    uint64_t cmd = rb->buf[i & kQueueMask].cmd;
    if (cmd == 0) {
      fprintf(stderr, "Error: cmd at index %zu is zero, my_tail: %lu\n", i,
              my_tail);
      exit(1);
    }
    uint64_t expected_cmd = (static_cast<uint64_t>(block_idx) << 32) | (i + 1);
    if (cmd != expected_cmd) {
      fprintf(stderr, "Error: block %d, expected cmd %llu, got %llu\n",
              block_idx, static_cast<unsigned long long>(expected_cmd),
              static_cast<unsigned long long>(cmd));
      exit(1);
    }
    wrs_to_post.push_back(i);
    wr_id_to_start_time[i] =
        std::chrono::high_resolution_clock::now();  // Record start time for
                                                    // this block
  }
  seen = cur_head;

  if (wrs_to_post.size() != batch_size) {
    fprintf(stderr, "Error: wrs_to_post size %zu exceeds batch size %zu\n",
            wrs_to_post.size(), batch_size);
    exit(1);
  }

  if (!wrs_to_post.empty()) {
    auto start = std::chrono::high_resolution_clock::now();
#ifdef RDMA_BATCH_TOKENS
    post_rdma_async_batched(gpu_buffer, kObjectSize, batch_size, wrs_to_post,
                            cq, finished_wrs, finished_wrs_mutex);
#else
    post_rdma_async_chained(gpu_buffer, kObjectSize, batch_size, wrs_to_post,
                            cq, finished_wrs, finished_wrs_mutex);
#endif
    auto end = std::chrono::high_resolution_clock::now();
    total_rdma_write_durations +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  }
}

void cpu_proxy(DeviceToHostCmdBuffer* rb, int block_idx, void* gpu_buffer,
               size_t total_size, int rank, char const* peer_ip) {
  printf("CPU thread for block %d started\n", block_idx + 1);
  per_thread_rdma_init(gpu_buffer, total_size, rank, block_idx);
  printf("Pinned thread to CPU core %d\n", block_idx + 1);
  pin_thread_to_cpu(block_idx + 1);
  int cpu = sched_getcpu();
  if (cpu == -1) {
    perror("sched_getcpu");
  } else {
    printf("Thread pinned to CPU core %d\n", cpu);
  }
  printf("Before creating CQ for block %d\n", block_idx + 1);
  ibv_cq* cq = create_per_thread_cq();
  RDMAConnectionInfo local_info, remote_info;

  printf("Created CQ for block %d: %p\n", block_idx + 1, cq);
  create_per_thread_qp(gpu_buffer, total_size, &local_info, rank, cq);
  printf("Created QP for block %d\n", block_idx + 1);
  modify_qp_to_init();
  // printf("Local RDMA info: addr=0x%lx, rkey=0x%x\n", local_info.addr,
  //        local_info.rkey);
  exchange_connection_info(rank, peer_ip, block_idx, &local_info, &remote_info);
  printf(
      "Exchanged remote_addr %d: 0x%lx, remote_rkey: 0x%x, Local RDMA info: "
      "addr=0x%lx, rkey=0x%x\n",
      block_idx + 1, local_info.addr, local_info.rkey, remote_info.addr,
      remote_info.rkey);

  modify_qp_to_rtr(&remote_info);
  modify_qp_to_rts(&local_info);
  local_init_ack_recv_ring(pd, kSenderAckQueueDepth);

  remote_addr = remote_info.addr;
  remote_rkey = remote_info.rkey;

  std::unordered_set<uint64_t> finished_wrs;
  std::mutex finished_wrs_mutex;

#ifdef SEPARATE_POLLING
  std::vector<std::thread> cq_threads;
  int cpu_id = kPollingThreadStartPort + block_idx;
  cq_threads.emplace_back(per_thread_polling, cpu_id, cq, &finished_wrs,
                          &finished_wrs_mutex);
#endif

  uint64_t my_tail = 0;
  auto total_rdma_write_durations =
      std::chrono::duration<double, std::micro>::zero();

  for (size_t seen = 0; my_tail < kIterations;) {
    local_poll_completions(cq, finished_wrs, finished_wrs_mutex, block_idx);
    notify_gpu_completion(finished_wrs, finished_wrs_mutex, rb, block_idx,
                          my_tail);
    post_gpu_command(rb, my_tail, seen, block_idx, gpu_buffer, cq, finished_wrs,
                     finished_wrs_mutex, total_rdma_write_durations);
  }

  printf(
      "CPU thread for block %d finished consuming %d commands, my_tail: %ld\n",
      block_idx, kIterations, my_tail);

  printf("Average rdma write duration: %.2f us\n",
         total_rdma_write_durations.count() / kIterations);

  if (check_cq_completion()) {
    // poll completion for 1 seconds
    auto start = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count() < 1) {
      local_poll_completions(cq, finished_wrs, finished_wrs_mutex, block_idx);
      notify_gpu_completion(finished_wrs, finished_wrs_mutex, rb, block_idx,
                            my_tail);
    }
    g_progress_run.store(false);
  }
#ifdef SEPARATE_POLLING
  for (auto& t : cq_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
#endif
  printf("CPU thread for block %d finished, joined all CQ threads\n",
         block_idx);

  printf(
      "Per-wr time: %.2f us, total wr time: %lu us, "
      "completion count: %lu\n",
      (float)wr_time_total / completion_count, wr_time_total, completion_count);
}

void cpu_proxy_local(DeviceToHostCmdBuffer* rb, int block_idx) {
  // printf("CPU thread for block %d started\n", block_idx);
  pin_thread_to_cpu(block_idx + 1);

  uint64_t my_tail = 0;
  for (int seen = 0; seen < kIterations; ++seen) {
    // TODO: here, if CPU caches fifo->head, it may not see the updates from
    // GPU.
    while (rb->head == my_tail) {
#ifdef DEBUG_PRINT
      if (block_idx == 0) {
        printf(
            "CPU thread for block %d, waiting for head to advance: my_tail: "
            "%lu, head: %lu\n",
            block_idx, my_tail, rb->head);
      }
#endif
      /* spin */
    }
    uint64_t idx = my_tail & kQueueMask;
    uint64_t cmd;
    do {
      cmd = rb->buf[idx].cmd;
      cpu_relax();  // Avoid hammering the cacheline.
    } while (cmd == 0);

#ifdef DEBUG_PRINT
    printf(
        "CPU thread for block %d, seen: %d, my_head: %lu, my_tail: %lu, "
        "consuming cmd %llu\n",
        block_idx, seen, rb->head, my_tail,
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

    rb->buf[idx].cmd = 0;
    my_tail++;
    rb->tail = my_tail;
  }
}
#ifndef RDMA_HPP
#define RDMA_HPP
#include "common.hpp"
#include "ring_buffer.hip.h"
#include "unistd.h"
#include <infiniband/verbs.h>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <vector>

// Global RDMA resources
extern thread_local struct ibv_context* context;
extern thread_local struct ibv_pd* pd;
extern thread_local struct ibv_mr* mr;
extern thread_local uint32_t rkey;
extern thread_local struct ibv_qp* qp;
extern thread_local struct ibv_qp* ack_qp;
extern thread_local uintptr_t remote_addr;
extern thread_local uint32_t remote_rkey;
extern thread_local std::atomic<bool> g_progress_run;
extern thread_local uint64_t largest_completed_wr;
extern thread_local bool has_received_ack;
// extern thread_local std::unordered_set<uint64_t> finished_wrs;
// extern thread_local std::mutex finished_wrs_mutex;
// extern thread_local std::unordered_set<uint64_t> finished_wrs;

struct RDMAConnectionInfo {
  uint32_t qp_num;  // Queue pair number
  uint32_t psn;     // Packet sequence number
  uint32_t ack_qp_num;
  uint32_t ack_psn;
  uint32_t rkey;    // Memory region key
  uintptr_t addr;   // Buffer address
  uint16_t lid;     // Local ID
  uint8_t gid[16];  // Global ID for RoCE (optional)
};

// Setup RDMA resources (register GPU memory, create QP, etc.)
void setup_rdma(void* gpu_buffer, size_t size, RDMAConnectionInfo* local_info,
                int rank);

// Post an RDMA write
void rdma_write_stub(void* local_dev_ptr, size_t bytes);
void post_rdma_async(void* buf, size_t bytes, uint64_t wr_id, ibv_cq* cq,
                     std::unordered_set<uint64_t>& finished_wrs,
                     std::mutex& finished_wrs_mutex);
void post_rdma_async_chained(void* buf, size_t bytes, size_t num_wrs,
                             std::vector<uint64_t> wrs_to_post, ibv_cq* cq,
                             std::unordered_set<uint64_t>& finished_wrs,
                             std::mutex& finished_wrs_mutex);

bool GdrSupportInitOnce();
void post_receive_buffer_for_imm();

void exchange_connection_info(int rank, char const* peer_ip, int tid,
                              RDMAConnectionInfo* local,
                              RDMAConnectionInfo* remote);

void modify_qp_to_rtr(RDMAConnectionInfo* remote);

void modify_qp_to_rts(RDMAConnectionInfo* local_info);

void modify_qp_to_init();
bool check_cq_completion();
void local_poll_completions(ibv_cq* cq,
                            std::unordered_set<uint64_t>& finished_wrs,
                            std::mutex& finished_wrs_mutex, int thread_idx);

void global_rdma_init(void* gpu_buf, size_t bytes, RDMAConnectionInfo* local,
                      int rank);
void create_per_thread_qp(void* gpu_buffer, size_t size,
                          RDMAConnectionInfo* local_info, int rank, ibv_cq* cq);
ibv_cq* create_per_thread_cq();
void per_thread_polling(int thread_idx, struct ibv_cq* per_thread_cq,
                        std::unordered_set<uint64_t>* per_thread_finished_wrs,
                        std::mutex* per_thread_finished_wrs_mutex);
void remote_cpu_proxy_poll_write_with_immediate(int idx, ibv_cq* cq,
                                                CopyRingBuffer& g_ring);
void handle_peer_copy(uint64_t wr_id, uint32_t imm, int src_dev, int dst_dev,
                      void* src_ptr, void* dst_ptr, size_t num_bytes);

void per_thread_rdma_init(void* gpu_buf, size_t bytes, int rank, int block_idx);

extern void* per_GPU_device_buf[NUM_GPUS];

#ifdef ENABLE_PROXY_HIP_MEMCPY
void print_average_async_memcpy_time();
#endif

void remote_notify_sender_that_wr_id_has_completed(struct ibv_qp* local_ack_qp,
                                                   uint64_t& wr_id,
                                                   ibv_mr* local_ack_mr,
                                                   uint64_t* ack_buf,
                                                   int worker_idx);
void local_init_ack_recv_ring(struct ibv_pd* pd, int depth);
void remote_ensure_ack_sender_resources(ibv_pd* pd, uint64_t* ack_buf,
                                        ibv_mr*& ack_mr);
void remote_notify_sender_batch(struct ibv_qp* ack_qp,
                                std::vector<uint64_t> const& wr_ids,
                                ibv_mr* ack_mr, uint64_t* ack_buf);
void create_per_thread_ack_qp(void* gpu_buffer, size_t size,
                              RDMAConnectionInfo* local_info, int rank,
                              ibv_cq* cq);
void post_rdma_async_batched(void* buf, size_t bytes, size_t num_wrs,
                             std::vector<uint64_t> wrs_to_post, ibv_cq* cq,
                             std::unordered_set<uint64_t>& finished_wrs,
                             std::mutex& finished_wrs_mutex);
#endif  // RDMA_HPP
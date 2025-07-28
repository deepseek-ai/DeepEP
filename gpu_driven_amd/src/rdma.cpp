#include "rdma.hpp"
#include "common.hpp"
#include "peer_copy_worker.hpp"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include <hip/hip_runtime.h>
#include <fcntl.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include "util/util_simple.h"
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#define MAX_RETRIES 20
#define RETRY_DELAY_MS 200

// Define globals
thread_local struct ibv_context* context = nullptr;
thread_local struct ibv_pd* pd = nullptr;
thread_local struct ibv_mr* mr = nullptr;
thread_local uint32_t rkey = 0;
// Define thread_local structs
thread_local struct ibv_qp* qp = nullptr;
thread_local struct ibv_qp* ack_qp = nullptr;
thread_local uintptr_t remote_addr = 0;
thread_local uint32_t remote_rkey = 0;

constexpr int TCP_PORT = 18515;
static thread_local std::atomic<uint64_t> g_posted = 0;     // WRs posted
static thread_local std::atomic<uint64_t> g_completed = 0;  // CQEs seen
thread_local std::atomic<bool> g_progress_run{true};
std::atomic<uint64_t> send_ack_posted{0}, send_ack_completed{0};

thread_local std::vector<uint64_t> ack_recv_buf;
thread_local struct ibv_mr* ack_recv_mr;
thread_local uint64_t largest_completed_wr = 0;
thread_local bool has_received_ack = false;
thread_local std::unordered_map<uint64_t, std::vector<uint64_t>>
    wr_id_to_wr_ids;

void* per_GPU_device_buf[NUM_GPUS];

void exchange_connection_info(int rank, char const* peer_ip, int tid,
                              RDMAConnectionInfo* local,
                              RDMAConnectionInfo* remote) {
  int sockfd;
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));

  printf("Rank %d exchanging RDMA connection info with peer %s\n", rank,
         peer_ip);
  if (rank == 0) {
    // Listen
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT + tid);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(listenfd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
      perror("bind failed");
      exit(1);
    }
    listen(listenfd, 1);

    socklen_t len = sizeof(addr);
    sockfd = accept(listenfd, (struct sockaddr*)&addr, &len);
    close(listenfd);
  } else {
    // Connect
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT + tid);
    inet_pton(AF_INET, peer_ip, &addr.sin_addr);

    int retry = 0;
    while (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
      if (errno == ECONNREFUSED || errno == ENETUNREACH) {
        if (++retry > MAX_RETRIES) {
          fprintf(stderr, "Rank %d: failed to connect after %d retries\n", rank,
                  retry);
          exit(1);
        }
        usleep(RETRY_DELAY_MS * 1000);  // sleep 200 ms
        continue;
      } else {
        perror("connect failed");
        exit(1);
      }
    }
    printf("Rank %d connected to peer %s on port %d\n", rank, peer_ip,
           TCP_PORT + tid);
  }

  // Exchange info
  send(sockfd, local, sizeof(*local), 0);
  recv(sockfd, remote, sizeof(*remote), MSG_WAITALL);
  close(sockfd);

  printf(
      "Rank %d exchanged RDMA info: addr=0x%lx, rkey=0x%x, "
      "qp_num=%u, psn=%u\n",
      rank, remote->addr, remote->rkey, remote->qp_num, remote->psn);
}

void per_thread_rdma_init(void* gpu_buf, size_t bytes, int rank,
                          int block_idx) {
  if (context) return;  // already initialized

  struct ibv_device** dev_list = ibv_get_device_list(NULL);
  if (!dev_list) {
    perror("Failed to get IB devices list");
    exit(1);
  }

  // Get GPU idx
  int gpu_idx = 0;
  auto gpu_cards = uccl::get_gpu_cards();
  auto ib_nics = uccl::get_rdma_nics();
  auto gpu_device_path = gpu_cards[gpu_idx];
  auto ib_nic_it = std::min_element(
      ib_nics.begin(), ib_nics.end(), [&](auto const& a, auto const& b) {
        return uccl::cal_pcie_distance(gpu_device_path, a.second) <
               uccl::cal_pcie_distance(gpu_device_path, b.second);
      });
  int selected_idx = ib_nic_it - ib_nics.begin();
  printf("[RDMA] Selected NIC %s for GPU %s\n", ib_nic_it->first.c_str(),
         gpu_device_path.c_str());

  context = ibv_open_device(dev_list[selected_idx]);
  if (!context) {
    perror("Failed to open device");
    exit(1);
  }
  printf("[RDMA] Selected NIC: %s (index %d)\n",
         ibv_get_device_name(dev_list[selected_idx]), selected_idx);

  ibv_free_device_list(dev_list);

  pd = ibv_alloc_pd(context);
  if (!pd) {
    perror("Failed to allocate PD");
    exit(1);
  }
  mr = ibv_reg_mr(pd, gpu_buf, bytes,
                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                      IBV_ACCESS_RELAXED_ORDERING);

  if (!mr) {
    perror("ibv_reg_mr failed");
    exit(1);
  }

  if (rkey != 0) {
    fprintf(stderr, "Warning: rkey already set (%x), overwriting\n", rkey);
  }
  rkey = mr->rkey;
}

void global_rdma_init(void* gpu_buf, size_t bytes, RDMAConnectionInfo* local,
                      int rank) {
  static std::once_flag flag;
  std::call_once(flag, [&] {
    setup_rdma(gpu_buf, bytes, local, rank);  // your existing function
  });
}

ibv_cq* create_per_thread_cq() {
  ibv_cq* cq;
  int cq_depth = kMaxOutstandingSends * 2;
  cq = ibv_create_cq(context, /* cqe */ cq_depth, /* user_context */ nullptr,
                     /* channel */ nullptr, /* comp_vector */ 0);
  if (!cq) {
    perror("Failed to create CQ");
    exit(1);
  }
  return cq;
}

void create_per_thread_ack_qp(void* gpu_buffer, size_t size,
                              RDMAConnectionInfo* local_info, int rank,
                              ibv_cq* cq) {
  if (ack_qp) return;
  struct ibv_qp_init_attr qp_init_attr = {};
  qp_init_attr.send_cq = cq;
  qp_init_attr.recv_cq = cq;
  qp_init_attr.qp_type = IBV_QPT_RC;  // Reliable Connection
  qp_init_attr.cap.max_send_wr =
      kMaxOutstandingSends * 2;  // max outstanding sends
  qp_init_attr.cap.max_recv_wr =
      kMaxOutstandingSends * 2;  // max outstanding recvs
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;
  qp_init_attr.sq_sig_all = 0;
  ack_qp = ibv_create_qp(pd, &qp_init_attr);
  if (!ack_qp) {
    perror("Failed to create QP");
    exit(1);
  }
}

void create_per_thread_qp(void* gpu_buffer, size_t size,
                          RDMAConnectionInfo* local_info, int rank,
                          ibv_cq* cq) {
  if (qp) return;  // Already initialized for this thread
  if (ack_qp) return;
  struct ibv_qp_init_attr qp_init_attr = {};
  qp_init_attr.send_cq = cq;
  qp_init_attr.recv_cq = cq;
  qp_init_attr.qp_type = IBV_QPT_RC;  // Reliable Connection
  qp_init_attr.cap.max_send_wr =
      kMaxOutstandingSends * 2;  // max outstanding sends
  qp_init_attr.cap.max_recv_wr =
      kMaxOutstandingSends * 2;  // max outstanding recvs
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;
  qp_init_attr.sq_sig_all = 0;

  qp = ibv_create_qp(pd, &qp_init_attr);
  if (!qp) {
    perror("Failed to create QP");
    exit(1);
  }

  ack_qp = ibv_create_qp(pd, &qp_init_attr);
  if (!ack_qp) {
    perror("Failed to create Ack QP");
    exit(1);
  }

  // Query port
  struct ibv_port_attr port_attr;
  if (ibv_query_port(context, 1, &port_attr)) {
    perror("Failed to query port");
    exit(1);
  }
  printf("Local LID: 0x%x\n", port_attr.lid);
  // Fill local connection info
  local_info->qp_num = qp->qp_num;
  local_info->ack_qp_num = ack_qp->qp_num;
  local_info->lid = port_attr.lid;
  local_info->rkey = rkey;
  local_info->addr = reinterpret_cast<uintptr_t>(gpu_buffer);
  local_info->psn = rand() & 0xffffff;      // random psn
  local_info->ack_psn = rand() & 0xffffff;  // random ack psn
  memset(local_info->gid, 0, 16);
  printf(
      "Local RDMA info: addr=0x%lx, rkey=0x%x, qp_num=%u, psn=%u, "
      "ack_qp_num=%u, ack_psn: %u\n",
      local_info->addr, local_info->rkey, local_info->qp_num, local_info->psn,
      local_info->ack_qp_num, local_info->ack_psn);
}

void setup_rdma(void* gpu_buffer, size_t size, RDMAConnectionInfo* local_info,
                int rank) {
  if (qp) return;

  srand(time(NULL) + getpid() + rank * 1000);
  struct ibv_device** dev_list = ibv_get_device_list(NULL);
  if (!dev_list) {
    perror("Failed to get IB devices list");
    exit(1);
  }

  context = ibv_open_device(dev_list[0]);
  if (!context) {
    perror("Failed to open device");
    exit(1);
  }
  printf("[RDMA] Selected NIC: %s\n", ibv_get_device_name(dev_list[0]));
  // Print out all the NICs
  for (int i = 0; dev_list[i]; ++i) {
    printf("[RDMA] NIC %d: %s\n", i, ibv_get_device_name(dev_list[i]));
  }
  ibv_free_device_list(dev_list);

  // 2. Allocate a Protection Domain
  pd = ibv_alloc_pd(context);
  if (!pd) {
    perror("Failed to allocate PD");
    exit(1);
  }

  // 3. Register the GPU memory
  mr = ibv_reg_mr(pd, gpu_buffer, size,
                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                      IBV_ACCESS_RELAXED_ORDERING);

  if (!mr) {
    perror("ibv_reg_mr (GPUDirect) failed");
    exit(1);
  }
  if (rkey != 0) {
    perror("rkey already set, this should not happen");
    exit(1);
  }
  rkey = mr->rkey;
}

void modify_qp_to_init() {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));

  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = 1;  // HCA port you use
  attr.pkey_index = 0;
  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

  int flags =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

  if (ibv_modify_qp(qp, &attr, flags)) {
    perror("Failed to modify QP to INIT");
    exit(1);
  }

  if (ack_qp) {
    int ret = ibv_modify_qp(ack_qp, &attr, flags);
    if (ret) {
      perror("Failed to modify Ack QP to INIT");
      fprintf(stderr, "errno: %d\n", errno);
      exit(1);
    }
  }

  printf("QP modified to INIT state\n");
}

void modify_qp_to_rtr(RDMAConnectionInfo* remote) {
  int is_roce = 0;

  struct ibv_port_attr port_attr;
  if (ibv_query_port(context, 1, &port_attr)) {
    perror("Failed to query port");
    exit(1);
  }

  if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
    printf("RoCE detected (Ethernet)\n");
    is_roce = 1;
  } else if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    printf("InfiniBand detected\n");
    is_roce = 0;
  } else {
    printf("Unknown link layer: %d\n", port_attr.link_layer);
    exit(1);
  }

  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = port_attr.active_mtu;
  attr.dest_qp_num = remote->qp_num;
  attr.rq_psn = remote->psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;

  if (is_roce) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.grh.hop_limit = 1;
    // Fill GID from remote_info
    memcpy(&attr.ah_attr.grh.dgid, remote->gid, 16);
    attr.ah_attr.grh.sgid_index = 0;  // Assume GID index 0
  } else {
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = remote->lid;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.static_rate = 0;
    memset(&attr.ah_attr.grh, 0, sizeof(attr.ah_attr.grh));  // Safe
  }

  int flags = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN |
              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  printf("Remote LID: 0x%x, QPN: %u, PSN: %u\n", remote->lid, remote->qp_num,
         remote->psn);
  printf("Verifying port state:\n");
  printf("  link_layer: %s\n", (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET)
                                   ? "Ethernet (RoCE)"
                                   : "InfiniBand");
  printf("  port_state: %s\n",
         (port_attr.state == IBV_PORT_ACTIVE) ? "ACTIVE" : "NOT ACTIVE");
  printf("  max_mtu: %d\n", port_attr.max_mtu);
  printf("  active_mtu: %d\n", port_attr.active_mtu);
  printf("  lid: 0x%x\n", port_attr.lid);

  int ret = ibv_modify_qp(qp, &attr, flags);
  if (ret) {
    perror("Failed to modify QP to RTR");
    fprintf(stderr, "errno: %d\n", errno);
    exit(1);
  }
  printf("QP modified to RTR state\n");

  if (ack_qp) {
    attr.dest_qp_num = remote->ack_qp_num;
    attr.rq_psn = remote->ack_psn;
    ret = ibv_modify_qp(ack_qp, &attr, flags);
    if (ret) {
      perror("Failed to modify Ack QP to RTR");
      fprintf(stderr, "errno: %d\n", errno);
      exit(1);
    }
  }
  printf("ACK-QP modified to RTR state\n");
}

void modify_qp_to_rts(RDMAConnectionInfo* local_info) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = local_info->psn;
  attr.max_rd_atomic = 1;

  int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

  if (ibv_modify_qp(qp, &attr, flags)) {
    perror("Failed to modify QP to RTS");
    exit(1);
  }
  printf("QP modified to RTS state\n");

  attr.sq_psn = local_info->ack_psn;
  int ret = ibv_modify_qp(ack_qp, &attr, flags);
  if (ret) {
    perror("Failed to modify Ack QP to RTS");
    fprintf(stderr, "errno: %d\n", errno);
    exit(1);
  }
  printf("ACK-QP modified to RTS state\n");
}

void post_receive_buffer_for_imm() {
  std::vector<ibv_recv_wr> wrs(kMaxOutstandingRecvs);
  std::vector<ibv_sge> sges(kMaxOutstandingRecvs);

  for (size_t i = 0; i < kMaxOutstandingRecvs; ++i) {
    int offset = kNumThBlocks > i ? i : (i % kNumThBlocks);

    sges[i] = {.addr = (uintptr_t)mr->addr + offset * kObjectSize,
               .length = kObjectSize,
               .lkey = mr->lkey};
    wrs[i] = {.wr_id = i,  // choose something meaningful
              .next = (i + 1 < kMaxOutstandingRecvs) ? &wrs[i + 1] : nullptr,
              .sg_list = &sges[i],
              .num_sge = 1};
  }

  /* Post the whole chain with ONE verbs call */
  ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(qp, &wrs[0], &bad)) {
    perror("ibv_post_recv");
    abort();
  }
}

uint32_t build_imm_data(int src_addr_offset, int destination_gpu,
                        uint32_t destination_addr_offset) {
  uint32_t imm_data = 0;
  imm_data |= (src_addr_offset & 0xFF) << 24;      // 8 bits.
  imm_data |= (destination_gpu & 0xFF) << 16;      // 8 bits.
  imm_data |= (destination_addr_offset & 0xFFFF);  // 16 bits.
  return imm_data;
}

void unpack_imm_data(int& src_addr_offset, int& destination_gpu,
                     uint32_t& destination_addr_offset, uint32_t imm_data) {
  src_addr_offset = (imm_data >> 24) & 0xFF;    // 8 bits
  destination_gpu = (imm_data >> 16) & 0xFF;    // 8 bits
  destination_addr_offset = imm_data & 0xFFFF;  // 16 bits
}

void post_rdma_async_batched(void* buf, size_t bytes, size_t num_wrs,
                             std::vector<uint64_t> wrs_to_post, ibv_cq* cq,
                             std::unordered_set<uint64_t>& finished_wrs,
                             std::mutex& finished_wrs_mutex) {
  struct ibv_sge sge {
    .addr = (uintptr_t)buf /*+ start_offset * bytes*/,
    .length = (uint32_t)(bytes * num_wrs), .lkey = mr->lkey
  };
  uint64_t largest_wr = wrs_to_post.back();
  struct ibv_send_wr wr {};
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.wr.rdma.remote_addr = remote_addr /*+ start_offset * bytes*/;
  wr.wr.rdma.rkey = remote_rkey;
  wr.wr_id = largest_wr;
#ifdef ENABLE_WRITE_WITH_IMMEDIATE
  wr.imm_data = largest_wr;
#endif
  if (largest_wr % kSignalledEvery == 0)
    wr.send_flags = IBV_SEND_SIGNALED;
  else
    wr.send_flags = 0;

  ibv_send_wr* bad = nullptr;
  int ret = ibv_post_send(qp, &wr, &bad);
  if (ret) {
    fprintf(stderr, "ibv_post_send failed: %s (ret=%d)\n", strerror(ret), ret);
    if (bad) {
      fprintf(stderr, "Bad WR at address: %p\n", bad);
    }
    exit(1);
  }
  g_posted.fetch_add(num_wrs, std::memory_order_relaxed);
  if (wr_id_to_wr_ids.find(largest_wr) != wr_id_to_wr_ids.end()) {
    fprintf(stderr, "Error: largest_wr %lu already exists in wr_id_to_wr_ids\n",
            largest_wr);
    exit(1);
  }
  wr_id_to_wr_ids[largest_wr] = wrs_to_post;
}

void post_rdma_async_chained(void* buf, size_t bytes, size_t num_wrs,
                             std::vector<uint64_t> wrs_to_post, ibv_cq* cq,
                             std::unordered_set<uint64_t>& finished_wrs,
                             std::mutex& finished_wrs_mutex) {
  std::vector<struct ibv_sge> sges(num_wrs);
  std::vector<struct ibv_send_wr> wrs(num_wrs);
  if (num_wrs != wrs_to_post.size()) {
    fprintf(stderr,
            "Error: num_wrs (%ld) does not match wrs_to_post size (%zu)\n",
            num_wrs, wrs_to_post.size());
    exit(1);
  }

  for (size_t i = 0; i < num_wrs; ++i) {
    int wr = wrs_to_post[i];
    int offset = wr % (kRemoteBufferSize / bytes);
    sges[i].addr = (uintptr_t)buf + offset * bytes;
    sges[i].length = (uint32_t)bytes;
    sges[i].lkey = mr->lkey;

    wrs[i].sg_list = &sges[i];
    wrs[i].num_sge = 1;
    wrs[i].wr.rdma.remote_addr = remote_addr + offset * bytes;
    wrs[i].wr.rdma.rkey = remote_rkey;
    wrs[i].wr_id = wrs_to_post[i];
    assert(wrs[i].wr_id <= kIterations);
#ifdef ENABLE_WRITE_WITH_IMMEDIATE
    wrs[i].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wrs[i].imm_data = wrs[i].wr_id;
#else
    wrs[i].opcode = IBV_WR_RDMA_WRITE;
#endif
    if ((i + 1) % kSignalledEvery == 0)
      wrs[i].send_flags = IBV_SEND_SIGNALED;
    else
      wrs[i].send_flags = 0;

    if (i < num_wrs - 1) {
      wrs[i].next = &wrs[i + 1];
    } else {
      wrs[i].next = nullptr;
    }
  }
  ibv_send_wr* bad = nullptr;
  int ret = ibv_post_send(qp, &wrs[0], &bad);
  if (ret) {
    fprintf(stderr,
            "ibv_post_send failed: %s (ret=%d), num_wrs: %ld, g_posted: %ld. "
            "g_completed: %ld\n",
            strerror(ret), ret, num_wrs, g_posted.load(), g_completed.load());
    uint64_t inflight = g_posted.load(std::memory_order_acquire) -
                        g_completed.load(std::memory_order_acquire);
    printf("Currently outstanding send WRs: %lu\n", inflight);

    if (bad) {
      fprintf(stderr, "Bad WR at address: %p\n", bad);
    }
    exit(1);
  }
  g_posted.fetch_add(num_wrs, std::memory_order_relaxed);
}

void post_rdma_async(void* buf, size_t bytes, uint64_t wr_id, ibv_cq* cq,
                     std::unordered_set<uint64_t>& finished_wrs,
                     std::mutex& finished_wrs_mutex) {
  /* Make it a closed loop to limit the maximum outstanding sends. */
  // while (g_posted.load() - g_completed.load() > kMaxOutstandingSends) {
  //   local_poll_completions(cq, finished_wrs, finished_wrs_mutex);
  // }

  struct ibv_sge sge {
    .addr = (uintptr_t)buf, .length = (uint32_t)bytes, .lkey = mr->lkey
  };

  struct ibv_send_wr wr {};
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = remote_rkey;
  wr.wr_id = wr_id;

  if (wr_id % kSignalledEvery == 0)
    wr.send_flags = IBV_SEND_SIGNALED;  // generate a CQE
  else
    wr.send_flags = 0;

  ibv_send_wr* bad = nullptr;
  int ret = ibv_post_send(qp, &wr, &bad);
  if (ret) {
    fprintf(stderr, "ibv_post_send failed: %s (ret=%d)\n", strerror(ret), ret);
    if (bad) {
      fprintf(stderr, "Bad WR at address: %p\n", bad);
    }
    // Optionally query QP state here for more info
    exit(1);
  }

  g_posted.fetch_add(1, std::memory_order_relaxed);
}

void rdma_write_stub(void* local_dev_ptr, size_t bytes) {
  struct ibv_qp_attr qattr;
  struct ibv_qp_init_attr qinit;
  ibv_query_qp(qp, &qattr, IBV_QP_STATE, &qinit);

  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = reinterpret_cast<uintptr_t>(local_dev_ptr);  // GPU memory address
  sge.length = bytes;
  sge.lkey = mr->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = 0;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = remote_rkey;

  struct ibv_send_wr* bad_wr = nullptr;
  int ret = ibv_post_send(qp, &wr, &bad_wr);
  if (ret) {
    perror("ibv_post_send failed");
    exit(1);
  }
}

#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
bool GdrSupportInitOnce() {
  // Check for the nv_peer_mem module being loaded
  return KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem/version") ||
         KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
         KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}

void local_poll_completions(ibv_cq* cq,
                            std::unordered_set<uint64_t>& finished_wrs,
                            std::mutex& finished_wrs_mutex, int thread_idx) {
  struct ibv_wc wc[kMaxOutstandingSends];  // batch poll
  int ne = ibv_poll_cq(cq, kMaxOutstandingSends, wc);
  if (ne == 0) return;
  int write_ack = 0;

  assert(ack_qp->send_cq == cq);
  assert(ack_qp->recv_cq == cq);
  for (int i = 0; i < ne; ++i) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      fprintf(stderr, "CQE error wr_id=%llu status=%s\n",
              (unsigned long long)wc[i].wr_id, ibv_wc_status_str(wc[i].status));
      std::abort();
    }

    switch (wc[i].opcode) {
      case IBV_WC_SEND:
      case IBV_WC_RDMA_WRITE: {
        std::lock_guard<std::mutex> lock(finished_wrs_mutex);
#ifdef RDMA_BATCH_TOKENS
        for (auto const& wr_id : wr_id_to_wr_ids[wc[i].wr_id]) {
          finished_wrs.insert(wr_id);
        }
        // printf("[WR] %d completed on peer, wr_id=%llu, num_wrs=%zu\n",
        //        thread_idx, (unsigned long long)wc[i].wr_id,
        //        wr_id_to_wr_ids[wc[i].wr_id].size());
        wr_id_to_wr_ids.erase(wc[i].wr_id);
#else
        finished_wrs.insert(wc[i].wr_id);
#endif
      } break;
      case IBV_WC_RECV:
        if (wc[i].wc_flags & IBV_WC_WITH_IMM) {
          uint64_t slot = static_cast<uint64_t>(wc[i].wr_id);
          write_ack++;

          uint64_t wr_done = static_cast<uint64_t>(wc[i].imm_data);
          // printf("[ACK - %d] Received ACK for WR %lu in slot %lu\n",
          // thread_idx, wr_done, slot);
          if (!has_received_ack || wr_done >= largest_completed_wr) {
            largest_completed_wr = wr_done;
            has_received_ack = true;
            // printf("New largest completed WR: %lu\n", largest_completed_wr);
          } else {
            fprintf(stderr,
                    "Warning: received ACK for WR %lu, but largest completed "
                    "WR is %lu\n",
                    wr_done, largest_completed_wr);
            std::abort();
          }

          ibv_sge sge = {
              .addr = reinterpret_cast<uintptr_t>(&ack_recv_buf[slot]),
              .length = sizeof(uint64_t),
              .lkey = ack_recv_mr->lkey,
          };
          ibv_recv_wr rwr = {};
          ibv_recv_wr* bad = nullptr;
          rwr.wr_id = static_cast<uint64_t>(slot);
          rwr.sg_list = &sge;
          rwr.num_sge = 1;
          if (ibv_post_recv(ack_qp, &rwr, &bad)) {
            perror("ibv_post_recv(repost ACK)");
            std::abort();
          }
        }
        break;

      default:
        break;
    }
  }
  g_completed.fetch_add(ne, std::memory_order_relaxed);
  // if (write_ack > 0) {
  //   printf("[ACK] %d completed on peer\n", write_ack);
  // }
}

void per_thread_polling(int thread_idx, struct ibv_cq* per_thread_cq,
                        std::unordered_set<uint64_t>* per_thread_finished_wrs,
                        std::mutex* per_thread_finished_wrs_mutex) {
  pin_thread_to_cpu(thread_idx);
  printf("Progress thread started on CPU %d\n", sched_getcpu());

  while (per_thread_cq == nullptr && g_progress_run.load()) cpu_relax();
  printf("Progress thread %d: cq=%p\n", thread_idx, per_thread_cq);

  while (g_progress_run.load(std::memory_order_acquire)) {
    local_poll_completions(per_thread_cq, *per_thread_finished_wrs,
                           *per_thread_finished_wrs_mutex, thread_idx);
  }
}

bool check_cq_completion() {
  uint64_t posted = g_posted.load(std::memory_order_acquire);
  uint64_t completed = g_completed.load(std::memory_order_acquire);
  printf("check_cq_completion: g_completed: %ld, g_posted: %ld, total: %d\n",
         completed, posted, kIterations * kNumThBlocks);
  return completed * kSignalledEvery == posted && kIterations == completed;
}

void handle_peer_copy(uint64_t wr_id, int src_dev, int dst_dev, void* src_ptr,
                      void* dst_ptr, size_t num_bytes) {
  if (src_dev == dst_dev) {
    return;
  }
  static thread_local hipStream_t copy_stream = nullptr;
  if (copy_stream == nullptr) {
    hipStreamCreate(&copy_stream);
  }

  static thread_local bool peer_enabled[NUM_GPUS][NUM_GPUS] = {};
  if (!peer_enabled[src_dev][dst_dev]) {
    hipDeviceEnablePeerAccess(dst_dev, 0);
    hipSetDevice(dst_dev);
    hipDeviceEnablePeerAccess(src_dev, 0);
    peer_enabled[src_dev][dst_dev] = true;
    hipSetDevice(src_dev);
  }
#ifdef ENABLE_PROXY_HIP_MEMCPY
  auto start_time = std::chrono::high_resolution_clock::now();
  hipError_t err = hipMemcpyPeerAsync(dst_ptr, dst_dev, src_ptr, src_dev,
                                        num_bytes, copy_stream);
  async_memcpy_count++;
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  async_memcpy_total_time += duration.count();
#else
  hipError_t err = hipMemcpyPeerAsync(dst_ptr, dst_dev, src_ptr, src_dev,
                                        num_bytes, copy_stream);
#endif
  if (err != hipSuccess) {
    fprintf(stderr,
            "hipMemcpyPeerAsync failed (%s)\n"
            "  wr_id=%llu  %zu B  GPU%d→GPU%d\n",
            hipGetErrorString(err), (unsigned long long)wr_id, num_bytes,
            src_dev, dst_dev);
    std::abort();
  }
}

void remote_cpu_proxy_poll_write_with_immediate(int idx, ibv_cq* cq,
                                                CopyRingBuffer& g_ring) {
  struct ibv_wc wc[kMaxOutstandingRecvs];
  struct ibv_sge sges[kMaxOutstandingRecvs];
  struct ibv_recv_wr wrs[kMaxOutstandingRecvs];
  size_t pool_index = 0;
  assert(ack_qp->send_cq == cq);
  assert(qp->send_cq == cq);
  while (g_progress_run.load(std::memory_order_acquire)) {
    int ne = ibv_poll_cq(cq, kMaxOutstandingRecvs, wc);
    if (ne == 0) continue;
    int num_wr_imm = 0;
    for (int i = 0; i < ne; ++i) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        // if (wc[i].status == IBV_WC_WR_FLUSH_ERR) {
        //   std::abort();
        // }
        // if (wc[i].status == IBV_WC_RNR_RETRY_EXC_ERR) {
        //   continue;
        // }
        fprintf(stderr, "RDMA error: %s\n", ibv_wc_status_str(wc[i].status));
        std::abort();
      }
      if (wc[i].opcode == IBV_WC_SEND) {
        send_ack_completed++;
        // printf("[ACK] remote_cpu_proxy_poll_write_with_immediate: %ld
        // onflight on peer\n",
        //        send_ack_posted - send_ack_completed);
        continue;
      }
      if (wc[i].opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
        fprintf(stderr, "Unexpected opcode: %d\n", wc[i].opcode);
        exit(1);
      }
      // int src_addr_offset;
      // int destination_gpu;
      // uint32_t destination_addr_offset;

      // unpack_imm_data(src_addr_offset, destination_gpu,
      // destination_addr_offset,
      //                 wc[i].imm_data);

      // int wr_gpu = (int)(wc[i].wr_id % NUM_GPUS);
      // int wr_gpu = wc[i].imm_data
      // if (destination_gpu != wr_gpu) {
      //   fprintf(stderr,
      //           "Unexpected immediate data: dest=%u  wr_id=%d  full "
      //           "wr_id=%lu\n",
      //           destination_gpu, wr_gpu, wc[i].wr_id);
      //   exit(EXIT_FAILURE);
      // }

      pool_index = (pool_index + 1) % (kRemoteBufferSize / kObjectSize - 1);
      char* next_buf = static_cast<char*>(mr->addr) + pool_index * kObjectSize;

      sges[num_wr_imm] = {.addr = reinterpret_cast<uintptr_t>(next_buf),
                          .length = kObjectSize,
                          .lkey = mr->lkey};

      wrs[num_wr_imm] = {.wr_id = wc[i].wr_id + 0x10000000ULL,
                         .next = nullptr,
                         .sg_list = &sges[num_wr_imm],
                         .num_sge = 1};
      if (num_wr_imm >= 1) {
        wrs[num_wr_imm - 1].next = &wrs[num_wr_imm];
      }
      num_wr_imm++;
    }
    ibv_recv_wr* bad = nullptr;
    if (num_wr_imm > 0) {
      int ret = ibv_post_recv(qp, &wrs[0], &bad);
      if (ret) {
        fprintf(stderr, "ibv_post_recv failed: %s\n", strerror(ret));
        std::abort();
      }
    }

#ifdef ENABLE_PROXY_HIP_MEMCPY
    std::vector<CopyTask> task_vec;
    task_vec.reserve(num_wr_imm);
    for (int i = 0; i < ne; ++i) {
      int src_addr_offset = 0;
      // int destination_gpu;
      uint32_t destination_addr_offset = 0;
      if (wc[i].opcode == IBV_WC_SEND) {
        continue;
      }
      // unpack_imm_data(src_addr_offset, destination_gpu,
      // destination_addr_offset,
      //                 wc[i].imm_data);
      int destination_gpu = wc[i].imm_data % NUM_GPUS;
      if (per_GPU_device_buf[destination_gpu] == nullptr) {
        fprintf(stderr, "per_GPU_device_buf[%d] is null\n", destination_gpu);
        std::abort();
      }
#ifndef RDMA_BATCH_TOKENS
      if (wc[i].byte_len != kObjectSize) {
        fprintf(stderr, "Unexpected byte length: %u, expected: %u\n",
                wc[i].byte_len, kObjectSize);
        std::abort();
      }
#endif
      if (wc[i].imm_data > kIterations) {
        fprintf(stderr, "Unexpected imm_data: %u, expected <= %d\n",
                wc[i].imm_data, kIterations);
        std::abort();
      }
      CopyTask task{
          .wr_id = wc[i].imm_data,
          .dst_dev = destination_gpu,
          .src_ptr = static_cast<char*>(mr->addr) + src_addr_offset,
          .dst_ptr = static_cast<char*>(per_GPU_device_buf[destination_gpu]) +
                     destination_addr_offset,
          .bytes = wc[i].byte_len};
      task_vec.push_back(task);
    }
    if (!task_vec.empty()) {
      while (!g_ring.pushN(task_vec.data(), task_vec.size())) { /* Busy spin. */
      }
    }
#endif
  }
}

#ifdef ENABLE_PROXY_HIP_MEMCPY
void print_average_async_memcpy_time() {
  printf("Total async memcpy calls: %lu\n", async_memcpy_count);
  if (async_memcpy_count == 0) {
    printf("No async memcpy calls were made.\n");
    return;
  }
  printf("Average async memcpy time: %lu us\n",
         async_memcpy_total_time / async_memcpy_count);
}
#endif

void remote_ensure_ack_sender_resources(ibv_pd* pd, uint64_t* ack_buf,
                                        ibv_mr*& ack_mr) {
  if (ack_mr) return;
  ack_mr = ibv_reg_mr(pd, ack_buf, sizeof(uint64_t) * RECEIVER_BATCH_SIZE,
                      IBV_ACCESS_LOCAL_WRITE);  // host-only

  if (!ack_mr) {
    perror("ibv_reg_mr(ack_buf)");
    std::abort();
  }
}

void remote_notify_sender_that_wr_id_has_completed(struct ibv_qp* local_ack_qp,
                                                   uint64_t& wr_id,
                                                   ibv_mr* local_ack_mr,
                                                   uint64_t* ack_buf,
                                                   int worker_idx) {
  if (!local_ack_qp || !local_ack_mr) {
    if (!local_ack_qp) {
      fprintf(stderr, "QP not initialised\n");
      std::abort();
    }
    if (!local_ack_mr) {
      fprintf(stderr, "ACK MR not initialised\n");
      std::abort();
    }
    fprintf(stderr, "ACK resources not initialised\n");
    std::abort();
  }

  *reinterpret_cast<uint64_t*>(ack_buf) = wr_id;
  ibv_sge sge = {
      .addr = reinterpret_cast<uintptr_t>(ack_buf),
      .length = sizeof(uint64_t),
      .lkey = local_ack_mr->lkey,
  };

  ibv_send_wr wr = {};
  ibv_send_wr* bad = nullptr;
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;  // generate a CQE
  // wr.send_flags         = IBV_SEND_INLINE;
  wr.imm_data = static_cast<uint32_t>(wr_id);

  int ret = ibv_post_send(local_ack_qp, &wr, &bad);
  send_ack_posted.fetch_add(1, std::memory_order_relaxed);

  if (ret) {  // ret is already an errno value
    fprintf(stderr, "ibv_post_send(SEND_WITH_IMM) failed: %d (%s)\n", ret,
            strerror(ret));  // strerror(ret) gives the text
    if (bad) {
      fprintf(stderr,
              "  first bad WR: wr_id=%llu  opcode=%u  addr=0x%llx  lkey=0x%x\n",
              (unsigned long long)bad->wr_id, bad->opcode,
              (unsigned long long)bad->sg_list[0].addr, bad->sg_list[0].lkey);
    }
    std::abort();
  }

  // printf("[ACK - %d]  wr_id=%lu posted to ACK sender\n", worker_idx,
  //        static_cast<unsigned long>(wr_id));
  // printf("[ACK] remote_notify_sender_that_wr_id_has_completed: %ld onflight
  // on peer, posted: %ld, completed: %ld\n", send_ack_posted -
  // send_ack_completed, send_ack_posted.load(), send_ack_completed.load());
}

void remote_notify_sender_batch(struct ibv_qp* ack_qp,
                                std::vector<uint64_t> const& wr_ids,
                                ibv_mr* ack_mr, uint64_t* ack_buf) {
  if (!ack_qp || !ack_mr || wr_ids.empty()) {
    fprintf(stderr, "ACK: bad arguments\n");
    std::abort();
  }
  size_t const n = wr_ids.size();

  std::vector<ibv_sge> sge(n);
  std::vector<ibv_send_wr> wr(n);

  for (size_t i = 0; i < n; ++i) {
    ack_buf[i] = wr_ids[i];

    sge[i].addr = reinterpret_cast<uintptr_t>(&ack_buf[i]);
    sge[i].length = sizeof(uint64_t);
    sge[i].lkey = ack_mr->lkey;

    wr[i] = {};
    wr[i].wr_id = wr_ids[i];
    wr[i].opcode = IBV_WR_SEND_WITH_IMM;
    wr[i].sg_list = &sge[i];
    wr[i].num_sge = 1;
    wr[i].imm_data = static_cast<uint32_t>(wr_ids[i]);
    wr[i].send_flags = IBV_SEND_SIGNALED;
    wr[i].next = (i + 1 < n) ? &wr[i + 1] : nullptr;
  }

  ibv_send_wr* bad = nullptr;
  int ret = ibv_post_send(ack_qp, &wr[0], &bad);
  send_ack_posted.fetch_add(n, std::memory_order_relaxed);
  if (ret) {
    fprintf(stderr, "ACK ibv_post_send failed: %d (%s)\n", ret, strerror(ret));
    if (bad) {
      fprintf(stderr, "  first bad wr_id=%llu\n",
              static_cast<unsigned long long>(bad->wr_id));
    }
    std::abort();
  } else {
    printf("[ACK] %zu WRs posted to ACK sender\n", n);
    printf("[ACK] %ld onflight on peer\n",
           send_ack_posted - send_ack_completed);
  }
}

void local_init_ack_recv_ring(struct ibv_pd* pd, int depth) {
  printf("Initializing ACK receive ring with depth %d\n", depth);
  ack_recv_buf.resize(static_cast<size_t>(depth), 0);
  ack_recv_mr = ibv_reg_mr(pd, ack_recv_buf.data(),
                           ack_recv_buf.size() * sizeof(uint64_t),
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  assert(ack_qp->recv_cq != nullptr);
  assert(ack_qp->send_cq != nullptr);

  if (!ack_recv_mr) {
    perror("ibv_reg_mr(ack_recv)");
    std::abort();
  }

  for (int i = 0; i < depth; ++i) {
    ibv_sge sge = {
        .addr = reinterpret_cast<uintptr_t>(&ack_recv_buf[i]),
        .length = sizeof(uint64_t),
        .lkey = ack_recv_mr->lkey,
    };

    ibv_recv_wr rwr = {};
    ibv_recv_wr* bad = nullptr;

    rwr.wr_id = static_cast<uint64_t>(i);
    rwr.sg_list = &sge;
    rwr.num_sge = 1;

    if (ibv_post_recv(ack_qp, &rwr, &bad)) {
      perror("ibv_post_recv(ack)");
      std::abort();
    }
    // printf("ack_qp ACK receive buffer %d initialized at %p\n", i,
    //        &ack_recv_buf[i]);
  }
}
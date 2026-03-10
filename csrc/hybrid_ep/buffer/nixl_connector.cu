// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#ifdef USE_NIXL

#include "buffer/nixl_connector.h"
#include <fstream>
#include <thread>
#include <chrono>
#include <ifaddrs.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <sys/stat.h>
#include <cstring>
#include <cassert>
#include <algorithm>

#define NIXL_ETCD_WATCH_TIMEOUT std::chrono::microseconds(1000000000)  // 1000 seconds

// Uncomment to enable detailed debug logging.
// #define NIXL_VERBOSE

#ifdef NIXL_VERBOSE
  #define NIXL_LOG(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
  #define NIXL_LOG(fmt, ...) do {} while(0)
#endif

#define NIXL_LOG_CRITICAL(fmt, ...) printf(fmt, ##__VA_ARGS__)

namespace hybrid_ep {

static void sleep_ms(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

static std::string get_local_ip() {
    struct ifaddrs *ifaddr, *ifa;
    char host[NI_MAXHOST];

    if (getifaddrs(&ifaddr) == -1) {
        perror("getifaddrs");
        return "127.0.0.1";
    }

    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL || ifa->ifa_addr->sa_family != AF_INET)
            continue;

        if ((ifa->ifa_flags & IFF_UP) && !(ifa->ifa_flags & IFF_LOOPBACK)) {
            if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST) == 0) {
                freeifaddrs(ifaddr);
                return std::string(host);
            }
        }
    }

    freeifaddrs(ifaddr);
    return "127.0.0.1";
}

static std::string boot_id_get() {
    std::ifstream boot_id_file("/proc/sys/kernel/random/boot_id");
    if (!boot_id_file.is_open()) {
        return "";
    }

    std::string boot_id;
    std::getline(boot_id_file, boot_id);

    if (!boot_id.empty() && boot_id.back() == '\n') {
        boot_id.pop_back();
    }

    return boot_id;
}

static ino_t ipc_namespace_inode_get() {
    struct stat st;
    if (stat("/proc/self/ns/ipc", &st) != 0) {
        return 0;
    }
    return st.st_ino;
}

HybridEP_NIXLConnector::HybridEP_NIXLConnector(int rank_uuid, int local_device_id)
    : rank_uuid(rank_uuid),
      local_device_id(local_device_id),
      num_ranks(0),
      num_experts_per_rank(0),
      num_nodes(0),
      ranks_per_node(0),
      num_channels(0),
      initialized(false),
      connected(false),
      d_dispatch_nixl_ctx(nullptr),
      d_combine_nixl_ctx(nullptr),
      d_dispatch_flag_counters(nullptr),
      d_combine_flag_counters(nullptr) {

    NIXL_LOG("  [Rank %d] HybridEP_NIXLConnector: Constructor called (local_device_id=%d)\n", rank_uuid, local_device_id);

    my_peer_info = {};
    strncpy(my_peer_info.ip, get_local_ip().c_str(), MAX_IP_LENGTH - 1);
    my_peer_info.ip[MAX_IP_LENGTH - 1] = '\0';
    strncpy(my_peer_info.boot_id, boot_id_get().c_str(), MAX_BOOT_ID_LENGTH - 1);
    my_peer_info.boot_id[MAX_BOOT_ID_LENGTH - 1] = '\0';
    my_peer_info.ipc_namespace_inode = ipc_namespace_inode_get();
    my_peer_info.device_id = local_device_id;
    my_peer_info.rank = rank_uuid;

    NIXL_LOG("  [Rank %d] HybridEP_NIXLConnector: My info - IP=%s, device_id=%d, boot_id=%s\n",
           rank_uuid, my_peer_info.ip, my_peer_info.device_id, my_peer_info.boot_id);
}

HybridEP_NIXLConnector::~HybridEP_NIXLConnector() {
    NIXL_LOG("  [Rank %d] ~HybridEP_NIXLConnector: Destructor called - cleaning up resources...\n", rank_uuid);

    cudaDeviceSynchronize();

    NIXL_LOG("  [Rank %d] ~HybridEP_NIXLConnector: Destroying NIXL agents...\n", rank_uuid);
    for (auto& agent_info : nixl_agent_infos) {
        agent_info.agent.reset();
    }
    nixl_agent_infos.clear();
    NIXL_LOG("  [Rank %d] ~HybridEP_NIXLConnector: NIXL agents destroyed\n", rank_uuid);

    cudaDeviceSynchronize();

    if (d_dispatch_flag_counters) {
        cudaFree(d_dispatch_flag_counters);
        d_dispatch_flag_counters = nullptr;
    }
    if (d_combine_flag_counters) {
        cudaFree(d_combine_flag_counters);
        d_combine_flag_counters = nullptr;
    }

    if (d_dispatch_nixl_ctx) {
        cudaFree(d_dispatch_nixl_ctx);
        d_dispatch_nixl_ctx = nullptr;
    }
    if (d_combine_nixl_ctx) {
        cudaFree(d_combine_nixl_ctx);
        d_combine_nixl_ctx = nullptr;
    }

    cudaDeviceSynchronize();

    NIXL_LOG("  [Rank %d] ~HybridEP_NIXLConnector: Cleanup complete\n", rank_uuid);
}

void HybridEP_NIXLConnector::updateMemoryBuffers(
    int num_ranks, int num_experts_per_rank, int num_nodes, int ranks_per_node,
    int num_dispatch_blocks, int num_combine_blocks,
    void* attn_input_token_d, size_t attn_input_token_sz,
    void* attn_input_prob_d, size_t attn_input_prob_sz,
    void* attn_input_token_scaling_factor_d, size_t attn_input_token_scaling_factor_sz,
    void* rdma_inter_node_group_token_d, size_t rdma_inter_node_group_token_sz,
    void* rdma_inter_node_group_flags_d, size_t rdma_inter_node_group_flags_sz,
    void* rdma_inter_node_group_prob_d, size_t rdma_inter_node_group_prob_sz,
    void* rdma_inter_node_group_scaling_factor_d, size_t rdma_inter_node_group_scaling_factor_sz,
    void* rdma_intra_node_red_token_d, size_t rdma_intra_node_red_token_sz,
    void* rdma_intra_node_red_prob_d, size_t rdma_intra_node_red_prob_sz,
    void* combine_rdma_inter_node_group_token_d, size_t combine_rdma_inter_node_group_token_sz,
    void* combine_rdma_inter_node_group_flags_d, size_t combine_rdma_inter_node_group_flags_sz,
    void* combine_rdma_inter_node_group_prob_d, size_t combine_rdma_inter_node_group_prob_sz,
    bool forward_dispatch, bool backward_combine, bool use_fp8) {

    NIXL_LOG("  [Rank %d] updateMemoryBuffers: Validating parameters...\n", rank_uuid);
    assert(rank_uuid >= 0 && rank_uuid < num_ranks && "Invalid rank_uuid");
    assert(!initialized && "updateMemoryBuffers can only be called once");
    NIXL_LOG("  [Rank %d] updateMemoryBuffers: Parameters valid\n", rank_uuid);

    this->num_ranks = num_ranks;
    this->num_experts_per_rank = num_experts_per_rank;
    this->num_nodes = num_nodes;
    this->ranks_per_node = ranks_per_node;

    this->num_channels = std::max(num_dispatch_blocks, num_combine_blocks);
    NIXL_LOG("  [Rank %d] updateMemoryBuffers: num_channels=%d\n", rank_uuid, num_channels);

    my_peer_info.rdma_buffer_ptr = rdma_inter_node_group_token_d;
    my_peer_info.rdma_prob_buffer_ptr = rdma_inter_node_group_prob_d;
    my_peer_info.rdma_scaling_factor_buffer_ptr = rdma_inter_node_group_scaling_factor_d;
    my_peer_info.dispatch_flags_ptr = static_cast<uint64_t*>(rdma_inter_node_group_flags_d);
    my_peer_info.combine_rdma_buffer_ptr = combine_rdma_inter_node_group_token_d;
    my_peer_info.combine_rdma_prob_buffer_ptr = combine_rdma_inter_node_group_prob_d;
    my_peer_info.combine_flags_ptr = static_cast<uint64_t*>(combine_rdma_inter_node_group_flags_d);
    my_peer_info.device_id = local_device_id;
    my_peer_info.rank = rank_uuid;
    nixl_peer_info.resize(num_ranks);

    buffers.attn_input_token_d = attn_input_token_d;
    buffers.attn_input_token_sz = attn_input_token_sz;
    buffers.attn_input_prob_d = attn_input_prob_d;
    buffers.attn_input_prob_sz = attn_input_prob_sz;
    buffers.attn_input_token_scaling_factor_d = attn_input_token_scaling_factor_d;
    buffers.attn_input_token_scaling_factor_sz = attn_input_token_scaling_factor_sz;
    buffers.rdma_inter_node_group_token_d = rdma_inter_node_group_token_d;
    buffers.rdma_inter_node_group_token_sz = rdma_inter_node_group_token_sz;
    buffers.rdma_inter_node_group_flags_d = rdma_inter_node_group_flags_d;
    buffers.rdma_inter_node_group_flags_sz = rdma_inter_node_group_flags_sz;
    buffers.rdma_inter_node_group_prob_d = rdma_inter_node_group_prob_d;
    buffers.rdma_inter_node_group_prob_sz = rdma_inter_node_group_prob_sz;
    buffers.rdma_inter_node_group_scaling_factor_d = rdma_inter_node_group_scaling_factor_d;
    buffers.rdma_inter_node_group_scaling_factor_sz = rdma_inter_node_group_scaling_factor_sz;
    buffers.rdma_intra_node_red_token_d = rdma_intra_node_red_token_d;
    buffers.rdma_intra_node_red_token_sz = rdma_intra_node_red_token_sz;
    buffers.rdma_intra_node_red_prob_d = rdma_intra_node_red_prob_d;
    buffers.rdma_intra_node_red_prob_sz = rdma_intra_node_red_prob_sz;
    buffers.combine_rdma_inter_node_group_token_d = combine_rdma_inter_node_group_token_d;
    buffers.combine_rdma_inter_node_group_token_sz = combine_rdma_inter_node_group_token_sz;
    buffers.combine_rdma_inter_node_group_flags_d = combine_rdma_inter_node_group_flags_d;
    buffers.combine_rdma_inter_node_group_flags_sz = combine_rdma_inter_node_group_flags_sz;
    buffers.combine_rdma_inter_node_group_prob_d = combine_rdma_inter_node_group_prob_d;
    buffers.combine_rdma_inter_node_group_prob_sz = combine_rdma_inter_node_group_prob_sz;
    buffers.forward_dispatch = forward_dispatch;
    buffers.backward_combine = backward_combine;
    buffers.use_fp8 = use_fp8;

    initialized = true;
    NIXL_LOG("  [Rank %d] updateMemoryBuffers: All buffer pointers stored\n", rank_uuid);

    NIXL_LOG("  [Rank %d] updateMemoryBuffers: Creating NIXL agent and registering buffers...\n", rank_uuid);
    _nixl_agents_init(1);
    _register_buffers_with_agents();
    NIXL_LOG("  [Rank %d] updateMemoryBuffers: NIXL agent created and buffers registered\n", rank_uuid);
}

void HybridEP_NIXLConnector::connectRanks(const std::vector<int>& remote_rank_uuids) {
    NIXL_LOG("  [Rank %d] connectRanks: Starting connection process...\n", rank_uuid);
    assert(initialized && "Must call updateMemoryBuffers before connectRanks");
    assert(!connected && "connectRanks can only be called once");
    assert(num_channels > 0 && "num_channels must be set in updateMemoryBuffers");

    std::vector<int> ranks_to_connect;
    for (int remote_rank : remote_rank_uuids) {
        if (remote_rank != rank_uuid) {
            ranks_to_connect.push_back(remote_rank);
        }
    }

    if (ranks_to_connect.empty()) {
        NIXL_LOG("  [Rank %d] connectRanks: No remote ranks to connect to\n", rank_uuid);
        connected = true;
        return;
    }

    NIXL_LOG("  [Rank %d] connectRanks: Connecting to %zu remote agents\n", rank_uuid, ranks_to_connect.size());
    _nixl_agents_connect(ranks_to_connect);

    NIXL_LOG("  [Rank %d] connectRanks: Peer info exchange\n", rank_uuid);
    _nixl_agents_wireup(ranks_to_connect);

    NIXL_LOG("  [Rank %d] connectRanks: Creating memory views\n", rank_uuid);
    _nixl_create_memory_views(ranks_to_connect);

    NIXL_LOG("  [Rank %d] connectRanks: Building GPU contexts\n", rank_uuid);
    _nixl_build_gpu_contexts(num_channels, num_channels);

    connected_ranks = ranks_to_connect;
    connected = true;

    NIXL_LOG("  [Rank %d] connectRanks: Successfully connected to %zu ranks\n", rank_uuid, connected_ranks.size());
}

void HybridEP_NIXLConnector::disconnectRanks(const std::vector<int>& remote_rank_uuids) {
    assert(connected && "Must be connected before disconnecting");
    _nixl_agents_wiredown(remote_rank_uuids);
}

dispatch_gpu_nixl_ctx* HybridEP_NIXLConnector::get_dispatch_gpu_ctx() {
    return d_dispatch_nixl_ctx;
}

combine_gpu_nixl_ctx* HybridEP_NIXLConnector::get_combine_gpu_ctx() {
    return d_combine_nixl_ctx;
}

void HybridEP_NIXLConnector::_nixl_agents_init(int num_agents) {
    NIXL_LOG("    [Rank %d] _nixl_agents_init: creating %d agent(s)\n", rank_uuid, num_agents);
    nixl_agent_infos.clear();
    nixl_agent_infos.reserve(num_agents);

    std::string agent_name = std::to_string(rank_uuid);

    const char* etcd_endpoint = std::getenv("NIXL_ETCD_ENDPOINTS");
    if (etcd_endpoint) {
        NIXL_LOG("    [Rank %d] _nixl_agents_init: etcd=%s\n", rank_uuid, etcd_endpoint);
    } else {
        NIXL_LOG("    [Rank %d] _nixl_agents_init: NIXL_ETCD_ENDPOINTS not set, using default\n", rank_uuid);
    }

    nixlAgentConfig cfg(true, false, 0,
                       nixl_thread_sync_t::NIXL_THREAD_SYNC_RW, 1, 0, 100000, false, NIXL_ETCD_WATCH_TIMEOUT);

    auto agent = std::make_shared<nixlAgent>(agent_name, cfg);

    nixl_mem_list_t mems;
    nixl_b_params_t init_params;
    nixl_status_t status = agent->getPluginParams("UCX", mems, init_params);
    assert(status == NIXL_SUCCESS);

    init_params["ucx_error_handling_mode"] = "none";
    init_params["num_workers"] = std::to_string(1);

    nixlBackendH* ucx_backend = nullptr;
    status = agent->createBackend("UCX", init_params, ucx_backend);
    assert(status == NIXL_SUCCESS && ucx_backend != nullptr);

    int num_remote_nodes = num_nodes - 1;

    nixl_agent_infos.emplace_back(num_remote_nodes, num_ranks);
    nixl_agent_infos[0].agent = agent;
    nixl_agent_infos[0].agent_name = agent_name;
    nixl_agent_infos[0].backend = ucx_backend;
    nixl_agent_infos[0].extra_params.backends.push_back(ucx_backend);
    NIXL_LOG("    [Rank %d] _nixl_agents_init: done (%d remote nodes)\n", rank_uuid, num_remote_nodes);
}

void HybridEP_NIXLConnector::_nixl_agents_connect(const std::vector<int>& ranks) {
    NIXL_LOG("    [Rank %d] _nixl_agents_connect: Connecting to %zu remote agents...\n", rank_uuid, ranks.size());
    int agent_idx = 0;

    for (int remote_rank : ranks) {
        std::string remote_agent_name = std::to_string(remote_rank);
        nixl_agent_infos[agent_idx].dst_agent_names[remote_rank] = remote_agent_name;

        NIXL_LOG("    [Rank %d] _nixl_agents_connect: Fetching metadata for remote agent '%s'...\n",
               rank_uuid, remote_agent_name.c_str());
        nixl_status_t fetch_status = nixl_agent_infos[agent_idx].agent->fetchRemoteMD(remote_agent_name);
        assert(fetch_status == NIXL_SUCCESS);

        nixl_xfer_dlist_t empty_descs(VRAM_SEG);
        int wait_count = 0;
        while (nixl_agent_infos[agent_idx].agent->checkRemoteMD(remote_agent_name, empty_descs) != NIXL_SUCCESS) {
            sleep_ms(10);
            wait_count++;
            if (wait_count % 100 == 0) {
                NIXL_LOG("    [Rank %d] _nixl_agents_connect: Still waiting for rank %d metadata (%d waits)...\n",
                       rank_uuid, remote_rank, wait_count);
            }
        }
        NIXL_LOG("    [Rank %d] _nixl_agents_connect: Metadata available from rank %d\n", rank_uuid, remote_rank);
    }
    NIXL_LOG("    [Rank %d] _nixl_agents_connect: Remote metadata fetched for all %zu ranks\n", rank_uuid, ranks.size());
}

void HybridEP_NIXLConnector::_nixl_agents_wireup(const std::vector<int>& ranks) {
    NIXL_LOG("    [Rank %d] _nixl_agents_wireup: Starting wireup for %zu ranks...\n", rank_uuid, ranks.size());
    int agent_idx = 0;

    for (int remote_rank : ranks) {
        std::string remote_agent_name = std::to_string(remote_rank);
        std::string my_peer_info_str(reinterpret_cast<const char*>(&my_peer_info), sizeof(NixlPeerInfo));
        nixl_agent_infos[agent_idx].agent->genNotif(remote_agent_name, my_peer_info_str);
        NIXL_LOG("    [Rank %d] _nixl_agents_wireup: Sent peer info notification to rank %d\n", rank_uuid, remote_rank);
    }

    int received_count = 0;
    for (int remote_rank : ranks) {
        int poll_count = 0;
        while (!nixl_agent_infos[agent_idx].wire_up_done[remote_rank]) {
            nixl_notifs_t notif_map;
            nixl_agent_infos[agent_idx].agent->getNotifs(notif_map);

            for (auto &notif : notif_map) {
                std::string peer_info_payload = notif.second[0];
                NixlPeerInfo remote_peer_info;
                memcpy(&remote_peer_info, peer_info_payload.c_str(), sizeof(NixlPeerInfo));
                nixl_peer_info[remote_peer_info.rank] = remote_peer_info;
                nixl_agent_infos[agent_idx].wire_up_done[remote_peer_info.rank] = true;
                received_count++;

                NIXL_LOG("    [Rank %d] _nixl_agents_wireup: Received peer info from rank %d (%d/%zu)\n",
                       rank_uuid, remote_peer_info.rank, received_count, ranks.size());
            }

            poll_count++;
            if (poll_count % 1000 == 0) {
                NIXL_LOG("    [Rank %d] _nixl_agents_wireup: Still waiting for rank %d (%d polls)...\n",
                       rank_uuid, remote_rank, poll_count);
            }
            sleep_ms(1);
        }
    }
    _nixl_ucx_wireup(ranks);
    NIXL_LOG("    [Rank %d] _nixl_agents_wireup: done\n", rank_uuid);
}

void HybridEP_NIXLConnector::_nixl_ucx_wireup(const std::vector<int>& ranks) {
    int agent_idx = 0;
    assert(buffers.attn_input_token_d != nullptr && "attn_input_token_d required for UCX wireup");

    nixl_xfer_dlist_t dummy_src_dlist(VRAM_SEG);
    dummy_src_dlist.addDesc(nixlBlobDesc((uintptr_t)buffers.attn_input_token_d, sizeof(uint64_t), local_device_id, ""));

    for (int remote_rank : ranks) {
        assert(nixl_peer_info[remote_rank].rdma_buffer_ptr != nullptr && "Remote rdma_buffer_ptr required for UCX wireup");

        nixl_xfer_dlist_t dummy_dst_dlist(VRAM_SEG);
        dummy_dst_dlist.addDesc(nixlBlobDesc((uintptr_t)nixl_peer_info[remote_rank].rdma_buffer_ptr,
                                             sizeof(uint64_t),
                                             nixl_peer_info[remote_rank].device_id, ""));

        nixl_opt_args_t wireup_params = {};
        wireup_params.backends.push_back(nixl_agent_infos[agent_idx].backend);

        nixlXferReqH* wireup_req = nullptr;
        nixl_status_t status = nixl_agent_infos[agent_idx].agent->createXferReq(
            NIXL_WRITE, dummy_src_dlist, dummy_dst_dlist,
            std::to_string(remote_rank), wireup_req, &wireup_params);
        assert(status == NIXL_SUCCESS);

        status = nixl_agent_infos[agent_idx].agent->postXferReq(wireup_req);
        assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
        while ((status = nixl_agent_infos[agent_idx].agent->getXferStatus(wireup_req)) == NIXL_IN_PROG) {
            sleep_ms(1);
        }
        assert(status == NIXL_SUCCESS);

        status = nixl_agent_infos[agent_idx].agent->releaseXferReq(wireup_req);
        assert(status == NIXL_SUCCESS);

        NIXL_LOG("    [Rank %d] _nixl_ucx_wireup: Connected to rank %d\n", rank_uuid, remote_rank);
    }
}

void HybridEP_NIXLConnector::_nixl_agents_wiredown(const std::vector<int>& ranks) {
    int agent_idx = 0;

    for (int remote_rank : ranks) {
        nixl_status_t status = nixl_agent_infos[agent_idx].agent->invalidateRemoteMD(
            nixl_agent_infos[agent_idx].dst_agent_names[remote_rank]);
        if (status != NIXL_SUCCESS) {
            fprintf(stderr, "WARNING: Failed to invalidate remote metadata for rank %d\n", remote_rank);
        }
        nixl_agent_infos[agent_idx].dst_agent_names[remote_rank].clear();
        nixl_agent_infos[agent_idx].wire_up_done[remote_rank] = false;
    }
}

void HybridEP_NIXLConnector::_nixl_create_memory_views(const std::vector<int>& ranks) {
    NIXL_LOG("    [Rank %d] _nixl_create_memory_views: Creating memory views...\n", rank_uuid);

    int agent_idx = 0;
    int node_rank = rank_uuid / ranks_per_node;
    int local_nvl_rank = rank_uuid % ranks_per_node;
    int num_remote_nodes = num_nodes - 1;

    for (int peer_idx = 0; peer_idx < num_remote_nodes; ++peer_idx) {
        int actual_node_rank = peer_idx < node_rank ? peer_idx : (peer_idx + 1);
        int expected_remote_rank = actual_node_rank * ranks_per_node + local_nvl_rank;
        bool found = std::find(ranks.begin(), ranks.end(), expected_remote_rank) != ranks.end();
        assert(found && "Expected remote rank not found in connected ranks list");
    }

    size_t token_stride = buffers.attn_input_token_sz;
    NIXL_LOG("    [Rank %d]   Token stride=%zu bytes\n", rank_uuid, token_stride);

    // -- Dispatch memory views --
    NIXL_LOG("    [Rank %d]   Creating dispatch memory views...\n", rank_uuid);
    nixl_xfer_dlist_t dispatch_local_descs(VRAM_SEG);
    if (buffers.attn_input_token_d && buffers.attn_input_token_sz > 0) {
        dispatch_local_descs.addDesc(nixlBlobDesc((uintptr_t)buffers.attn_input_token_d,
                                                   buffers.attn_input_token_sz, local_device_id, ""));
    }

    if (buffers.attn_input_prob_d && buffers.forward_dispatch) {
        dispatch_local_descs.addDesc(nixlBlobDesc((uintptr_t)buffers.attn_input_prob_d,
                                                   buffers.attn_input_prob_sz, local_device_id, ""));
    }

    if (buffers.attn_input_token_scaling_factor_d && buffers.use_fp8) {
        dispatch_local_descs.addDesc(nixlBlobDesc((uintptr_t)buffers.attn_input_token_scaling_factor_d,
                                                   buffers.attn_input_token_scaling_factor_sz, local_device_id, ""));
    }

    nixl_remote_dlist_t dispatch_remote_data_descs(VRAM_SEG);
    nixl_remote_dlist_t dispatch_remote_signal_descs(VRAM_SEG);

    for (int peer_idx = 0; peer_idx < num_remote_nodes; ++peer_idx) {
        int actual_node_rank = peer_idx < node_rank ? peer_idx : (peer_idx + 1);
        int remote_rank = actual_node_rank * ranks_per_node + local_nvl_rank;

        int my_node_rank_in_remote = (node_rank < actual_node_rank) ? node_rank : (node_rank - 1);
        std::string remote_agent_name = nixl_agent_infos[agent_idx].dst_agent_names[remote_rank];

        void* remote_dispatch_token_addr = (uint8_t*)nixl_peer_info[remote_rank].rdma_buffer_ptr +
                                           my_node_rank_in_remote * token_stride;
        dispatch_remote_data_descs.addDesc(nixlRemoteDesc(
            (uintptr_t)remote_dispatch_token_addr,
            token_stride,
            nixl_peer_info[remote_rank].device_id,
            remote_agent_name));

        if (buffers.rdma_inter_node_group_prob_d && buffers.forward_dispatch) {
            size_t prob_stride = buffers.rdma_inter_node_group_prob_sz/(num_nodes-1);
            void* remote_dispatch_prob_addr = (uint8_t*)nixl_peer_info[remote_rank].rdma_prob_buffer_ptr +
                                                my_node_rank_in_remote * prob_stride;
            dispatch_remote_data_descs.addDesc(nixlRemoteDesc(
                (uintptr_t)remote_dispatch_prob_addr,
                prob_stride,
                nixl_peer_info[remote_rank].device_id,
                remote_agent_name));
        }

        if (buffers.rdma_inter_node_group_scaling_factor_d && buffers.use_fp8) {
            size_t sf_stride = buffers.attn_input_token_scaling_factor_sz;
            void* remote_dispatch_scaling_factor_addr = (uint8_t*)nixl_peer_info[remote_rank].rdma_scaling_factor_buffer_ptr +
                                                my_node_rank_in_remote * sf_stride;
            dispatch_remote_data_descs.addDesc(nixlRemoteDesc(
                (uintptr_t)remote_dispatch_scaling_factor_addr,
                sf_stride,
                nixl_peer_info[remote_rank].device_id,
                remote_agent_name));
        }

        uint64_t* remote_dispatch_flag_addr = nixl_peer_info[remote_rank].dispatch_flags_ptr;
        dispatch_remote_signal_descs.addDesc(nixlRemoteDesc(
            (uintptr_t)remote_dispatch_flag_addr,
            buffers.rdma_inter_node_group_flags_sz,
            nixl_peer_info[remote_rank].device_id,
            remote_agent_name));

        NIXL_LOG("    [Rank %d]     dispatch[%d] -> remote_rank=%d, data=%p, signal=%p\n",
               rank_uuid, peer_idx, remote_rank, remote_dispatch_token_addr, (void*)remote_dispatch_flag_addr);
    }

    nixl_status_t status;
    status = nixl_agent_infos[agent_idx].agent->prepMemView(
        dispatch_local_descs,
        nixl_agent_infos[agent_idx].dispatch_local_mvh,
        &nixl_agent_infos[agent_idx].extra_params);
    assert(status == NIXL_SUCCESS && "Failed to create dispatch local memory view");

    status = nixl_agent_infos[agent_idx].agent->prepMemView(
        dispatch_remote_data_descs,
        nixl_agent_infos[agent_idx].dispatch_remote_data_mvh,
        &nixl_agent_infos[agent_idx].extra_params);
    assert(status == NIXL_SUCCESS && "Failed to create dispatch remote data memory view");

    status = nixl_agent_infos[agent_idx].agent->prepMemView(
        dispatch_remote_signal_descs,
        nixl_agent_infos[agent_idx].dispatch_remote_signal_mvh,
        &nixl_agent_infos[agent_idx].extra_params);
    assert(status == NIXL_SUCCESS && "Failed to create dispatch remote signal memory view");

    NIXL_LOG("    [Rank %d]   Dispatch memory views created\n", rank_uuid);

    // -- Combine memory views --
    NIXL_LOG("    [Rank %d]   Creating combine memory views...\n", rank_uuid);

    nixl_xfer_dlist_t combine_local_descs(VRAM_SEG);
    if (buffers.rdma_intra_node_red_token_d && buffers.rdma_intra_node_red_token_sz > 0) {
        combine_local_descs.addDesc(nixlBlobDesc((uintptr_t)buffers.rdma_intra_node_red_token_d,
                                                  buffers.rdma_intra_node_red_token_sz, local_device_id, ""));
    }
    if (buffers.rdma_intra_node_red_prob_d && buffers.backward_combine) {
        combine_local_descs.addDesc(nixlBlobDesc((uintptr_t)buffers.rdma_intra_node_red_prob_d,
                                                  buffers.rdma_intra_node_red_prob_sz, local_device_id, ""));
    }

    nixl_remote_dlist_t combine_remote_data_descs(VRAM_SEG);
    nixl_remote_dlist_t combine_remote_signal_descs(VRAM_SEG);

    for (int peer_idx = 0; peer_idx < num_remote_nodes; ++peer_idx) {
        int actual_node_rank = peer_idx < node_rank ? peer_idx : (peer_idx + 1);
        int remote_rank = actual_node_rank * ranks_per_node + local_nvl_rank;
        int my_node_rank_in_remote = (node_rank < actual_node_rank) ? node_rank : (node_rank - 1);
        std::string remote_agent_name = nixl_agent_infos[agent_idx].dst_agent_names[remote_rank];

        size_t combine_token_stride = buffers.rdma_intra_node_red_token_sz / (num_nodes - 1);

        void* remote_combine_token_addr = (uint8_t*)nixl_peer_info[remote_rank].combine_rdma_buffer_ptr +
                                          my_node_rank_in_remote * combine_token_stride;
        combine_remote_data_descs.addDesc(nixlRemoteDesc(
            (uintptr_t)remote_combine_token_addr,
            combine_token_stride,
            nixl_peer_info[remote_rank].device_id,
            remote_agent_name));

        if (buffers.combine_rdma_inter_node_group_prob_d && buffers.backward_combine) {
            size_t combine_prob_stride = buffers.rdma_inter_node_group_prob_sz/(num_nodes-1);

            void* remote_combine_prob_addr = (uint8_t*)nixl_peer_info[remote_rank].combine_rdma_prob_buffer_ptr +
                                              my_node_rank_in_remote * combine_prob_stride;
            combine_remote_data_descs.addDesc(nixlRemoteDesc(
                (uintptr_t)remote_combine_prob_addr,
                combine_prob_stride,
                nixl_peer_info[remote_rank].device_id,
                remote_agent_name));

        }

        uint64_t* remote_combine_flag_addr = nixl_peer_info[remote_rank].combine_flags_ptr;
        combine_remote_signal_descs.addDesc(nixlRemoteDesc(
            (uintptr_t)remote_combine_flag_addr,
            buffers.combine_rdma_inter_node_group_flags_sz,
            nixl_peer_info[remote_rank].device_id,
            remote_agent_name));

        NIXL_LOG("    [Rank %d]     combine[%d] -> remote_rank=%d, data=%p, signal=%p\n",
               rank_uuid, peer_idx, remote_rank, remote_combine_token_addr, (void*)remote_combine_flag_addr);
    }

    status = nixl_agent_infos[agent_idx].agent->prepMemView(
        combine_local_descs,
        nixl_agent_infos[agent_idx].combine_local_mvh,
        &nixl_agent_infos[agent_idx].extra_params);
    assert(status == NIXL_SUCCESS && "Failed to create combine local memory view");

    status = nixl_agent_infos[agent_idx].agent->prepMemView(
        combine_remote_data_descs,
        nixl_agent_infos[agent_idx].combine_remote_data_mvh,
        &nixl_agent_infos[agent_idx].extra_params);
    assert(status == NIXL_SUCCESS && "Failed to create combine remote data memory view");

    status = nixl_agent_infos[agent_idx].agent->prepMemView(
        combine_remote_signal_descs,
        nixl_agent_infos[agent_idx].combine_remote_signal_mvh,
        &nixl_agent_infos[agent_idx].extra_params);
    assert(status == NIXL_SUCCESS && "Failed to create combine remote signal memory view");

    NIXL_LOG("    [Rank %d]   Combine memory views created\n", rank_uuid);

    NIXL_LOG("    [Rank %d] _nixl_create_memory_views: All memory views created for %d remote nodes\n",
           rank_uuid, num_remote_nodes);
}

void HybridEP_NIXLConnector::_nixl_build_gpu_contexts(int num_dispatch_blocks, int num_combine_blocks) {
    NIXL_LOG("    [Rank %d] _nixl_build_gpu_contexts: Building GPU contexts...\n", rank_uuid);

    int agent_idx = 0;
    int num_remote_nodes = num_nodes - 1;

    int ucx_num_channels = 1;
    const char* env_num_channels = std::getenv("UCX_RC_GDA_NUM_CHANNELS");
    if (env_num_channels) {
        ucx_num_channels = std::atoi(env_num_channels);
        if (ucx_num_channels <= 0) ucx_num_channels = 1;
    }
    NIXL_LOG("    [Rank %d] _nixl_build_gpu_contexts: UCX_RC_GDA_NUM_CHANNELS=%d\n", rank_uuid, ucx_num_channels);

    // -- Dispatch context --
    NIXL_LOG("    [Rank %d] _nixl_build_gpu_contexts: Building dispatch GPU context...\n", rank_uuid);
    cudaMalloc(&d_dispatch_flag_counters, sizeof(uint64_t) * num_remote_nodes);
    cudaMemset(d_dispatch_flag_counters, 0, sizeof(uint64_t) * num_remote_nodes);

    dispatch_gpu_nixl_ctx h_dispatch_ctx = {};
    h_dispatch_ctx.local_mvh = nixl_agent_infos[agent_idx].dispatch_local_mvh;
    h_dispatch_ctx.remote_data_mvh = nixl_agent_infos[agent_idx].dispatch_remote_data_mvh;
    h_dispatch_ctx.remote_signal_mvh = nixl_agent_infos[agent_idx].dispatch_remote_signal_mvh;
    h_dispatch_ctx.local_flag_counters = d_dispatch_flag_counters;
    h_dispatch_ctx.num_remote_nodes = num_remote_nodes;
    h_dispatch_ctx.num_channels = ucx_num_channels;
    h_dispatch_ctx.rank = rank_uuid;

    cudaMalloc(&d_dispatch_nixl_ctx, sizeof(dispatch_gpu_nixl_ctx));
    cudaMemcpy(d_dispatch_nixl_ctx, &h_dispatch_ctx,
               sizeof(dispatch_gpu_nixl_ctx),
               cudaMemcpyHostToDevice);
    NIXL_LOG("    [Rank %d]   Dispatch GPU context built\n", rank_uuid);

    // -- Combine context --
    NIXL_LOG("    [Rank %d] _nixl_build_gpu_contexts: Building combine GPU context...\n", rank_uuid);
    cudaMalloc(&d_combine_flag_counters, sizeof(uint64_t) * num_remote_nodes);
    cudaMemset(d_combine_flag_counters, 0, sizeof(uint64_t) * num_remote_nodes);

    combine_gpu_nixl_ctx h_combine_ctx = {};
    h_combine_ctx.local_mvh = nixl_agent_infos[agent_idx].combine_local_mvh;
    h_combine_ctx.remote_data_mvh = nixl_agent_infos[agent_idx].combine_remote_data_mvh;
    h_combine_ctx.remote_signal_mvh = nixl_agent_infos[agent_idx].combine_remote_signal_mvh;
    h_combine_ctx.local_flag_counters = d_combine_flag_counters;
    h_combine_ctx.num_remote_nodes = num_remote_nodes;
    h_combine_ctx.num_channels = ucx_num_channels;
    h_combine_ctx.rank = rank_uuid;

    cudaMalloc(&d_combine_nixl_ctx, sizeof(combine_gpu_nixl_ctx));
    cudaMemcpy(d_combine_nixl_ctx, &h_combine_ctx,
               sizeof(combine_gpu_nixl_ctx),
               cudaMemcpyHostToDevice);
    NIXL_LOG("    [Rank %d]   Combine GPU context built\n", rank_uuid);

    NIXL_LOG("    [Rank %d] _nixl_build_gpu_contexts: NIXL GPU contexts built (num_channels=%d)\n",
           rank_uuid, ucx_num_channels);
}

void HybridEP_NIXLConnector::_register_buffers_with_agents() {
    NIXL_LOG("    [Rank %d] _register_buffers_with_agents\n", rank_uuid);
    int agent_idx = 0;
    int buffer_count = 0;
    nixl_reg_dlist_t reg_dlist(VRAM_SEG);
    if (buffers.attn_input_token_d && buffers.attn_input_token_sz > 0) {
        buffer_count++;
        nixlBlobDesc desc((uintptr_t)buffers.attn_input_token_d,
                         buffers.attn_input_token_sz,
                         local_device_id, "attn_input_token");
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc);
        reg_dlist.addDesc(desc);
    }

    if (buffers.attn_input_prob_d && buffers.attn_input_prob_sz > 0) {
        buffer_count++;
        nixlBlobDesc desc((uintptr_t)buffers.attn_input_prob_d,
                         buffers.attn_input_prob_sz,
                         local_device_id, "attn_input_prob");
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc);
        reg_dlist.addDesc(desc);
    }

    if (buffers.attn_input_token_scaling_factor_d && buffers.attn_input_token_scaling_factor_sz > 0) {
        buffer_count++;
        nixlBlobDesc desc((uintptr_t)buffers.attn_input_token_scaling_factor_d,
                         buffers.attn_input_token_scaling_factor_sz,
                         local_device_id, "attn_input_token_scaling_factor");
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc);
        reg_dlist.addDesc(desc);
    }

    if (buffers.rdma_inter_node_group_token_d && buffers.rdma_inter_node_group_token_sz > 0) {
        buffer_count++;
        nixlBlobDesc desc((uintptr_t)buffers.rdma_inter_node_group_token_d,
                         buffers.rdma_inter_node_group_token_sz,
                         local_device_id, "rdma_inter_node_group_token");
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc);
        reg_dlist.addDesc(desc);
    }

    if (buffers.rdma_inter_node_group_flags_d && buffers.rdma_inter_node_group_flags_sz > 0) {
        buffer_count++;
        nixlBlobDesc desc((uintptr_t)buffers.rdma_inter_node_group_flags_d,
                         buffers.rdma_inter_node_group_flags_sz,
                         local_device_id, "rdma_inter_node_group_flags");
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc);
        reg_dlist.addDesc(desc);
    }

    if (buffers.rdma_inter_node_group_prob_d && buffers.rdma_inter_node_group_prob_sz > 0) {
        buffer_count++;
        nixlBlobDesc desc((uintptr_t)buffers.rdma_inter_node_group_prob_d,
                         buffers.rdma_inter_node_group_prob_sz,
                         local_device_id, "rdma_inter_node_group_prob");
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc);
        reg_dlist.addDesc(desc);
    }

    if (buffers.rdma_inter_node_group_scaling_factor_d && buffers.rdma_inter_node_group_scaling_factor_sz > 0) {
        buffer_count++;
        nixlBlobDesc desc((uintptr_t)buffers.rdma_inter_node_group_scaling_factor_d,
                         buffers.rdma_inter_node_group_scaling_factor_sz,
                         local_device_id, "rdma_inter_node_group_scaling_factor");
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc);
        reg_dlist.addDesc(desc);
    }

    if (buffers.rdma_intra_node_red_token_d && buffers.rdma_intra_node_red_token_sz > 0) {
        buffer_count++;
        nixlBlobDesc desc((uintptr_t)buffers.rdma_intra_node_red_token_d,
                         buffers.rdma_intra_node_red_token_sz,
                         local_device_id, "rdma_intra_node_red_token");
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc);
        reg_dlist.addDesc(desc);
    }

    if (buffers.rdma_intra_node_red_prob_d && buffers.rdma_intra_node_red_prob_sz > 0) {
        buffer_count++;
        nixlBlobDesc desc((uintptr_t)buffers.rdma_intra_node_red_prob_d,
                         buffers.rdma_intra_node_red_prob_sz,
                         local_device_id, "rdma_intra_node_red_prob");
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc);
        reg_dlist.addDesc(desc);
    }

    if (buffers.combine_rdma_inter_node_group_token_d && buffers.combine_rdma_inter_node_group_token_sz > 0) {
        buffer_count++;
        nixlBlobDesc desc((uintptr_t)buffers.combine_rdma_inter_node_group_token_d,
                         buffers.combine_rdma_inter_node_group_token_sz,
                         local_device_id, "combine_rdma_inter_node_group_token");
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc);
        reg_dlist.addDesc(desc);
    }

    if (buffers.combine_rdma_inter_node_group_flags_d && buffers.combine_rdma_inter_node_group_flags_sz > 0) {
        buffer_count++;
        nixlBlobDesc desc((uintptr_t)buffers.combine_rdma_inter_node_group_flags_d,
                         buffers.combine_rdma_inter_node_group_flags_sz,
                         local_device_id, "combine_rdma_inter_node_group_flags");
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc);
        reg_dlist.addDesc(desc);
    }

    if (buffers.combine_rdma_inter_node_group_prob_d && buffers.combine_rdma_inter_node_group_prob_sz > 0) {
        buffer_count++;
        nixlBlobDesc desc((uintptr_t)buffers.combine_rdma_inter_node_group_prob_d,
                         buffers.combine_rdma_inter_node_group_prob_sz,
                         local_device_id, "combine_rdma_inter_node_group_prob");
        nixl_agent_infos[agent_idx].src_vram.addDesc(desc);
        reg_dlist.addDesc(desc);
    }

    NIXL_LOG("    [Rank %d] _register_buffers_with_agents: registering %d buffers\n", rank_uuid, buffer_count);
    nixl_status_t status = nixl_agent_infos[agent_idx].agent->registerMem(reg_dlist);
    assert(status == NIXL_SUCCESS);

    size_t signal_size = 0;
    status = nixl_agent_infos[agent_idx].agent->getGpuSignalSize(signal_size, &nixl_agent_infos[agent_idx].extra_params);
    assert(status == NIXL_SUCCESS);
    assert(signal_size == sizeof(uint64_t));
    status = nixl_agent_infos[agent_idx].agent->prepGpuSignal(reg_dlist, &nixl_agent_infos[agent_idx].extra_params);
    assert(status == NIXL_SUCCESS);

    status = nixl_agent_infos[agent_idx].agent->sendLocalMD(&nixl_agent_infos[agent_idx].extra_params);
    assert(status == NIXL_SUCCESS);
    NIXL_LOG("    [Rank %d] _register_buffers_with_agents: done (%d buffers)\n", rank_uuid, buffer_count);
}

}  // namespace hybrid_ep

#endif  // USE_NIXL

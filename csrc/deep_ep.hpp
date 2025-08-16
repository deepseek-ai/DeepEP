#pragma once

// Forcibly disable NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>
#include <tuple>
#include <vector>
#include <string>

#include <memory>
#include "config.hpp"
#include "event.hpp"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

#define EP_REMOVE_ONCE
#define EP_EXECUTE_ONCE(func) do { static bool _ = ((func), true); } while(0)
#include "nixl.h"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME deep_ep_cpp
#endif

namespace deep_ep {
#define MAX_IP_LENGTH 16
#define MAX_BOOT_ID_LENGTH 37

struct NixlPeerInfo {
    char ip[MAX_IP_LENGTH];
    char boot_id[MAX_BOOT_ID_LENGTH];
    ino_t ipc_namespace_inode;
    void *rdma_buffer_ptr;
    uint64_t *counters_buffer_ptr;
    cudaIpcMemHandle_t rdma_ipc_handle;
    cudaIpcMemHandle_t counters_ipc_handle;
    uint64_t* barrier_ptr;
    int device_id;
    int rank;
};

/// @brief NixlAgentInfo is a struct that contains the information of a Nixl agent (per channel)
/// @param agent The Nixl agent
/// @param head_counters The head counters of the Nixl agent (per remote rank)
/// @param head_counter_handles The handles of the head counters of the Nixl agent (per remote rank)
/// @param tail_counters The tail counters of the Nixl agent (per remote rank)
/// @param tail_counter_handles The handles of the tail counters of the Nixl agent (per remote rank)
/// @param src_vram The source VRAM of the Nixl agent
/// @param dst_vram The destination VRAM of the Nixl agent (per remote rank)
struct NixlAgentInfo
{
    NixlAgentInfo(std::shared_ptr<nixlAgent> agent, std::string agent_name, nixlBackendH* backend, int num_peers, int max_num_ranks): agent(agent), agent_name(agent_name), backend(backend), src_vram(VRAM_SEG) {
        dst_vram.resize(num_peers, nixl_xfer_dlist_t(VRAM_SEG));
        cpu_batch_reqs.resize(num_peers);
        gpu_batch_reqs.resize(num_peers);
        cpu_counter_reqs.resize(num_peers);
        cpu_barrier_reqs.resize(num_peers);
        dst_agent_names.resize(max_num_ranks);
        wire_up_done.resize(max_num_ranks, false);
    }

    /* Common fields */
    std::shared_ptr<nixlAgent> agent;
    std::string agent_name;
    nixl_opt_args_t extra_params;
    nixlBackendH* backend;
    nixl_xfer_dlist_t src_vram;
    std::vector<nixl_xfer_dlist_t> dst_vram;
    std::vector<std::string> dst_agent_names;
    std::vector<nixlXferReqH*> cpu_batch_reqs;
    std::vector<nixlGpuXferReqH> gpu_batch_reqs;
    std::vector<nixlXferReqH*> cpu_counter_reqs;
    std::vector<nixlXferReqH*> cpu_barrier_reqs;
    std::vector<bool> wire_up_done; // [num_peers]
};

struct nixl_low_latency_ctx {
    std::vector<nixlXferReqH *> cpu_remote_counter_reqs_0; // [dest_expert_id,remote_rank], cpu ptrs to nixlXferReqH
    std::vector<nixlXferReqH *> cpu_remote_counter_reqs_1; // [dest_expert_id,remote_rank], cpu ptrs to nixlXferReqH
    std::vector<nixlGpuXferReqH> gpu_remote_counter_reqs_0; // [dest_expert_id,remote_rank], gpu ptrs to nixlGpuXferReqH
    std::vector<nixlGpuXferReqH> gpu_remote_counter_reqs_1; // [dest_expert_id,remote_rank], gpu ptrs to nixlGpuXferReqH
    std::vector<nixlXferReqH *> cpu_sync_counters;
    std::vector<nixlGpuXferReqH> gpu_sync_counters;
    std::vector<void *> rdma_p2p_ptrs; // [num_ranks]
    std::vector<uint64_t *> counters_p2p_ptrs; // [num_ranks]
    internode_ll::gpu_nixl_ctx nixl_ctx[2]; // Double buffering
};

class nixl_internode_ctx {
public:
    std::vector<internode::gpu_channel_nixl_ctx> cpu_channel_ctxs;
    internode::gpu_nixl_ctx gpu_ctx;
    int num_rdma_ranks = 0;
    int num_channels = 0;
    int rank = 0;

    nixl_internode_ctx(unsigned int num_channels = 0, unsigned int num_rdma_ranks = 0, int rank = 0): num_channels(num_channels), num_rdma_ranks(num_rdma_ranks), rank(rank) {
        cpu_channel_ctxs.resize(num_channels);
        for (int i = 0; i < num_channels; i++) {
            CUDA_CHECK(cudaMalloc(&cpu_channel_ctxs[i].data_request_handles, sizeof(nixlGpuXferReqH) * num_rdma_ranks));
            CUDA_CHECK(cudaMalloc(&cpu_channel_ctxs[i].remote_head_counter_handles, sizeof(nixlGpuXferReqH) * num_rdma_ranks));
            CUDA_CHECK(cudaMalloc(&cpu_channel_ctxs[i].remote_barrier_handles, sizeof(nixlGpuXferReqH) * num_rdma_ranks));
            CUDA_CHECK(cudaMemset(cpu_channel_ctxs[i].remote_barrier_handles, 0, sizeof(nixlGpuXferReqH) * num_rdma_ranks));
        }
        CUDA_CHECK(cudaMalloc(&gpu_ctx.channel_ctxs, sizeof(internode::gpu_channel_nixl_ctx) * num_channels));
    }

    ~nixl_internode_ctx() noexcept(false) {
        for (int i = 0; i < num_channels; i++) {
            CUDA_CHECK(cudaFree(cpu_channel_ctxs[i].data_request_handles));
            CUDA_CHECK(cudaFree(cpu_channel_ctxs[i].remote_head_counter_handles));
            CUDA_CHECK(cudaFree(cpu_channel_ctxs[i].remote_barrier_handles));
        }
        CUDA_CHECK(cudaFree(gpu_ctx.channel_ctxs));
    }

    void copy_to_gpu() {
        CUDA_CHECK(cudaMemcpy(gpu_ctx.channel_ctxs, cpu_channel_ctxs.data(), sizeof(internode::gpu_channel_nixl_ctx) * num_channels, cudaMemcpyHostToDevice));
        gpu_ctx.num_channels = num_channels;
        gpu_ctx.num_rdma_ranks = num_rdma_ranks;
        gpu_ctx.num_channels = num_channels;
        gpu_ctx.rank = rank;
    }
};

struct Buffer {
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

private:
    // Low-latency mode buffer
    int low_latency_buffer_idx = 0;
    bool low_latency_mode = false;

    // NVLink Buffer
    int64_t num_nvl_bytes;
    void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;

    // NVSHMEM Buffer
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;

    // Device info and communication
    int device_id;
    int num_device_sms;
    int rank, rdma_rank, nvl_rank;
    int num_ranks, num_rdma_ranks, num_nvl_ranks;
    std::vector<int> remote_ranks; /* global ranks */
    cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS];

    // Stream for communication
    at::cuda::CUDAStream comm_stream;

    // After IPC/NVSHMEM synchronization, this flag will be true
    bool available = false;

    // Whether explicit `destroy()` is required.
    bool explicitly_destroy;
    // After `destroy()` be called, this flag will be true
    bool destroyed = false;

    // Barrier signals
    int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    int** barrier_signal_ptrs_gpu = nullptr;

    // Workspace
    void* workspace = nullptr;

    // Host-side MoE info
    volatile int* moe_recv_counter = nullptr;
    int* moe_recv_counter_mapped = nullptr;

    // Host-side expert-level MoE info
    volatile int* moe_recv_expert_counter = nullptr;
    int* moe_recv_expert_counter_mapped = nullptr;

    // Host-side RDMA-level MoE info
    volatile int* moe_recv_rdma_counter = nullptr;
    int* moe_recv_rdma_counter_mapped = nullptr;

    std::vector<NixlAgentInfo> nixl_agent_infos;
    std::vector<std::string> nixl_agent_metadata;
    std::vector<NixlPeerInfo> nixl_peer_info;
    uint64_t *counters_buffer_ptr = nullptr;
    NixlPeerInfo my_peer_info;
    uint64_t num_counters;
    uint64_t max_num_ranks;
    int env_num_channels;
    uint64_t* last_barrier_counter = nullptr;
    uint64_t* local_barrier_counter = nullptr;
    nixl_xfer_dlist_t dummy_src_dlist; // TODO: Remove once NIXL supports null src dlist for signals

    nixl_internode_ctx *nixl_kernel_params = nullptr;
    struct nixl_low_latency_ctx nllc;

    /* Common private funcs */
    void _nixl_agents_init();
    void _nixl_agents_connect(const std::vector<int>& ranks);
    void _nixl_agents_wireup();
    void _nixl_agents_wiredown(const std::vector<int>& ranks_to_remove);

    /* Low-latency mode private funcs */
    void _nixl_ll_init(const std::vector<int>& ranks_to_setup);
    void _nixl_ll_nllc_init();
    void _nixl_ll_counters_prepare(const std::vector<int>& ranks_to_setup);
    void _nixl_ll_batches_prepare(const std::vector<int>& ranks_to_setup);
    void _nixl_ll_p2p_ptrs_prepare(const std::vector<int>& ranks_to_setup);
    void _nixl_ll_gpu_ctx_prepare();
    
    /* Low-latency mode cleanup funcs */
    void _nixl_ll_cleanup(const std::vector<int>& ranks_to_remove);
    void _nixl_ll_counters_ranks_cleanup(const std::vector<int>& ranks_to_remove);
    void _nixl_ll_batches_cleanup(const std::vector<int>& ranks_to_remove);
    void _nixl_ll_p2p_ptrs_cleanup(const std::vector<int>& ranks_to_remove);
    void _nixl_ll_gpu_ctx_cleanup();

    /* Internode mode private funcs */
    void _nixl_internode_init();
    void _nixl_internode_local_data_init();
    void _nixl_remote_counters_prepare();
    void _nixl_internode_batches_prepare();
    void _nixl_kernels_params_free();
    void _nixl_establish_new_connections();
    void _ipc_handles_sync(const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles);
public:
    Buffer(int rank, bool low_latency_mode, bool explicitly_destroy);
    void update_memory_buffers(int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes);
    void connect_ranks(const std::vector<int>& remote_ranks_list, const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles = {});
    void remove_ranks(const std::vector<int>& remote_ranks_list);

    void init(int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes);

    ~Buffer() noexcept(false);

    bool is_available() const;

    bool is_internode_available() const;

    int get_num_rdma_ranks() const;

    int get_rdma_rank() const;

    int get_root_rdma_rank(bool global) const;

    int get_local_device_id() const;

    pybind11::bytearray get_local_ipc_handle() const;

    pybind11::bytearray get_local_nvshmem_unique_id() const;

    torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const;

    torch::Stream get_comm_stream() const;

    void sync(const std::vector<int>& device_ids, const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles, const std::optional<pybind11::bytearray>& root_unique_id_opt);

    void destroy();

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
    get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts, std::optional<EventHandle>& previous_event,
                        bool async, bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
    intranode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank, const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                       int cached_num_recv_tokens, const std::optional<torch::Tensor>& cached_rank_prefix_matrix, const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                       int expert_alignment, int num_worst_tokens, const Config& config,
                       std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
    intranode_combine(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                      const std::optional<torch::Tensor>& bias_0, const std::optional<torch::Tensor>& bias_1,
                      const torch::Tensor& src_idx, const torch::Tensor& rank_prefix_matrix, const torch::Tensor& channel_prefix_matrix,
                      const torch::Tensor& send_head, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<EventHandle>>
    internode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank, const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                       const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                       int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
                       const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                       const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                       int expert_alignment, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
    internode_combine(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                      const std::optional<torch::Tensor>& bias_0, const std::optional<torch::Tensor>& bias_1,
                      const torch::Tensor& src_meta, const torch::Tensor& is_combined_token_in_rank,
                      const torch::Tensor& rdma_channel_prefix_matrix, const torch::Tensor& rdma_rank_prefix_sum, const torch::Tensor& gbl_channel_prefix_matrix,
                      const torch::Tensor& combined_rdma_head, const torch::Tensor& combined_nvl_head,
                      const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream);

    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
    low_latency_dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx,
                         const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                         const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                         int num_max_dispatch_tokens_per_rank, int num_experts,
                         bool use_fp8, bool round_scale, bool use_ue8m0,
                         bool async, bool return_recv_hook);

    std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
    low_latency_combine(const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                        const torch::Tensor& src_info, const torch::Tensor& layout_range,
                        const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
                        int num_max_dispatch_tokens_per_rank, int num_experts,
                        bool use_logfmt, bool zero_copy, bool async, bool return_recv_hook,
                        const std::optional<torch::Tensor>& out = std::nullopt);

    void low_latency_sync();

    torch::Tensor
    get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const;
};

} // namespace deep_ep

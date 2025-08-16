#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <pybind11/functional.h>
#include <torch/python.h>

#include "deep_ep.hpp"
#include "kernels/api.cuh"
#include "kernels/configs.cuh"
#include <cstdio>
#include <fstream>
#include <unistd.h>
#include <stdio.h>
#include "nixl.h"
#include <ifaddrs.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <sstream>

#define NIXL_ETCD_WATCH_TIMEOUT std::chrono::microseconds(1000000000) // 1000 seconds

#ifdef ENABLE_DEBUG_LOGS
#define HOST_LOG_DEBUG(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define HOST_LOG_DEBUG(...)
#endif

namespace deep_ep {

static void sleep_ms(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

static std::string _get_local_ip() {
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

void Buffer::update_memory_buffers(int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes)
{
    if (!available) {
        init(num_ranks, num_nvl_bytes, num_rdma_bytes);
        available = true;
    } else {
        throw std::runtime_error("Multiple calls to update_memory_buffers are not supported");
    }
}

Buffer::Buffer(int rank, bool low_latency_mode, bool explicitly_destroy):
        rank(rank), num_ranks(1), low_latency_mode(low_latency_mode),
        explicitly_destroy(explicitly_destroy),
        comm_stream(at::cuda::getStreamFromPool(true)),
        dummy_src_dlist(VRAM_SEG) {}

void Buffer::init(int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes)
{
    // Update buffer attributes
    this->max_num_ranks = num_ranks;
    this->num_nvl_bytes = num_nvl_bytes;
    this->num_rdma_bytes = num_rdma_bytes;

    // Metadata memory
    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

    // Common checks
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);

    // Get ranks
    CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);

    // Get device info
    cudaDeviceProp device_prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    num_device_sms = device_prop.multiProcessorCount;

    if (num_nvl_bytes > 0) {
        // Local IPC: alloc local memory and set local IPC handles
        CUDA_CHECK(cudaMalloc(&buffer_ptrs[nvl_rank], num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes));
        CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));
        buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

        // Set barrier signals
        barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        barrier_signal_ptrs_gpu = reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
    }

    // Create 32 MiB workspace
    CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    // MoE counter
    CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    // MoE expert-level counter
    CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++ i)
        moe_recv_expert_counter[i] = -1;

    // MoE RDMA-level counter
    if (num_rdma_ranks > 0) {
        CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
        *moe_recv_rdma_counter = -1;
    }

    env_num_channels = std::getenv("NIXL_DEEPEP_NUM_CHANNELS") ? std::stoi(std::getenv("NIXL_DEEPEP_NUM_CHANNELS")) : 1;
    EP_HOST_ASSERT(env_num_channels > 0);
    num_counters = env_num_channels * (low_latency_mode ? max_num_ranks : num_rdma_ranks) * 2 + max_num_ranks;
    rdma_buffer_ptr = internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);
    CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));
    counters_buffer_ptr = reinterpret_cast<uint64_t*>(internode::alloc(num_counters * sizeof(uint64_t), NUM_BUFFER_ALIGNMENT_BYTES));
    CUDA_CHECK(cudaMemset(counters_buffer_ptr, 0, num_counters * sizeof(uint64_t)));
    nllc = {};

    /* Initialize dummy src dlist with a dummy device address */
    dummy_src_dlist.addDesc(nixlBlobDesc((uintptr_t)counters_buffer_ptr, sizeof(uint64_t), device_id, ""));

    /* Prepare internode barrier */
    last_barrier_counter = counters_buffer_ptr + (num_counters - max_num_ranks);
    local_barrier_counter = counters_buffer_ptr + (num_counters - max_num_ranks + 1);

    strncpy(my_peer_info.ip, _get_local_ip().c_str(), MAX_IP_LENGTH - 1);
    my_peer_info.ip[MAX_IP_LENGTH - 1] = '\0';
    my_peer_info.rdma_buffer_ptr = rdma_buffer_ptr;
    my_peer_info.counters_buffer_ptr = counters_buffer_ptr;
    my_peer_info.barrier_ptr = local_barrier_counter;
    my_peer_info.device_id = get_local_device_id();
    my_peer_info.rank = rank;

    // Create IPC handles for rdma buffer and counters
    CUDA_CHECK(cudaIpcGetMemHandle(&my_peer_info.rdma_ipc_handle, rdma_buffer_ptr));
    CUDA_CHECK(cudaIpcGetMemHandle(&my_peer_info.counters_ipc_handle, counters_buffer_ptr));

    strncpy(my_peer_info.boot_id, boot_id_get().c_str(), MAX_BOOT_ID_LENGTH - 1);
    my_peer_info.boot_id[MAX_BOOT_ID_LENGTH - 1] = '\0';
    my_peer_info.ipc_namespace_inode = ipc_namespace_inode_get();

    nixl_peer_info.resize(max_num_ranks);
    nixl_peer_info[rank] = my_peer_info;

    _nixl_agents_init();

    // non-low-latency mode will be initialized once in connect_ranks
    if (low_latency_mode)
        _nixl_ll_init(std::vector<int>{rank});
}

Buffer::~Buffer() noexcept(false) {
    if (not explicitly_destroy) {
        destroy();
    } else if (not destroyed) {
        printf("WARNING: destroy() was not called before DeepEP buffer destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

bool Buffer::is_available() const {
    return available;
}

bool Buffer::is_internode_available() const {
    return is_available() and num_ranks > NUM_MAX_NVL_PEERS;
}

int Buffer::get_num_rdma_ranks() const {
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
    return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const {
    return device_id;
}

pybind11::bytearray Buffer::get_local_ipc_handle() const {
    return {ipc_handles[nvl_rank].reserved, CUDA_IPC_HANDLE_SIZE};
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
    return pybind11::bytearray();
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) + offset;
    auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
    return torch::from_blob(base_ptr, num_bytes / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));
}

torch::Stream Buffer::get_comm_stream() const {
    return comm_stream;
}

void Buffer::destroy() {
    EP_HOST_ASSERT(not destroyed);

    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    if (num_nvl_bytes > 0) {
        // Barrier
        intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Close remote IPC
        if (is_available()) {
            for (int i = 0; i < num_nvl_ranks; ++ i) if (i != nvl_rank)
                CUDA_CHECK(cudaIpcCloseMemHandle(buffer_ptrs[i]));
        }

        // Free local buffer and error flag
        CUDA_CHECK(cudaFree(buffer_ptrs[nvl_rank]));
    }

    internode::free(counters_buffer_ptr);
    internode::free(rdma_buffer_ptr);
    _nixl_kernels_params_free();
    rdma_buffer_ptr = nullptr;
    counters_buffer_ptr = nullptr;

    // Free workspace and MoE counter
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));

    // Free chunked mode staffs
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));

    destroyed = true;
    available = false;
}

void Buffer::sync(const std::vector<int> &device_ids,
                  const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles,
                  const std::optional<pybind11::bytearray>& root_unique_id_opt) {}

void Buffer::low_latency_sync() {
    EP_HOST_ASSERT(low_latency_mode);
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    internode_ll::sync(nllc.nixl_ctx[0], compute_stream);
}

void Buffer::_nixl_agents_connect(const std::vector<int>& ranks) {
    EP_HOST_ASSERT(!ranks.empty());

    // Assuming ranks vector does not include current rank and has only new ranks
    remote_ranks.insert(remote_ranks.end(), ranks.begin(), ranks.end());

    for (int remote_rank : ranks) {
        for (int channel = 0; channel < env_num_channels; ++channel) {
            std::string remote_agent_name = std::to_string(remote_rank) + "_ch" + std::to_string(channel);
            nixl_status_t fetch_status = nixl_agent_infos[channel].agent->fetchRemoteMD(remote_agent_name);
            if (fetch_status != NIXL_SUCCESS) {
                throw std::runtime_error("Failed to fetch metadata for remote agent " + remote_agent_name +
                                       ", status: " + std::to_string(fetch_status));
            }

            // Wait for remote metadata to be available
            nixl_xfer_dlist_t empty_descs(VRAM_SEG);
            while (nixl_agent_infos[channel].agent->checkRemoteMD(remote_agent_name, empty_descs) != NIXL_SUCCESS) {
                sleep_ms(10);
            }
        }
    }
}

void Buffer::_nixl_agents_wireup() {
    // for rank in remote ranks, we send a notification to the remote rank in which we send my_peer_info
    for (int remote_rank : remote_ranks) {
        for (int channel = 0; channel < env_num_channels; ++channel) {
            std::string remote_agent_name = std::to_string(remote_rank) + "_ch" + std::to_string(channel);
            // turn my_peer_info into a string
            std::string my_peer_info_str(reinterpret_cast<const char*>(&my_peer_info), sizeof(NixlPeerInfo));
            nixl_agent_infos[channel].agent->genNotif(remote_agent_name, my_peer_info_str);
        }
    }

    // we wait for a notifications from the remote ranks on every channel
    for (int channel = 0; channel < env_num_channels; ++channel) {
        for (int remote_rank : remote_ranks) {
            do {
                nixl_notifs_t notif_map;
                nixl_agent_infos[channel].agent->getNotifs(notif_map);
                for (auto &notif : notif_map) {
                    std::string remote_agent_name = notif.first;
                    std::string my_peer_info_str = notif.second[0];
                    NixlPeerInfo remote_peer_info;
                    memcpy(&remote_peer_info, my_peer_info_str.c_str(), sizeof(NixlPeerInfo));
                    nixl_peer_info[remote_peer_info.rank] = remote_peer_info;
                    nixl_agent_infos[channel].wire_up_done[remote_peer_info.rank] = true;
                }
            } while (!nixl_agent_infos[channel].wire_up_done[remote_rank]);
        }
    }
}

void Buffer::_ipc_handles_sync(const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles = {}) {
    if (num_nvl_bytes > 0) {
        EP_HOST_ASSERT(all_gathered_handles.size() == max_num_ranks);
        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++ i) {
            EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            EP_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE);
                CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs[i], ipc_handles[i], cudaIpcMemLazyEnablePeerAccess));
                barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
            } else {
                EP_HOST_ASSERT(std::memcmp(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE) == 0);
            }
        }

        // Copy all buffer and barrier signal pointers to GPU
        CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void Buffer::connect_ranks(const std::vector<int>& remote_ranks_list, const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles) {
    EP_HOST_ASSERT(!remote_ranks_list.empty());
    std::vector<int> new_ranks;

    if (all_gathered_handles.size() > 0)
        _ipc_handles_sync(all_gathered_handles);

    for (int remote_rank : remote_ranks_list) {
        // Skip self and ranks we are already connected to
        if (remote_rank == rank or std::find(remote_ranks.begin(), remote_ranks.end(), remote_rank) != remote_ranks.end())
            continue;

        // For non-low-latency mode, only establish NIXL connections to ranks that are the same nvl rank as us
        if (low_latency_mode or remote_rank % NUM_MAX_NVL_PEERS == nvl_rank)
            new_ranks.push_back(remote_rank);

        num_ranks++;
    }

    if (new_ranks.empty())
        return;

    // Set up dst_agent_names now that remote_ranks is populated
    for (int i = 0; i < env_num_channels; ++i) {
        for (int j = 0; j < new_ranks.size(); j++) {
            int remote_rank = new_ranks[j];
            nixl_agent_infos[i].dst_agent_names[remote_rank] = std::to_string(remote_rank) + "_ch" + std::to_string(i);
        }
    }

    _nixl_agents_connect(new_ranks);

    _nixl_agents_wireup();

    low_latency_mode ? _nixl_ll_init(new_ranks) : _nixl_internode_init();

    CUDA_CHECK(cudaDeviceSynchronize());

    // Ready to use
    available = true;
}

void Buffer::remove_ranks(const std::vector<int>& remote_ranks_list) {
    EP_HOST_ASSERT(low_latency_mode);
    EP_HOST_ASSERT(!remote_ranks_list.empty());
    EP_HOST_ASSERT(remote_ranks_list.size() <= remote_ranks.size());
    
    // Validate that the ranks to remove are at the end of remote_ranks
    size_t start_idx = remote_ranks.size() - remote_ranks_list.size();
    for (size_t i = 0; i < remote_ranks_list.size(); ++i) {
        EP_HOST_ASSERT(remote_ranks[start_idx + i] == remote_ranks_list[i] && 
                      "Ranks to remove must be at the end of remote_ranks in the same order");
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    _nixl_ll_cleanup(remote_ranks_list);
    _nixl_agents_wiredown(remote_ranks_list);

    remote_ranks.erase(remote_ranks.end() - remote_ranks_list.size(), remote_ranks.end());
    num_ranks -= remote_ranks_list.size();
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts,
                            std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kCUDA));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA));
    if (is_internode_available())
        num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    layout::get_dispatch_layout(topk_idx.data_ptr<int64_t>(),
                                num_tokens_per_rank.data_ptr<int>(),
                                num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,
                                num_tokens_per_expert.data_ptr<int>(),
                                is_token_in_rank.data_ptr<bool>(),
                                num_tokens, num_topk, num_ranks, num_experts,
                                comm_stream);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {num_tokens_per_rdma_rank}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::intranode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank, const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens, const std::optional<torch::Tensor>& cached_rank_prefix_matrix, const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                           int expert_alignment, int num_worst_tokens, const Config& config,
                           std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    bool cached_mode = cached_rank_prefix_matrix.has_value();

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and cached_rank_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and cached_rank_prefix_matrix->size(1) == num_ranks);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and cached_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and cached_channel_prefix_matrix->size(1) == num_channels);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    int64_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<int64_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1;
    auto rank_prefix_matrix = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    int num_memset_int = num_channels * num_ranks * 4;
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        rank_prefix_matrix = cached_rank_prefix_matrix.value();
        channel_prefix_matrix = cached_channel_prefix_matrix.value();

        // Copy rank prefix matrix and clean flags
        intranode::cached_notify_dispatch(rank_prefix_matrix.data_ptr<int>(), num_memset_int,
                                          buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank, num_ranks,
                                          comm_stream);
    } else {
        rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));

        // Send sizes
        // Meta information:
        //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
        // NOTES: no more token dropping in this version
        *moe_recv_counter = -1;
        for (int i = 0; i < num_local_experts; ++ i)
            moe_recv_expert_counter[i] = -1;
        EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes);
        intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
                                   num_tokens_per_expert->data_ptr<int>(), moe_recv_expert_counter_mapped, num_experts,
                                   num_tokens, is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
                                   rank_prefix_matrix.data_ptr<int>(),
                                   num_memset_int, expert_alignment,
                                   buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank,
                                   comm_stream, num_channels);

        if (num_worst_tokens > 0) {
            // No CPU sync, just allocate the worst case
            num_recv_tokens = num_worst_tokens;

            // Must be forward with top-k stuffs
            EP_HOST_ASSERT(topk_idx.has_value());
            EP_HOST_ASSERT(topk_weights.has_value());
        } else {
            // Synchronize total received tokens and tokens per expert
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() > NUM_CPU_TIMEOUT_SECS)
                    throw std::runtime_error("DeepEP error: CPU recv timeout");
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kCUDA));
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(), recv_x_scales = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    // Assign pointers
    int64_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<int64_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ?
                        torch::empty({num_recv_tokens}, x_scales->options()) :
                        torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Dispatch
    EP_HOST_ASSERT(num_ranks * num_ranks * sizeof(int) +                                                                    // Size prefix matrix
                   num_channels * num_ranks * sizeof(int) +                                                                 // Channel start offset
                   num_channels * num_ranks * sizeof(int) +                                                                 // Channel end offset
                   num_channels * num_ranks * sizeof(int) * 2 +                                                             // Queue head and tail
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * recv_x.element_size() +     // Data buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +                        // Source index buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(int64_t) +         // Top-k index buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) +           // Top-k weight buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales           // FP8 scale buffer
                   <= num_nvl_bytes);
    intranode::dispatch(recv_x.data_ptr(), recv_x_scales_ptr, recv_src_idx.data_ptr<int>(), recv_topk_idx_ptr, recv_topk_weights_ptr, recv_channel_prefix_matrix.data_ptr<int>(),
                        send_head.data_ptr<int>(),
                        x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
                        is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
                        num_tokens, num_worst_tokens, static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),
                        num_topk, num_experts, num_scales,
                        scale_token_stride, scale_hidden_stride,
                        buffer_ptrs_gpu, rank, num_ranks, comm_stream, config.num_sms,
                        config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix, recv_x, recv_src_idx, recv_channel_prefix_matrix, send_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {x_scales, topk_idx, topk_weights, num_tokens_per_rank, num_tokens_per_expert, cached_channel_prefix_matrix, cached_rank_prefix_matrix, recv_topk_idx, recv_topk_weights, recv_x_scales}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    // Return values
    return {recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::intranode_combine(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                          const std::optional<torch::Tensor>& bias_0, const std::optional<torch::Tensor>& bias_1,
                          const torch::Tensor& src_idx, const torch::Tensor& rank_prefix_matrix, const torch::Tensor& channel_prefix_matrix,
                          const torch::Tensor& send_head, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and rank_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and channel_prefix_matrix.scalar_type() == torch::kInt32);

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));
    EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    int num_topk = 0;
    auto recv_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }

    // Launch barrier and reset queue head and tail
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
    intranode::cached_notify_combine(buffer_ptrs_gpu, send_head.data_ptr<int>(),
                                     num_channels, num_recv_tokens, num_channels * num_ranks * 2,
                                     barrier_signal_ptrs_gpu, rank, num_ranks,
                                     comm_stream);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++ i) if (bias_opts[i].has_value()) {
        auto bias = bias_opts[i].value();
        EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
        EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
        EP_HOST_ASSERT(bias.size(0) == num_recv_tokens and bias.size(1) == hidden);
        bias_ptrs[i] = bias.data_ptr();
    }

    // Combine data
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 +                                                      // Queue head and tail
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * x.element_size() +    // Data buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +                  // Source index buffer
                   num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float)       // Top-k weight buffer
                   <= num_nvl_bytes);
    intranode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                       recv_x.data_ptr(), recv_topk_weights_ptr,
                       x.data_ptr(), topk_weights_ptr, bias_ptrs[0], bias_ptrs[1],
                       src_idx.data_ptr<int>(), rank_prefix_matrix.data_ptr<int>(), channel_prefix_matrix.data_ptr<int>(),
                       send_head.data_ptr<int>(), num_tokens, num_recv_tokens, hidden, num_topk,
                       buffer_ptrs_gpu, rank, num_ranks,
                       comm_stream, config.num_sms,
                       config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {topk_weights, recv_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    return {recv_x, recv_topk_weights, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::internode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank, const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                           const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
                           const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                           const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                           int expert_alignment, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
    // If users of DeepEP need to execute other Python code on other threads, such as KV transfer, their code will get stuck due to GIL
    // unless we release GIL here.
    pybind11::gil_scoped_release release;
    HOST_LOG_DEBUG("internode_dispatch");

    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and cached_rdma_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and cached_rdma_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and cached_gbl_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)), hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    int64_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<int64_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
    auto rdma_channel_prefix_matrix = torch::Tensor();
    auto recv_rdma_rank_prefix_sum = torch::Tensor();
    auto gbl_channel_prefix_matrix = torch::Tensor();
    auto recv_gbl_rank_prefix_sum = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
        rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
        recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
        gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
        recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

        // Just a barrier and clean flags
        internode::cached_notify(hidden_int4, num_scales, num_topk, num_topk,
                                 num_ranks, num_channels, 0, nullptr,
                                 nullptr, nullptr, nullptr,
                                 rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                                 buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
                                 barrier_signal_ptrs_gpu, rank, comm_stream,
                                 config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                 num_nvl_bytes, true, low_latency_mode, nixl_kernel_params->gpu_ctx);
    } else {
        rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

        // Send sizes
        *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++ i)
            moe_recv_expert_counter[i] = -1;
        internode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
                                   num_tokens_per_rdma_rank->data_ptr<int>(), moe_recv_rdma_counter_mapped,
                                   num_tokens_per_expert->data_ptr<int>(), moe_recv_expert_counter_mapped, num_experts,
                                   is_token_in_rank.data_ptr<bool>(), num_tokens, num_channels,
                                   hidden_int4, num_scales, num_topk, expert_alignment,
                                   rdma_channel_prefix_matrix.data_ptr<int>(), recv_rdma_rank_prefix_sum.data_ptr<int>(),
                                   gbl_channel_prefix_matrix.data_ptr<int>(), recv_gbl_rank_prefix_sum.data_ptr<int>(),
                                   rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                                   buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
                                   barrier_signal_ptrs_gpu, rank, comm_stream,
                                   config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                   num_nvl_bytes, low_latency_mode, nixl_kernel_params->gpu_ctx);

        // Synchronize total received tokens and tokens per expert
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            // Read total count
            num_recv_tokens = static_cast<int>(*moe_recv_counter);
            num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

            // Read per-expert count
            bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
            for (int i = 0; i < num_local_experts and ready; ++ i)
                ready &= moe_recv_expert_counter[i] >= 0;

            if (ready)
                break;

            // Timeout check
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() > NUM_CPU_TIMEOUT_SECS) {
                // printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank, num_recv_tokens, num_rdma_recv_tokens);
                for (int i = 0; i < num_local_experts; ++ i)
                    printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
            }
        }
        num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(), recv_x_scales = std::optional<torch::Tensor>();
    auto recv_src_meta = std::optional<torch::Tensor>();
    auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto send_rdma_head = std::optional<torch::Tensor>();
    auto send_nvl_head = std::optional<torch::Tensor>();
    if (not cached_mode) {
        recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kCUDA));
        recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(torch::kCUDA));
    }

    // Assign pointers
    int64_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<int64_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ?
                        torch::empty({num_recv_tokens}, x_scales->options()) :
                        torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Launch data dispatch
    // NOTES: the buffer size checks are moved into the `.cu` file
    internode::dispatch(recv_x.data_ptr(), recv_x_scales_ptr, recv_topk_idx_ptr, recv_topk_weights_ptr,
                        cached_mode ? nullptr : recv_src_meta->data_ptr(),
                        x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
                        cached_mode ? nullptr : send_rdma_head->data_ptr<int>(), cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
                        cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
                        cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
                        rdma_channel_prefix_matrix.data_ptr<int>(), recv_rdma_rank_prefix_sum.data_ptr<int>(),
                        gbl_channel_prefix_matrix.data_ptr<int>(), recv_gbl_rank_prefix_sum.data_ptr<int>(),
                        is_token_in_rank.data_ptr<bool>(),
                        num_tokens, hidden_int4, num_scales, num_topk, num_experts,
                        scale_token_stride, scale_hidden_stride,
                        rdma_buffer_ptr, config.num_max_rdma_chunked_send_tokens, config.num_max_rdma_chunked_recv_tokens,
                        buffer_ptrs_gpu, config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens,
                        rank, num_ranks, cached_mode,
                        comm_stream, num_channels, low_latency_mode, nixl_kernel_params->gpu_ctx);

    HOST_LOG_DEBUG("internode_dispatch finished");

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, is_token_in_rank, recv_x,
                       rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {x_scales, topk_idx, topk_weights,
                        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert,
                        cached_rdma_channel_prefix_matrix, cached_recv_rdma_rank_prefix_sum,
                        cached_gbl_channel_prefix_matrix, cached_recv_gbl_rank_prefix_sum,
                        recv_topk_idx, recv_topk_weights, recv_x_scales,
                        recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix, send_rdma_head, send_nvl_head,
                        recv_src_meta}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    // Return values
    return {recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
            recv_src_meta, send_rdma_head, send_nvl_head, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::internode_combine(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                          const std::optional<torch::Tensor>& bias_0, const std::optional<torch::Tensor>& bias_1,
                          const torch::Tensor& src_meta, const torch::Tensor& is_combined_token_in_rank,
                          const torch::Tensor& rdma_channel_prefix_matrix, const torch::Tensor& rdma_rank_prefix_sum, const torch::Tensor& gbl_channel_prefix_matrix,
                          const torch::Tensor& combined_rdma_head, const torch::Tensor& combined_nvl_head,
                          const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    const int num_channels = config.num_sms / 2;
    HOST_LOG_DEBUG("internode_combine | num_channels: %d", num_channels);
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);
    EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and is_combined_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and combined_rdma_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and combined_nvl_head.scalar_type() == torch::kInt32);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)), hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());
    EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and rdma_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and combined_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Top-k checks
    int num_topk = 0;
    auto combined_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
        combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
    }

    // Extra check for avoid-dead-lock design
    EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

    // Launch barrier and reset queue head and tail
    internode::cached_notify(hidden_int4, 0, 0, num_topk,
                             num_ranks, num_channels,
                             num_combined_tokens, combined_rdma_head.data_ptr<int>(),
                             rdma_channel_prefix_matrix.data_ptr<int>(), rdma_rank_prefix_sum.data_ptr<int>(), combined_nvl_head.data_ptr<int>(),
                             rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                             buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
                             barrier_signal_ptrs_gpu, rank, comm_stream,
                             config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                             num_nvl_bytes, false, low_latency_mode, nixl_kernel_params->gpu_ctx);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++ i) if (bias_opts[i].has_value()) {
        auto bias = bias_opts[i].value();
        EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
        EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
        EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
        bias_ptrs[i] = bias.data_ptr();
    }

    // Launch data combine
    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    internode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                       combined_x.data_ptr(), combined_topk_weights_ptr,
                       is_combined_token_in_rank.data_ptr<bool>(),
                       x.data_ptr(), topk_weights_ptr, bias_ptrs[0], bias_ptrs[1],
                       combined_rdma_head.data_ptr<int>(), combined_nvl_head.data_ptr<int>(),
                       src_meta.data_ptr(), rdma_channel_prefix_matrix.data_ptr<int>(), rdma_rank_prefix_sum.data_ptr<int>(), gbl_channel_prefix_matrix.data_ptr<int>(),
                       num_tokens, num_combined_tokens, hidden, num_topk,
                       rdma_buffer_ptr, config.num_max_rdma_chunked_send_tokens, config.num_max_rdma_chunked_recv_tokens,
                       buffer_ptrs_gpu, config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens,
                       rank, num_ranks, comm_stream, num_channels, low_latency_mode, nixl_kernel_params->gpu_ctx);

    HOST_LOG_DEBUG("internode_combine finished");

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, src_meta,
                       is_combined_token_in_rank, rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix,
                       combined_x, combined_rdma_head, combined_nvl_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {topk_weights, combined_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    // Return values
    return {combined_x, combined_topk_weights, event};
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
    EP_HOST_ASSERT(low_latency_mode);

    auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    auto clean_meta_0 = layout.buffers[0].clean_meta();
    auto clean_meta_1 = layout.buffers[1].clean_meta();

    auto check_boundary = [=](void* ptr, size_t num_bytes) {
        auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
        EP_HOST_ASSERT(0 <= offset and offset + num_bytes <= num_rdma_bytes);
    };
    check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
    check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

    internode_ll::clean_low_latency_buffer(clean_meta_0.first, clean_meta_0.second,
                                           clean_meta_1.first, clean_meta_1.second,
                                           at::cuda::getCurrentCUDAStream());
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx,
                             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                             const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                             int num_max_dispatch_tokens_per_rank, int num_experts,
                             bool use_fp8, bool round_scale, bool use_ue8m0,
                             bool async, bool return_recv_hook) {
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    // By default using `ptp128c` FP8 cast
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);

    // Diagnosis tensors
    if (cumulative_local_expert_recv_stats.has_value()) {
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts / num_ranks);
    }
    if (dispatch_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_topk = static_cast<int>(topk_idx.size(1));
    int num_local_experts = num_experts / (remote_ranks.size() + 1);

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    internode_ll::gpu_nixl_ctx nixl_ctx = nllc.nixl_ctx[low_latency_buffer_idx];
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not (async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate packed tensors
    auto packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                      x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn: torch::kBFloat16));
    auto packed_recv_src_info = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto packed_recv_count = torch::empty({num_local_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // Allocate column-majored scales
    auto packed_recv_x_scales = std::optional<torch::Tensor>();
    void* packed_recv_x_scales_ptr = nullptr;
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

    if (use_fp8) {
        // TODO: support unaligned cases
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (not use_ue8m0) {
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 128, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
        } else {
            EP_HOST_ASSERT(round_scale);
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 512, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kInt).device(torch::kCUDA));
        }
        packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::dispatch(packed_recv_x.data_ptr(), packed_recv_x_scales_ptr,
                               packed_recv_src_info.data_ptr<int>(), packed_recv_layout_range.data_ptr<int64_t>(),
                               packed_recv_count.data_ptr<int>(),
                               cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
                               dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                               buffer.dispatch_rdma_recv_data_buffer, buffer.dispatch_rdma_recv_count_buffer,
                               buffer.dispatch_rdma_send_buffer,
                               x.data_ptr(), topk_idx.data_ptr<int64_t>(),
                               next_clean_meta.first, next_clean_meta.second,
                               num_tokens, hidden, num_max_dispatch_tokens_per_rank,
                               num_topk, num_experts, rank, num_ranks,
                               use_fp8, round_scale, use_ue8m0,
                               workspace, num_device_sms,
                               launch_stream, phases, nixl_ctx);
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::low_latency_combine(const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                            const torch::Tensor& src_info, const torch::Tensor& layout_range,
                            const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
                            int num_max_dispatch_tokens_per_rank, int num_experts,
                            bool use_logfmt, bool zero_copy, bool async, bool return_recv_hook,
                            const std::optional<torch::Tensor>& out) {
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
    EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and layout_range.size(1) == num_ranks);

    if (combine_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and combine_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto hidden = static_cast<int>(x.size(2));
    auto num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    internode_ll::gpu_nixl_ctx nixl_ctx = nllc.nixl_ctx[low_latency_buffer_idx];
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not (async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate output tensor
    torch::Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        combined_x = out.value();
    } else {
        combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::combine(combined_x.data_ptr(),
                              buffer.combine_rdma_recv_data_buffer, buffer.combine_rdma_recv_flag_buffer,
                              buffer.combine_rdma_send_buffer,
                              x.data_ptr(), topk_idx.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
                              src_info.data_ptr<int>(), layout_range.data_ptr<int64_t>(),
                              combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                              next_clean_meta.first, next_clean_meta.second,
                              num_combined_tokens, hidden, num_max_dispatch_tokens_per_rank,
                              num_topk, num_experts, rank, num_ranks,
                              use_logfmt,
                              workspace, num_device_sms,
                              launch_stream, phases, zero_copy, nixl_ctx);
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {combined_x, event, recv_hook};
}

torch::Tensor
Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);

    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                            {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                            {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                            torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
}

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

void Buffer::_nixl_ll_gpu_ctx_prepare() {
    int num_local_experts = nixl_agent_infos.size();

    /* Initialize local counter arrays */
    nllc.nixl_ctx[0].local_counters = counters_buffer_ptr;
    nllc.nixl_ctx[1].local_counters = counters_buffer_ptr + max_num_ranks * num_local_experts;

    /* Each context cleans the counters of the other context */
    nllc.nixl_ctx[0].clean_counters = nllc.nixl_ctx[1].local_counters;
    nllc.nixl_ctx[1].clean_counters = nllc.nixl_ctx[0].local_counters;

    /* Copy remote counter reqs to device */
    if (nllc.nixl_ctx[0].remote_counter_reqs == nullptr) {
        CUDA_CHECK(cudaMalloc(&nllc.nixl_ctx[0].remote_counter_reqs, num_local_experts * max_num_ranks * sizeof(nixlGpuXferReqH)));
        CUDA_CHECK(cudaMalloc(&nllc.nixl_ctx[1].remote_counter_reqs, num_local_experts * max_num_ranks * sizeof(nixlGpuXferReqH)));
    }

    // Always copy the updated arrays to GPU (since new ranks may have been added)
    CUDA_CHECK(cudaMemcpy(nllc.nixl_ctx[0].remote_counter_reqs, nllc.gpu_remote_counter_reqs_0.data(), num_local_experts * max_num_ranks * sizeof(nixlGpuXferReqH), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nllc.nixl_ctx[1].remote_counter_reqs, nllc.gpu_remote_counter_reqs_1.data(), num_local_experts * max_num_ranks * sizeof(nixlGpuXferReqH), cudaMemcpyHostToDevice));

    /* Copy batch reqs to device */
    if (nllc.nixl_ctx[0].batch_reqs == nullptr)
        CUDA_CHECK(cudaMalloc(&nllc.nixl_ctx[0].batch_reqs, num_local_experts * max_num_ranks * sizeof(nixlGpuXferReqH)));

    // Always copy the updated batch arrays to GPU (since new ranks may have been added)
    for (int dest_expert_idx = 0; dest_expert_idx < num_local_experts; dest_expert_idx++)
        CUDA_CHECK(cudaMemcpy(nllc.nixl_ctx[0].batch_reqs + dest_expert_idx * max_num_ranks, nixl_agent_infos[dest_expert_idx].gpu_batch_reqs.data(), max_num_ranks * sizeof(nixlGpuXferReqH), cudaMemcpyHostToDevice));
    nllc.nixl_ctx[1].batch_reqs = nllc.nixl_ctx[0].batch_reqs; // Both contexts share the same batch handles, no need to duplicate them

    /* Initialize counters P2P pointers */
    if (nllc.nixl_ctx[0].counters_p2p_ptrs == nullptr)
        CUDA_CHECK(cudaMalloc(&nllc.nixl_ctx[0].counters_p2p_ptrs, max_num_ranks * sizeof(uint64_t *)));

    if (nllc.nixl_ctx[1].counters_p2p_ptrs == nullptr)
        CUDA_CHECK(cudaMalloc(&nllc.nixl_ctx[1].counters_p2p_ptrs, max_num_ranks * sizeof(uint64_t *)));

    CUDA_CHECK(cudaMemcpy(nllc.nixl_ctx[0].counters_p2p_ptrs, nllc.counters_p2p_ptrs.data(), num_ranks * sizeof(uint64_t *), cudaMemcpyHostToDevice));
    for (int i = 0; i < num_ranks; i++) if (nllc.counters_p2p_ptrs[i] != 0) nllc.counters_p2p_ptrs[i] += max_num_ranks * num_local_experts;
    CUDA_CHECK(cudaMemcpy(nllc.nixl_ctx[1].counters_p2p_ptrs, nllc.counters_p2p_ptrs.data(), num_ranks * sizeof(uint64_t *), cudaMemcpyHostToDevice));
    for (int i = 0; i < num_ranks; i++) if (nllc.counters_p2p_ptrs[i] != 0) nllc.counters_p2p_ptrs[i] -= max_num_ranks * num_local_experts;

    /* Initialize RDMA P2P pointers */
    if (nllc.nixl_ctx[0].rdma_p2p_ptrs == nullptr)
        CUDA_CHECK(cudaMalloc(&nllc.nixl_ctx[0].rdma_p2p_ptrs, max_num_ranks * sizeof(void *)));

    if (nllc.nixl_ctx[1].rdma_p2p_ptrs == nullptr)
        CUDA_CHECK(cudaMalloc(&nllc.nixl_ctx[1].rdma_p2p_ptrs, max_num_ranks * sizeof(void *)));

    CUDA_CHECK(cudaMemcpy(nllc.nixl_ctx[0].rdma_p2p_ptrs, nllc.rdma_p2p_ptrs.data(), num_ranks * sizeof(void *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nllc.nixl_ctx[1].rdma_p2p_ptrs, nllc.rdma_p2p_ptrs.data(), num_ranks * sizeof(void *), cudaMemcpyHostToDevice));

    /* Initialize sync counters */
    nllc.nixl_ctx[0].local_sync_counters = counters_buffer_ptr + 2 * max_num_ranks * num_local_experts;

    if (nllc.nixl_ctx[0].remote_sync_counters == nullptr)
        CUDA_CHECK(cudaMalloc(&nllc.nixl_ctx[0].remote_sync_counters, max_num_ranks * sizeof(nixlGpuXferReqH)));

    CUDA_CHECK(cudaMemcpy(nllc.nixl_ctx[0].remote_sync_counters, nllc.gpu_sync_counters.data(), max_num_ranks * sizeof(nixlGpuXferReqH), cudaMemcpyHostToDevice));

    /* Initialize info fields */
    nllc.nixl_ctx[0].rdma_buffer_ptr = rdma_buffer_ptr;
    nllc.nixl_ctx[1].rdma_buffer_ptr = rdma_buffer_ptr;
    nllc.nixl_ctx[0].num_local_experts = num_local_experts;
    nllc.nixl_ctx[1].num_local_experts = num_local_experts;
    nllc.nixl_ctx[0].num_ranks = max_num_ranks;
    nllc.nixl_ctx[1].num_ranks = max_num_ranks;
    nllc.nixl_ctx[0].rank = rank;
    nllc.nixl_ctx[1].rank = rank;
}

void Buffer::_nixl_internode_init() {
    nixl_kernel_params = new nixl_internode_ctx(env_num_channels, num_rdma_ranks, rank);
    _nixl_internode_local_data_init();
    _nixl_remote_counters_prepare();
    _nixl_internode_batches_prepare();
    nixl_kernel_params->copy_to_gpu();
}

void Buffer::_nixl_ll_nllc_init() {
    int num_local_experts = nixl_agent_infos.size();

    nllc.cpu_remote_counter_reqs_0.resize(num_local_experts * max_num_ranks);
    nllc.cpu_remote_counter_reqs_1.resize(num_local_experts * max_num_ranks);
    nllc.gpu_remote_counter_reqs_0.resize(num_local_experts * max_num_ranks);
    nllc.gpu_remote_counter_reqs_1.resize(num_local_experts * max_num_ranks);
    nllc.cpu_sync_counters.resize(max_num_ranks);
    nllc.gpu_sync_counters.resize(max_num_ranks);
    nllc.rdma_p2p_ptrs.resize(max_num_ranks);
    nllc.counters_p2p_ptrs.resize(max_num_ranks);
}

void Buffer::_nixl_ll_init(const std::vector<int>& ranks_to_setup) {
    EP_EXECUTE_ONCE(_nixl_ll_nllc_init());
    _nixl_ll_counters_prepare(ranks_to_setup);
    _nixl_ll_batches_prepare(ranks_to_setup);
    _nixl_ll_p2p_ptrs_prepare(ranks_to_setup);
    _nixl_ll_gpu_ctx_prepare();
}

void Buffer::_nixl_kernels_params_free() {
     //TODO: the subsequent xfer releases are commented out because of nixl/ucx bug on the second xfer release, needs to be uncommented when the bug is fixed

    // for (size_t i = 0; i < nixl_agent_infos.size(); i++) {
            // for (size_t j = 0; j < num_rdma_ranks; j++) {
            //     if (j == rdma_rank || nixl_agent_infos[i].cpu_counter_reqs[j] == nullptr) continue;
            //     printf("[DEEPEP-NIXL] RANK %2d | RELEASE_COUNTER_XFER_REQ: agent='%s', rank=%d, req=%p\n", rank, nixl_agent_infos[i].agent_name.c_str(), j, nixl_agent_infos[i].cpu_counter_reqs[j]);
            //     EP_HOST_ASSERT(nixl_agent_infos[i].agent->releaseXferReqtoGPU(nixl_agent_infos[i].cpu_counter_reqs[j]) == NIXL_SUCCESS and "Failed to destroy xfer req");
            //     EP_HOST_ASSERT(nixl_agent_infos[i].agent->releaseXferReq(nixl_agent_infos[i].cpu_counter_reqs[j]) == NIXL_SUCCESS and "Failed to destroy xfer req");

            // if (j == rdma_rank || nixl_agent_infos[i].cpu_batch_reqs[j] == nullptr) continue;
            // printf("[DEEPEP-NIXL] RANK %2d | RELEASE_BATCH_XFER_REQ: agent='%s', rank=%d, req=%p\n", rank, nixl_agent_infos[i].agent_name.c_str(), j, nixl_agent_infos[i].cpu_batch_reqs[j]);
            // EP_HOST_ASSERT(nixl_agent_infos[i].agent->releaseXferReqtoGPU(nixl_agent_infos[i].cpu_batch_reqs[j]) == NIXL_SUCCESS and "Failed to destroy xfer req");
            // EP_HOST_ASSERT(nixl_agent_infos[i].agent->releaseXferReq(nixl_agent_infos[i].cpu_batch_reqs[j]) == NIXL_SUCCESS and "Failed to destroy xfer req");
        // }
     
        // for (size_t j = 0; j < num_rdma_ranks; j++) {
            // if (j == rdma_rank || nixl_agent_infos[i].cpu_barrier_reqs[j] == nullptr) continue;
            // printf("[DEEPEP-NIXL] RANK %2d | RELEASE_BARRIER_XFER_REQ: agent='%s', rank=%d, req=%p\n", rank, nixl_agent_infos[i].agent_name.c_str(), j, nixl_agent_infos[i].cpu_barrier_reqs[j]);
            // EP_HOST_ASSERT(nixl_agent_infos[i].agent->releaseXferReqtoGPU(nixl_agent_infos[i].cpu_barrier_reqs[j]) == NIXL_SUCCESS and "Failed to destroy xfer req");
            // EP_HOST_ASSERT(nixl_agent_infos[i].agent->releaseXferReq(nixl_agent_infos[i].cpu_barrier_reqs[j]) == NIXL_SUCCESS and "Failed to destroy xfer req");
    //     }
    // }
    nixl_agent_infos.clear();
    if (nixl_kernel_params == nullptr) return;
    delete nixl_kernel_params;
}

void Buffer::_nixl_agents_init() {
    nixl_agent_infos.clear();
    nixl_agent_infos.reserve(env_num_channels);

    for (int channel = 0; channel < env_num_channels; ++channel) {
        // Create agent with unique name
        std::string agent_name = std::to_string(rank) + "_ch" + std::to_string(channel);
        nixlAgentConfig cfg(true, false, 0,
                           nixl_thread_sync_t::NIXL_THREAD_SYNC_RW, 1, 0, 100000, false, NIXL_ETCD_WATCH_TIMEOUT);
        auto agent = std::make_shared<nixlAgent>(agent_name, cfg);

        // Create UCX backend
        nixl_mem_list_t mems;
        nixl_b_params_t init_params;

        nixl_status_t status = agent->getPluginParams("UCX", mems, init_params);
        if (status != NIXL_SUCCESS) {
            throw std::runtime_error("Failed to get UCX plugin parameters for agent " + agent_name +
                                   ", status: " + std::to_string(status));
        }

        // Set UCX-specific parameters
        init_params["ucx_error_handling_mode"] = "none";

        nixlBackendH* ucx_backend = nullptr;
        status = agent->createBackend("UCX", init_params, ucx_backend);
        if (status != NIXL_SUCCESS || !ucx_backend) {
            throw std::runtime_error("Failed to create UCX backend for agent " + agent_name +
                                   ", status: " + std::to_string(status));
        }

        nixl_agent_infos.emplace_back(agent, agent_name, ucx_backend, low_latency_mode ? max_num_ranks : num_rdma_ranks, max_num_ranks);
        nixl_agent_infos[channel].extra_params.backends.push_back(ucx_backend);

        /* Register RDMA buffer */
        nixl_reg_dlist_t rdma_ptr_dlist(VRAM_SEG);
        rdma_ptr_dlist.addDesc(nixlBlobDesc((uintptr_t)(rdma_buffer_ptr), num_rdma_bytes, get_local_device_id(), ""));
        EP_HOST_ASSERT(agent->registerMem(rdma_ptr_dlist) == NIXL_SUCCESS);
        nixl_agent_infos[channel].src_vram = rdma_ptr_dlist.trim();

        /* Register counters buffer */
        nixl_reg_dlist_t counters_dlist(VRAM_SEG);
        counters_dlist.addDesc(nixlBlobDesc((uintptr_t)(counters_buffer_ptr), num_counters * sizeof(uint64_t), get_local_device_id(), ""));
        EP_HOST_ASSERT(agent->registerMem(counters_dlist) == NIXL_SUCCESS);

        if (channel == 0) {
            // Initialize GPU signals only on agent 0
            // For now, assumes UCX signal size is 8 bytes
            size_t signal_size = 0;
            EP_HOST_ASSERT(nixl_agent_infos[0].agent->getGpuSignalSize(signal_size, &nixl_agent_infos[0].extra_params) == NIXL_SUCCESS);
            EP_HOST_ASSERT(signal_size == sizeof(uint64_t));
            EP_HOST_ASSERT(agent->prepGpuSignal(counters_dlist, &nixl_agent_infos[0].extra_params) == NIXL_SUCCESS);
        }

        // Send local metadata
        status = nixl_agent_infos[channel].agent->sendLocalMD();
        if (status != NIXL_SUCCESS) {
            throw std::runtime_error("Failed to send local metadata for agent " +
                                   nixl_agent_infos[channel].agent_name + ", status: " + std::to_string(status));
        }
    }
}

/* TODO_Roey: review and delete this function */
void Buffer::_nixl_internode_local_data_init() {
    for (int i = 0; i < env_num_channels; ++ i) {
        uint64_t head_counter_elem_offset = 2 * i * num_rdma_ranks;
        uint64_t tail_counter_elem_offset = (2 * i + 1) * num_rdma_ranks;
        // copying the pointer to the relevant counter for each channel
        uint64_t* head_counter_ptr = &counters_buffer_ptr[head_counter_elem_offset];
        uint64_t* tail_counter_ptr = &counters_buffer_ptr[tail_counter_elem_offset];

        HOST_LOG_DEBUG("channel %d, local head_counter_ptr: %p, local tail_counter_ptr: %p", i, (void*)head_counter_ptr, (void*)tail_counter_ptr);

        nixl_kernel_params->cpu_channel_ctxs[i].local_head_counters = head_counter_ptr;
        nixl_kernel_params->cpu_channel_ctxs[i].local_tail_counters = tail_counter_ptr;
        nixl_kernel_params->cpu_channel_ctxs[i].last_barrier_counter = last_barrier_counter;
        nixl_kernel_params->cpu_channel_ctxs[i].local_barrier_counter_ptr = local_barrier_counter;
    }
}

void Buffer::_nixl_internode_batches_prepare() {
    for (int i = 0; i < env_num_channels; ++i) {
        // TODO: Remove once NIXL supports null src dlist for signals
        nixl_agent_infos[i].src_vram.addDesc(nixlBlobDesc((uintptr_t)counters_buffer_ptr, sizeof(uint64_t), device_id, ""));
        for (int j = 0; j < num_rdma_ranks; ++j) {
            if (j == rdma_rank) continue;
            int remote_rank = nvl_rank + j * NUM_MAX_NVL_PEERS;
            nixl_agent_infos[i].dst_vram[j].addDesc(nixlBlobDesc((uintptr_t)(nixl_peer_info[remote_rank].rdma_buffer_ptr), num_rdma_bytes, nixl_peer_info[remote_rank].device_id, ""));
            // setting the signal address to the tail counter of the remote rank
            uint64_t *counter_addr = &nixl_peer_info[remote_rank].counters_buffer_ptr[(2 * i + 1) * num_rdma_ranks + rdma_rank];
            HOST_LOG_DEBUG("channel %d, remote_rank %d, remote tail counter signal addr: %p", i, j, (void *)counter_addr);
            nixl_agent_infos[i].dst_vram[j].addDesc(nixlBlobDesc((uintptr_t)(counter_addr), sizeof(uint64_t), nixl_peer_info[remote_rank].device_id, ""));
            nixl_opt_args_t extra_params;
            extra_params.backends.push_back(nixl_agent_infos[i].extra_params.backends[0]);
            nixlXferReqH* xfer_req;
            EP_HOST_ASSERT(nixl_agent_infos[i].agent->createXferReq(NIXL_WRITE, nixl_agent_infos[i].src_vram,
                    nixl_agent_infos[i].dst_vram[j], nixl_agent_infos[i].dst_agent_names[remote_rank],
                    xfer_req, &extra_params) == NIXL_SUCCESS and "Failed to create xfer req");
            nixl_agent_infos[i].cpu_batch_reqs[j] = xfer_req;
            nixlGpuXferReqH gpu_xfer_req;
            EP_HOST_ASSERT(nixl_agent_infos[i].agent->createGpuXferReq(*xfer_req, gpu_xfer_req) == NIXL_SUCCESS);
            CUDA_CHECK(cudaMemcpy(&nixl_kernel_params->cpu_channel_ctxs[i].data_request_handles[j], &gpu_xfer_req, sizeof(nixlGpuXferReqH), cudaMemcpyHostToDevice));
        }
    }
}

void Buffer::_nixl_ll_batches_prepare(const std::vector<int>& ranks_to_setup) {
    nixl_status_t status;

    for (int i = 0; i < env_num_channels; ++i) {
        for (int j : ranks_to_setup) {
            if (j == rank) continue; // Skip self
            if (nixl_agent_infos[i].gpu_batch_reqs[j]) continue; // Skip if already exported
            nixl_xfer_dlist_t src_vram(VRAM_SEG);
            src_vram.addDesc(nixlBlobDesc((uintptr_t)(rdma_buffer_ptr), num_rdma_bytes, get_local_device_id(), ""));
            nixl_agent_infos[i].dst_vram[j].addDesc(nixlBlobDesc((uintptr_t)(nixl_peer_info[j].rdma_buffer_ptr), num_rdma_bytes, nixl_peer_info[j].device_id, ""));
            nixl_opt_args_t extra_params = {};
            extra_params.backends.push_back(nixl_agent_infos[i].backend);
            status = nixl_agent_infos[i].agent->createXferReq(NIXL_WRITE, src_vram, nixl_agent_infos[i].dst_vram[j], nixl_agent_infos[i].dst_agent_names[j], nixl_agent_infos[i].cpu_batch_reqs[j], &extra_params);
            EP_HOST_ASSERT(status == NIXL_SUCCESS);
            EP_HOST_ASSERT(nixl_agent_infos[i].agent->createGpuXferReq(*nixl_agent_infos[i].cpu_batch_reqs[j], nixl_agent_infos[i].gpu_batch_reqs[j]) == NIXL_SUCCESS);
        }
    }
}

void Buffer::_nixl_ll_p2p_ptrs_prepare(const std::vector<int>& ranks_to_setup) {
    for (int i : ranks_to_setup) {
        if (i == rank) {
            nllc.rdma_p2p_ptrs[i] = rdma_buffer_ptr;
            nllc.counters_p2p_ptrs[i] = counters_buffer_ptr;
        } else if (std::string(nixl_peer_info[i].boot_id) == std::string(my_peer_info.boot_id) &&
                   nixl_peer_info[i].ipc_namespace_inode == my_peer_info.ipc_namespace_inode &&
                   std::string(std::getenv("DEEPEP_LL_NVLINK_IPC")) == "1") {
            CUDA_CHECK(cudaIpcOpenMemHandle((void **)&nllc.rdma_p2p_ptrs[i], nixl_peer_info[i].rdma_ipc_handle, cudaIpcMemLazyEnablePeerAccess));
            CUDA_CHECK(cudaIpcOpenMemHandle((void **)&nllc.counters_p2p_ptrs[i], nixl_peer_info[i].counters_ipc_handle, cudaIpcMemLazyEnablePeerAccess));
        } else {
            nllc.rdma_p2p_ptrs[i] = nullptr;
            nllc.counters_p2p_ptrs[i] = nullptr;
        }
    }
}

void Buffer::_nixl_remote_counters_prepare() {
    uint64_t* remote_head_counter_ptr;
    nixlXferReqH* xfer_req;
    nixlGpuXferReqH gpu_xfer_req;

    for (int i = 0; i < env_num_channels; ++ i) {
        for (int j = 0; j < num_rdma_ranks; ++ j) {
            if (j == rdma_rank) continue;

            int remote_rank = nvl_rank + j * NUM_MAX_NVL_PEERS;
            int remote_head_elem_offset = 2 * i * num_rdma_ranks + rdma_rank;
            remote_head_counter_ptr = nixl_peer_info[remote_rank].counters_buffer_ptr + remote_head_elem_offset;
            HOST_LOG_DEBUG("channel %d, remote_rank %d, remote head_counter_ptr: %p", i, j, (void*)remote_head_counter_ptr);

            nixl_opt_args_t eparams = { .backends = {nixl_agent_infos[i].extra_params.backends[0]}};
            nixl_xfer_dlist_t dst_dlist(VRAM_SEG);
            dst_dlist.addDesc(nixlBlobDesc((uintptr_t)remote_head_counter_ptr, sizeof(uint64_t), nixl_peer_info[remote_rank].device_id, ""));
            EP_HOST_ASSERT(nixl_agent_infos[i].agent->createXferReq(NIXL_WRITE, dummy_src_dlist, dst_dlist, nixl_agent_infos[i].dst_agent_names[remote_rank], xfer_req, &eparams) == NIXL_SUCCESS and "Failed to create signal xfer req");
            nixl_agent_infos[i].cpu_counter_reqs[j] = xfer_req;
            EP_HOST_ASSERT(nixl_agent_infos[i].agent->createGpuXferReq(*xfer_req, gpu_xfer_req) == NIXL_SUCCESS);
            HOST_LOG_DEBUG("channel %d, remote_rank %d, cpu counter xfer_req: %p, gpu_xfer_req: %p, signal_addr: %p", i, remote_rank, xfer_req, &gpu_xfer_req, remote_head_counter_ptr);
            CUDA_CHECK(cudaMemcpy(&nixl_kernel_params->cpu_channel_ctxs[i].remote_head_counter_handles[j], &gpu_xfer_req, sizeof(nixlGpuXferReqH), cudaMemcpyHostToDevice));

            // Initialize internode barrier
            EP_HOST_ASSERT(num_rdma_ranks <= 2 && "Internode barrier is not supported for more than 2 rdma ranks"); // TODO_Roey: fix internode's nixl barrier impl for more than 2 rdma ranks
            nixlXferReqH* barrier_xfer_req_cpu = nullptr;
            nixlGpuXferReqH barrier_xfer_req_gpu;
            nixl_xfer_dlist_t barrier_dst_dlist(VRAM_SEG);
            barrier_dst_dlist.addDesc(nixlBlobDesc((uintptr_t)nixl_peer_info[remote_rank].barrier_ptr, sizeof(uint64_t), nixl_peer_info[remote_rank].device_id, ""));
            EP_HOST_ASSERT(nixl_agent_infos[i].agent->createXferReq(NIXL_WRITE, dummy_src_dlist, barrier_dst_dlist, nixl_agent_infos[i].dst_agent_names[remote_rank], barrier_xfer_req_cpu, &eparams) == NIXL_SUCCESS);
            nixl_agent_infos[i].cpu_barrier_reqs[j] = barrier_xfer_req_cpu;
            HOST_LOG_DEBUG("channel %d, remote_rank %d, cpu barrier xfer_req: %p, nixl_agent_infos[i].cpu_barrier_reqs[j]: %p", i, j, barrier_xfer_req_cpu, nixl_agent_infos[i].cpu_barrier_reqs[j]);

            EP_HOST_ASSERT(nixl_agent_infos[i].agent->createGpuXferReq(*barrier_xfer_req_cpu, barrier_xfer_req_gpu) == NIXL_SUCCESS);
            CUDA_CHECK(cudaMemcpy(&nixl_kernel_params->cpu_channel_ctxs[i].remote_barrier_handles[j], &barrier_xfer_req_gpu, sizeof(nixlGpuXferReqH), cudaMemcpyHostToDevice));
        };
    }
}

void Buffer::_nixl_ll_counters_prepare(const std::vector<int>& ranks_to_setup) {
    int num_local_experts = nixl_agent_infos.size();

    for (int expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
        for (int remote_rank : ranks_to_setup) {
            if (remote_rank == rank)
                continue;

            NixlAgentInfo &agent_info = nixl_agent_infos[expert_idx];
            int remote_counter_idx = expert_idx * max_num_ranks + rank; // remote rank's counters array is indexed by [local_expert_idx, src_rank]
            int local_counter_idx = expert_idx * max_num_ranks + remote_rank; // remote_counter_reqs is indexed by [local_expert_idx, dst_rank]

            if (nixl_peer_info[remote_rank].counters_buffer_ptr == nullptr) {
                printf("[ERROR] _nixl_ll_counters_prepare: nixl_peer_info[%d].counters_buffer_ptr is NULL!\n", remote_rank);
                exit(1);
            }

            // Fetch the first counter (double buffering)
            uint64_t *remote_counter_addr = nixl_peer_info[remote_rank].counters_buffer_ptr + remote_counter_idx;
            nixl_opt_args_t eparams = {};
            eparams.backends.push_back(agent_info.backend);
            nixl_xfer_dlist_t dst_dlist(VRAM_SEG);
            dst_dlist.addDesc(nixlBlobDesc((uintptr_t)remote_counter_addr, sizeof(uint64_t), nixl_peer_info[remote_rank].device_id, ""));
            EP_HOST_ASSERT(agent_info.agent->createXferReq(NIXL_WRITE, dummy_src_dlist, dst_dlist, agent_info.dst_agent_names[remote_rank], nllc.cpu_remote_counter_reqs_0[local_counter_idx], &eparams) == NIXL_SUCCESS);
            EP_HOST_ASSERT(agent_info.agent->createGpuXferReq(*nllc.cpu_remote_counter_reqs_0[local_counter_idx], nllc.gpu_remote_counter_reqs_0[local_counter_idx]) == NIXL_SUCCESS);

            // Fetch the second counter (double buffering)
            remote_counter_addr += max_num_ranks * num_local_experts;
            nixl_xfer_dlist_t dst_dlist_2(VRAM_SEG);
            dst_dlist_2.addDesc(nixlBlobDesc((uintptr_t)remote_counter_addr, sizeof(uint64_t), nixl_peer_info[remote_rank].device_id, ""));
            EP_HOST_ASSERT(agent_info.agent->createXferReq(NIXL_WRITE, dummy_src_dlist, dst_dlist_2, agent_info.dst_agent_names[remote_rank], nllc.cpu_remote_counter_reqs_1[local_counter_idx], &eparams) == NIXL_SUCCESS);
            EP_HOST_ASSERT(agent_info.agent->createGpuXferReq(*nllc.cpu_remote_counter_reqs_1[local_counter_idx], nllc.gpu_remote_counter_reqs_1[local_counter_idx]) == NIXL_SUCCESS);
        }
    }

    // Initialize sync counters
    uint64_t sync_counter_offset = num_local_experts * max_num_ranks * 2;

    for (int remote_rank : ranks_to_setup) {
        if (remote_rank == rank) continue;
        nixl_opt_args_t eparams = {};
        eparams.backends.push_back(nixl_agent_infos[0].backend);
        nixl_xfer_dlist_t dst_dlist(VRAM_SEG);
        dst_dlist.addDesc(nixlBlobDesc((uintptr_t)(nixl_peer_info[remote_rank].counters_buffer_ptr + (sync_counter_offset + rank)), sizeof(uint64_t), nixl_peer_info[remote_rank].device_id, ""));
        EP_HOST_ASSERT(nixl_agent_infos[0].agent->createXferReq(NIXL_WRITE, dummy_src_dlist, dst_dlist, nixl_agent_infos[0].dst_agent_names[remote_rank], nllc.cpu_sync_counters[remote_rank], &eparams) == NIXL_SUCCESS);
        EP_HOST_ASSERT(nixl_agent_infos[0].agent->createGpuXferReq(*nllc.cpu_sync_counters[remote_rank], nllc.gpu_sync_counters[remote_rank]) == NIXL_SUCCESS);
    }
}

void Buffer::_nixl_agents_wiredown(const std::vector<int>& ranks_to_remove) {
    // Clean up wire_up_done flags and agent names for removed ranks
    for (int channel = 0; channel < env_num_channels; ++channel) {
        for (int remote_rank : ranks_to_remove) {
            EP_HOST_ASSERT(remote_rank != rank);
            EP_HOST_ASSERT(remote_rank < num_ranks);
            
            nixl_status_t status = nixl_agent_infos[channel].agent->invalidateRemoteMD(nixl_agent_infos[channel].dst_agent_names[remote_rank]);
            if (status != NIXL_SUCCESS) {
                printf("WARNING: Failed to invalidate remote metadata for agent %s, status: %d\n", 
                    nixl_agent_infos[channel].dst_agent_names[remote_rank].c_str(), status); fflush(stdout);
            }
            
            nixl_agent_infos[channel].dst_agent_names[remote_rank].clear();
            nixl_agent_infos[channel].wire_up_done[remote_rank] = false;
            
            // Clear nixl_peer_info for removed ranks (only do this once per rank, on first channel)
            if (channel == 0) {
                nixl_peer_info[remote_rank] = NixlPeerInfo{};
            }
        }
    }
}

void Buffer::_nixl_ll_cleanup(const std::vector<int>& ranks_to_remove) {
    _nixl_ll_p2p_ptrs_cleanup(ranks_to_remove);
    _nixl_ll_batches_cleanup(ranks_to_remove);
    _nixl_ll_counters_ranks_cleanup(ranks_to_remove);
    _nixl_ll_gpu_ctx_cleanup();
}

void Buffer::_nixl_ll_counters_ranks_cleanup(const std::vector<int>& ranks_to_remove) {

    int num_local_experts = nixl_agent_infos.size(); 
    
    for (int expert_idx = 0; expert_idx < num_local_experts; expert_idx++) {
        for (int remote_rank : ranks_to_remove) {
            EP_HOST_ASSERT(remote_rank != rank);
            
            int local_counter_idx = expert_idx * max_num_ranks + remote_rank;
            
            // Clean up remote counter requests (double buffering)
            if (nllc.cpu_remote_counter_reqs_0[local_counter_idx] != nullptr) {

#ifndef EP_REMOVE_ONCE
                nixl_agent_infos[expert_idx].agent->releaseXferReqtoGPU(nllc.cpu_remote_counter_reqs_0[local_counter_idx]);
                nixl_agent_infos[expert_idx].agent->releaseXferReq(nllc.cpu_remote_counter_reqs_0[local_counter_idx]);
#endif
                nllc.cpu_remote_counter_reqs_0[local_counter_idx] = nullptr;
                nllc.gpu_remote_counter_reqs_0[local_counter_idx] = nullptr;
            }
            
            if (nllc.cpu_remote_counter_reqs_1[local_counter_idx] != nullptr) {
#ifndef EP_REMOVE_ONCE
                nixl_agent_infos[expert_idx].agent->releaseXferReqtoGPU(nllc.cpu_remote_counter_reqs_1[local_counter_idx]);
                nixl_agent_infos[expert_idx].agent->releaseXferReq(nllc.cpu_remote_counter_reqs_1[local_counter_idx]);
#endif
                nllc.cpu_remote_counter_reqs_1[local_counter_idx] = nullptr;
                nllc.gpu_remote_counter_reqs_1[local_counter_idx] = nullptr;
            }
        }
    }
    
    // Clean up sync counters
    for (int remote_rank : ranks_to_remove) {
        if (remote_rank == rank) continue;
        if (nllc.cpu_sync_counters[remote_rank] != nullptr) {
#ifndef EP_REMOVE_ONCE
            nixl_agent_infos[0].agent->releaseXferReqtoGPU(nllc.cpu_sync_counters[remote_rank]);
            nixl_agent_infos[0].agent->releaseXferReq(nllc.cpu_sync_counters[remote_rank]);
#endif
            nllc.cpu_sync_counters[remote_rank] = nullptr;
            nllc.gpu_sync_counters[remote_rank] = nullptr;
        }
    }
}

void Buffer::_nixl_ll_batches_cleanup(const std::vector<int>& ranks_to_remove) {
    for (int channel = 0; channel < env_num_channels; ++channel) {
        for (int remote_rank : ranks_to_remove) {
            if (remote_rank == rank) continue;
            
            // Clean up cpu_batch_reqs and gpu_batch_reqs
            if (remote_rank < nixl_agent_infos[channel].cpu_batch_reqs.size() && 
                nixl_agent_infos[channel].cpu_batch_reqs[remote_rank] != nullptr) {
                
                // Release GPU transfer request first
                if (nixl_agent_infos[channel].gpu_batch_reqs[remote_rank] != nullptr) {
#ifndef EP_REMOVE_ONCE
                    nixl_status_t status = nixl_agent_infos[channel].agent->releaseXferReqtoGPU(nixl_agent_infos[channel].cpu_batch_reqs[remote_rank]);
                    if (status != NIXL_SUCCESS) {
                        printf("[WARNING] _nixl_ll_batches_cleanup: Failed to release GPU batch xfer req for rank %d on channel %d, status: %d\n", 
                               remote_rank, channel, status);
                    }
#endif
                    nixl_agent_infos[channel].gpu_batch_reqs[remote_rank] = nullptr;
                }
                
                // Release CPU transfer request
#ifndef EP_REMOVE_ONCE
                nixl_status_t status = nixl_agent_infos[channel].agent->releaseXferReq(nixl_agent_infos[channel].cpu_batch_reqs[remote_rank]);
                if (status != NIXL_SUCCESS) {
                    printf("[WARNING] _nixl_ll_batches_cleanup: Failed to release CPU batch xfer req for rank %d on channel %d, status: %d\n", 
                           remote_rank, channel, status);
                }
#endif
                nixl_agent_infos[channel].cpu_batch_reqs[remote_rank] = nullptr;
            }

            // Clear destination VRAM descriptor for this rank
            if (remote_rank < nixl_agent_infos[channel].dst_vram.size()) {
                nixl_agent_infos[channel].dst_vram[remote_rank] = nixl_xfer_dlist_t(VRAM_SEG);
            }
        }
    }
}

void Buffer::_nixl_ll_p2p_ptrs_cleanup(const std::vector<int>& ranks_to_remove) {
    for (int remote_rank : ranks_to_remove) {
        EP_HOST_ASSERT(remote_rank < num_ranks);
        // Close P2P memory mappings if they exist
        if (nllc.rdma_p2p_ptrs[remote_rank] != nullptr && 
            nllc.rdma_p2p_ptrs[remote_rank] != rdma_buffer_ptr) {
            CUDA_CHECK(cudaIpcCloseMemHandle(nllc.rdma_p2p_ptrs[remote_rank]));
            nllc.rdma_p2p_ptrs[remote_rank] = nullptr;
        }
        
        if (nllc.counters_p2p_ptrs[remote_rank] != nullptr &&
            nllc.counters_p2p_ptrs[remote_rank] != counters_buffer_ptr) {
            CUDA_CHECK(cudaIpcCloseMemHandle(nllc.counters_p2p_ptrs[remote_rank]));
            nllc.counters_p2p_ptrs[remote_rank] = nullptr;
        }
    }
}

void Buffer::_nixl_ll_gpu_ctx_cleanup() {
    // Update GPU context with the cleaned data (nullptr entries for removed ranks)
    _nixl_ll_gpu_ctx_prepare();
}

} // namespace deep_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    pybind11::class_<deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6, py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6, py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, bool, bool>())
        .def("update_memory_buffers", &deep_ep::Buffer::update_memory_buffers)
        .def("low_latency_sync", &deep_ep::Buffer::low_latency_sync)
        .def("connect_ranks", &deep_ep::Buffer::connect_ranks,
                py::arg("remote_ranks"),
                py::arg("ipc_handles") = std::vector<std::optional<pybind11::bytearray>>{})
        .def("remove_ranks", &deep_ep::Buffer::remove_ranks)
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_nvshmem_unique_id", &deep_ep::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &deep_ep::Buffer::get_comm_stream)
        .def("sync", &deep_ep::Buffer::sync)
        .def("destroy", &deep_ep::Buffer::destroy)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine)
        .def("get_next_low_latency_combine_buffer", &deep_ep::Buffer::get_next_low_latency_combine_buffer);
    m.def("is_sm90_compiled", deep_ep::is_sm90_compiled);
}

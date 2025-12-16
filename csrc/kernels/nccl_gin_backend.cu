#ifdef ENABLE_NCCL

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "configs.cuh"
#include "cuda_runtime.h"
#include "exception.cuh"
#include "nccl.h"
#include "nccl_device/core.h"  // For NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER
#include "nccl_device/gin.h"
#include "nccl_device/gin/gin_device_api.h"
#include "nccl_gin_backend.h"
#include "utils.cuh"

// NCCL error checking macro - pattern from official NCCL examples
// See: nccl/examples/common/include/nccl_utils.h
#define NCCLCHECK(cmd)                                                                                      \
    do {                                                                                                    \
        ncclResult_t res = cmd;                                                                             \
        if (res != ncclSuccess) {                                                                           \
            fprintf(stderr, "DeepEP NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
            throw std::runtime_error("NCCL operation failed");                                              \
        }                                                                                                   \
    } while (0)

// Use NCCL's NCCLCHECK for functions that can return errors

namespace deep_ep {
namespace internode {

NCCLGINBackend::~NCCLGINBackend() {
    if (initialized_) {
        finalize();
    }
}

int NCCLGINBackend::init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode, int qps_per_rank) {
    if (initialized_) {
        return rank_;
    }

    // Check if P2P/NVLink is disabled via environment variable
    const char* nccl_disable_p2p = std::getenv("NCCL_DISABLE_P2P");
    p2p_disabled_ = (nccl_disable_p2p != nullptr && std::string(nccl_disable_p2p) == "1");

    try {
        // Determine communication topology based on mode
        const int gpus_per_server = NUM_MAX_NVL_PEERS;
        int comm_rank;        // Rank to use for NCCL initialization
        int comm_nranks;      // Number of ranks in communicator
        int color = -1;       // Symmetric group ID (only for high throughput mode)
        int group_rank = -1;  // Rank within symmetric group

        if (low_latency_mode) {
            // LOW LATENCY MODE: Connect to all ranks
            comm_rank = rank;
            comm_nranks = num_ranks;
            if (rank == 0)
                printf("[NCCL Backend] LOW LATENCY MODE: Rank %d connecting to all %d ranks\n", rank, num_ranks);
        } else {
            // HIGH THROUGHPUT MODE: Connect only to symmetric RDMA ranks
            color = rank % gpus_per_server;
            group_rank = rank / gpus_per_server;
            comm_nranks = (num_ranks + gpus_per_server - 1) / gpus_per_server;
            comm_rank = group_rank;

            printf(
                "[NCCL Backend] HIGH THROUGHPUT MODE: Rank %d (color=%d, group_rank=%d) "
                "connecting to %d symmetric ranks [",
                rank,
                color,
                group_rank,
                comm_nranks);
            for (int s = 0; s < comm_nranks; s++) {
                printf("%d%s", color + s * gpus_per_server, s < comm_nranks - 1 ? ", " : "");
            }
            printf("]\n");
        }

        // Total comms (add 1 if there's a remainder)
        num_comms_ = (qps_per_rank / DEEP_EP_NCCL_GIN_CTXS_PER_COMM) + ((qps_per_rank % DEEP_EP_NCCL_GIN_CTXS_PER_COMM) > 0 ? 1 : 0);

        // Verify we received the right number of unique IDs
        // We always receive NUM_MAX_NVL_PEERS * qps_per_rank IDs from runtime.cu
        // (generated for worst-case HT mode, LL mode uses only the first qps_per_rank IDs)
        size_t single_id_size = sizeof(ncclUniqueId);
        size_t expected_ids = gpus_per_server * qps_per_rank;  // Always NUM_MAX_NVL_PEERS * qps_per_rank
        // printf("[NCCL Backend] Expected IDs: %zu, Actual IDs: %zu\n", expected_ids, root_unique_id_val.size() / single_id_size);
        EP_HOST_ASSERT(root_unique_id_val.size() == expected_ids * single_id_size &&
                       "Number of unique IDs doesn't match NUM_MAX_NVL_PEERS * qps_per_rank");

        if (rank == 0) {
            // Print NCCL version from the actually loaded library
            int nccl_version;
            ncclGetVersion(&nccl_version);
            printf("[NCCL Backend] NCCL version: %d.%d.%d (loaded library)\n",
                   nccl_version / 10000,
                   (nccl_version % 10000) / 100,
                   nccl_version % 100);
            printf("[NCCL Backend] Initializing %d communicator(s) (qps_per_rank=%d) for rank %d/%d\n",
                   num_comms_,
                   qps_per_rank,
                   rank,
                   num_ranks);
        }

        // Get current device
        int current_device = -1;
        CUDA_CHECK(cudaGetDevice(&current_device));
        CUDA_CHECK(cudaSetDevice(current_device));

        nccl_comms_.resize(num_comms_);
        for (int i = 0; i < num_comms_; i++) {
            ncclUniqueId id;
            size_t id_offset;
            if (low_latency_mode) {
                // Low latency: IDs are sequential
                id_offset = i * single_id_size;
            } else {
                // High throughput: IDs are organized by color group
                id_offset = (color * num_comms_ + i) * single_id_size;
            }
            std::memcpy(&id, root_unique_id_val.data() + id_offset, single_id_size);

            // Initialize NCCL communicator using public API (no config needed for GIN)
            // GIN support is automatically available in the communicator
            NCCLCHECK(ncclCommInitRank(&nccl_comms_[i], comm_nranks, id, comm_rank));
            cudaGetLastError();  // Clear any pending errors
        }

        if (rank == 0)
            printf("[NCCL Backend] Rank %d successfully initialized %d communicator(s)\n", comm_rank, num_comms_);

        // Verify we have at least one communicator initialized
        EP_HOST_ASSERT(num_comms_ > 0);

        // Allocate signals per context per buffer (double buffered total)
        // Only Low Latency mode uses dispatch signals
        if (low_latency_mode) {
            int num_local_experts = qps_per_rank;
            int signals_per_buffer_per_ctx = num_local_experts * comm_nranks;  // dispatch path (and combine uses num_experts equal to this)
            int signals_per_ctx_total = signals_per_buffer_per_ctx * 2;        // double buffered (per buffer)
            num_dispatch_signals_ = signals_per_ctx_total;
        } else {
            // High Throughput mode doesn't use dispatch signals
            num_dispatch_signals_ = 0;
        }

        // The assumption is that kDecoupled is false when initializing SymBuffers in internode.cu
        // IMPORTANT: Use global num_ranks, not comm_nranks, because kernels use global topology
        const auto num_rdma_ranks = std::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
        int rdma_channel_head_signals = num_rdma_ranks * DEEP_EP_NCCL_MAX_NUM_CHANNELS;
        int rdma_channel_tail_signals = num_rdma_ranks * DEEP_EP_NCCL_MAX_NUM_CHANNELS;
        // Adding signals for high throughput and low latency kernels
        num_total_signals_ = rdma_channel_head_signals + rdma_channel_tail_signals + num_dispatch_signals_;

        if (num_comms_ > 0) {
            // Allocate device window arrays based on num_comms_
            // since we need one window per communicator in multi-communicator mode
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nccl_dev_wins_), num_comms_ * sizeof(ncclWindow_t)));
        }

        // Initialize Device Communicators
        dcomms_ = new ncclDevComm_t[num_comms_];
        for (int c = 0; c < num_comms_; ++c) {
            dcomms_[c] = ncclDevComm_t{};  // Initialize to default
            ncclDevCommRequirements reqs{}; 
            reqs.barrierCount = MAX_BARRIER_SESSIONS;
            reqs.ginSignalCount = num_total_signals_ + MAX_BARRIER_SESSIONS;
            reqs.ginForceEnable = true;
            NCCLCHECK(ncclDevCommCreate(nccl_comms_[c], &reqs, &dcomms_[c]));
        }
        if (rank == 0)
            printf("[NCCL Backend] Rank %d created %d device communication(s) with %d barrier sessions each\n",
                   comm_rank,
                   num_comms_,
                   MAX_BARRIER_SESSIONS);

        // Allocate device memory for dcomms and copy data
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dcomms_), num_comms_ * sizeof(ncclDevComm_t)));
        CUDA_CHECK(cudaMemcpy(d_dcomms_, dcomms_, num_comms_ * sizeof(ncclDevComm_t), cudaMemcpyHostToDevice));

        // Allocate barrier dummy variable
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_barrier_var_), sizeof(int)));
        CUDA_CHECK(cudaMemset(d_barrier_var_, 0, sizeof(int)));

        // Store global rank and num_ranks (for external API)
        rank_ = rank;
        num_ranks_ = num_ranks;

        // Store communicator-specific ranks for internal use
        comm_rank_ = comm_rank;
        comm_nranks_ = comm_nranks;

        initialized_ = true;

        if (rank == 0)
            printf("[NCCL Backend] Initialized global rank %d/%d (comm rank %d/%d)\n", rank_, num_ranks_, comm_rank_, comm_nranks_);
        return rank_;

    } catch (const std::exception& e) {
        fprintf(stderr, "[NCCL Backend] Initialization failed: %s\n", e.what());
        throw;
    }
}

void NCCLGINBackend::finalize() {
    if (initialized_) {
        if (rank_ == 0)
            printf("[NCCL Backend] Finalizing rank %d\n", rank_);

        try {
            // Destroy device communicators
            if (dcomms_ != nullptr) {
                for (int c = 0; c < num_comms_; ++c) {
                    if (c < static_cast<int>(nccl_comms_.size()) && nccl_comms_[c]) {
                        ncclResult_t res = ncclDevCommDestroy(nccl_comms_[c], &dcomms_[c]);
                        if (res != ncclSuccess) {
                            fprintf(stderr,
                                    "[NCCL Backend] Warning: Failed to destroy device communication %d: %s\n",
                                    c,
                                    ncclGetErrorString(res));
                        }
                    }
                }
                delete[] dcomms_;
                dcomms_ = nullptr;
            }
            // Free device memory for dcomms
            if (d_dcomms_ != nullptr) {
                cudaFree(d_dcomms_);
                d_dcomms_ = nullptr;
            }
            // Free barrier dummy variable
            if (d_barrier_var_ != nullptr) {
                cudaFree(d_barrier_var_);
                d_barrier_var_ = nullptr;
            }
            // Destroy all communicators
            for (auto& c : nccl_comms_) {
                if (c) {
                    ncclCommFinalize(c);
                    ncclCommDestroy(c);
                }
            }
            nccl_comms_.clear();
            // Free device arrays
            if (d_nccl_dev_wins_) {
                cudaFree(d_nccl_dev_wins_);
                d_nccl_dev_wins_ = nullptr;
            }
            if (rank_ == 0)
                printf("[NCCL Backend] Destroyed NCCL communicator\n");
        } catch (const std::exception& e) {
            fprintf(stderr, "[NCCL Backend] Error during finalization: %s\n", e.what());
        }

        initialized_ = false;
    }
}

void NCCLGINBackend::barrier() {
    if (!initialized_) {
        throw std::runtime_error("NCCL backend not initialized");
    }
    if (d_barrier_var_ == nullptr) {
        throw std::runtime_error("Barrier variable not allocated");
    }

    // Use default stream for barrier
    cudaStream_t stream = 0;

    // Perform AllReduce with device memory
    ncclResult_t result = ncclAllReduce(d_barrier_var_,  // sendbuff (device memory)
                                        d_barrier_var_,  // recvbuff (device memory, in-place)
                                        1,               // count
                                        ncclInt,         // datatype
                                        ncclSum,         // operation
                                        nccl_comms_[0],  // communicator
                                        stream           // CUDA stream
    );
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string("NCCL barrier failed: ") + ncclGetErrorString(result));
    }

    // Wait for completion
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void* NCCLGINBackend::alloc(size_t size, size_t alignment) {
    if (!initialized_) {
        throw std::runtime_error("NCCL backend not initialized");
    }

    void* ptr = nullptr;
    // NCCL memory is already aligned to page size, so alignment parameter is ignored for now.
    ncclResult_t res = ncclMemAlloc(&ptr, size);
    if (res != ncclSuccess) {
        throw std::runtime_error(std::string("Failed to allocate NCCL memory: ") + ncclGetErrorString(res));
    }
    if (rank_ == 0)
        printf("[NCCL Backend - Memory Alloc] Rank %d: Allocated ptr=%p, size=%lu\n", rank_, ptr, size);

    return ptr;
}

void NCCLGINBackend::register_memory(void* ptr, size_t size) {
    if (!initialized_) {
        throw std::runtime_error("NCCL backend not initialized");
    }
    if (mem_handle_.ptr != nullptr) {
        throw std::runtime_error("NCCL backend only supports a single registration at a time.");
    }

    mem_handle_.ptr = ptr;

    // Multi-communicator: register with each communicator and gather ctx0 windows
    dev_wins_multi_nccl_.clear();
    dev_wins_multi_nccl_.resize(num_comms_);
    std::vector<ncclWindow_t> wins_nccl;
    wins_nccl.reserve(num_comms_);

    for (int c = 0; c < num_comms_; ++c) {
        // printf("[NCCL Backend - Memory Register] Rank %d: Registering comm %d/%d\n", rank_, c, num_comms_);
        //  Register with ncclCommWindowRegister
        ncclResult_t r = ncclCommWindowRegister(nccl_comms_[c], ptr, size, dev_wins_multi_nccl_[c].data(), 0);
        if (r != ncclSuccess) {
            // Best-effort rollback of registrations
            for (int j = 0; j < c; ++j) {
                ncclCommWindowDeregister(nccl_comms_[j], dev_wins_multi_nccl_[j][0]);
            }
            throw std::runtime_error(std::string("Failed to register NCCL comm windows (multi): ") + ncclGetErrorString(r));
        }

        wins_nccl.push_back(dev_wins_multi_nccl_[c][0]);
    }

    if (d_nccl_dev_wins_ != nullptr && num_comms_ > 0) {
        if (rank_ == 0) {
            printf("[NCCL Backend - Memory Register] Rank %d: Copying %lu NCCL windows to GPU\n", rank_, wins_nccl.size());
            fflush(stdout);
        }

        cudaError_t e2 = cudaMemcpy(d_nccl_dev_wins_, wins_nccl.data(), wins_nccl.size() * sizeof(ncclWindow_t), cudaMemcpyHostToDevice);
        if (e2 != cudaSuccess) {
            printf("[NCCL Backend - Memory Register] Rank %d: NCCL window copy FAILED: %s (error %d)\n", rank_, cudaGetErrorString(e2), e2);
            fflush(stdout);
            for (int c = 0; c < num_comms_; ++c) {
                ncclCommWindowDeregister(nccl_comms_[c], dev_wins_multi_nccl_[c][0]);
            }
            throw std::runtime_error(std::string("Failed to copy NCCL windows to GPU: ") + cudaGetErrorString(e2));
        }

        if (rank_ == 0) {
            printf("[NCCL Backend - Memory Register] Rank %d: Successfully copied windows to GPU\n", rank_);
            fflush(stdout);
        }
    }
    if (rank_ == 0) {
        printf("[NCCL Backend - Memory Register] Rank %d: Registered windows for ptr=%p, size=%lu\n", rank_, ptr, size);
        fflush(stdout);
    }
}

void NCCLGINBackend::free(void* ptr) {
    if (!initialized_) {
        // Don't throw an error during shutdown
        return;
    }

    if (ptr != nullptr && ptr == mem_handle_.ptr) {
        // Deregister memory windows from all communicators
        for (int c = 0; c < num_comms_; ++c) {
            ncclResult_t r = ncclCommWindowDeregister(nccl_comms_[c], dev_wins_multi_nccl_[c][0]);
            if (r != ncclSuccess) {
                fprintf(stderr, "[NCCL Backend] Warning: Failed to deregister NCCL comm windows (comm %d): %s\n", c, ncclGetErrorString(r));
            }
        }
        dev_wins_multi_nccl_.clear();

        // Free the memory
        ncclResult_t res2 = ncclMemFree(mem_handle_.ptr);
        if (res2 != ncclSuccess) {
            fprintf(stderr, "[NCCL Backend] Warning: Failed to free NCCL memory: %s\n", ncclGetErrorString(res2));
        }

        // Reset the handle
        mem_handle_ = {};
    }
}

void* NCCLGINBackend::get_gin_base_ptr() {
    if (!initialized_ || mem_handle_.ptr == nullptr) {
        throw std::runtime_error("NCCL memory not allocated or backend not initialized.");
    }
    return mem_handle_.ptr;
}

unsigned NCCLGINBackend::get_signals_base(int buffer_idx) const {
    EP_HOST_ASSERT(buffer_idx == 0 || buffer_idx == 1 || buffer_idx == 2);
    if (!initialized_ || num_total_signals_ == 0) {
        throw std::runtime_error("NCCL backend not initialized or no signals allocated");
    }

    // Signal layout: [dispatch buffer 0][dispatch buffer 1][channel signals]
    // - buffer_idx 0/1: LL mode dispatch signals (double buffered)
    // - buffer_idx 2: Channel signals (used by both LL and HT modes)
    //
    // For HT mode: num_dispatch_signals_ = 0, so buffer_idx 2 returns 0
    // For LL mode: buffer_idx 2 skips past dispatch signals to reach channel signals
    int signals_per_buffer = num_dispatch_signals_ / 2;  // 0 for HT mode
    return buffer_idx * signals_per_buffer;
}

int NCCLGINBackend::get_num_gin_comms() const {
    return initialized_ ? num_comms_ : 0;
}

ncclDevComm_t* NCCLGINBackend::get_device_communicators() const {
    if (!initialized_) {
        throw std::runtime_error("NCCL backend not initialized");
    }
    if (d_dcomms_ == nullptr) {
        throw std::runtime_error("Device communicators not allocated");
    }
    return d_dcomms_;
}

int NCCLGINBackend::get_rank() const {
    return initialized_ ? rank_ : -1;
}

int NCCLGINBackend::get_num_ranks() const {
    return initialized_ ? num_ranks_ : -1;
}

BackendType NCCLGINBackend::get_backend_type() const {
    return BackendType::NCCL;
}

bool NCCLGINBackend::is_p2p_disabled() const {
    return p2p_disabled_;
}

void NCCLGINBackend::get_unique_id(void* unique_id) {
    ncclGetUniqueId(static_cast<ncclUniqueId*>(unique_id));
}

ncclWindow_t* NCCLGINBackend::get_device_nccl_windows() {
    EP_HOST_ASSERT(initialized_);
    return d_nccl_dev_wins_;
}

}  // namespace internode
}  // namespace deep_ep

#endif  // ENABLE_NCCL
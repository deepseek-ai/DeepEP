#ifdef ENABLE_NCCL_GIN

#include <cuda.h>  // For CUDA Driver API (cuInit)
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "comm.h"  // For access to ncclComm internal structure
#include "configs.cuh"
#include "cuda_runtime.h"
#include "exception.cuh"
#include "ibgda_device.cuh"
#include "launch.cuh"
#include "mpi.h"
#include "nccl.h"
#include "nccl_device/gin.h"
#include "nccl_device/gin/gin_device_api.h"  // Only include device API in .cu files
#include "nccl_gin_backend.h"
#include "utils.cuh"

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
        // Calculate localRank based on hostname (like working examples)
        /* - the device is already set by the caller in init_dist(). By setting the device here, we need to do MPI_Allgather and we don't
        have MPI here. int localRank = 0; uint64_t* hostHashs = new uint64_t[num_ranks]; char hostname[1024]; ncclGinGetHostName(hostname,
        1024); hostHashs[rank] = ncclGinGetHostHash(hostname);

        // Use MPI to exchange hostnames
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
        for (int p = 0; p < num_ranks; p++) {
            if (p == rank) break;
            if (hostHashs[p] == hostHashs[rank]) localRank++;
        }
        delete[] hostHashs;

        std::cout << "[NCCL GIN Backend] Rank " << rank << " using GPU " << localRank << std::endl;

        // Set CUDA device BEFORE NCCL initialization (critical!)
        cudaError_t cuda_result = cudaSetDevice(localRank);
        if (cuda_result != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device " + std::to_string(localRank) +
                                   ": " + std::string(cudaGetErrorString(cuda_result)));
        }
        */

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
                printf("[NCCL GIN Backend] LOW LATENCY MODE: Rank %d connecting to all %d ranks\n", rank, num_ranks);
        } else {
            // HIGH THROUGHPUT MODE: Connect only to symmetric RDMA ranks
            color = rank % gpus_per_server;
            group_rank = rank / gpus_per_server;
            comm_nranks = (num_ranks + gpus_per_server - 1) / gpus_per_server;
            comm_rank = group_rank;

            printf(
                "[NCCL GIN Backend] HIGH THROUGHPUT MODE: Rank %d (color=%d, group_rank=%d) "
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
        num_comms_ = (qps_per_rank / NCCL_GIN_NUM_CONTEXTS_PER_COMM) + ((qps_per_rank % NCCL_GIN_NUM_CONTEXTS_PER_COMM) > 0 ? 1 : 0);
        num_gin_ctxs_ = num_comms_;

        // Verify we received the right number of unique IDs
        // We always receive NUM_MAX_NVL_PEERS * qps_per_rank IDs from runtime.cu
        // (generated for worst-case HT mode, LL mode uses only the first qps_per_rank IDs)
        size_t single_id_size = sizeof(ncclUniqueId);
        size_t expected_ids = gpus_per_server * qps_per_rank;  // Always NUM_MAX_NVL_PEERS * qps_per_rank
        printf("[NCCL GIN Backend] Expected IDs: %zu, Actual IDs: %zu\n", expected_ids, root_unique_id_val.size() / single_id_size);
        EP_HOST_ASSERT(root_unique_id_val.size() == expected_ids * single_id_size &&
                       "Number of unique IDs doesn't match NUM_MAX_NVL_PEERS * qps_per_rank");

        if (rank == 0)
            printf("[NCCL GIN Backend] Initializing %d communicator(s) (qps_per_rank=%d) for rank %d/%d\n",
                   num_comms_,
                   qps_per_rank,
                   rank,
                   num_ranks);

        // Configure NCCL with GIN support
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = 1;

        // Get current device
        int current_device = -1;
        cudaError_t cuda_get_err = cudaGetDevice(&current_device);
        if (cuda_get_err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(cuda_get_err));
        }
        cudaError_t cuda_set_err = cudaSetDevice(current_device);
        if (cuda_set_err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cuda_set_err));
        }

        // Always use vector approach (even for single communicator)
        comms_multi_.resize(num_comms_);
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

            // printf("[NCCL GIN Backend] Rank %d, Device %d: initializing communicator %d/%d (ID first 8 bytes:
            // %02x%02x%02x%02x%02x%02x%02x%02x)\n",
            //        rank, current_device, i, num_comms_,
            //        id.internal[0], id.internal[1], id.internal[2], id.internal[3],
            //        id.internal[4], id.internal[5], id.internal[6], id.internal[7]);
            // fflush(stdout);

            ncclResult_t nccl_result = ncclCommInitRankConfig(&comms_multi_[i], comm_nranks, id, comm_rank, &config);
            // printf("[Rank %d] Comm %d: ncclCommInitRankConfig returned: %d (%s)\n", rank, i, nccl_result,
            // ncclGetErrorString(nccl_result)); fflush(stdout);

            if (nccl_result != ncclSuccess) {
                throw std::runtime_error("NCCL communicator initialization failed for comm " + std::to_string(i) + ": " +
                                         ncclGetErrorString(nccl_result));
            }

            cudaError_t cuda_err = cudaGetLastError();
            // printf("[Rank %d] Comm %d: after ncclCommInitRankConfig - cuda error: %d (%s)\n", rank, i, cuda_err,
            // cudaGetErrorString(cuda_err)); fflush(stdout);

            NCCLCHECK(ncclGinConnectOnce(comms_multi_[i]));
            cudaGetLastError();
            // printf("[Rank %d] Comm %d: after ncclGinConnectOnce - cuda error: %s\n", rank, i, cudaGetErrorString(cudaGetLastError()));
            // fflush(stdout);
        }

        if (rank == 0)
            printf("[NCCL GIN Backend] Rank %d successfully initialized %d communicator(s)\n", comm_rank, num_comms_);

        // Get num_gin_ctxs_ from first communicator
        if (num_comms_ > 0) {
            num_gin_ctxs_ = comms_multi_[0]->sharedRes->ginState.ginCommCount;
            if (num_gin_ctxs_ <= 0 || num_gin_ctxs_ > DEEP_EP_GIN_MAX_CONTEXTS) {
                num_gin_ctxs_ = DEEP_EP_GIN_MAX_CONTEXTS;
            }
            // Verify we have at least as many contexts as expected per communicator
            EP_HOST_ASSERT(num_gin_ctxs_ >= NCCL_GIN_NUM_CONTEXTS_PER_COMM);
        }

        // Allocate signals per context per buffer (double buffered total)
        // Only Low Latency mode uses dispatch signals
        if (low_latency_mode) {
            int num_local_experts = qps_per_rank;
            signals_per_buffer_per_ctx_ = num_local_experts * comm_nranks;  // dispatch path (and combine uses num_experts equal to this)
            int signals_per_ctx_total = signals_per_buffer_per_ctx_ * 2;    // double buffered (per buffer)
            num_dispatch_signals_ = signals_per_ctx_total;
        } else {
            // High Throughput mode doesn't use dispatch signals
            num_dispatch_signals_ = 0;
            signals_per_buffer_per_ctx_ = 0;
        }
        num_dispatch_counters_ = 0;  // We don't use NCCL GIN counters - only signals

        // The assumption is that kDecoupled is false when initializing SymBuffers in internode.cu
        // IMPORTANT: Use global num_ranks, not comm_nranks, because kernels use global topology
        const auto num_rdma_ranks = std::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
        int rdma_channel_head_signals = num_rdma_ranks * max_num_channels_;
        int rdma_channel_tail_signals = num_rdma_ranks * max_num_channels_;
        // Adding signals for high throughput and low latency kernels
        num_total_signals_ = rdma_channel_head_signals + rdma_channel_tail_signals + num_dispatch_signals_;

        // Build and upload device arrays for contexts; windows copied after registration
        /* {
            std::vector<ncclGinCtx_M<-1u>> h_ctxs;
            h_ctxs.reserve(num_comms_);

            // For now, use the first communicator's contexts
            // In multi-comm mode, this can be extended to use one ctx per communicator
            for (int i = 0; i < num_comms_; ++i) {
                ncclGinCtx_M<-1u> gctx;
                gctx.backend = comms_multi_[i]->sharedRes->ginState.ginDevHandles[0]->netDeviceType;
                gctx.handle = comms_multi_[i]->sharedRes->ginState.ginDevHandles[0]->handle;
                gctx.rank = comm_rank;
                gctx.nRanks = comm_nranks;
                h_ctxs.push_back(gctx);
            }
            if (!h_ctxs.empty()) {
                cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&d_gin_ctxs_), h_ctxs.size() * sizeof(ncclGinCtx_M<-1u>));
                if (e1 != cudaSuccess)
                    throw std::runtime_error("Failed to cudaMalloc d_gin_ctxs_");
                cudaError_t e2 = cudaMemcpy(d_gin_ctxs_, h_ctxs.data(), h_ctxs.size() * sizeof(ncclGinCtx_M<-1u>), cudaMemcpyHostToDevice);
                if (e2 != cudaSuccess)
                    throw std::runtime_error("Failed to cudaMemcpy d_gin_ctxs_");
            }
        } */

        {
            if (num_comms_ > 0) {
                // Allocate device window arrays based on num_comms_
                // since we need one window per communicator in multi-communicator mode
                cudaError_t e3 = cudaMalloc(reinterpret_cast<void**>(&d_nccl_dev_wins_), num_comms_ * sizeof(ncclWindow_t));
                if (e3 != cudaSuccess) {
                    cudaFree(d_nccl_dev_wins_);
                    d_nccl_dev_wins_ = nullptr;
                    throw std::runtime_error("Failed to cudaMalloc d_nccl_dev_wins_");
                }
            }
        }

        // Initialize Device Communicators
        dcomms_ = new ncclDevComm_t[num_comms_];
        for (int c = 0; c < num_comms_; ++c) {
            dcomms_[c] = ncclDevComm_t{};  // Initialize to default
            ncclDevCommRequirements reqs = {};
            reqs.barrierCount = MAX_BARRIER_SESSIONS;
            reqs.ginSignalCount = num_total_signals_ + MAX_BARRIER_SESSIONS;
            reqs.ginForceEnable = true;
            NCCLCHECK(ncclDevCommCreate(comms_multi_[c], &reqs, &dcomms_[c]));
        }
        if (rank == 0)
            std::cout << "[NCCL GIN Backend] Rank " << comm_rank << " created " << num_comms_ << " device communication(s) with "
                      << MAX_BARRIER_SESSIONS << " barrier sessions each" << std::endl;

        // Allocate device memory for dcomms and copy data
        cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&d_dcomms_), num_comms_ * sizeof(ncclDevComm_t));
        if (e1 != cudaSuccess)
            throw std::runtime_error("Failed to cudaMalloc d_dcomms_");
        cudaError_t e2 = cudaMemcpy(d_dcomms_, dcomms_, num_comms_ * sizeof(ncclDevComm_t), cudaMemcpyHostToDevice);
        if (e2 != cudaSuccess)
            throw std::runtime_error("Failed to cudaMemcpy d_dcomms_");

        // Allocate barrier dummy variable
        cudaError_t e3 = cudaMalloc(reinterpret_cast<void**>(&d_barrier_var_), sizeof(int));
        if (e3 != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to cudaMalloc d_barrier_var_: ") + cudaGetErrorString(e3));
        }
        cudaError_t e4 = cudaMemset(d_barrier_var_, 0, sizeof(int));
        if (e4 != cudaSuccess) {
            cudaFree(d_barrier_var_);
            d_barrier_var_ = nullptr;
            throw std::runtime_error(std::string("Failed to cudaMemset d_barrier_var_: ") + cudaGetErrorString(e4));
        }

        // Store global rank and num_ranks (for external API)
        rank_ = rank;
        num_ranks_ = num_ranks;

        // Store communicator-specific ranks for internal use
        comm_rank_ = comm_rank;
        comm_nranks_ = comm_nranks;

        initialized_ = true;

        if (rank == 0)
            std::cout << "[NCCL GIN Backend] Initialized global rank " << rank_ << "/" << num_ranks_ << " (comm rank " << comm_rank_ << "/"
                      << comm_nranks_ << ")" << std::endl;
        return rank_;

    } catch (const std::exception& e) {
        std::cerr << "[NCCL GIN Backend] Initialization failed: " << e.what() << std::endl;
        throw;
    }
}

void NCCLGINBackend::finalize() {
    if (initialized_) {
        if (rank_ == 0)
            std::cout << "[NCCL GIN Backend] Finalizing rank " << rank_ << std::endl;

        try {
            // Destroy device communicators
            if (dcomms_ != nullptr) {
                for (int c = 0; c < num_comms_; ++c) {
                    if (c < static_cast<int>(comms_multi_.size()) && comms_multi_[c]) {
                        ncclResult_t res = ncclDevCommDestroy(comms_multi_[c], &dcomms_[c]);
                        if (res != ncclSuccess) {
                            std::cerr << "[NCCL GIN Backend] Warning: Failed to destroy device communication " << c << ": "
                                      << ncclGetErrorString(res) << std::endl;
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
            for (auto& c : comms_multi_) {
                if (c) {
                    ncclGinFinalize(c);
                    ncclCommFinalize(c);
                    ncclCommDestroy(c);
                }
            }
            comms_multi_.clear();
            // Free device arrays
            if (d_gin_ctxs_) {
                cudaFree(d_gin_ctxs_);
                d_gin_ctxs_ = nullptr;
            }
            if (d_nccl_dev_wins_) {
                cudaFree(d_nccl_dev_wins_);
                d_nccl_dev_wins_ = nullptr;
            }
            if (rank_ == 0)
                std::cout << "[NCCL GIN Backend] Destroyed NCCL communicator" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[NCCL GIN Backend] Error during finalization: " << e.what() << std::endl;
        }

        initialized_ = false;
    }
}

void NCCLGINBackend::barrier() {
    if (!initialized_) {
        throw std::runtime_error("NCCL GIN backend not initialized");
    }
    if (d_barrier_var_ == nullptr) {
        throw std::runtime_error("Barrier variable not allocated");
    }

    // Use default stream for barrier
    cudaStream_t stream = 0;

    // Perform AllReduce with device memory
    ncclResult_t result = ncclAllReduce(d_barrier_var_,   // sendbuff (device memory)
                                        d_barrier_var_,   // recvbuff (device memory, in-place)
                                        1,                // count
                                        ncclInt,          // datatype
                                        ncclSum,          // operation
                                        comms_multi_[0],  // communicator
                                        stream            // CUDA stream
    );
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string("NCCL barrier failed: ") + ncclGetErrorString(result));
    }

    // Wait for completion
    cudaError_t cuda_err = cudaStreamSynchronize(stream);
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaStreamSynchronize failed in barrier: ") + cudaGetErrorString(cuda_err));
    }
}

void* NCCLGINBackend::alloc(size_t size, size_t alignment) {
    if (!initialized_) {
        throw std::runtime_error("NCCL GIN backend not initialized");
    }
    if (mem_handle_.ptr != nullptr) {
        throw std::runtime_error("NCCL GIN backend only supports a single allocation at a time.");
    }

    void* ptr = nullptr;
    // NCCL memory is already aligned to page size, so alignment parameter is ignored for now.
    ncclResult_t res = ncclMemAlloc(&ptr, size);
    if (res != ncclSuccess) {
        throw std::runtime_error(std::string("Failed to allocate NCCL memory: ") + ncclGetErrorString(res));
    }
    if (rank_ == 0)
        printf("[NCCL GIN Backend - Memory Alloc] Rank %d: Allocated ptr=%p, size=%lu\n", rank_, ptr, size);

    // Multi-communicator: register with each communicator and gather ctx0 windows
    dev_wins_multi_nccl_.clear();
    dev_wins_multi_nccl_.resize(num_comms_);
    std::vector<ncclWindow_t> wins_nccl;
    wins_nccl.reserve(num_comms_);

    for (int c = 0; c < num_comms_; ++c) {
        printf("[NCCL GIN Backend - Memory Alloc] Rank %d: Registering comm %d/%d\n", rank_, c, num_comms_);
        // Register with ncclCommWindowRegister
        ncclResult_t r = ncclCommWindowRegister(comms_multi_[c], ptr, size, dev_wins_multi_nccl_[c].data(), 0);
        if (r != ncclSuccess) {
            // Best-effort rollback of registrations
            for (int j = 0; j < c; ++j) {
                ncclCommWindowDeregister(comms_multi_[j], dev_wins_multi_nccl_[j][0]);
            }
            ncclMemFree(ptr);
            throw std::runtime_error(std::string("Failed to register NCCL comm windows (multi): ") + ncclGetErrorString(r));
        }

        wins_nccl.push_back(dev_wins_multi_nccl_[c][0]);
    }
    mem_handle_.ptr = ptr;
    mem_handle_.size = size;
    if (d_nccl_dev_wins_ != nullptr && num_comms_ > 0) {
        if (rank_ == 0) {
            printf("[NCCL GIN Backend - Memory Alloc] Rank %d: Copying %lu NCCL windows to GPU\n", rank_, wins_nccl.size());
            fflush(stdout);
        }

        cudaError_t e2 = cudaMemcpy(d_nccl_dev_wins_, wins_nccl.data(), wins_nccl.size() * sizeof(ncclWindow_t), cudaMemcpyHostToDevice);
        if (e2 != cudaSuccess) {
            printf(
                "[NCCL GIN Backend - Memory Alloc] Rank %d: NCCL window copy FAILED: %s (error %d)\n", rank_, cudaGetErrorString(e2), e2);
            fflush(stdout);
            for (int c = 0; c < num_comms_; ++c) {
                ncclCommWindowDeregister(comms_multi_[c], dev_wins_multi_nccl_[c][0]);
            }
            ncclMemFree(mem_handle_.ptr);
            mem_handle_ = {};
            throw std::runtime_error(std::string("Failed to copy NCCL windows to GPU: ") + cudaGetErrorString(e2));
        }

        if (rank_ == 0) {
            printf("[NCCL GIN Backend - Memory Alloc] Rank %d: Successfully copied windows to GPU\n", rank_);
            fflush(stdout);
        }
    }
    if (rank_ == 0) {
        printf("[NCCL GIN Backend - Memory Alloc] Rank %d: Registered windows and returning ptr=%p, size=%lu\n", rank_, ptr, size);
        fflush(stdout);
    }
    return ptr;
}

void NCCLGINBackend::free(void* ptr) {
    if (!initialized_) {
        // Don't throw an error during shutdown
        return;
    }

    if (ptr != nullptr && ptr == mem_handle_.ptr) {
        // Deregister memory windows from all communicators
        for (int c = 0; c < num_comms_; ++c) {
            ncclResult_t r = ncclCommWindowDeregister(comms_multi_[c], dev_wins_multi_nccl_[c][0]);
            if (r != ncclSuccess) {
                std::cerr << "[NCCL GIN Backend] Warning: Failed to deregister NCCL comm windows (comm " << c
                          << "): " << ncclGetErrorString(r) << std::endl;
            }
        }
        dev_wins_multi_nccl_.clear();

        // Free the memory
        ncclResult_t res2 = ncclMemFree(mem_handle_.ptr);
        if (res2 != ncclSuccess) {
            std::cerr << "[NCCL GIN Backend] Warning: Failed to free NCCL memory: " << ncclGetErrorString(res2) << std::endl;
        }

        // Reset the handle
        mem_handle_ = {};
    }
}

void* NCCLGINBackend::get_gin_base_ptr() {
    if (!initialized_ || mem_handle_.ptr == nullptr) {
        throw std::runtime_error("NCCL GIN memory not allocated or backend not initialized.");
    }
    return mem_handle_.ptr;
}

unsigned NCCLGINBackend::get_signals_base(int buffer_idx) const {
    EP_HOST_ASSERT(buffer_idx == 0 || buffer_idx == 1 || buffer_idx == 2);
    if (!initialized_ || num_total_signals_ == 0) {
        throw std::runtime_error("NCCL GIN backend not initialized or no signals allocated");
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
        throw std::runtime_error("NCCL GIN backend not initialized");
    }
    if (d_dcomms_ == nullptr) {
        throw std::runtime_error("Device communicators not allocated");
    }
    return d_dcomms_;
}

int NCCLGINBackend::get_max_num_channels() const {
    return max_num_channels_;
}

int NCCLGINBackend::get_rank() const {
    return initialized_ ? rank_ : -1;
}

int NCCLGINBackend::get_num_ranks() const {
    return initialized_ ? num_ranks_ : -1;
}

BackendType NCCLGINBackend::get_backend_type() const {
    return BackendType::NCCL_GIN;
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

// Factory function for creating NCCL GIN backend
std::unique_ptr<CommunicationBackend> create_nccl_gin_backend() {
    return std::make_unique<NCCLGINBackend>();
}

}  // namespace internode
}  // namespace deep_ep

#endif  // ENABLE_NCCL_GIN
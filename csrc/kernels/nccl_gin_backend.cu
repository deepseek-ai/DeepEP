#ifdef ENABLE_NCCL_GIN

#include <vector>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <iostream>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"
#include "ibgda_device.cuh"
#include "nccl_gin_backend.h"
#include "nccl.h" 
#include "nccl_device/gin.h"
#include "nccl_device/gin/gin_device_api.h"  // Only include device API in .cu files
#include "comm.h"            // For access to ncclComm internal structure
#include "mpi.h"
#include "cuda_runtime.h"
#include <cuda.h>            // For CUDA Driver API (cuInit)
#include <unistd.h>

// Use NCCL's NCCLCHECK for functions that can return errors

namespace deep_ep {
namespace internode {

// Helper functions (from working examples)
static void ncclGinGetHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

static uint64_t ncclGinGetHostHash(const char* string) {
    // djb2 hash
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

NCCLGINBackend::~NCCLGINBackend() {
    if (initialized_) {
        finalize();
    }
}

int NCCLGINBackend::init(const std::vector<uint8_t>& root_unique_id_val, 
                         int rank, int num_ranks, bool low_latency_mode, int qps_per_rank) {
    if (initialized_) {
        return rank_;
    }
    
    const char* env_vars[] = {
        "DEEP_EP_BACKEND", "NCCL_GIN_TYPE", "NCCL_GIN_ENABLE", "UCX_IB_DM_COUNT", "NCCL_SHM_DISABLE", "NCCL_P2P_DISABLE", "NCCL_NET_PLUGIN"
    };    
    
    //const char* env_vars[] = {
    //    "DEEP_EP_BACKEND", "NCCL_GIN_TYPE", "NCCL_SPCX_COLL_ENABLE", "NCCL_GIN_ENABLE",
    //    "UCX_IB_DM_COUNT", "NCCL_SHM_DISABLE", "NCCL_COLLNET_ENABLE", 
    //    "NCCL_COLLNET_NODE_THRESHOLD", "NCCL_P2P_DISABLE", "NCCL_NET_PLUGIN"
    //};/print all environment variables
    /*
    for (const char* var : env_vars) {
        const char* value = getenv(var);
        if (value) {
            std::cout << "[NCCL GIN Backend] Rank " << rank << " " << var << "=" << value << std::endl;
        } else {
            std::cout << "[NCCL GIN Backend] Rank " << rank << " " << var << " is NOT SET" << std::endl;
        }
    }
    */    
    
    // Assert all GIN environment variables are set
    for (const char* var : env_vars) {
        //EP_HOST_ASSERT(getenv(var) != nullptr);
    }

    try {
        // Calculate localRank based on hostname (like working examples)
        /* - the device is already set by the caller in init_dist(). By setting the device here, we need to do MPI_Allgather and we don't have MPI 
             here. 
        int localRank = 0;
        uint64_t* hostHashs = new uint64_t[num_ranks];
        char hostname[1024];
        ncclGinGetHostName(hostname, 1024);
        hostHashs[rank] = ncclGinGetHostHash(hostname);
        
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

        // Number of communicators equals qps_per_rank
        num_comms_ = qps_per_rank;
        num_gin_ctxs_ = num_comms_;
        
        // Verify we received the right number of unique IDs
        size_t single_id_size = sizeof(ncclUniqueId);
        EP_HOST_ASSERT(root_unique_id_val.size() % single_id_size == 0 && 
                       "Invalid unique ID vector size");
        EP_HOST_ASSERT(root_unique_id_val.size() == num_comms_ * single_id_size &&
                       "Number of unique IDs doesn't match qps_per_rank");
        
        printf("[NCCL GIN Backend] Initializing %d communicator(s) (qps_per_rank=%d) for rank %d/%d\n", 
               num_comms_, qps_per_rank, rank, num_ranks);
        
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
            std::memcpy(&id, 
                       root_unique_id_val.data() + i * single_id_size, 
                       single_id_size);
            
            //printf("[NCCL GIN Backend] Rank %d, Device %d: initializing communicator %d/%d (ID first 8 bytes: %02x%02x%02x%02x%02x%02x%02x%02x)\n", 
            //       rank, current_device, i, num_comms_,
            //       id.internal[0], id.internal[1], id.internal[2], id.internal[3],
            //       id.internal[4], id.internal[5], id.internal[6], id.internal[7]); 
            //fflush(stdout);
            
            ncclResult_t nccl_result = ncclCommInitRankConfig(&comms_multi_[i], num_ranks, id, rank, &config);
            //printf("[Rank %d] Comm %d: ncclCommInitRankConfig returned: %d (%s)\n", rank, i, nccl_result, ncclGetErrorString(nccl_result));
            //fflush(stdout);
            
            if (nccl_result != ncclSuccess) {
                throw std::runtime_error("NCCL communicator initialization failed for comm " + 
                                       std::to_string(i) + ": " + ncclGetErrorString(nccl_result));
            }
            
            cudaError_t cuda_err = cudaGetLastError();
            //printf("[Rank %d] Comm %d: after ncclCommInitRankConfig - cuda error: %d (%s)\n", rank, i, cuda_err, cudaGetErrorString(cuda_err));
            //fflush(stdout);
            
            NCCLCHECK(ncclGinConnectOnce(comms_multi_[i]));
            cudaGetLastError();
            //printf("[Rank %d] Comm %d: after ncclGinConnectOnce - cuda error: %s\n", rank, i, cudaGetErrorString(cudaGetLastError()));
            //fflush(stdout);
        }
        
        printf("[NCCL GIN Backend] Rank %d successfully initialized %d communicator(s)\n", rank, num_comms_);
        
        // Get num_gin_ctxs_ from first communicator
        if (num_comms_ > 0) {
            num_gin_ctxs_ = comms_multi_[0]->sharedRes->ginState.ginCommCount;
            if (num_gin_ctxs_ <= 0 || num_gin_ctxs_ > DEEP_EP_GIN_MAX_CONTEXTS) {
                num_gin_ctxs_ = DEEP_EP_GIN_MAX_CONTEXTS;
            }
        }

        // Allocate signals per context per buffer (double buffered total)
        int num_local_experts = qps_per_rank; //num_experts / num_ranks;
        signals_per_buffer_per_ctx_ = num_local_experts * num_ranks; // dispatch path (and combine uses num_experts equal to this)
        int signals_per_ctx_total = signals_per_buffer_per_ctx_ * 2;  // double buffered (per buffer)
        // One shared signal range per buffer reused across contexts
        num_dispatch_signals_ = signals_per_ctx_total;
        num_dispatch_counters_ = 0;    // We don't use NCCL GIN counters - only signals

        // The assumption is that kDecoupled is false when initializing SymBuffers in internode.cu
        const auto num_rdma_ranks = std::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
        int rdma_channel_head_signals = num_rdma_ranks * max_num_channels_;
        int rdma_channel_tail_signals = num_rdma_ranks * max_num_channels_;
        // Adding signals for high throughput and low latency kernels
        num_total_signals_ = rdma_channel_head_signals + rdma_channel_tail_signals + num_dispatch_signals_;

        // Build and upload device arrays for contexts; windows copied after registration
        {
            std::vector<ncclGinCtx_M<-1u>> h_ctxs;
            h_ctxs.reserve(num_gin_ctxs_);
            
            // For now, use the first communicator's contexts
            // In multi-comm mode, this can be extended to use one ctx per communicator
            for (int i = 0; i < num_gin_ctxs_; ++i) {
                ncclGinCtx_M<-1u> gctx;
                gctx.backend = comms_multi_[0]->sharedRes->ginState.ginDevHandles[i]->netDeviceType;
                gctx.handle = comms_multi_[0]->sharedRes->ginState.ginDevHandles[i]->handle;
                gctx.rank = rank;
                gctx.nRanks = num_ranks;
                h_ctxs.push_back(gctx);
            }
            if (!h_ctxs.empty()) {
                cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&d_gin_ctxs_), h_ctxs.size() * sizeof(ncclGinCtx_M<-1u>));
                if (e1 != cudaSuccess) throw std::runtime_error("Failed to cudaMalloc d_gin_ctxs_");
                cudaError_t e2 = cudaMemcpy(d_gin_ctxs_, h_ctxs.data(), h_ctxs.size() * sizeof(ncclGinCtx_M<-1u>), cudaMemcpyHostToDevice);
                if (e2 != cudaSuccess) throw std::runtime_error("Failed to cudaMemcpy d_gin_ctxs_");
            }
            if (num_gin_ctxs_ > 0) {
                // Allocate device window arrays based on num_comms_ (not num_gin_ctxs_)
                // since we need one window per communicator in multi-communicator mode
                cudaError_t e3 = cudaMalloc(reinterpret_cast<void**>(&d_nccl_dev_wins_), num_comms_ * sizeof(ncclWindow_t));
                if (e3 != cudaSuccess) {
                    cudaFree(d_nccl_dev_wins_);
                    d_nccl_dev_wins_ = nullptr;
                    throw std::runtime_error("Failed to cudaMalloc d_nccl_dev_wins_");
                }
            }
        }

        // Initialize GIN barriers - always use vector approach
        dcomms_ = new ncclDevComm_t[num_comms_];
        for (int c = 0; c < num_comms_; ++c) {
            dcomms_[c] = ncclDevComm_t{};  // Initialize to default
            ncclDevCommRequirements reqs = {};
            reqs.barrierCount = MAX_BARRIER_SESSIONS;
            reqs.ginSignalCount = num_total_signals_ + MAX_BARRIER_SESSIONS;
            reqs.ginForceEnable = true;
            NCCLCHECK(ncclDevCommCreate(comms_multi_[c], &reqs, &dcomms_[c]));
        }
        std::cout << "[NCCL GIN Backend] Rank " << rank << " created " << num_comms_ 
                << " device communication(s) with " << MAX_BARRIER_SESSIONS << " barrier sessions each" << std::endl;
        
        // Allocate device memory for dcomms and copy data
        cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&d_dcomms_), num_comms_ * sizeof(ncclDevComm_t));
        if (e1 != cudaSuccess) throw std::runtime_error("Failed to cudaMalloc d_dcomms_");
        cudaError_t e2 = cudaMemcpy(d_dcomms_, dcomms_, num_comms_ * sizeof(ncclDevComm_t), cudaMemcpyHostToDevice);
        if (e2 != cudaSuccess) throw std::runtime_error("Failed to cudaMemcpy d_dcomms_");

        // Allocate barrier dummy variable
        cudaError_t e3 = cudaMalloc(reinterpret_cast<void**>(&d_barrier_var_), sizeof(int));
        if (e3 != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to cudaMalloc d_barrier_var_: ") +
                                   cudaGetErrorString(e3));
        }
        cudaError_t e4 = cudaMemset(d_barrier_var_, 0, sizeof(int));
        if (e4 != cudaSuccess) {
            cudaFree(d_barrier_var_);
            d_barrier_var_ = nullptr;
            throw std::runtime_error(std::string("Failed to cudaMemset d_barrier_var_: ") +
                                   cudaGetErrorString(e4));
        }

        rank_ = rank;
        num_ranks_ = num_ranks;
        initialized_ = true;

        std::cout << "[NCCL GIN Backend] Initialized rank " << rank_ << "/" << num_ranks_ << std::endl;
        return rank_;

    } catch (const std::exception& e) {
        std::cerr << "[NCCL GIN Backend] Initialization failed: " << e.what() << std::endl;
        throw;
    }
}

void NCCLGINBackend::finalize() {
    if (initialized_) {
        std::cout << "[NCCL GIN Backend] Finalizing rank " << rank_ << std::endl;
        
        try {
            // Destroy device communicators
            if (dcomms_ != nullptr) {
                for (int c = 0; c < num_comms_; ++c) {
                    if (c < static_cast<int>(comms_multi_.size()) && comms_multi_[c]) {
                        ncclResult_t res = ncclDevCommDestroy(comms_multi_[c], &dcomms_[c]);
                        if (res != ncclSuccess) {
                            std::cerr << "[NCCL GIN Backend] Warning: Failed to destroy device communication " << c 
                                      << ": " << ncclGetErrorString(res) << std::endl;
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
    ncclResult_t result = ncclAllReduce(
        d_barrier_var_,         // sendbuff (device memory)
        d_barrier_var_,         // recvbuff (device memory, in-place)
        1,                        // count
        ncclInt,                  // datatype
        ncclSum,                  // operation
        comms_multi_[0],          // communicator
        stream                    // CUDA stream
    );
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string("NCCL barrier failed: ") +
                                ncclGetErrorString(result));
    }

    // Wait for completion
    cudaError_t cuda_err = cudaStreamSynchronize(stream);
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaStreamSynchronize failed in barrier: ") +
                                cudaGetErrorString(cuda_err));
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
    printf("[NCCL GIN Backend - Memory Alloc] Rank %d: Allocated ptr=%p, size=%lu\n", rank_, ptr, size); fflush(stdout);
    
    // Multi-communicator: register with each communicator and gather ctx0 windows
    dev_wins_multi_nccl_.clear();
    dev_wins_multi_nccl_.resize(num_comms_);
    std::vector<ncclWindow_t> wins_nccl;
    wins_nccl.reserve(num_comms_);
    
    for (int c = 0; c < num_comms_; ++c) {
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
    if (d_nccl_dev_wins_ != nullptr && num_gin_ctxs_ > 0) {
        printf("[NCCL GIN Backend - Memory Alloc] Rank %d: Copying %lu NCCL windows to GPU\n",
               rank_, wins_nccl.size()); fflush(stdout);
        
        cudaError_t e2 = cudaMemcpy(d_nccl_dev_wins_, wins_nccl.data(), wins_nccl.size() * sizeof(ncclWindow_t), cudaMemcpyHostToDevice);
        if (e2 != cudaSuccess) {
            printf("[NCCL GIN Backend - Memory Alloc] Rank %d: NCCL window copy FAILED: %s (error %d)\n", 
                   rank_, cudaGetErrorString(e2), e2); fflush(stdout);
            for (int c = 0; c < num_comms_; ++c) {
                ncclCommWindowDeregister(comms_multi_[c], dev_wins_multi_nccl_[c][0]);
            }
            ncclMemFree(mem_handle_.ptr);
            mem_handle_ = {};
            throw std::runtime_error(std::string("Failed to copy NCCL windows to GPU: ") + cudaGetErrorString(e2));
        }
        
        printf("[NCCL GIN Backend - Memory Alloc] Rank %d: Successfully copied windows to GPU\n", rank_); fflush(stdout);
    }
    printf("[NCCL GIN Backend - Memory Alloc] Rank %d: Registered windows and returning ptr=%p, size=%lu\n", rank_, ptr, size); fflush(stdout);
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
                std::cerr << "[NCCL GIN Backend] Warning: Failed to deregister NCCL comm windows (comm " << c << "): " << ncclGetErrorString(r) << std::endl;
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
    if (!initialized_ || num_dispatch_signals_ == 0 || num_total_signals_ == num_dispatch_signals_) {
        throw std::runtime_error("NCCL GIN backend not initialized or no signals allocated");
    }
    // Calculate signal offset based on buffer index (without signal_base_id_)
    // It works for the high-throughput kernels, because when buffer_idx == 2 it skips all num_dispatch_signals_ 
    int signals_per_buffer = num_dispatch_signals_ / 2;  // We allocated double-buffered signals
    return buffer_idx * signals_per_buffer;
}

int NCCLGINBackend::get_num_gin_ctxs() const {
    return initialized_ ? num_gin_ctxs_ : 0;
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

} // namespace internode
} // namespace deep_ep

#endif // ENABLE_NCCL_GIN 
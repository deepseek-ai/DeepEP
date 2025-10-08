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

        // Determine multi-communicator mode from env
        num_comms_ = qps_per_rank;  //(num_experts / num_ranks); 
        multi_mode_ = false; //FIXME: For the time being, trying to make it work with single communicator. By initializing multiple communicators, we need do 
                             // MPI_Allgather and we don't have MPI here. 

        // todo: remove this and take the max between (num_experts / num_ranks) and number of SMs
        if (const char* env_num = std::getenv("DEEP_EP_GIN_NUM_COMMS")) {
            int requested = std::atoi(env_num);
            if (requested > 1) {
                num_comms_ = std::min(requested, DEEP_EP_GIN_MAX_CONTEXTS);
                multi_mode_ = true;
            }
        }

        // Configure NCCL with GIN support
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = 1;
        //config.ginSupport = 1;  // Enable GIN support

        if (!multi_mode_) {
            // Single-communicator path (backward compatible)
            // Root unique id provided by caller
            ncclUniqueId id;
            if (root_unique_id_val.size() != sizeof(ncclUniqueId)) {
                throw std::runtime_error("Invalid NCCL unique ID size");
            }
            std::memcpy(&id, root_unique_id_val.data(), sizeof(ncclUniqueId));
            //config.numGinCtxs = DEEP_EP_GIN_MAX_CONTEXTS;  // request multiple ctxs in one comm

            // Get current device and explicitly set it (matching working test behavior)
            int current_device = -1;
            cudaError_t cuda_get_err = cudaGetDevice(&current_device);
            if (cuda_get_err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(cuda_get_err));
            }
            cudaError_t cuda_set_err = cudaSetDevice(current_device);
            if (cuda_set_err != cudaSuccess) {
                throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cuda_set_err));
            }
            
            printf("[NCCL GIN Backend] Rank %d, Num Ranks %d, Device %d: initializing NCCL communicator (ID first 8 bytes: %02x%02x%02x%02x%02x%02x%02x%02x)\n", 
                   rank, num_ranks, current_device,
                   id.internal[0], id.internal[1], id.internal[2], id.internal[3],
                   id.internal[4], id.internal[5], id.internal[6], id.internal[7]); 
            fflush(stdout);
            
            ncclResult_t nccl_result = ncclCommInitRankConfig(&comm_, num_ranks, id, rank, &config);
            printf("[Rank %d] ncclCommInitRankConfig returned: %d (%s)\n", rank, nccl_result, ncclGetErrorString(nccl_result));
            fflush(stdout);
            if (nccl_result != ncclSuccess) {
                throw std::runtime_error(std::string("ncclCommInitRankConfig failed: ") + ncclGetErrorString(nccl_result));
            }
            cudaError_t cuda_err = cudaGetLastError();
            printf("[Rank %d] after ncclCommInitRankConfig - cuda error: %d (%s)\n", rank, cuda_err, cudaGetErrorString(cuda_err));
            fflush(stdout);
            NCCLCHECK(ncclGinConnectOnce(comm_));
            printf("[Rank %d] after ncclGinConnectOnce - cuda error: %s\n", rank, cudaGetErrorString(cudaGetLastError()));
            std::cout << "[NCCL GIN Backend] Rank " << rank << " ncclGinConnectOnce completed" << std::endl;
            //cudaGetLastError();

            num_gin_ctxs_ = comm_->sharedRes->ginState.ginCommCount;
            if (num_gin_ctxs_ <= 0 || num_gin_ctxs_ > DEEP_EP_GIN_MAX_CONTEXTS) num_gin_ctxs_ = DEEP_EP_GIN_MAX_CONTEXTS;
        } else {
            /* comment out the multi-communicator path for the time being. 
               By initializing multiple communicators, we need do MPI_Allgather and we don't have MPI here. 
            // Multi-communicator path: create num_comms_ comms, each with 1 ctx (ctx 0)
            std::vector<ncclUniqueId> ids(num_comms_);
            if (rank == 0) {
                for (int c = 0; c < num_comms_; ++c) ncclGetUniqueId(&ids[c]);
            }
            for (int c = 0; c < num_comms_; ++c) {
                MPI_Bcast(&ids[c], sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
            }
            comms_multi_.resize(num_comms_);
            //config.numGinCtxs = 1; // one ctx per communicator
            for (int c = 0; c < num_comms_; ++c) {
                NCCLCHECK(ncclCommInitRankConfig(&comms_multi_[c], num_ranks, ids[c], rank, &config));
                cudaGetLastError();
                //printf("[Rank %d] after ncclCommInitRankConfig - cuda error: %s\n", rank, cudaGetErrorString(cudaGetLastError()));
                NCCLCHECK(ncclGinConnectOnce(comms_multi_[c]));
                cudaGetLastError();
                //printf("[Rank %d] after ncclGinConnectOnce - cuda error: %s\n", rank, cudaGetErrorString(cudaGetLastError()));
            }
            num_gin_ctxs_ = num_comms_;
            */
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
            if (!multi_mode_) {
                for (int i = 0; i < num_gin_ctxs_; ++i) {
                    ncclGinCtx_M<-1u> gctx;
                    gctx.backend = comm_->sharedRes->ginState.ginDevHandles[i]->netDeviceType;
                    gctx.handle = comm_->sharedRes->ginState.ginDevHandles[i]->handle;
                    gctx.rank = rank;
                    gctx.nRanks = num_ranks;
                    h_ctxs.push_back(gctx);
                }
            } else {
                // One ctx (index 0) per communicator
                for (int c = 0; c < num_comms_; ++c) {
                    ncclGinCtx_M<-1u> gctx;
                    gctx.backend = comms_multi_[c]->sharedRes->ginState.ginDevHandles[0]->netDeviceType;
                    gctx.handle = comms_multi_[c]->sharedRes->ginState.ginDevHandles[0]->handle;
                    gctx.rank = rank;
                    gctx.nRanks = num_ranks;
                    h_ctxs.push_back(gctx);
                }
            }
            if (!h_ctxs.empty()) {
                cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&d_gin_ctxs_), h_ctxs.size() * sizeof(ncclGinCtx_M<-1u>));
                if (e1 != cudaSuccess) throw std::runtime_error("Failed to cudaMalloc d_gin_ctxs_");
                cudaError_t e2 = cudaMemcpy(d_gin_ctxs_, h_ctxs.data(), h_ctxs.size() * sizeof(ncclGinCtx_M<-1u>), cudaMemcpyHostToDevice);
                if (e2 != cudaSuccess) throw std::runtime_error("Failed to cudaMemcpy d_gin_ctxs_");
            }
            if (num_gin_ctxs_ > 0) {
                cudaError_t e3 = cudaMalloc(reinterpret_cast<void**>(&d_gin_dev_wins_), num_gin_ctxs_ * sizeof(ncclGinWindow_t));
                if (e3 != cudaSuccess) {
                    cudaFree(d_gin_dev_wins_);
                    d_gin_dev_wins_ = nullptr;
                    throw std::runtime_error("Failed to cudaMalloc d_gin_dev_wins_");
                }
                cudaError_t e4 = cudaMalloc(reinterpret_cast<void**>(&d_nccl_dev_wins_), num_gin_ctxs_ * sizeof(ncclWindow_t));
                if (e4 != cudaSuccess) {
                    cudaFree(d_nccl_dev_wins_);
                    d_nccl_dev_wins_ = nullptr;
                    throw std::runtime_error("Failed to cudaMalloc d_nccl_dev_wins_");
                }
            }
        }

        // Initialize GIN barriers       
        if (multi_mode_) {
            // Multi-communicator mode: create one device comm per communicator
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
                    << " device communications with " << MAX_BARRIER_SESSIONS << " barrier sessions each" << std::endl;
        } else {
            // Single-communicator mode: create one device comm
            dcomms_ = new ncclDevComm_t[1];  // Only allocate what you need
            dcomms_[0] = ncclDevComm_t{};    // Initialize to default
            ncclDevCommRequirements reqs = {};
            reqs.barrierCount = MAX_BARRIER_SESSIONS;
            reqs.ginSignalCount = num_total_signals_ + MAX_BARRIER_SESSIONS;
            reqs.ginForceEnable = true;
            NCCLCHECK(ncclDevCommCreate(comm_, &reqs, &dcomms_[0]));
            std::cout << "[NCCL GIN Backend] Rank " << rank << " created device communication with " 
                    << MAX_BARRIER_SESSIONS << " barrier sessions" << std::endl;
        }
        
        // Allocate device memory for dcomms and copy data
        int num_dcomms = multi_mode_ ? num_comms_ : 1;
        cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&d_dcomms_), num_dcomms * sizeof(ncclDevComm_t));
        if (e1 != cudaSuccess) throw std::runtime_error("Failed to cudaMalloc d_dcomms_");
        cudaError_t e2 = cudaMemcpy(d_dcomms_, dcomms_, num_dcomms * sizeof(ncclDevComm_t), cudaMemcpyHostToDevice);
        if (e2 != cudaSuccess) throw std::runtime_error("Failed to cudaMemcpy d_dcomms_");
                    
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
            // Destroy device communicators based on mode
            if (dcomms_ != nullptr) {
                if (multi_mode_) {
                    // Multi-communicator mode: destroy each device comm
                    for (int c = 0; c < num_comms_; ++c) {
                        ncclResult_t res = ncclDevCommDestroy(comms_multi_[c], &dcomms_[c]);
                        if (res != ncclSuccess) {
                            std::cerr << "[NCCL GIN Backend] Warning: Failed to destroy device communication " << c 
                                      << ": " << ncclGetErrorString(res) << std::endl;
                        }
                    }
                } else {
                    // Single-communicator mode: destroy single device comm
                    ncclResult_t res = ncclDevCommDestroy(comm_, &dcomms_[0]);
                    if (res != ncclSuccess) {
                        std::cerr << "[NCCL GIN Backend] Warning: Failed to destroy device communication: " << ncclGetErrorString(res) << std::endl;
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
            if (comm_) {
                // Finalize GIN before destroying communicator (new API requirement)
                ncclResult_t res1 = ncclGinFinalize(comm_);
                ncclResult_t res2 = ncclCommFinalize(comm_);
                ncclResult_t res3 = ncclCommDestroy(comm_);
                if (res1 != ncclSuccess || res2 != ncclSuccess || res3 != ncclSuccess) {
                    std::cerr << "[NCCL GIN Backend] Warning: Errors during finalization" << std::endl;
                }
                comm_ = nullptr;
            }
            if (!comms_multi_.empty()) {
                for (auto& c : comms_multi_) {
                    if (c) {
                        ncclGinFinalize(c);
                        ncclCommFinalize(c);
                        ncclCommDestroy(c);
                    }
                }
                comms_multi_.clear();
            }
            // Free device arrays
            if (d_gin_ctxs_) {
                cudaFree(d_gin_ctxs_);
                d_gin_ctxs_ = nullptr;
            }
            if (d_gin_dev_wins_) {
                cudaFree(d_gin_dev_wins_);
                d_gin_dev_wins_ = nullptr;
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
    
    // For now, use MPI barrier since we're not implementing NCCL barrier yet
    // In later phases, we can implement this with NCCL primitives
    //MPI_Barrier(MPI_COMM_WORLD); FIXME: We don't have MPI here. Need the equivalent of nvshmem_barrier_all() here with NCCL primitives. Requested George to look into it.
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
    // Register the allocated memory to get window handles
    if (!multi_mode_) {
        res = ncclGinRegister(comm_, ptr, size, mem_handle_.ginHostWins, mem_handle_.ginDevWins);
        if (res != ncclSuccess) {
            ncclMemFree(ptr);  // Clean up on failure
            throw std::runtime_error(std::string("Failed to register GIN memory: ") + ncclGetErrorString(res));
        }
        
        // Also register NCCL windows (similar to multi-mode)
        ncclResult_t res2 = ncclCommWindowRegister(comm_, ptr, size, mem_handle_.ncclWins, 0);
        if (res2 != ncclSuccess) {
            ncclGinDeregister(comm_, mem_handle_.ginHostWins);
            ncclMemFree(ptr);
            throw std::runtime_error(std::string("Failed to register NCCL comm windows: ") + ncclGetErrorString(res2));
        }
        
        mem_handle_.ptr = ptr;
        mem_handle_.size = size;
        
        // Refresh device-visible windows now that registration populated mem_handle_.ginDevWins
        if (d_gin_dev_wins_ != nullptr && d_nccl_dev_wins_ != nullptr && num_gin_ctxs_ > 0) {
            cudaError_t e1 = cudaMemcpy(d_gin_dev_wins_, mem_handle_.ginDevWins,
                                       num_gin_ctxs_ * sizeof(ncclGinWindow_t), cudaMemcpyHostToDevice);
            cudaError_t e2 = cudaMemcpy(d_nccl_dev_wins_, mem_handle_.ncclWins,
                                       num_gin_ctxs_ * sizeof(ncclWindow_t), cudaMemcpyHostToDevice);
            if (e1 != cudaSuccess || e2 != cudaSuccess) {
                // Best effort cleanup before throwing
                ncclCommWindowDeregister(comm_, mem_handle_.ncclWins[0]);
                ncclGinDeregister(comm_, mem_handle_.ginHostWins);
                ncclMemFree(mem_handle_.ptr);
                mem_handle_ = {};
                throw std::runtime_error("Failed to copy GIN/NCCL device windows to GPU");
            }
        }
    } else {
        // Multi-communicator: register with each communicator and gather ctx0 windows
        host_wins_multi_.clear();
        dev_wins_multi_.clear();
        host_wins_multi_.resize(num_comms_);
        dev_wins_multi_.resize(num_comms_);
        std::vector<ncclGinWindow_t> wins0;
        wins0.reserve(num_comms_);
        // New windows wrapper for nccl
        dev_wins_multi_nccl_.clear();
        dev_wins_multi_nccl_.resize(num_comms_);
        std::vector<ncclWindow_t>  wins_nccl;
        wins_nccl.reserve(num_comms_);
        
        for (int c = 0; c < num_comms_; ++c) {
            ncclResult_t r = ncclGinRegister(comms_multi_[c], ptr, size, host_wins_multi_[c].data(), dev_wins_multi_[c].data());
            if (r != ncclSuccess) {
                // Best-effort rollback of registrations
                for (int j = 0; j < c; ++j) ncclGinDeregister(comms_multi_[j], host_wins_multi_[j].data());
                ncclMemFree(ptr);
                throw std::runtime_error(std::string("Failed to register GIN memory (multi): ") + ncclGetErrorString(r));
            }
            
            // Register with ncclCommWindowRegister (symmetric to ncclGinRegister)
            ncclResult_t r2 = ncclCommWindowRegister(comms_multi_[c], ptr, size, dev_wins_multi_nccl_[c].data(), 0);
            if (r2 != ncclSuccess) {
                // Best-effort rollback of registrations
                for (int j = 0; j <= c; ++j) ncclGinDeregister(comms_multi_[j], host_wins_multi_[j].data());
                ncclMemFree(ptr);
                throw std::runtime_error(std::string("Failed to register NCCL comm windows (multi): ") + ncclGetErrorString(r2));
            }
            
            wins0.push_back(dev_wins_multi_[c][0]);
            wins_nccl.push_back(dev_wins_multi_nccl_[c][0]);
        }
        mem_handle_.ptr = ptr;
        mem_handle_.size = size;
        if (d_gin_dev_wins_ != nullptr && d_nccl_dev_wins_ != nullptr && num_gin_ctxs_ > 0) {
            cudaError_t e1 = cudaMemcpy(d_gin_dev_wins_, wins0.data(), wins0.size() * sizeof(ncclGinWindow_t), cudaMemcpyHostToDevice);
            cudaError_t e2 = cudaMemcpy(d_nccl_dev_wins_, wins_nccl.data(), wins_nccl.size() * sizeof(ncclWindow_t), cudaMemcpyHostToDevice);
            if (e1 != cudaSuccess || e2 != cudaSuccess) {
                for (int c = 0; c < num_comms_; ++c) {
                    ncclGinDeregister(comms_multi_[c], host_wins_multi_[c].data());
                    ncclCommWindowDeregister(comms_multi_[c], dev_wins_multi_nccl_[c][0]);
                }
                ncclMemFree(mem_handle_.ptr);
                mem_handle_ = {};
                throw std::runtime_error("Failed to copy multi-comm windows to GPU");
            }
        }
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
        // Deregister memory windows first
        if (!multi_mode_) {
            ncclResult_t res1 = ncclGinDeregister(comm_, mem_handle_.ginHostWins);
            if (res1 != ncclSuccess) {
                std::cerr << "[NCCL GIN Backend] Warning: Failed to deregister GIN memory: " << ncclGetErrorString(res1) << std::endl;
            }
            
            ncclResult_t res2 = ncclCommWindowDeregister(comm_, mem_handle_.ncclWins[0]);
            if (res2 != ncclSuccess) {
                std::cerr << "[NCCL GIN Backend] Warning: Failed to deregister NCCL comm windows: " << ncclGetErrorString(res2) << std::endl;
            }
        } else {
            for (int c = 0; c < num_comms_; ++c) {
                ncclResult_t r = ncclGinDeregister(comms_multi_[c], host_wins_multi_[c].data());
                if (r != ncclSuccess) {
                    std::cerr << "[NCCL GIN Backend] Warning: Failed to deregister GIN memory (comm " << c << "): " << ncclGetErrorString(r) << std::endl;
                }
                
                ncclResult_t r2 = ncclCommWindowDeregister(comms_multi_[c], dev_wins_multi_nccl_[c][0]);
                if (r2 != ncclSuccess) {
                    std::cerr << "[NCCL GIN Backend] Warning: Failed to deregister NCCL comm windows (comm " << c << "): " << ncclGetErrorString(r2) << std::endl;
                }
            }
            host_wins_multi_.clear();
            dev_wins_multi_.clear();
            dev_wins_multi_nccl_.clear();
        }
        
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
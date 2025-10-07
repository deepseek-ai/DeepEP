#pragma once

#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <array>
#include <nccl.h>
#include "nccl_device.h"
#include "nccl_device/gin.h"
#include "nccl_device/gin/gin_device_host_common.h"  // For type definitions

// Define API version flag BEFORE any conditional includes
#ifndef NCCL_GIN_NEW_API
#define NCCL_GIN_NEW_API 1
#endif

#if NCCL_GIN_NEW_API
#include "nccl_device/gin/gin_device_common.h"       // For new API types
#endif
// Note: gin_device_api.h is only included in .cu files for device compilation

#include "communication_backend.h"

#define DEEP_EP_GIN_MAX_CONTEXTS 32

#ifdef ENABLE_NCCL_GIN

namespace deep_ep {
namespace internode {

#define DISPATCH_WITH_IMMEDIATE_SIGNALS 0
#define DISPATCH_WITH_IMMEDIATE_SIGNALS_CORRECT 0

struct NcclGinMemHandle {
    void* ptr = nullptr;
    size_t size = 0;
    void* ginHostWins[DEEP_EP_GIN_MAX_CONTEXTS] = {}; // Array of host window handles
    ncclGinWindow_t ginDevWins[DEEP_EP_GIN_MAX_CONTEXTS] = {}; // Array of device window handles
};

class NCCLGINBackend : public CommunicationBackend {
public:
    NCCLGINBackend() : comm_(nullptr), rank_(-1), num_ranks_(-1), initialized_(false) {}
    ~NCCLGINBackend() override;

    // Required interface methods
    int init(const std::vector<uint8_t>& root_unique_id_val, 
             int rank, int num_ranks, bool low_latency_mode, int qps_per_rank) override;
    void finalize() override;
    void barrier() override;
    
    // Memory management interface methods
    void* alloc(size_t size, size_t alignment) override;
    void free(void* ptr) override;
    
    int get_rank() const override;
    int get_num_ranks() const override;
    BackendType get_backend_type() const override;

    // NCCL GIN-specific method for unique ID generation
    void get_unique_id(void* unique_id) override;
    
    // NCCL GIN specific methods
    // Single-context accessors (backwards compatible; use ctx_idx=0)
    ncclGinWindow_t get_gin_window_handle();
#if !NCCL_GIN_NEW_API
    ncclGinGpuCtx get_gin_gpu_context();  // Old API only
#endif
    ncclGinSignal_t get_gin_signals(int buffer_idx) const;

    // Multi-context accessors
    int get_num_gin_ctxs() const;
    ncclGinWindow_t get_gin_window_handle(int ctx_idx);
#if !NCCL_GIN_NEW_API
    ncclGinGpuCtx get_gin_gpu_context(int ctx_idx);  // Old API only
#endif
    ncclGinSignal_t get_gin_signals(int ctx_idx, int buffer_idx) const;
    // Device arrays for kernels
#if NCCL_GIN_NEW_API
    ncclGinCtx_M<-1u>* get_device_gin_ctxs();  // New API device contexts
#else
    ncclGinGpuCtx* get_device_gin_ctxs();       // Old API device contexts
#endif
    ncclGinWindow_t* get_device_gin_windows();
    void* get_gin_base_ptr();
    int get_max_num_channels() const;
    ncclDevComm* get_device_communicators() const;

#if NCCL_GIN_NEW_API
    // New API methods (placeholders for future implementation)
    ncclGinCtx_M<-1u> get_gin_gpu_context_new();
    ncclGinCtx_M<-1u> get_gin_gpu_context_new(int ctx_idx);
    // TODO: Add device arrays for new API contexts when implementing
#endif

    // For future phases - access to the NCCL communicator
    // ncclComm_t get_nccl_comm() const;

private:
    bool initialized_ = false;
    int rank_ = -1;
    int num_ranks_ = -1;
    
    // NCCL communicator(s) for GIN support
    ncclComm_t comm_ = nullptr; // single-communicator path (default)
    std::vector<ncclComm_t> comms_multi_; // multi-communicator path (size = num_gin_ctxs_)
    
    NcclGinMemHandle mem_handle_;
    // Per-communicator registration (multi-communicator path)
    std::vector<std::array<void*, DEEP_EP_GIN_MAX_CONTEXTS>> host_wins_multi_;
    std::vector<std::array<ncclGinWindow_t, DEEP_EP_GIN_MAX_CONTEXTS>> dev_wins_multi_;

    // GIN signal and counter management (new API)
    uint32_t signal_base_id_ = 0;
    uint32_t counter_base_id_ = 0;
    int num_dispatch_signals_ = 0;   // total allocated signals across all contexts and buffers
    int num_total_signals_ = 0;
    int num_dispatch_counters_ = 0;  // currently unused

    // Multi-context tracking
    int num_gin_ctxs_ = 1;
    int num_comms_ = 1; // equals num_gin_ctxs_ in multi-communicator mode
    bool multi_mode_ = false;
    int signals_per_buffer_per_ctx_ = 0; // number of signals per buffer for one context

    // Device arrays for contexts and windows
#if NCCL_GIN_NEW_API
    ncclGinCtx_M<-1u>* d_gin_ctxs_ = nullptr;  // New API contexts
#else
    ncclGinGpuCtx* d_gin_ctxs_ = nullptr;       // Old API contexts
#endif
    ncclGinWindow_t* d_gin_dev_wins_ = nullptr;

    // The assumption is that DeepSeek (256 experts) runs on at least 8 GPUs, hence 32 channels
    const int max_num_channels_ = 32; // Max number of local experts per GPU

    // GIN barriers -- assume 32 rdma ranks
    ncclDevComm_t* dcomms_ = nullptr;  // Host array
    ncclDevComm_t* d_dcomms_ = nullptr;  // Device array
    const int MAX_BARRIER_SESSIONS = 32;

};

} // namespace internode
} // namespace deep_ep 

#endif // ENABLE_NCCL_GIN 

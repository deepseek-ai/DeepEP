#pragma once

#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <array>
#include <nccl.h>
#include "nccl_device.h"
#include "nccl_device/gin.h"
#include "nccl_device/gin/gin_device_host_common.h"  // For type definitions
#include "nccl_device/gin/gin_device_common.h"       // For new API types
// Note: gin_device_api.h is only included in .cu files for device compilation

#include "communication_backend.h"

#define DEEP_EP_GIN_MAX_CONTEXTS 32

#ifdef ENABLE_NCCL_GIN

namespace deep_ep {
namespace internode {

struct NcclGinMemHandle {
    void* ptr = nullptr;
    size_t size = 0;
    void* ginHostWins[DEEP_EP_GIN_MAX_CONTEXTS] = {}; // Array of host window handles
    ncclGinWindow_t ginDevWins[DEEP_EP_GIN_MAX_CONTEXTS] = {}; // Array of device window handles
    ncclWindow_t ncclWins[DEEP_EP_GIN_MAX_CONTEXTS] = {}; // Array of NCCL window handles
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
    unsigned get_signals_base(int buffer_idx) const;
    int get_num_gin_ctxs() const;

    // Device arrays for kernels
    ncclWindow_t* get_device_nccl_windows();
    void* get_gin_base_ptr();
    int get_max_num_channels() const;
    ncclDevComm* get_device_communicators() const;

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
    std::vector<std::array<ncclWindow_t, DEEP_EP_GIN_MAX_CONTEXTS>> dev_wins_multi_nccl_;

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
    ncclGinCtx_M<-1u>* d_gin_ctxs_ = nullptr;  // New API contexts
    ncclGinWindow_t* d_gin_dev_wins_ = nullptr;
    ncclWindow_t* d_nccl_dev_wins_ = nullptr;

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

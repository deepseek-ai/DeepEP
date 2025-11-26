#pragma once

#include <cuda_runtime.h>
#include <nccl.h>

#include <array>
#include <vector>

#include "nccl_device.h"
#include "nccl_device/gin.h"
#include "nccl_device/gin/gin_device_common.h"       // For new API types
#include "nccl_device/gin/gin_device_host_common.h"  // For type definitions
// Note: gin_device_api.h is only included in .cu files for device compilation

#include "communication_backend.h"

#define DEEP_EP_GIN_MAX_CONTEXTS 32
#define NCCL_GIN_NUM_CONTEXTS_PER_COMM 4
#define NCCL_MAX_NUM_CHANNELS 32  // Max number of local experts per GPU

#ifdef ENABLE_NCCL

namespace deep_ep {
namespace internode {

struct NcclGinMemHandle {
    void* ptr = nullptr;
};

class NCCLGINBackend : public CommunicationBackend {
public:
    NCCLGINBackend() : rank_(-1), num_ranks_(-1), initialized_(false) {}
    ~NCCLGINBackend() override;

    // Required interface methods
    int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode, int qps_per_rank) override;
    void finalize() override;
    void barrier() override;

    // Memory management interface methods
    void* alloc(size_t size, size_t alignment) override;
    void free(void* ptr) override;

    int get_rank() const override;
    int get_num_ranks() const override;
    BackendType get_backend_type() const override;

    // NCCL-specific methods
    bool is_p2p_disabled() const;

    // NCCL-specific method for unique ID generation
    void get_unique_id(void* unique_id) override;

    // NCCL specific methods
    unsigned get_signals_base(int buffer_idx) const;
    int get_num_gin_comms() const;

    // Device arrays for kernels
    ncclWindow_t* get_device_nccl_windows();
    void* get_gin_base_ptr();
    ncclDevComm* get_device_communicators() const;

private:
    bool initialized_ = false;
    bool p2p_disabled_ = false;  // True if P2P/NVLink is disabled
    int rank_ = -1;              // Global rank (for external API)
    int num_ranks_ = -1;         // Global num_ranks (for external API)
    int comm_rank_ = -1;         // Rank within NCCL communicator
    int comm_nranks_ = -1;       // Number of ranks in NCCL communicator

    // NCCL communicators for GIN support (always use vector, even for single comm)
    std::vector<ncclComm_t> nccl_comms_;

    NcclGinMemHandle mem_handle_;
    // Per-communicator registration (multi-communicator path)
    std::vector<std::array<ncclWindow_t, DEEP_EP_GIN_MAX_CONTEXTS>> dev_wins_multi_nccl_;

    // GIN signal management
    int num_dispatch_signals_ = 0;  // total allocated signals across all contexts and buffers
    int num_total_signals_ = 0;

    // Multi-context tracking
    int num_comms_ = 1;

    // Device arrays for windows
    ncclWindow_t* d_nccl_dev_wins_ = nullptr;

    // GIN barriers -- assume 32 rdma ranks
    ncclDevComm_t* dcomms_ = nullptr;    // Host array
    ncclDevComm_t* d_dcomms_ = nullptr;  // Device array
    const int MAX_BARRIER_SESSIONS = 32;

    // Barrier variable
    int* d_barrier_var_ = nullptr;
};

}  // namespace internode
}  // namespace deep_ep

#endif  // ENABLE_NCCL

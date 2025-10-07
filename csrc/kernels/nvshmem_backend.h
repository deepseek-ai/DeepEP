#pragma once

#include "communication_backend.h"
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

namespace deep_ep {
namespace internode {

class NVSHMEMBackend : public CommunicationBackend {
public:
    NVSHMEMBackend() = default;
    ~NVSHMEMBackend() override;

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

    // NVSHMEM-specific methods (not part of base interface)
    void get_unique_id(void* unique_id) override;
    
    // Low-latency mode and team management
    void setup_low_latency_mode(int rank, int num_ranks);

private:
    bool initialized_ = false;
    int rank_ = -1;
    int num_ranks_ = -1;
    bool low_latency_mode_ = false;
    
    // Team management for low-latency mode
    nvshmem_team_t cpu_rdma_team_ = NVSHMEM_TEAM_INVALID;
    nvshmem_team_config_t cpu_rdma_team_config_;
};

} // namespace internode
} // namespace deep_ep 
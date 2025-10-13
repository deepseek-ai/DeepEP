/*

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "configs.cuh"
#include "exception.cuh"
#include "ibgda_device.cuh"
#include "launch.cuh"
#include "mpi.h"
#include "nvshmem_backend.h"
#include "utils.cuh"

// Forward declaration for global team variable (for internode.cu compatibility)
extern nvshmem_team_t cpu_rdma_team;
extern nvshmem_team_config_t cpu_rdma_team_config;

namespace deep_ep {
namespace internode {

NVSHMEMBackend::~NVSHMEMBackend() {
    if (initialized_) {
        finalize();
    }
}

int NVSHMEMBackend::init(const std::vector<uint8_t>& root_unique_id_val,
                         int rank, int num_ranks, bool low_latency_mode, int qps_per_rank) {
    if (initialized_) {
        return rank_;
    }

    // Note: num_experts parameter is not used by NVSHMEM backend
    // but is required for interface consistency with NCCLGINBackend
    (void)num_experts; // Suppress unused parameter warning

    // Real NVSHMEM initialization
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    rank_ = nvshmem_my_pe();
    num_ranks_ = nvshmem_n_pes();
    initialized_ = true;

    std::cout << "[NVSHMEM Backend] Initialized rank " << rank_ << "/" << num_ranks_ << std::endl;

    return rank_;
}

void NVSHMEMBackend::finalize() {
    if (initialized_) {
        std::cout << "[NVSHMEM Backend] Finalizing rank " << rank_ << std::endl;

        // Clean up teams if they exist
        if (low_latency_mode_ && cpu_rdma_team_ != NVSHMEM_TEAM_INVALID) {
            nvshmem_team_destroy(cpu_rdma_team_);
            cpu_rdma_team_ = NVSHMEM_TEAM_INVALID;

            // Update global variables for internode.cu compatibility
            cpu_rdma_team = NVSHMEM_TEAM_INVALID;

            std::cout << "[NVSHMEM Backend] Destroyed NVSHMEM teams" << std::endl;
        }

        nvshmem_finalize();
        initialized_ = false;
        low_latency_mode_ = false;
    }
}

void NVSHMEMBackend::barrier() {
    if (!initialized_) {
        throw std::runtime_error("NVSHMEM backend not initialized");
    }
    nvshmem_barrier_all();
}

void* NVSHMEMBackend::alloc(size_t size, size_t alignment) {
    if (!initialized_) {
        throw std::runtime_error("NVSHMEM backend not initialized");
    }
    return nvshmem_align(alignment, size);
}

void NVSHMEMBackend::free(void* ptr) {
    if (!initialized_) {
        throw std::runtime_error("NVSHMEM backend not initialized");
    }
    if (ptr != nullptr) {
        nvshmem_free(ptr);
    }
}

int NVSHMEMBackend::get_rank() const {
    return initialized_ ? rank_ : -1;
}

int NVSHMEMBackend::get_num_ranks() const {
    return initialized_ ? num_ranks_ : -1;
}

BackendType NVSHMEMBackend::get_backend_type() const {
    return BackendType::NVSHMEM;
}

void NVSHMEMBackend::get_unique_id(void* unique_id) {
    nvshmemx_uniqueid_t* nvshmem_id = static_cast<nvshmemx_uniqueid_t*>(unique_id);
    nvshmemx_get_uniqueid(nvshmem_id);
}



void NVSHMEMBackend::setup_low_latency_mode(int rank, int num_ranks) {
    if (!initialized_) {
        throw std::runtime_error("NVSHMEM backend not initialized");
    }

    if (num_ranks <= NUM_MAX_NVL_PEERS) {
        // No team creation needed for small number of ranks
        return;
    }

    // Validate preconditions
    EP_HOST_ASSERT(cpu_rdma_team_ == NVSHMEM_TEAM_INVALID);
    EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);

    // Create NVSHMEM team for low-latency mode
    EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD,
                                              rank % NUM_MAX_NVL_PEERS,
                                              NUM_MAX_NVL_PEERS,
                                              num_ranks / NUM_MAX_NVL_PEERS,
                                              &cpu_rdma_team_config_,
                                              0,
                                              &cpu_rdma_team_) == 0);
    EP_HOST_ASSERT(cpu_rdma_team_ != NVSHMEM_TEAM_INVALID);

    // Update global variables for internode.cu compatibility
    cpu_rdma_team = cpu_rdma_team_;
    cpu_rdma_team_config = cpu_rdma_team_config_;

    // Device state manipulation for IBGDA mode
    nvshmemi_device_host_state_t* dev_state_ptr = nullptr;
    CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&dev_state_ptr), nvshmemi_device_state_d));

    bool ibgda_is_initialized = false;
    CUDA_CHECK(cudaMemcpy(&dev_state_ptr->ibgda_is_initialized, &ibgda_is_initialized, sizeof(bool), cudaMemcpyHostToDevice));

    low_latency_mode_ = true;

    std::cout << "[NVSHMEM Backend] Created teams and applied device state for low-latency mode" << std::endl;
}

} // namespace internode
} // namespace deep_ep
*/
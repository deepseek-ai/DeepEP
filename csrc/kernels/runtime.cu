#include <cstring>
#include <vector>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

#ifdef ENABLE_NCCL_GIN
#include "nccl_gin_backend.h"
#include <nccl.h>
#endif

#ifndef DISABLE_NVSHMEM
#include "ibgda_device.cuh"
#include "nvshmem.h"
#endif

namespace deep_ep {

namespace intranode {

template <int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                  \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, 32, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

}  // namespace intranode

namespace internode {

#ifndef DISABLE_NVSHMEM
nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
nvshmem_team_config_t cpu_rdma_team_config;
#endif

std::vector<uint8_t> get_unique_id() {

#ifdef ENABLE_NCCL_GIN
    //FIXME: this generates 1 unique id. Since we are creating multiple communicators, we will need to generate multiple unique ids.
    ncclUniqueId unique_id;
    ncclGetUniqueId(&unique_id);
    std::vector<uint8_t> result(sizeof(ncclUniqueId));
    std::memcpy(result.data(), &unique_id, sizeof(ncclUniqueId));
    return result;
#else 
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
#endif

}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode, int qps_per_rank) {

    //printf("runtime::init() called\n"); fflush(stdout);
#ifdef ENABLE_NCCL_GIN
    //printf("NCCL: init()\n"); fflush(stdout);
    internode::BackendType backend_type = internode::detect_backend_type();
    // Pass 0 for num_experts as it's not available at this point. Backend can use DEEP_EP_GIN_NUM_COMMS env var if needed.
    internode::initialize_backend(backend_type, root_unique_id_val, rank, num_ranks, low_latency_mode, qps_per_rank);     
    internode::CommunicationBackend* backend = internode::get_backend();
    backend->barrier();
    return backend->get_rank();
#else
    //printf("NVSHMEM: init()\n"); fflush(stdout);
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    // Create sub-RDMA teams
    // NOTES: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency kernels are used
    if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
        EP_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD,
                                                  rank % NUM_MAX_NVL_PEERS,
                                                  NUM_MAX_NVL_PEERS,
                                                  num_ranks / NUM_MAX_NVL_PEERS,
                                                  &cpu_rdma_team_config,
                                                  0,
                                                  &cpu_rdma_team) == 0);
        EP_HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
    }

    nvshmem_barrier_all();
    return nvshmem_my_pe();
#endif

}

void* alloc(size_t size, size_t alignment) {
#ifdef ENABLE_NCCL_GIN
    //printf("NCCL: alloc()\n");
    internode::CommunicationBackend* backend = internode::get_backend();
    if (backend == nullptr) {
        throw std::runtime_error("Backend not initialized");
    }
    return backend->alloc(size, alignment);
#else
    return nvshmem_align(alignment, size);
#endif
}

void free(void* ptr) {
#ifdef ENABLE_NCCL_GIN
    //printf("NCCL: free()\n");
    internode::CommunicationBackend* backend = internode::get_backend();
    if (backend == nullptr) {
        throw std::runtime_error("Backend not initialized");
    }
    backend->free(ptr);
#else
    nvshmem_free(ptr);
#endif
}

void barrier() {
#ifdef ENABLE_NCCL_GIN
    printf("NCCL: barrier()\n");
    internode::CommunicationBackend* backend = internode::get_backend();
    backend->barrier();
#else
    nvshmem_barrier_all();
#endif
}

void finalize() {
#ifdef ENABLE_NCCL_GIN
    //printf("NCCL: finalize()\n");
    internode::CommunicationBackend* backend = internode::get_backend();
    if (backend) {
        backend->finalize();
    }
    internode::finalize_backend();
#else
    if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
        nvshmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = NVSHMEM_TEAM_INVALID;
    }
    nvshmem_finalize();
#endif
}

}  // namespace internode

}  // namespace deep_ep

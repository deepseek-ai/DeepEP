#include <cstring>
#include <vector>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

#ifdef ENABLE_NCCL
#include <nccl.h>

#include "nccl_gin_backend.h"

// Forward declaration for internal initialization function
namespace deep_ep {
namespace internode_ll {
void set_p2p_disabled_flag(bool disabled);
}
}  // namespace deep_ep
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

std::vector<uint8_t> get_unique_id(int qps_per_rank, int num_ranks) {
    std::vector<uint8_t> result;

#ifdef ENABLE_NCCL
    // For NCCL: Generate enough IDs for both LL and HT modes
    // At this stage, we don't know which mode will be used, so generate for worst case (HT mode)
    // - Low Latency mode: will use only first qps_per_rank IDs
    // - High Throughput mode: will use all NUM_MAX_NVL_PEERS * qps_per_rank IDs
    //   (each of the 8 color groups needs qps_per_rank IDs)

    int num_total_ids = NUM_MAX_NVL_PEERS * qps_per_rank;  // 8 * qps_per_rank

    // Generate unique IDs and pack them
    for (int i = 0; i < num_total_ids; i++) {
        ncclUniqueId unique_id;
        ncclGetUniqueId(&unique_id);

        size_t offset = result.size();
        result.resize(offset + sizeof(ncclUniqueId));
        std::memcpy(result.data() + offset, &unique_id, sizeof(ncclUniqueId));
    }

    return result;
#else
    // NVSHMEM: always return exactly 1 unique ID (qps_per_rank is ignored)
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    result.resize(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
#endif
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode, int qps_per_rank) {
    // printf("runtime::init() called\n"); fflush(stdout);
#ifdef ENABLE_NCCL
    // printf("NCCL: init()\n"); fflush(stdout);

    // For NCCL: Verify we received the correct number of unique IDs
    // We always receive NUM_MAX_NVL_PEERS * qps_per_rank IDs (generated for HT mode worst case)
    // LL mode uses only the first qps_per_rank IDs, HT mode uses all of them
    size_t expected_size = NUM_MAX_NVL_PEERS * qps_per_rank * sizeof(ncclUniqueId);
    EP_HOST_ASSERT(root_unique_id_val.size() == expected_size && "Unique ID size mismatch");

    // Initialize backend with packed unique IDs (backend will unpack based on mode)
    internode::BackendType backend_type = internode::detect_backend_type();
    internode::initialize_backend(backend_type, root_unique_id_val, rank, num_ranks, low_latency_mode, qps_per_rank);
    internode::CommunicationBackend* backend = internode::get_backend();

    // Set the device constant for P2P disabled flag based on backend configuration
    auto* nccl_backend = dynamic_cast<internode::NCCLGINBackend*>(backend);
    if (nccl_backend) {
        bool p2p_disabled = nccl_backend->is_p2p_disabled();
        internode_ll::set_p2p_disabled_flag(p2p_disabled);
    }

    backend->barrier();
    return backend->get_rank();
#else
    // printf("NVSHMEM: init()\n"); fflush(stdout);
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
#ifdef ENABLE_NCCL
    // printf("NCCL: alloc()\n");
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
#ifdef ENABLE_NCCL
    // printf("NCCL: free()\n");
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
#ifdef ENABLE_NCCL
    // printf("NCCL: barrier()\n");
    internode::CommunicationBackend* backend = internode::get_backend();
    backend->barrier();
#else
    nvshmem_barrier_all();
#endif
}

void finalize() {
#ifdef ENABLE_NCCL
    // printf("NCCL: finalize()\n");
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

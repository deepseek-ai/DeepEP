#pragma once

#include <memory>
#include <vector>
#include <optional>

#include <nccl.h>
#include <nccl_device.h>

#include "symmetric.hpp"

// TODO: make a unified API
namespace deep_ep::nvshmem {

std::vector<uint8_t> get_unique_id();

int init(const std::vector<uint8_t>& root_unique_id_val,
         const int& rank,
         const int& num_ranks,
         const int& team_split_stride);

void* alloc(const size_t& size, const size_t& alignment);

void free(void* ptr);

void barrier(const bool& with_cpu_sync, const std::optional<cudaStream_t>& stream_opt = std::nullopt);

void finalize();

}  // deep_ep::nvshmem

namespace deep_ep::nccl {

pybind11::bytearray get_local_unique_id();

int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes,
                         const int& num_ranks, const int& rank_idx);

void destroy_nccl_comm(const int64_t& nccl_comm);

std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm);

std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode);

// TODO: make it header only?
struct NCCLSymmetricMemoryContext {
private:
    // Can not use this unmapped pointer from outside
    void* raw_window_ptr;
    std::shared_ptr<symmetric::SymmetricMemory> symmetric_memory;

public:
    // Global
    int rank_idx;
    int num_ranks;

    // Logical
    int num_scaleout_ranks, num_scaleup_ranks;
    int scaleout_rank_idx, scaleup_rank_idx;

    // Physical
    int num_rdma_ranks, num_nvl_ranks;
    int rdma_rank_idx, nvl_rank_idx;
    bool is_scaleup_nvlink;

    // NCCL handles
    ncclComm_t comm;
    ncclDevComm_t dev_comm;
    ncclWindow_t window;
    void* mapped_window_ptr;
    std::vector<void*> nvl_window_ptrs;

    // Configs
    int num_allocated_qps;

    // Buffer size
    int64_t num_gpu_bytes;
    int64_t num_cpu_bytes;

    NCCLSymmetricMemoryContext(const int64_t& nccl_comm, const symmetric::cpu_comm_t& cpu_comm,
                               const int& num_ranks, const int& rank_idx,
                               const int64_t& num_bytes, const int64_t& num_cpu_bytes,
                               const bool& allow_hybrid_mode,
                               const int& sl_idx, const int& num_allocated_qps);

    // TODO: finish this with `explicit_destroy`
    // ~NCCLSymmetricMemoryContext();

    void* get_sym_ptr(void* ptr, const int& dst_rank_idx) const;

    void finalize();
};

}  // deep_ep::nccl

namespace deep_ep::cuda_driver {

void batched_write(CUstream stream, const std::vector<void*>& ptrs, const int& value);

void batched_wait(CUstream stream, const std::vector<void*>& ptrs, const int& value);

void batched_write_and_wait(CUstream stream, const std::vector<void*>& write_ptrs, const std::vector<void*>& wait_ptrs, const int& value);

}  // namespace deep_ep::cuda_driver

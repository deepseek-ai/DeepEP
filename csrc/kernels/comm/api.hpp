#pragma once

#include <memory>
#include <optional>

#include <ATen/cuda/CUDAContext.h>
#include <torch/python.h>
#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/layout/bucket/workspace.cuh>
#include <deep_jit/utils/no_ref_ptr.hpp>

#include "symmetric.hpp"

namespace deep_ep::comm {

pybind11::bytearray get_local_unique_id();

int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes,
                         const int& num_ranks, const int& rank_idx);

void destroy_nccl_comm(const int64_t& nccl_comm);

std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm);

std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode);

std::tuple<int, int> get_gin_min_stride(const int64_t& nccl_comm);

class Context {
    void* raw_window_ptr;
    std::shared_ptr<symmetric::SymmetricMemory> symmetric_memory;
    bool finalized = false;

public:
    int rank_idx, num_ranks;
    int num_scaleout_ranks, num_scaleup_ranks;
    int scaleout_rank_idx, scaleup_rank_idx;
    int num_rdma_ranks, num_nvl_ranks;
    int rdma_rank_idx, nvl_rank_idx;
    bool is_scaleup_nvlink;
    int gin_min_stride;

    ncclComm_t comm;
    deep_jit::NoRefPtr dev_comm;
    ncclWindow_t window;

    int num_allocated_qps;
    int num_cpu_timeout_secs;
    int64_t num_gpu_timeout_cycles;

    int64_t num_workspace_bytes;
    int64_t num_gpu_buffer_bytes;
    int64_t num_rdma_storage_bytes;
    bool use_cpu_rdma_storage;
    void* workspace;
    void* barrier_signals = nullptr;
    void* buffer;
    void* buffer_multimem = nullptr;
    void* local_rdma_storage_ptr = nullptr;
    std::vector<ncclWindow_t> rdma_storage_windows;
    ncclWindow_t* rdma_storage_windows_gpu = nullptr;

    Context(const int64_t& nccl_comm, const symmetric::shared_comm_t& shared_comm,
            const int& num_ranks, const int& rank_idx,
            const int64_t& num_workspace_bytes,
            const int64_t& num_gpu_buffer_bytes,
            const int64_t& num_rdma_storage_bytes,
            const bool& use_cpu_rdma_storage,
            const bool& allow_hybrid_mode,
            const std::optional<int>& sl_idx, const int& num_allocated_qps, const int& qp_depth,
            const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
            const bool& enable_lsa_multimem = false,
            const std::shared_ptr<Context>& main_context = nullptr);

    std::tuple<int, int> get_physical_domain_size() const;

    std::tuple<int, int> get_logical_domain_size() const;

    void* get_sym_ptr(void* ptr, const int& dst_nvl_rank_idx) const;

    void set_barrier_signals(void* ptr) {
        barrier_signals = ptr;
    }

    void finalize();
};

}  // namespace deep_ep::comm

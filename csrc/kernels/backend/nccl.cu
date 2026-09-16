// MIT License
//
// Copyright (c) 2025 DeepSeek
// Changes and additions copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cstring>
#include <vector>
#include <string>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <sstream>

#include <nccl.h>
#include <nccl_device/core.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include "../elastic/kernel_select.hpp"

#include "api.cuh"
#include "../../utils/system.hpp"


namespace deep_ep::nccl {

pybind11::bytearray get_local_unique_id() {
    ncclUniqueId unique_id;
    NCCL_CHECK(ncclGetUniqueId(&unique_id));
    std::vector<char> result(sizeof(ncclUniqueId));
    std::memcpy(result.data(), &unique_id, sizeof(ncclUniqueId));
    return {result.data(), result.size()};
}

int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes,
                         const int& num_ranks, const int& rank_idx) {
    // Copy unique ID
    ncclUniqueId root_unique_id;
    const auto root_unique_id_str = root_unique_id_bytes.cast<std::string>();
    std::memcpy(&root_unique_id, root_unique_id_str.c_str(), sizeof(ncclUniqueId));

    // Init
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, root_unique_id, rank_idx));
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("New NCCL host communicator created (%d/%d)\n", rank_idx, num_ranks);
    return reinterpret_cast<int64_t>(comm);
}

void destroy_nccl_comm(const int64_t& nccl_comm) {
    NCCL_CHECK(ncclCommAbort(reinterpret_cast<ncclComm_t>(nccl_comm)));
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("NCCL host communicator aborted\n");
}

std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm) {
    const auto comm = reinterpret_cast<ncclComm_t>(nccl_comm);
    const int num_ranks = ncclTeamWorld(comm).nRanks, num_nvl_ranks = ncclTeamLsa(comm).nRanks;
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0);
    return {num_ranks / num_nvl_ranks, num_nvl_ranks};
}

std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode) {
    const auto [num_rdma_ranks, num_nvl_ranks] = get_physical_domain_size(nccl_comm);
    return {allow_hybrid_mode ? num_rdma_ranks : 1,
            allow_hybrid_mode ? num_nvl_ranks : num_rdma_ranks * num_nvl_ranks};
}

NCCLSymmetricMemoryContext::NCCLSymmetricMemoryContext(const int64_t& nccl_comm, const symmetric::cpu_comm_t& cpu_comm,
                                                       const int& num_ranks, const int& rank_idx,
                                                       const int64_t& num_bytes, const int64_t& num_cpu_bytes,
                                                       const bool& allow_hybrid_mode,
                                                       const int& sl_idx, const int& num_allocated_qps):
    rank_idx(rank_idx), num_ranks(num_ranks), num_allocated_qps(num_allocated_qps) {
    int nccl_runtime_version;
    NCCL_CHECK(ncclGetVersion(&nccl_runtime_version));
    if (get_env("EP_BUFFER_DEBUG", 0)) {
        printf("DeepEP initialized with NCCL version: %d.%d.%d (loaded library)\n",
               nccl_runtime_version / 10000, (nccl_runtime_version % 10000) / 100, nccl_runtime_version % 100);
    }

    // Reuse the NCCL communicator
    comm = reinterpret_cast<ncclComm_t>(nccl_comm);

    // Print number of allocated QPs
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("EP NCCL device communicator has %d allocated QPs\n", num_allocated_qps);

    const bool gin_disabled = get_env("EP_DISABLE_GIN", 0) != 0;

    // Initialize NCCL device communicator
    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    if (num_ranks > 1 and not gin_disabled) {
        EP_HOST_ASSERT(
            (allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE and
            "NCCL GIN is unavailable. This is usually due to a network configuration issue, "
            "such as `allow_hybrid_mode=0` (disable direct RDMA kernels) in multi-plane network.");

        const int num_rdma_ranks = ncclTeamRail(comm).nRanks;
        EP_HOST_ASSERT(num_ranks == ncclTeamLsa(comm).nRanks * num_rdma_ranks);

        const bool scaleout_active = num_rdma_ranks > 1;

        // Only the unordered hybrid kernels use the shared-context weak-signal GIN
        // configuration; direct mode and the ordered hybrid kernels keep the
        // upstream requirements untouched.
        if (allow_hybrid_mode and not elastic::use_ordered_hybrid_kernel()) {
        auto resolve_gin_context_cnt = [&]() -> int {
            const int requested = (this->num_allocated_qps == 0)
                ? elastic::gin_alloc::kDefaultGinContextCnt
                : this->num_allocated_qps;
            int ctx = requested;
            if (ctx < elastic::gin_alloc::kMinGinContextCnt)
                ctx = elastic::gin_alloc::kMinGinContextCnt;
            if (ctx > elastic::gin_alloc::kMaxGinContextCnt)
                ctx = elastic::gin_alloc::kMaxGinContextCnt;
            if (ctx != requested)
                printf("[WARN] DeepEP clamped num_allocated_qps from %d to %d: the unordered "
                       "GIN layout supports [%d, %d] contexts (one GIN context supplies one QP)\n",
                       requested, ctx,
                       elastic::gin_alloc::kMinGinContextCnt,
                       elastic::gin_alloc::kMaxGinContextCnt);
            return ctx;
        };

        if (scaleout_active) {
            gin_config = elastic::gin_alloc::make_gin_resources(resolve_gin_context_cnt());
        } else {
            gin_config.gin_context_cnt = resolve_gin_context_cnt();
            gin_config.gin_indexed_signals_cnt = 0;
        }

        EP_HOST_ASSERT(gin_config.gin_indexed_signals_cnt >= (num_rdma_ranks - 1) and
                       "GIN indexed-signal budget cannot give each peer rail team a dedicated "
                       "signal; reduce num_allocated_qps to raise the per-context signal count");

        this->num_allocated_qps = gin_config.gin_context_cnt;

        if (get_env<int>("EP_BUFFER_DEBUG"))
            printf("GIN layout: gin_context_cnt=%d, gin_indexed_signals_cnt=%d, num_qp=%d\n",
                   gin_config.gin_context_cnt,
                   gin_config.gin_indexed_signals_cnt,
                   this->num_allocated_qps);

        reqs.ginExclusiveContexts = false;
        reqs.ginQueueDepth = kGinQPDepth;
        reqs.ginTrafficClass = sl_idx;
        if (scaleout_active) {
            reqs.ginContextCount = gin_config.gin_context_cnt;
            reqs.ginSignalCount = gin_config.gin_indexed_signals_cnt;
        } else if (gin_config.gin_context_cnt > 0) {
            reqs.ginContextCount = gin_config.gin_context_cnt;
        }

        // The unordered kernels synchronize through counting signals only; VA and
        // strong signals are not required.
        reqs.ginVaSignalsRequired = false;
        reqs.ginStrongSignalsRequired = false;
        } else {
            // Direct-mode / ordered-kernel path: the upstream GIN requirements,
            // unchanged. The automatic QP count is resolved on the Python side
            // (65/129 for hybrid mode by fast-RDMA-atomic support, 17 otherwise).
            reqs.ginContextCount = this->num_allocated_qps;
            reqs.ginExclusiveContexts = true;
            reqs.ginQueueDepth = kGinQPDepth;
            reqs.ginTrafficClass = sl_idx;
            // Customized RDMA barrier needs extra signals
            reqs.ginSignalCount = num_ranks + 2 * 2;
        }

        reqs.ginConnectionType = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL: NCCL_GIN_CONNECTION_FULL;
    }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)
    reqs.useRuntimeVersion = true;
    dev_comm.ptr = malloc(props.devCommRuntimeVersionSize);
#else
    EP_HOST_ASSERT(NCCL_VERSION_CODE == nccl_runtime_version and "Prior to NCCL 2.31, NCCL compile-time and runtime versions must be the same. Please re-compile DeepEP.");
    dev_comm.ptr = malloc(sizeof(ncclDevComm_t));
#endif
    EP_HOST_ASSERT(dev_comm.ptr != nullptr);
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, static_cast<ncclDevComm_t*>(dev_comm.ptr)));

    // Now we know the NVLink domain size
    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    num_nvl_ranks = lsaTeam.nRanks, nvl_rank_idx = lsaTeam.rank;
    num_rdma_ranks = num_ranks / num_nvl_ranks, rdma_rank_idx = rank_idx / num_nvl_ranks;
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0 and nvl_rank_idx == rank_idx % num_nvl_ranks);
    EP_HOST_ASSERT(rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx);

    // Calculate scaleout/up domain size
    if (allow_hybrid_mode) {
        num_scaleout_ranks = num_rdma_ranks, num_scaleup_ranks = num_nvl_ranks;
        scaleout_rank_idx = rdma_rank_idx, scaleup_rank_idx = nvl_rank_idx;
    } else {
        num_scaleout_ranks = 1, num_scaleup_ranks = num_ranks;
        scaleout_rank_idx = 0, scaleup_rank_idx = rank_idx;
    }
    is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;

    // Create symmetric memory
    // num_bytes = GPU + CPU, derive GPU portion
    this->symmetric_memory = symmetric::alloc(
        num_bytes - num_cpu_bytes, num_cpu_bytes,
        allow_hybrid_mode, num_scaleup_ranks, scaleout_rank_idx,
        cpu_comm);

    // Create window
    // NOTES: `ncclCommWindowRegister` is collective: it internally calls bootstrapBarrier
    // across all ranks, so no explicit barrier is needed after this call.
    raw_window_ptr = this->symmetric_memory->ptr;
    this->num_gpu_bytes = this->symmetric_memory->num_gpu_bytes;
    this->num_cpu_bytes = this->symmetric_memory->num_cpu_bytes;
    NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, this->symmetric_memory->num_bytes, &window, NCCL_WIN_STRICT_ORDERING));
    NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr));

    // Get LSA pointers for all LSA peers
    // TODO: check whether this is correct for network with RDMA
    nvl_window_ptrs.resize(num_nvl_ranks);
    for (int i = 0; i < num_nvl_ranks; ++ i)
        NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, i, &nvl_window_ptrs[i]));
}

void* NCCLSymmetricMemoryContext::get_sym_ptr(void* ptr, const int& dst_rank_idx) const {
    const auto offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(mapped_window_ptr);
    return static_cast<uint8_t*>(nvl_window_ptrs[dst_rank_idx]) + offset;
}

void NCCLSymmetricMemoryContext::finalize() {
    // Deregister window
    NCCL_CHECK(ncclCommWindowDeregister(comm, window));
    symmetric_memory.reset();

    // Destroy device communicator
    NCCL_CHECK(ncclDevCommDestroy(comm, static_cast<ncclDevComm_t*>(dev_comm.ptr)));
    free(dev_comm.ptr);
}

}  // namespace deep_ep::nccl

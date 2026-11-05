#pragma once

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>
#include <pybind11/functional.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/layout/engram/workspace.cuh>

#include "base.hpp"
#include "../comm/api.hpp"
#include "../kernels/engram/engram.hpp"
#include "../utils/tensor.hpp"

namespace deep_ep::engram {

class EngramBuffer: public BufferBase {
    static constexpr int kFlushDepthReserve = 256;

    bool allow_hybrid_mode;
    int qp_depth;

    std::vector<int> num_entries_per_layer;
    torch::ScalarType sf_dtype;
    int hidden = 0, elem_size = 0;
    int num_max_tokens = 0;
    int num_sf_packs = 0;
    int64_t num_sf_shard_bytes = 0;

public:
    std::shared_ptr<comm::Context> context;

    EngramBuffer(const int& rank_idx, const int& num_ranks,
                 const int64_t& nccl_comm, const symmetric::shared_comm_t& shared_comm,
                 const int64_t& num_gpu_buffer_bytes, const int64_t& num_rdma_storage_bytes,
                 const bool& use_cpu_rdma_storage,
                 const bool& allow_hybrid_mode,
                 const std::optional<int>& sl_idx, const int& num_allocated_qps, const int& qp_depth,
                 const int& num_gpu_timeout_secs,
                 const bool& explicitly_destroy):
        BufferBase(explicitly_destroy),
        allow_hybrid_mode(allow_hybrid_mode),
        qp_depth(qp_depth) {
        const auto num_workspace_bytes = math::align<int64_t>(
            sizeof(layout::EngramWorkspace),
            kNumAllocationAlignmentBytes);
        context = std::make_shared<comm::Context>(
            nccl_comm, shared_comm, num_ranks, rank_idx,
            num_workspace_bytes, num_gpu_buffer_bytes, num_rdma_storage_bytes, use_cpu_rdma_storage,
            allow_hybrid_mode, sl_idx, num_allocated_qps, qp_depth,
            0, num_gpu_timeout_secs);
        main_context = context;
        auto& workspace = *static_cast<layout::EngramWorkspace*>(context->workspace);
        context->set_barrier_signals(&workspace.barrier);
    }

    ~EngramBuffer() noexcept(false) override {
        destroy_on_destruction("Engram");
    }

    void destroy() override {
        EP_HOST_ASSERT(not destroyed);
        comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                      context->num_gpu_timeout_cycles, true, true);
        context->finalize();
        context.reset();
        destroyed = true;
    }

    void set_config(const std::vector<int>& new_num_entries_per_layer,
                    const int& new_hidden, const int& new_elem_size,
                    const int& new_num_max_tokens, const int& num_entries_per_token,
                    const int& new_num_sf_packs) {
        comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                      context->num_gpu_timeout_cycles, true, true);

        EP_HOST_ASSERT(new_num_max_tokens > 0 and num_entries_per_token > 0);
        EP_HOST_ASSERT(new_num_entries_per_layer.size() <= layout::EngramWorkspace::kNumMaxLayers);
        EP_HOST_ASSERT(new_hidden * new_elem_size % kNumRDMAAlignmentBytes == 0);
        int64_t num_total_entries = 0;
        for (const auto& num_entries: new_num_entries_per_layer) {
            EP_HOST_ASSERT(num_entries > 0);
            num_total_entries += num_entries;
        }
        EP_HOST_ASSERT(num_total_entries * new_hidden * new_elem_size <= context->num_rdma_storage_bytes);

        const auto num_recv_bytes = static_cast<int64_t>(new_num_entries_per_layer.size()) *
                                    new_num_max_tokens * num_entries_per_token * new_hidden * new_elem_size;
        num_sf_shard_bytes = new_num_sf_packs == 0 ? 0 : math::align<int64_t>(
            num_total_entries * context->num_rdma_ranks * new_num_sf_packs * sizeof(sf_pack_t),
            kNumAllocationAlignmentBytes);
        EP_HOST_ASSERT(num_recv_bytes + num_sf_shard_bytes <= context->num_gpu_buffer_bytes);

        num_entries_per_layer = new_num_entries_per_layer;
        hidden = new_hidden, elem_size = new_elem_size;
        num_max_tokens = new_num_max_tokens;
        num_sf_packs = new_num_sf_packs;
    }

    void write(const std::vector<torch::Tensor>& storages,
               const std::optional<std::vector<torch::Tensor>>& sfs) {
        comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                      context->num_gpu_timeout_cycles, true, true);

        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        const auto num_layers = num_entries_per_layer.size();
        EP_HOST_ASSERT(num_layers > 0 and storages.size() == num_layers);

        auto* storage_ptr = context->local_rdma_storage_ptr;
        for (int i = 0; i < num_layers; ++ i) {
            const auto& storage = storages[i];
            const auto [num_entries, storage_hidden] = get_shape<2>(storage);
            EP_HOST_ASSERT(num_entries == num_entries_per_layer[i] and storage_hidden == hidden);
            EP_HOST_ASSERT(storage.scalar_type() == torch::kBFloat16 or
                           storage.scalar_type() == torch::kFloat8_e4m3fn);
            EP_HOST_ASSERT(storage.is_cuda() and storage.is_contiguous());
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                storage_ptr,
                storage.data_ptr(), storage.nbytes(),
                cudaMemcpyDefault, compute_stream));
            storage_ptr = math::advance_ptr(storage_ptr, storage.nbytes());
        }

        if (sfs.has_value()) {
            EP_HOST_ASSERT(sfs->size() == num_layers);
            sf_dtype = sfs->front().scalar_type();
            auto* shard_ptr = math::advance_ptr(context->buffer,
                                                context->num_gpu_buffer_bytes - num_sf_shard_bytes);
            for (int i = 0; i < num_layers; ++ i) {
                const auto& sf = sfs->at(i);
                const auto [num_rows, num_packs] = get_shape<2>(sf);
                EP_HOST_ASSERT(sf.is_cuda() and sf.is_contiguous() and sf.element_size() == sizeof(sf_pack_t));
                EP_HOST_ASSERT(num_packs == num_sf_packs);

                const auto num_shard_bytes = sf.nbytes() / context->num_nvl_ranks;
                CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                    shard_ptr,
                    math::advance_ptr(sf.data_ptr(), context->nvl_rank_idx * num_shard_bytes),
                    num_shard_bytes, cudaMemcpyDefault, compute_stream));
                shard_ptr = math::advance_ptr(shard_ptr, num_shard_bytes);
            }
        }

        comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                      context->num_gpu_timeout_cycles, true, true);
    }

    std::vector<std::function<pybind11::object()>>
    fetch(const torch::Tensor& indices, const int& num_qps,
          const bool& use_tma_aligned_col_major_sf) const {
        const auto use_fp8 = num_sf_packs > 0;
        const auto fetched_dtype = use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16;
        const auto [num_layers, num_tokens, num_entries_per_token] = get_shape<3>(indices);
        EP_HOST_ASSERT(indices.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(indices.is_cuda() and indices.is_contiguous());
        EP_HOST_ASSERT(num_layers == num_entries_per_layer.size());
        EP_HOST_ASSERT(num_tokens <= num_max_tokens);

        const auto num_layer_bytes = static_cast<int64_t>(num_max_tokens) * num_entries_per_token *
                                     hidden * elem_size;
        std::vector<torch::Tensor> fetched;
        for (int i = 0; i < num_layers; ++ i)
            fetched.push_back(torch::from_blob(
                math::advance_ptr(context->buffer, i * num_layer_bytes),
                {num_tokens, hidden * num_entries_per_token},
                torch::TensorOptions().dtype(fetched_dtype).device(torch::kCUDA)
            ));

        std::vector<torch::Tensor> fetched_sf;
        torch::Tensor sf_backing;
        void* sf_shards_ptr = nullptr;
        void* fetched_sf_ptr = nullptr;
        int sf_token_stride = 0, sf_hidden_stride = 0, sf_layer_stride = 0;
        if (use_fp8) {
            if (use_tma_aligned_col_major_sf) {
                sf_token_stride = 1, sf_hidden_stride = math::align(num_tokens, kNumAlignedSFPacks);
            } else {
                sf_token_stride = num_entries_per_token * num_sf_packs, sf_hidden_stride = 1;
            }
            sf_layer_stride = use_tma_aligned_col_major_sf
                ? num_entries_per_token * num_sf_packs * sf_hidden_stride
                : num_entries_per_token * num_sf_packs * num_tokens;
            sf_backing = torch::empty({num_layers, sf_layer_stride},
                torch::TensorOptions().dtype(sf_dtype).device(torch::kCUDA));
            for (int i = 0; i < num_layers; ++ i)
                fetched_sf.push_back(sf_backing[i].as_strided({num_tokens, num_entries_per_token * num_sf_packs},
                                                               {sf_token_stride, sf_hidden_stride}));
            fetched_sf_ptr = sf_backing.data_ptr();
            sf_shards_ptr = math::advance_ptr(context->buffer,
                                              context->num_gpu_buffer_bytes - num_sf_shard_bytes);
        }

        EP_HOST_ASSERT(context->num_ranks % context->gin_min_stride == 0);
        const auto num_rdma_peers = context->num_ranks / context->gin_min_stride;
        EP_HOST_ASSERT(num_rdma_peers * num_qps <= layout::EngramWorkspace::kNumMaxPeerContexts);
        const auto num_active_gin_contexts =
            std::min(num_qps, math::ceil_div(num_tokens, 32));

        const auto last_gin_requests = torch::empty(
            {num_layers, num_rdma_peers * num_qps, sizeof(ncclGinRequest_t)},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
        );

        launch_engram_fetch(
            context->dev_comm, context->window,
            context->rdma_storage_windows_gpu,
            context->buffer,
            context->workspace,
            indices.data_ptr<int>(),
            static_cast<ncclGinRequest_t*>(last_gin_requests.data_ptr()),
            sf_shards_ptr, fetched_sf_ptr,
            sf_token_stride, sf_hidden_stride, sf_layer_stride,
            context->num_nvl_ranks,
            num_entries_per_layer,
            hidden, elem_size, num_sf_packs,
            num_entries_per_token,
            num_tokens, num_max_tokens,
            num_rdma_peers,
            context->gin_min_stride,
            num_qps, qp_depth - kFlushDepthReserve,
            context->num_gpu_timeout_cycles,
            at::cuda::getCurrentCUDAStream()
        );

        std::vector<std::function<pybind11::object()>> hooks;
        for (int i = 0; i < num_layers; ++ i) {
            hooks.emplace_back([=, this]() -> pybind11::object {
                launch_engram_fetch_wait(
                    static_cast<ncclGinRequest_t*>(last_gin_requests[i].data_ptr()),
                    context->dev_comm,
                    context->window,
                    num_rdma_peers,
                    num_active_gin_contexts,
                    at::cuda::getCurrentCUDAStream()
                );
                return use_fp8 ? pybind11::make_tuple(fetched[i], fetched_sf[i]) : pybind11::cast(fetched[i]);
            });
        }
        return hooks;
    }

    static std::tuple<int64_t, int64_t> calculate_storage_size(
            const int64_t& nccl_comm,
            const std::vector<int>& num_entries_per_layer, const int& hidden, const int& elem_size,
            const int& num_max_tokens, const int& num_entries_per_token, const int& num_sf_packs) {
        const auto [num_rdma_ranks, num_nvl_ranks] = comm::get_physical_domain_size(nccl_comm);
        const auto num_total_entries = std::reduce(
            num_entries_per_layer.begin(), num_entries_per_layer.end(), int64_t(0));

        const auto num_bytes_per_entry = hidden * elem_size;
        EP_HOST_ASSERT(num_bytes_per_entry % kNumRDMAAlignmentBytes == 0);
        const auto num_recv_bytes = math::align<int64_t>(
            num_entries_per_layer.size() * num_max_tokens * num_entries_per_token * num_bytes_per_entry,
            kNumAllocationAlignmentBytes);
        const auto num_rdma_storage_bytes = math::align<int64_t>(
            num_total_entries * num_bytes_per_entry, kNumAllocationAlignmentBytes);

        const auto num_sf_bytes = num_sf_packs == 0 ? 0 : math::align<int64_t>(
            num_total_entries * num_rdma_ranks * num_sf_packs * sizeof(sf_pack_t),
            kNumAllocationAlignmentBytes);
        return {num_recv_bytes + num_sf_bytes, num_rdma_storage_bytes};
    }

    static std::tuple<int, int, bool> get_theoretical_config(const int64_t& nccl_comm,
                                                       const int& num_layers,
                                                       const int& num_max_tokens,
                                                       const int& num_entries_per_token,
                                                       const float& nic_read_mpps,
                                                       const int& num_max_read_requests_per_qp) {
        constexpr float kReadRTTSeconds = 10e-6f;

        EP_HOST_ASSERT(num_layers > 0 and num_max_tokens > 0 and num_entries_per_token > 0 and
                       nic_read_mpps > 0 and num_max_read_requests_per_qp > 0);

        const auto [_, num_rdma_peers] = comm::get_gin_min_stride(nccl_comm);

        const auto num_qps = std::max(static_cast<int>(std::ceil(
            nic_read_mpps * 1e6f * kReadRTTSeconds / num_max_read_requests_per_qp / num_rdma_peers)), 1);

        const auto num_requests = num_layers * num_max_tokens * num_entries_per_token;
        const auto num_required_slots = math::ceil_div(num_requests, num_qps * num_rdma_peers);
        if (num_required_slots > 32768) {
            const auto num_min_qps = static_cast<int>(math::ceil_div(num_requests, 32768 * num_rdma_peers) * 1.05);
            return {num_min_qps, 32768, true};
        }

        auto qp_depth = kDefaultQPDepth;
        while (qp_depth < num_required_slots and qp_depth < 32768) qp_depth *= 2;
        return {num_qps, qp_depth, false};
    }

    static symmetric::shared_handle_t create_shared_handle(const int64_t& num_rdma_storage_bytes,
                                                           const bool& use_cpu_rdma_storage) {
        return symmetric::HybridElasticSymmetricMemory::create_shared_handle(num_rdma_storage_bytes, use_cpu_rdma_storage);
    }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<EngramBuffer, BufferBase>(m, "EngramBuffer")
        .def(pybind11::init<int, int, int64_t, symmetric::shared_comm_t, int64_t, int64_t, bool, bool, std::optional<int>, int, int, int, bool>())
        .def_readonly("context", &EngramBuffer::context)
        .def("set_config", &EngramBuffer::set_config)
        .def("write", &EngramBuffer::write)
        .def("fetch", &EngramBuffer::fetch)
        .def_static("create_shared_handle", &EngramBuffer::create_shared_handle)
        .def_static("calculate_storage_size", &EngramBuffer::calculate_storage_size)
        .def_static("get_theoretical_config", &EngramBuffer::get_theoretical_config);
}

}  // namespace deep_ep::engram

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/layout/common/barrier.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>
#include <deep_ep/layout/bucket/reduce_scatter.cuh>

#include "base.hpp"
#include "../comm/api.hpp"
#include "../comm/barrier.hpp"
#include "../comm/stream.hpp"
#include "../kernels/bucket/all_gather.hpp"
#include "../kernels/bucket/all_reduce.hpp"
#include "../kernels/bucket/reduce_scatter.hpp"
#include "../runtime/jit.hpp"
#include "../utils/event.hpp"

namespace deep_ep::bucket {

class BucketBuffer: public BufferBase {
public:
    std::vector<std::shared_ptr<comm::Context>> contexts;
    torch::Tensor storage;

private:
    int64_t num_storage_bytes;
    float nvl_bandwidth, rdma_bandwidth;

    // Allocator for operation indices in sequence
    mutable std::array<uint64_t, kNumMaxContexts> seq_indices = {};

    static EventHandle stream_control_epilogue(const std::vector<torch::Tensor>& tensors = {}) {
        const auto comm_stream = comm::get_comm_stream();
        auto event = EventHandle(comm_stream);
        if (get_env<int>("EP_AVOID_RECORD_STREAM", 0)) {
            event.tensors_to_record.assign(tensors.begin(), tensors.end());
        } else {
            for (const auto& tensor: tensors)
                tensor.record_stream(comm_stream);
        }
        return event;
    }

public:
    BucketBuffer(const std::vector<int>& rank_indices,
                 const std::vector<int>& num_ranks,
                 const std::vector<int64_t>& nccl_comms,
                 const int64_t& num_storage_bytes,
                 const std::optional<int>& sl_idx,
                 const int& num_gpu_timeout_secs,
                 const float& nvl_gbs, const float& rdma_gbs,
                 const bool& explicitly_destroy):
        BufferBase(explicitly_destroy),
        num_storage_bytes(num_storage_bytes),
        nvl_bandwidth(nvl_gbs * 1e9f), rdma_bandwidth(rdma_gbs * 1e9f) {
        EP_HOST_ASSERT(nvl_gbs > 0 and rdma_gbs > 0);
        const auto num_groups = static_cast<int>(nccl_comms.size());
        EP_HOST_ASSERT(num_groups > 0 and rank_indices.size() == num_groups and num_ranks.size() == num_groups);
        EP_HOST_ASSERT(num_groups <= kNumMaxContexts and
                       "Too many contexts for the bucket workspace");
        EP_HOST_ASSERT(num_storage_bytes > 0 and num_storage_bytes % kNumAllocationAlignmentBytes == 0);

        const auto stream = at::cuda::getCurrentCUDAStream();
        const auto num_workspace_bytes = math::align<int64_t>(
            sizeof(layout::BucketWorkspace),
            kNumAllocationAlignmentBytes);
        contexts.reserve(num_groups);
        for (int i = 0; i < num_groups; ++ i) {
            const auto [num_rdma_ranks, num_nvl_ranks] = comm::get_physical_domain_size(nccl_comms[i]);
            const auto num_allocated_qps = num_rdma_ranks == 1 ? 1 : kNumMaxQPs / num_rdma_ranks;
            contexts.emplace_back(std::make_shared<comm::Context>(
                nccl_comms[i], symmetric::shared_comm_t{}, num_ranks[i], rank_indices[i],
                num_workspace_bytes, num_storage_bytes, 0, true,
                true, sl_idx, num_allocated_qps,
                0, 0, num_gpu_timeout_secs, num_nvl_ranks > 1,
                contexts.empty() ? nullptr : contexts.front()));
            auto& workspace = *static_cast<layout::BucketWorkspace*>(contexts.back()->workspace);
            contexts.back()->set_barrier_signals(&workspace.signals[i].barrier);

            // All ranks of a group must have the same context id
            const auto gathered = torch::empty({num_ranks[i]}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
            const auto send = torch::full({1}, i, gathered.options());
            NCCL_CHECK(ncclAllGather(send.data_ptr<int>(), gathered.data_ptr<int>(), 1, ncclInt, reinterpret_cast<ncclComm_t>(nccl_comms[i]), stream));
            EP_HOST_ASSERT(torch::eq(gathered, i).all().item<bool>());
        }
        main_context = contexts.front();
        storage = torch::from_blob(
            main_context->buffer,
            {num_storage_bytes},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    }

    ~BucketBuffer() noexcept(false) override {
        destroy_on_destruction("bucket");
    }

    std::tuple<std::vector<torch::Tensor>, EventHandle, int> reduce_scatter(
        const std::vector<torch::Tensor>& srcs,
        const int& context_idx,
        const int& num_sms,
        const std::string& comm_precision,
        const float& scale) const {
        EP_HOST_ASSERT(not destroyed);

        // Check config
        const auto num_buckets = static_cast<int>(srcs.size());
        EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets);
        EP_HOST_ASSERT(context_idx >= 0 and context_idx < static_cast<int>(contexts.size()));
        EP_HOST_ASSERT((comm_precision == "fp32" or comm_precision == "bf16") and
                       "Reduce-scatter comm_precision must be fp32 or bf16");
        EP_HOST_ASSERT(num_sms > 0 and "0-SM reduce-scatter is not implemented");
        EP_HOST_ASSERT(num_sms <= jit->device.get_num_sms());

        // Check context
        const auto& context = contexts[context_idx];

        // Create bucket list
        const auto storage_begin = reinterpret_cast<uintptr_t>(storage.data_ptr());
        const auto storage_end = storage_begin + num_storage_bytes;
        comm::BucketList buckets;
        std::vector<torch::Tensor> dsts;
        dsts.reserve(srcs.size());
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            // TODO: support other dtypes
            const auto& src = srcs[bucket_idx];
            EP_HOST_ASSERT(src.is_cuda() and src.is_contiguous());
            EP_HOST_ASSERT(src.scalar_type() == torch::kFloat and src.numel() > 0);
            EP_HOST_ASSERT(src.nbytes() % (context->num_ranks * sizeof(longlong4_t)) == 0);
            EP_HOST_ASSERT(src.nbytes() % (context->num_ranks * kNumTMAAlignmentBytes) == 0);

            // TODO: check overlapping
            const auto src_begin = reinterpret_cast<uintptr_t>(src.data_ptr());
            const auto src_end = src_begin + src.nbytes();
            EP_HOST_ASSERT(src_begin >= storage_begin and src_end <= storage_end and
                           "Input tensors must belong to this BucketBuffer");
            EP_HOST_ASSERT(src_begin % alignof(longlong4_t) == 0);
            EP_HOST_ASSERT(src_begin % kNumTMAAlignmentBytes == 0);

            // TODO: support >1D slicing
            const auto shard_numel = src.numel() / context->num_ranks;
            auto dst = src.view({-1}).slice(
                0, context->rank_idx * shard_numel, (context->rank_idx + 1) * shard_numel);

            // Create bucket (shard view)
            buckets[bucket_idx] = {
                .offset = static_cast<int64_t>(src_begin - storage_begin),
                .num_bytes = static_cast<int64_t>(dst.nbytes()),
            };
            dsts.push_back(dst);
        }

        // Launch
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        const auto comm_stream = comm::get_comm_stream();
        comm::stream_wait(comm_stream, compute_stream);
        launch_reduce_scatter(
            context->dev_comm, context->window,
            context->workspace, context->buffer, context->buffer_multimem,
            num_buckets, buckets,
            context->num_rdma_ranks, context->num_nvl_ranks,
            num_sms, context->num_allocated_qps, context->num_gpu_timeout_cycles,
            scale, comm_precision, context_idx, comm_stream);

        auto event = stream_control_epilogue();
        return {dsts, event, num_sms};
    }

    std::tuple<std::vector<torch::Tensor>, EventHandle, int> all_reduce(
        const std::vector<torch::Tensor>& srcs,
        const int& context_idx,
        const int& num_sms,
        const float& scale) const {
        EP_HOST_ASSERT(not destroyed);

        // Check config
        const auto num_buckets = static_cast<int>(srcs.size());
        EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets);
        EP_HOST_ASSERT(context_idx >= 0 and context_idx < static_cast<int>(contexts.size()));
        EP_HOST_ASSERT(num_sms > 0 and "0-SM all-reduce is not implemented");
        EP_HOST_ASSERT(num_sms <= jit->device.get_num_sms());

        // Check context
        const auto& context = contexts[context_idx];
        EP_HOST_ASSERT(context->num_nvl_ranks == 1 or context->buffer_multimem != nullptr);

        // Create bucket list
        const auto storage_begin = reinterpret_cast<uintptr_t>(storage.data_ptr());
        const auto storage_end = storage_begin + num_storage_bytes;
        comm::BucketList buckets;
        std::vector<torch::Tensor> dsts;
        dsts.reserve(num_buckets);
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& src = srcs[bucket_idx];
            EP_HOST_ASSERT(src.is_cuda() and src.is_contiguous());
            EP_HOST_ASSERT(src.scalar_type() == torch::kFloat and src.numel() > 0);
            // Each rank reduces one shard (numel / num_ranks) of float4 vectors
            EP_HOST_ASSERT(src.nbytes() % (context->num_ranks * sizeof(float4)) == 0);
            EP_HOST_ASSERT(src.nbytes() % (context->num_ranks * kNumTMAAlignmentBytes) == 0);

            const auto src_begin = reinterpret_cast<uintptr_t>(src.data_ptr());
            const auto src_end = src_begin + src.nbytes();
            EP_HOST_ASSERT(src_begin >= storage_begin and src_end <= storage_end and
                           "Input tensors must belong to this BucketBuffer");
            EP_HOST_ASSERT(src_begin % alignof(float4) == 0);
            EP_HOST_ASSERT(src_begin % kNumTMAAlignmentBytes == 0);

            // In-place all-reduce: the full reduced tensor overwrites the source
            // Create bucket (full view)
            buckets[bucket_idx] = {
                .offset = static_cast<int64_t>(src_begin - storage_begin),
                .num_bytes = static_cast<int64_t>(src.nbytes()),
            };
            dsts.push_back(src.view({-1}));
        }

        // Launch
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        const auto comm_stream = comm::get_comm_stream();
        comm::stream_wait(comm_stream, compute_stream);
        launch_all_reduce(
            *context, num_buckets, buckets, context_idx,
            num_sms, context->num_gpu_timeout_cycles, scale, comm_stream);

        auto event = stream_control_epilogue();
        return {dsts, event, num_sms};
    }

    // In-place without `dsts`, otherwise `srcs` can be out of the storage (pure NVLink only)
    pybind11::tuple all_gather(
        const std::vector<torch::Tensor>& srcs,
        const std::vector<torch::Tensor>& dsts,
        const int& context_idx,
        const int& num_sms) const {
        EP_HOST_ASSERT(not destroyed);

        // Check config
        const auto num_buckets = static_cast<int>(srcs.size());
        EP_HOST_ASSERT(num_buckets > 0 and num_buckets <= comm::kNumMaxBuckets);
        EP_HOST_ASSERT(dsts.empty() or static_cast<int>(dsts.size()) == num_buckets);
        EP_HOST_ASSERT(context_idx >= 0 and context_idx < static_cast<int>(contexts.size()));
        EP_HOST_ASSERT(num_sms == 0);

        // Check context
        const auto& context = contexts[context_idx];

        // Create bucket list
        comm::BucketList buckets;
        auto src_buckets = dsts.empty() ? std::nullopt : std::make_optional<comm::BucketList>();
        std::vector<torch::Tensor> gathered;
        gathered.reserve(num_buckets);
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& src = srcs[bucket_idx];
            EP_HOST_ASSERT(src.is_cuda() and src.is_contiguous());
            EP_HOST_ASSERT(src.numel() > 0);
            EP_HOST_ASSERT(dsts.empty() or (dsts[bucket_idx].is_cuda() and dsts[bucket_idx].is_contiguous()));

            const auto num_shard_bytes = static_cast<int64_t>(src.nbytes());
            const auto src_offset = math::ptr_diff(src.data_ptr(), storage.data_ptr());
            const auto bucket_offset = dsts.empty() ? src_offset - context->rank_idx * num_shard_bytes :
                                                     math::ptr_diff(dsts[bucket_idx].data_ptr(), storage.data_ptr());
            EP_HOST_ASSERT(dsts.empty() or static_cast<int64_t>(dsts[bucket_idx].nbytes()) == context->num_ranks * num_shard_bytes);
            EP_HOST_ASSERT((src_offset == bucket_offset + context->rank_idx * num_shard_bytes or
                            src_offset + num_shard_bytes <= bucket_offset or
                            src_offset >= bucket_offset + context->num_ranks * num_shard_bytes) and
                           "Input tensors overlapping with the output must be the local shard");
            EP_HOST_ASSERT(bucket_offset >= 0 and
                           bucket_offset + context->num_ranks * num_shard_bytes <= num_storage_bytes and
                           "Output tensors must belong to this BucketBuffer");

            // Create bucket (shard view)
            buckets[bucket_idx] = {
                .offset = bucket_offset,
                .num_bytes = num_shard_bytes,
            };
            if (src_buckets.has_value()) {
                // External sources must be decoded against the selected context's buffer
                const auto src_in_buffer = src_offset >= 0 and src_offset + num_shard_bytes <= num_storage_bytes;
                (*src_buckets)[bucket_idx] = {
                    .offset = src_in_buffer ? src_offset : math::ptr_diff(src.data_ptr(), context->buffer),
                    .num_bytes = num_shard_bytes,
                };
            }

            gathered.push_back(torch::from_blob(
                math::advance_ptr(storage.data_ptr(), bucket_offset),
                {src.numel() * context->num_ranks},
                torch::TensorOptions().dtype(src.dtype()).device(src.device())));
        }

        // Launch
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        const auto comm_stream = comm::get_comm_stream();
        comm::stream_wait(comm_stream, compute_stream);
        launch_all_gather(*context, num_buckets, buckets, src_buckets, context_idx,
                          seq_indices[context_idx], nvl_bandwidth, rdma_bandwidth, comm_stream);

        // Record and epilogue
        auto event = stream_control_epilogue(srcs);
        std::function epilogue = [context, gathered = std::move(gathered)]() {
            if (context->num_rdma_ranks > 1) {
                // NOTES: the hybrid path confirms every put by the tail signals before the barrier, no flush needed
                comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                              context->num_gpu_timeout_cycles, false, true, context->num_nvl_ranks == 1);
            }
            return pybind11::cast(gathered);
        };
        return pybind11::make_tuple(event, epilogue);
    }

    void destroy() override {
        EP_HOST_ASSERT(not destroyed);

        for (int context_idx = 0; context_idx < contexts.size(); ++ context_idx) {
            const auto& context = contexts[context_idx];
            comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                          context->num_gpu_timeout_cycles, true, true);
        }
        for (const auto& item: contexts)
            item->finalize();

        storage = torch::Tensor();
        contexts.clear();
        destroyed = true;
    }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<BucketBuffer, BufferBase>(m, "BucketBuffer")
        .def(pybind11::init<std::vector<int>, std::vector<int>, std::vector<int64_t>, int64_t, std::optional<int>, int, float, float, bool>(),
             py::arg("rank_indices"),
             py::arg("num_ranks"),
             py::arg("nccl_comms"),
             py::arg("num_buffer_bytes"),
             py::arg("sl_idx"),
             py::arg("num_gpu_timeout_secs"),
             py::arg("nvl_gbs"), py::arg("rdma_gbs"),
             py::arg("explicitly_destroy") = false)
        .def_readonly("contexts", &BucketBuffer::contexts)
        .def_readonly("storage", &BucketBuffer::storage)
        .def("reduce_scatter", &BucketBuffer::reduce_scatter,
             py::arg("srcs"),
             py::arg("context_idx"),
             py::arg("num_sms"),
             py::arg("comm_precision") = "fp32",
             py::arg("scale") = 1.0f)
        .def("all_reduce", &BucketBuffer::all_reduce,
             py::arg("srcs"),
             py::arg("context_idx"),
             py::arg("num_sms"),
             py::arg("scale") = 1.0f)
        .def("all_gather", &BucketBuffer::all_gather,
             py::arg("srcs"),
             py::arg("dsts"),
             py::arg("context_idx") = 0,
             py::arg("num_sms") = 0);
}

}  // namespace deep_ep::bucket

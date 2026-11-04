#pragma once

#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "../comm/stream.hpp"
#include "../kernels/magic/runtime.hpp"
#include "../utils/event.hpp"
#include "../utils/system.hpp"

namespace deep_ep::magic {

class MagicBuffer {
    Runtime runtime;
    at::cuda::CUDAStream comm_stream;
    const bool avoid_record_stream;

    at::cuda::CUDAStream stream_control_prologue(int device) const {
        TORCH_CHECK(device == comm_stream.device_index(), "input tensor must be on the same CUDA device as MagicBuffer");
        const auto compute_stream = at::cuda::getCurrentCUDAStream(device);
        comm::stream_wait(comm_stream, compute_stream);
        return compute_stream;
    }

    std::optional<EventHandle> stream_control_epilogue(const at::cuda::CUDAStream& compute_stream,
                                                       const std::vector<std::optional<torch::Tensor>>& tensors,
                                                       bool async_with_compute_stream) const {
        auto event = EventHandle(comm_stream);
        if (async_with_compute_stream) {
            if (avoid_record_stream) {
                event.tensors_to_record = tensors;
            } else {
                for (const auto& tensor : tensors)
                    if (tensor.has_value()) {
                        tensor->record_stream(compute_stream);
                        tensor->record_stream(comm_stream);
                    }
            }
            return event;
        }
        comm::stream_wait(compute_stream, event);
        return std::nullopt;
    }

public:
    MagicBuffer(unsigned rank, unsigned world_size, int64_t nccl_comm_ptr, unsigned max_tokens_per_rank, size_t workspace_size)
        : runtime(rank, world_size, nccl_comm_ptr, max_tokens_per_rank, workspace_size),
          comm_stream(at::cuda::getStreamFromPool(true)),
          avoid_record_stream(get_env<int>("EP_AVOID_RECORD_STREAM", 0) != 0) {}

    void destroy() { runtime.destroy(); }

    std::optional<EventHandle> dispatch(const torch::Tensor& x,
                                        const std::optional<torch::Tensor>& x_scale,
                                        const torch::Tensor& topk_ids,
                                        const torch::Tensor& topk_weights,
                                        const torch::Tensor& recv_x,
                                        const std::optional<torch::Tensor>& recv_x_scale,
                                        const torch::Tensor& recv_topk_ids,
                                        const torch::Tensor& recv_topk_weights,
                                        unsigned num_local_experts,
                                        bool async_with_compute_stream,
                                        const std::optional<torch::Tensor>& psum_recv_per_rank) {
        TORCH_CHECK(!x_scale || x_scale->dim() == 2, "x_scale must be a 2D tensor");
        TORCH_CHECK(x.dim() == 2 && topk_ids.dim() == 2 && topk_weights.dim() == 2, "x, topk_ids and topk_weights must be 2D tensors");
        const auto recv_capacity = recv_x.size(0);
        const auto num_tokens = x.size(0);
        const auto topk = topk_ids.size(1);
        const auto token_bytes = x.size(1) * x.element_size();
        const auto scale_bytes = x_scale ? x_scale->size(1) * x_scale->element_size() : 0;
        TORCH_CHECK(topk <= 32, "topk must be <= 32");
        const DispatchArgs args{
            x.data_ptr(),
            x_scale ? x_scale->data_ptr() : nullptr,
            topk_ids.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            recv_x.data_ptr(),
            recv_x_scale ? recv_x_scale->data_ptr() : nullptr,
            recv_topk_ids.data_ptr<int32_t>(),
            recv_topk_weights.data_ptr<float>(),
            psum_recv_per_rank ? psum_recv_per_rank->data_ptr<int32_t>() : nullptr,
            static_cast<unsigned>(num_tokens),
            static_cast<unsigned>(recv_capacity),
            static_cast<unsigned>(topk),
            static_cast<unsigned>(token_bytes),
            static_cast<unsigned>(scale_bytes),
            num_local_experts,
        };
        const auto compute_stream = stream_control_prologue(x.get_device());
        runtime.dispatch(args, comm_stream.stream());
        return stream_control_epilogue(compute_stream,
                                       {x,
                                        x_scale,
                                        topk_ids,
                                        topk_weights,
                                        recv_x,
                                        recv_x_scale,
                                        recv_topk_ids,
                                        recv_topk_weights,
                                        psum_recv_per_rank},
                                       async_with_compute_stream);
    }

    std::optional<EventHandle> combine(const torch::Tensor& x,
                                       const torch::Tensor& out,
                                       bool async_with_compute_stream,
                                       unsigned combine_num_sf_packs) {
        TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "x and out must be 2D tensors");
        const bool is_quantized_combine = combine_num_sf_packs != 0;
        TORCH_CHECK(x.scalar_type() == (is_quantized_combine ? torch::kUInt8 : torch::kBFloat16),
                    "x must have dtype uint8 for quantized combine or bfloat16 otherwise");
        TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must have dtype bfloat16");
        const auto recv_capacity = x.size(0);
        const auto num_tokens = out.size(0);
        const CombineArgs args{
            x.data_ptr(),
            out.data_ptr(),
            static_cast<unsigned>(num_tokens),
            static_cast<unsigned>(recv_capacity),
            static_cast<unsigned>(out.size(1)),
            static_cast<unsigned>(x.size(1) * x.element_size()),
            combine_num_sf_packs,
        };
        const auto compute_stream = stream_control_prologue(x.get_device());
        runtime.combine(args, comm_stream.stream());
        return stream_control_epilogue(compute_stream, {x, out}, async_with_compute_stream);
    }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<MagicBuffer>(m, "MagicBuffer")
        .def(pybind11::init<unsigned, unsigned, int64_t, unsigned, size_t>(), pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("destroy", &MagicBuffer::destroy, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("dispatch", &MagicBuffer::dispatch, pybind11::call_guard<pybind11::gil_scoped_release>())
        .def("combine", &MagicBuffer::combine, pybind11::call_guard<pybind11::gil_scoped_release>());
}

}

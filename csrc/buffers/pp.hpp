#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <pybind11/stl.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/layout/pp/workspace.cuh>

#include "base.hpp"
#include "../comm/api.hpp"
#include "../comm/barrier.hpp"
#include "../kernels/pp/pp_send_recv.hpp"
#include "../runtime/jit.hpp"

namespace deep_ep::pp {

class PPBuffer: public BufferBase {
public:
    std::shared_ptr<comm::Context> context;
    torch::Tensor storage;

private:
    // PP settings
    int64_t num_max_tensor_bytes;
    int num_max_inflight_tensors;

public:
    PPBuffer(const int& rank_idx, const int& num_ranks, const int64_t& nccl_comm,
             const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors,
             const std::optional<int>& sl_idx,
             const int& num_gpu_timeout_secs,
             const bool& explicitly_destroy):
        BufferBase(explicitly_destroy) {
        EP_HOST_ASSERT(num_max_tensor_bytes > 0 and num_max_inflight_tensors > 0);

        // Each send/recv buffer contains the previous and next rank in the ring
        this->num_max_tensor_bytes = math::align<int64_t>(num_max_tensor_bytes, kNumRDMAAlignmentBytes);
        this->num_max_inflight_tensors = num_max_inflight_tensors;
        const auto num_storage_bytes = math::align<int64_t>(
            4 * this->num_max_tensor_bytes * this->num_max_inflight_tensors,
            kNumAllocationAlignmentBytes);

        const auto num_workspace_bytes = math::align<int64_t>(
            sizeof(layout::PPSignals),
            kNumAllocationAlignmentBytes);
        const auto [num_rdma_ranks, _] = comm::get_physical_domain_size(nccl_comm);
        const auto num_allocated_qps = num_rdma_ranks == 1 ? 1 : kNumMaxQPs / num_rdma_ranks;
        context = std::make_shared<comm::Context>(
            nccl_comm, symmetric::shared_comm_t{}, num_ranks, rank_idx,
            num_workspace_bytes, num_storage_bytes, 0, true,
            false, sl_idx, num_allocated_qps,
            0, 0, num_gpu_timeout_secs, false);
        main_context = context;
        storage = torch::from_blob(
            context->buffer,
            {num_storage_bytes},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

        auto& pp = *static_cast<layout::PPSignals*>(context->workspace);
        context->set_barrier_signals(&pp.barrier);

        // We should call a barrier at the end
        // The barrier should be called by Python `dist.barrier`
        // NOTES: do not call our barrier, as the workspace is not ready yet
    }

    ~PPBuffer() noexcept(false) override {
        destroy_on_destruction("PP");
    }

    void send(const torch::Tensor& x, const int& dst_rank_idx, const int& num_sms) const {
        EP_HOST_ASSERT(not destroyed);

        // Check input
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() > 0);
        EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(x.data_ptr()) % kNumTMAAlignmentBytes == 0 and
                       x.nbytes() % kNumTMAAlignmentBytes == 0 and x.nbytes() <= num_max_tensor_bytes);
        EP_HOST_ASSERT((dst_rank_idx == (context->rank_idx + 1) % context->num_ranks or
                        dst_rank_idx == (context->rank_idx + context->num_ranks - 1) % context->num_ranks) and
                       "PP send requires an adjacent rank in the ring");

        launch_pp_send(*context, x.data_ptr(), x.nbytes(), dst_rank_idx,
                       num_max_tensor_bytes, num_max_inflight_tensors,
                       num_sms == 0 ? jit->device.get_num_sms() : num_sms,
                       at::cuda::getCurrentCUDAStream());
    }

    void recv(const torch::Tensor& x, const int& src_rank_idx, const int& num_sms) const {
        EP_HOST_ASSERT(not destroyed);

        // Check input
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() > 0);
        EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(x.data_ptr()) % kNumTMAAlignmentBytes == 0 and
                       x.nbytes() % kNumTMAAlignmentBytes == 0 and x.nbytes() <= num_max_tensor_bytes);
        EP_HOST_ASSERT((src_rank_idx == (context->rank_idx + 1) % context->num_ranks or
                        src_rank_idx == (context->rank_idx + context->num_ranks - 1) % context->num_ranks) and
                       "PP recv requires an adjacent rank in the ring");

        launch_pp_recv(*context, x.data_ptr(), x.nbytes(), src_rank_idx,
                       num_max_tensor_bytes, num_max_inflight_tensors,
                       num_sms == 0 ? jit->device.get_num_sms() : num_sms,
                       at::cuda::getCurrentCUDAStream());
    }

    void destroy() override {
        EP_HOST_ASSERT(not destroyed);

        // Flush outstanding RDMA operations before deregistering, as a peer may still signal us
        comm::barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                      context->num_gpu_timeout_cycles, true, true);
        context->finalize();
        storage = torch::Tensor();
        context = nullptr;
        destroyed = true;
    }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<PPBuffer, BufferBase>(m, "PPBuffer")
        .def(pybind11::init<int, int, int64_t, int64_t, int, std::optional<int>, int, bool>(),
             py::arg("rank_idx"),
             py::arg("num_ranks"),
             py::arg("nccl_comm"),
             py::arg("num_max_tensor_bytes"),
             py::arg("num_max_inflight_tensors"),
             py::arg("sl_idx"),
             py::arg("num_gpu_timeout_secs"),
             py::arg("explicitly_destroy") = false)
        .def_readonly("storage", &PPBuffer::storage)
        .def("send", &PPBuffer::send,
             py::arg("x"),
             py::arg("dst_rank_idx"),
             py::arg("num_sms") = 0)
        .def("recv", &PPBuffer::recv,
             py::arg("x"),
             py::arg("src_rank_idx"),
             py::arg("num_sms") = 0);
}

}  // namespace deep_ep::pp

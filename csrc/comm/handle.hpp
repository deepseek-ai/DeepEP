#pragma once

#include "barrier.hpp"
#include "../kernels/comm/api.hpp"

namespace deep_ep::comm {

static void register_apis(pybind11::module_& m) {
    pybind11::class_<Context, std::shared_ptr<Context>>(m, "Context")
        .def("get_physical_domain_size", &Context::get_physical_domain_size)
        .def("get_logical_domain_size", &Context::get_logical_domain_size)
        .def_readonly("num_workspace_bytes", &Context::num_workspace_bytes)
        .def_readonly("num_gpu_buffer_bytes", &Context::num_gpu_buffer_bytes)
        .def_readonly("num_rdma_storage_bytes", &Context::num_rdma_storage_bytes);

    m.def("barrier", [](const std::shared_ptr<Context>& context,
                        const bool& with_cpu_sync, const bool& sequential, const bool& wait_comm_stream) {
        EP_HOST_ASSERT(context != nullptr);
        EP_HOST_ASSERT(context->barrier_signals != nullptr);
        barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
                context->num_gpu_timeout_cycles, with_cpu_sync, sequential, true, wait_comm_stream);
    });

    m.def("get_local_nccl_unique_id", &get_local_unique_id);
    m.def("create_nccl_comm", &create_nccl_comm);
    m.def("destroy_nccl_comm", &destroy_nccl_comm);
    m.def("get_physical_domain_size", &get_physical_domain_size);
    m.def("get_logical_domain_size", &get_logical_domain_size);
    m.def("get_comm_stream", []() -> torch::Stream { return get_comm_stream(); });
}

}  // namespace deep_ep::comm

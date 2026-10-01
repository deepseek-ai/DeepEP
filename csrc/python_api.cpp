#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>

#include "buffers/base.hpp"
#include "buffers/bucket.hpp"
#include "buffers/engram.hpp"
#include "buffers/ep.hpp"
#include "buffers/pp.hpp"
#include "comm/api.hpp"
#include "runtime/jit.hpp"
#include "utils/event.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    // The integer type of top-k indices
    m.attr("topk_idx_t") = py::cast(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value);
    m.def("get_num_allocation_alignment", []() { return deep_ep::kNumAllocationAlignmentBytes; });
    m.def("get_num_tma_alignment", []() { return deep_ep::kNumTMAAlignmentBytes; });
    m.def("get_num_rdma_alignment", []() { return deep_ep::kNumRDMAAlignmentBytes; });

    // JIT API
    deep_ep::register_apis(m);

    // Event API
    deep_ep::register_event_apis(m);

    // Common communication APIs
    deep_ep::comm::register_apis(m);

    // Common buffer interface
    deep_ep::base::register_apis(m);

    // Register EP buffer APIs
    deep_ep::ep::register_apis(m);

    // Register Engram buffer APIs
    deep_ep::engram::register_apis(m);

    // Register bucket communication APIs
    deep_ep::bucket::register_apis(m);

    // Register pipeline-parallel send/recv APIs
    deep_ep::pp::register_apis(m);
}

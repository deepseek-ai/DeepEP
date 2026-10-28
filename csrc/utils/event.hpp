#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <utility>

#include <deep_ep/common/exception.cuh>

#include "tensor.hpp"

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;
    tensor_list_t tensors_to_record;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(at::cuda::getCurrentCUDAStream());
    }

    explicit EventHandle(const at::cuda::CUDAStream& stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const { at::cuda::getCurrentCUDAStream().unwrap().wait(*event); }
    void record_current_stream() {
        const auto stream = at::cuda::getCurrentCUDAStream();
        if (event->device_index() == stream.device_index()) {
            event->record(stream);
        } else {
            auto completion_event = torch::Event(torch::kCUDA);
            completion_event.record(stream);
            *event = std::move(completion_event);
        }
    }
};

static void register_event_apis(pybind11::module_& m) {
    pybind11::class_<EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("record_current_stream", &EventHandle::record_current_stream)
        .def("current_stream_wait", &EventHandle::current_stream_wait);
}

}  // namespace deep_ep

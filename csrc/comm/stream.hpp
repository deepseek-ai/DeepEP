#pragma once

#include <torch/python.h>

#include "../utils/event.hpp"

namespace deep_ep::comm {

static at::cuda::CUDAStream get_comm_stream() {
    static const auto comm_stream = at::cuda::getStreamFromPool(true);
    return comm_stream;
}

static torch::Event create_event(const at::cuda::CUDAStream& stream) {
    auto event = torch::Event(torch::kCUDA);
    event.record(stream);
    return event;
}

static void stream_wait(const at::cuda::CUDAStream& stream, const at::cuda::CUDAStream& event_stream) {
    EP_HOST_ASSERT(stream.id() != event_stream.id());
    stream.unwrap().wait(create_event(event_stream));
}

static void stream_wait(const at::cuda::CUDAStream& stream, const EventHandle& event) {
    stream.unwrap().wait(*event.event);
}

}  // namespace deep_ep::comm

#pragma once
// #include <ATen/cuda/CUDAContext.h>
#include <c10/core/Event.h>
#include <memory>

#include "kernels/exception.cuh"

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(at::cuda::getCurrentCUDAStream());
    }

    explicit EventHandle(const cudaStream_t& stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const {
        CUDA_CHECK(cudaStreamWaitEvent(
            at::cuda::getCurrentCUDAStream().raw_stream(),
            event->cuda_event(),
            0));
    }
};

torch::Event create_event(const cudaStream_t &s) {
    auto event = torch::Event(torch::kCUDA);
    event.record(s);
    return event;
}

inline void stream_wait(const cudaStream_t& s_0, const cudaStream_t& s_1) {
  EP_HOST_ASSERT(s_0 != s_1);
  CUDA_CHECK(cudaStreamWaitEvent(s_0, create_event(s_1).cuda_event(), 0));
}

inline void stream_wait(const cudaStream_t& s, const EventHandle& event) {
    CUDA_CHECK(cudaStreamWaitEvent(s, event.event->cuda_event(), 0));
}

} // namespace deep_ep

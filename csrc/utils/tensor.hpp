#pragma once

#include <optional>
#include <vector>

#include <torch/python.h>

#include <deep_ep/common/exception.cuh>

namespace deep_ep {

using tensor_list_t = std::vector<std::optional<torch::Tensor>>;

template <int kNumDims>
static auto get_shape(const torch::Tensor& t) {
    EP_HOST_ASSERT(t.dim() == kNumDims);
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<kNumDims>());
}

template <typename dtype_t = void>
static dtype_t* get_data_ptr(const std::optional<torch::Tensor>& t) {
    return t.has_value() ? t->data_ptr<dtype_t>() : nullptr;
}

}  // namespace deep_ep

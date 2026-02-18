// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#pragma once

#include <pybind11/pybind11.h>
#include "hybrid_ep.cuh"
#include "config.cuh"
#include <tuple>
#include <type_traits>
#include <vector>
#include <ATen/core/ivalue.h>
#include <ATen/record_function.h>

namespace py = pybind11;

// Convert a single argument to c10::IValue for tracing
template<typename T>
inline c10::IValue to_ivalue(T&& arg) {
    if constexpr (std::is_same_v<std::decay_t<T>, HybridEpConfigInstance>) {
        return std::forward<T>(arg).to_ivalue_tuple();
    } else if constexpr (std::is_same_v<std::decay_t<T>, BufferConfig>) {
        return std::forward<T>(arg).to_ivalue_tuple();
    } else if constexpr (std::is_same_v<std::decay_t<T>, c10::optional<torch::Tensor>>) {
        return arg.has_value() ? c10::IValue(arg.value()) : c10::IValue();
    } else if constexpr (std::is_same_v<std::decay_t<T>, c10::optional<int64_t>>) {
        return arg.has_value() ? c10::IValue(arg.value()) : c10::IValue();
    } else {
        return c10::IValue(std::forward<T>(arg));
    }
}

template<typename Result>
inline void record_result(c10::impl::GenericList& outputList, Result&& result) {
    std::apply([&outputList](const auto&... elem) {
        (outputList.emplace_back(c10::IValue(elem)), ...);
    }, result);
}

// Wrapper for combine
template<typename Func>
auto hybrid_ep_buffer_combine(Func func, const std::string& name) {
    return [func, name](HybridEPBuffer& self,
                        HybridEpConfigInstance config,
                        torch::Tensor hidden,
                        c10::optional<torch::Tensor> probs,
                        torch::Tensor sparse_to_dense_map,
                        torch::Tensor rdma_to_attn_map,
                        torch::Tensor attn_to_rdma_map,
                        int64_t num_of_tokens_per_rank,
                        bool with_probs) {
        auto result = (self.*func)(config, hidden, probs, sparse_to_dense_map,
                                   rdma_to_attn_map, attn_to_rdma_map,
                                   num_of_tokens_per_rank, with_probs);
        if (at::isRecordFunctionEnabled()) {
            c10::impl::GenericList inputList(c10::AnyType::get());
            inputList.emplace_back(to_ivalue(config));
            inputList.emplace_back(to_ivalue(hidden));
            inputList.emplace_back(to_ivalue(probs));
            inputList.emplace_back(to_ivalue(sparse_to_dense_map));
            inputList.emplace_back(to_ivalue(rdma_to_attn_map));
            inputList.emplace_back(to_ivalue(attn_to_rdma_map));
            inputList.emplace_back(to_ivalue(num_of_tokens_per_rank));
            inputList.emplace_back(to_ivalue(with_probs));
            c10::impl::GenericList outputList(c10::AnyType::get());
            record_result(outputList, result);
            c10::ArrayRef<const c10::IValue> inputsArray(inputList);
            c10::ArrayRef<const c10::IValue> outputsArray(outputList);
            RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(name, inputsArray, outputsArray);
        }
        return result;
    };
}

// Wrapper for dispatch (returns 3-tuple)
template<typename Func>
auto hybrid_ep_buffer_dispatch(Func func, const std::string& name) {
    return [func, name](HybridEPBuffer& self,
                        HybridEpConfigInstance config,
                        torch::Tensor hidden,
                        c10::optional<torch::Tensor> probs,
                        c10::optional<torch::Tensor> scaling_factor,
                        torch::Tensor sparse_to_dense_map,
                        torch::Tensor rdma_to_attn_map,
                        torch::Tensor attn_to_rdma_map,
                        c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
                        c10::optional<int64_t> num_dispatched_tokens,
                        int64_t num_of_tokens_per_rank,
                        bool with_probs) {
        auto result = (self.*func)(config, hidden, probs, scaling_factor,
                                   sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map,
                                   num_dispatched_tokens_tensor, num_dispatched_tokens,
                                   num_of_tokens_per_rank, with_probs);
        if (at::isRecordFunctionEnabled()) {
            c10::impl::GenericList inputList(c10::AnyType::get());
            inputList.emplace_back(to_ivalue(config));
            inputList.emplace_back(to_ivalue(hidden));
            inputList.emplace_back(to_ivalue(probs));
            inputList.emplace_back(to_ivalue(scaling_factor));
            inputList.emplace_back(to_ivalue(sparse_to_dense_map));
            inputList.emplace_back(to_ivalue(rdma_to_attn_map));
            inputList.emplace_back(to_ivalue(attn_to_rdma_map));
            inputList.emplace_back(to_ivalue(num_dispatched_tokens_tensor));
            inputList.emplace_back(to_ivalue(num_dispatched_tokens));
            inputList.emplace_back(to_ivalue(num_of_tokens_per_rank));
            inputList.emplace_back(to_ivalue(with_probs));
            c10::impl::GenericList outputList(c10::AnyType::get());
            record_result(outputList, result);
            c10::ArrayRef<const c10::IValue> inputsArray(inputList);
            c10::ArrayRef<const c10::IValue> outputsArray(outputList);
            RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(name, inputsArray, outputsArray);
        }
        return result;
    };
}

// Wrapper for dispatch_with_permute (returns 6-tuple)
template<typename Func>
auto hybrid_ep_buffer_dispatch_with_permute(Func func, const std::string& name) {
    return [func, name](HybridEPBuffer& self,
                        HybridEpConfigInstance config,
                        torch::Tensor hidden,
                        c10::optional<torch::Tensor> probs,
                        c10::optional<torch::Tensor> scaling_factor,
                        torch::Tensor sparse_to_dense_map,
                        torch::Tensor rdma_to_attn_map,
                        torch::Tensor attn_to_rdma_map,
                        c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
                        c10::optional<torch::Tensor> local_expert_routing_map,
                        c10::optional<torch::Tensor> row_id_map,
                        c10::optional<int64_t> num_permuted_tokens,
                        int64_t num_of_tokens_per_rank,
                        c10::optional<int64_t> pad_multiple,
                        bool non_blocking,
                        bool with_probs) {
        auto result = (self.*func)(config, hidden, probs, scaling_factor,
                                  sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map,
                                  num_dispatched_tokens_tensor, local_expert_routing_map, row_id_map,
                                  num_permuted_tokens, num_of_tokens_per_rank, pad_multiple,
                                  non_blocking, with_probs);
        if (at::isRecordFunctionEnabled()) {
            c10::impl::GenericList inputList(c10::AnyType::get());
            inputList.emplace_back(to_ivalue(config));
            inputList.emplace_back(to_ivalue(hidden));
            inputList.emplace_back(to_ivalue(probs));
            inputList.emplace_back(to_ivalue(scaling_factor));
            inputList.emplace_back(to_ivalue(sparse_to_dense_map));
            inputList.emplace_back(to_ivalue(rdma_to_attn_map));
            inputList.emplace_back(to_ivalue(attn_to_rdma_map));
            inputList.emplace_back(to_ivalue(num_dispatched_tokens_tensor));
            inputList.emplace_back(to_ivalue(local_expert_routing_map));
            inputList.emplace_back(to_ivalue(row_id_map));
            inputList.emplace_back(to_ivalue(num_permuted_tokens));
            inputList.emplace_back(to_ivalue(num_of_tokens_per_rank));
            inputList.emplace_back(to_ivalue(pad_multiple));
            inputList.emplace_back(to_ivalue(non_blocking));
            inputList.emplace_back(to_ivalue(with_probs));
            c10::impl::GenericList outputList(c10::AnyType::get());
            record_result(outputList, result);
            c10::ArrayRef<const c10::IValue> inputsArray(inputList);
            c10::ArrayRef<const c10::IValue> outputsArray(outputList);
            RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(name, inputsArray, outputsArray);
        }
        return result;
    };
}

// Wrapper for combine_with_unpermute (returns 2-tuple)
template<typename Func>
auto hybrid_ep_buffer_combine_with_unpermute(Func func, const std::string& name) {
    return [func, name](HybridEPBuffer& self,
                        HybridEpConfigInstance config,
                        torch::Tensor hidden,
                        c10::optional<torch::Tensor> probs,
                        torch::Tensor sparse_to_dense_map,
                        torch::Tensor rdma_to_attn_map,
                        torch::Tensor attn_to_rdma_map,
                        c10::optional<torch::Tensor> num_dispatched_tokens_tensor,
                        c10::optional<torch::Tensor> row_id_map,
                        int64_t num_of_tokens_per_rank,
                        c10::optional<int64_t> pad_multiple,
                        bool with_probs) {
        auto result = (self.*func)(config, hidden, probs, sparse_to_dense_map,
                                   rdma_to_attn_map, attn_to_rdma_map,
                                   num_dispatched_tokens_tensor, row_id_map,
                                   num_of_tokens_per_rank, pad_multiple, with_probs);
        if (at::isRecordFunctionEnabled()) {
            c10::impl::GenericList inputList(c10::AnyType::get());
            inputList.emplace_back(to_ivalue(config));
            inputList.emplace_back(to_ivalue(hidden));
            inputList.emplace_back(to_ivalue(probs));
            inputList.emplace_back(to_ivalue(sparse_to_dense_map));
            inputList.emplace_back(to_ivalue(rdma_to_attn_map));
            inputList.emplace_back(to_ivalue(attn_to_rdma_map));
            inputList.emplace_back(to_ivalue(num_dispatched_tokens_tensor));
            inputList.emplace_back(to_ivalue(row_id_map));
            inputList.emplace_back(to_ivalue(num_of_tokens_per_rank));
            inputList.emplace_back(to_ivalue(pad_multiple));
            inputList.emplace_back(to_ivalue(with_probs));
            c10::impl::GenericList outputList(c10::AnyType::get());
            record_result(outputList, result);
            c10::ArrayRef<const c10::IValue> inputsArray(inputList);
            c10::ArrayRef<const c10::IValue> outputsArray(outputList);
            RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(name, inputsArray, outputsArray);
        }
        return result;
    };
}

// HybridEPBuffer constructor with tracing
inline HybridEPBuffer* hybrid_ep_buffer_init(
    py::object process_group, BufferConfig config, int local_rank, int node_rank,
    int group_size, std::string base_path, bool load_cached_kernels,
    bool use_shared_buffer, bool enable_custom_allgather) {
    if (at::isRecordFunctionEnabled()) {
        c10::impl::GenericList inputList(c10::AnyType::get());
        inputList.emplace_back(to_ivalue(config));
        inputList.emplace_back(to_ivalue(local_rank));
        inputList.emplace_back(to_ivalue(node_rank));
        inputList.emplace_back(to_ivalue(group_size));
        inputList.emplace_back(to_ivalue(base_path));
        inputList.emplace_back(to_ivalue(load_cached_kernels));
        inputList.emplace_back(to_ivalue(use_shared_buffer));
        inputList.emplace_back(to_ivalue(enable_custom_allgather));
        c10::impl::GenericList outputList(c10::AnyType::get());
        c10::ArrayRef<const c10::IValue> inputsArray(inputList);
        c10::ArrayRef<const c10::IValue> outputsArray(outputList);
        RECORD_FUNCTION_WITH_INPUTS_OUTPUTS("HybridEPBuffer::__init__", inputsArray, outputsArray);
    }
    return new HybridEPBuffer(process_group, config, local_rank, node_rank, group_size,
                              base_path, load_cached_kernels, use_shared_buffer, enable_custom_allgather);
}

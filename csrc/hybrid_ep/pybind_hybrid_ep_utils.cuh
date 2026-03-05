// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#pragma once

#include <pybind11/pybind11.h>
#include "hybrid_ep.cuh"
#include "config.cuh"
#include <iostream>
#include <tuple>
#include <type_traits>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <optional>
#include <ATen/core/ivalue.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
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
inline void record_result(std::vector<c10::IValue>& outputList, Result&& result) {
    std::apply([&outputList](const auto&... elem) {
        (outputList.emplace_back(c10::IValue(elem)), ...);
    }, result);
}

// Struct to hold recording data for HybridEPBuffer::__init__ (can be consumed by record_hybrid_ep_buffer_init).
struct HybridEpBufferInitInfo {
    std::shared_mutex mutex;
    std::vector<c10::IValue> inputValues;
    std::optional<c10::FunctionSchema> schema;
    bool recorded = false;
};

HybridEpBufferInitInfo hybrid_ep_buffer_init_info;

// Call this to record HybridEPBuffer::__init__ using data stored in hybrid_ep_buffer_init_info.
inline void record_hybrid_ep_buffer_init() {
    {
        std::shared_lock lock(hybrid_ep_buffer_init_info.mutex);
        if (hybrid_ep_buffer_init_info.recorded) return;
    }
    std::unique_lock lock(hybrid_ep_buffer_init_info.mutex);
    if (hybrid_ep_buffer_init_info.recorded) return;

    std::vector<c10::IValue> outputValues;
    RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(
        hybrid_ep_buffer_init_info.schema.value(),
        &hybrid_ep_buffer_init_info.inputValues,
        outputValues);
    hybrid_ep_buffer_init_info.recorded = true;
}

inline HybridEPBuffer* hybrid_ep_buffer_init(
    py::object process_group, BufferConfig config, int local_rank, int node_rank,
    int group_size, std::string base_path, bool load_cached_kernels,
    bool use_shared_buffer, bool enable_custom_allgather) {

    // Initializing  HybridEPBuffer may happens before pytorch profiling starts
    // save the information to record it later when other operators are called. 
    // It is based on the assumption there is only one HybridEPBuffer instance 
    // in one process
    {
        std::unique_lock lock(hybrid_ep_buffer_init_info.mutex);

        hybrid_ep_buffer_init_info.inputValues.clear();
        hybrid_ep_buffer_init_info.inputValues.reserve(9);
        hybrid_ep_buffer_init_info.inputValues.emplace_back(c10::IValue(process_group.attr("group_name").cast<std::string>()));
        hybrid_ep_buffer_init_info.inputValues.emplace_back(to_ivalue(config));
        hybrid_ep_buffer_init_info.inputValues.emplace_back(to_ivalue(local_rank));
        hybrid_ep_buffer_init_info.inputValues.emplace_back(to_ivalue(node_rank));
        hybrid_ep_buffer_init_info.inputValues.emplace_back(to_ivalue(group_size));
        hybrid_ep_buffer_init_info.inputValues.emplace_back(to_ivalue(base_path));
        hybrid_ep_buffer_init_info.inputValues.emplace_back(to_ivalue(load_cached_kernels));
        hybrid_ep_buffer_init_info.inputValues.emplace_back(to_ivalue(use_shared_buffer));
        hybrid_ep_buffer_init_info.inputValues.emplace_back(to_ivalue(enable_custom_allgather));

        std::vector<c10::Argument> args;
        args.emplace_back("group_name", c10::StringType::get());
        args.emplace_back("config", c10::AnyType::get());
        args.emplace_back("local_rank", c10::IntType::get());
        args.emplace_back("node_rank", c10::IntType::get());
        args.emplace_back("group_size", c10::IntType::get());
        args.emplace_back("base_path", c10::StringType::get());
        args.emplace_back("load_cached_kernels", c10::BoolType::get());
        args.emplace_back("use_shared_buffer", c10::BoolType::get());
        args.emplace_back("enable_custom_allgather", c10::BoolType::get());
        std::vector<c10::Argument> returns;
        hybrid_ep_buffer_init_info.schema.emplace(
            "HybridEPBuffer::__init__", "",
            std::move(args), std::move(returns), false, false);

        hybrid_ep_buffer_init_info.recorded = false;
    }

    return new HybridEPBuffer(process_group, config, local_rank, node_rank, group_size,
                              base_path, load_cached_kernels, use_shared_buffer, enable_custom_allgather);
}

template<typename Func>
auto hybrid_ep_buffer_combine(Func func) {
    return [func](HybridEPBuffer& self,
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
            record_hybrid_ep_buffer_init();

            std::vector<c10::IValue> inputValues;
            inputValues.reserve(8);
            inputValues.push_back(to_ivalue(config));
            inputValues.push_back(to_ivalue(hidden));
            inputValues.push_back(to_ivalue(probs));
            inputValues.push_back(to_ivalue(sparse_to_dense_map));
            inputValues.push_back(to_ivalue(rdma_to_attn_map));
            inputValues.push_back(to_ivalue(attn_to_rdma_map));
            inputValues.push_back(to_ivalue(num_of_tokens_per_rank));
            inputValues.push_back(to_ivalue(with_probs));

            std::vector<c10::IValue> outputValues;
            record_result(outputValues, result);

            std::vector<c10::Argument> args;
            args.emplace_back("config", c10::AnyType::get());
            args.emplace_back("hidden", c10::TensorType::get());
            args.emplace_back("probs", c10::OptionalType::create(c10::TensorType::get()));
            args.emplace_back("sparse_to_dense_map", c10::TensorType::get());
            args.emplace_back("rdma_to_attn_map", c10::TensorType::get());
            args.emplace_back("attn_to_rdma_map", c10::TensorType::get());
            args.emplace_back("num_of_tokens_per_rank", c10::IntType::get());
            args.emplace_back("with_probs", c10::BoolType::get());
            std::vector<c10::Argument> returns;
            returns.emplace_back("", c10::TensorType::get());
            returns.emplace_back("", c10::TensorType::get());
            c10::FunctionSchema schema(
                "HybridEPBuffer::combine", "",
                std::move(args), std::move(returns), false, false);
            RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(schema, &inputValues, outputValues);
        }
        return result;
    };
}

// Wrapper for update_buffer (returns bool)
template<typename Func>
auto hybrid_ep_buffer_update_buffer(Func func) {
    return [func](HybridEPBuffer& self, HybridEpConfigInstance config) {
        auto result = (self.*func)(config);
        if (at::isRecordFunctionEnabled()) {
            record_hybrid_ep_buffer_init();

            std::vector<c10::IValue> inputValues;
            inputValues.reserve(1);
            inputValues.push_back(to_ivalue(config));

            std::vector<c10::IValue> outputValues;
            outputValues.push_back(c10::IValue(result));

            std::vector<c10::Argument> args;
            args.emplace_back("config", c10::AnyType::get());
            std::vector<c10::Argument> returns;
            returns.emplace_back("", c10::BoolType::get());
            c10::FunctionSchema schema(
                "HybridEPBuffer::update_buffer", "",
                std::move(args), std::move(returns), false, false);
            RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(schema, &inputValues, outputValues);
        }
        return result;
    };
}

// Wrapper for metadata_preprocessing (returns 5-tuple of tensors)
template<typename Func>
auto hybrid_ep_buffer_metadata_preprocessing(Func func) {
    return [func](HybridEPBuffer& self,
                  HybridEpConfigInstance config,
                  torch::Tensor routing_map,
                  int64_t num_of_tokens_per_rank,
                  bool non_blocking) {
        auto result = (self.*func)(config, routing_map, num_of_tokens_per_rank, non_blocking);
        if (at::isRecordFunctionEnabled()) {
            record_hybrid_ep_buffer_init();

            std::vector<c10::IValue> inputValues;
            inputValues.reserve(4);
            inputValues.push_back(to_ivalue(config));
            inputValues.push_back(to_ivalue(routing_map));
            inputValues.push_back(to_ivalue(num_of_tokens_per_rank));
            inputValues.push_back(to_ivalue(non_blocking));

            std::vector<c10::IValue> outputValues;
            record_result(outputValues, result);

            std::vector<c10::Argument> args;
            args.emplace_back("config", c10::AnyType::get());
            args.emplace_back("routing_map", c10::TensorType::get());
            args.emplace_back("num_of_tokens_per_rank", c10::IntType::get());
            args.emplace_back("non_blocking", c10::BoolType::get());
            std::vector<c10::Argument> returns;
            for (int i = 0; i < 5; ++i) {
                returns.emplace_back("", c10::TensorType::get());
            }
            c10::FunctionSchema schema(
                "HybridEPBuffer::metadata_preprocessing", "",
                std::move(args), std::move(returns), false, false);
            RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(schema, &inputValues, outputValues);
        }
        return result;
    };
}

// Wrapper for dispatch (returns 3-tuple)
template<typename Func>
auto hybrid_ep_buffer_dispatch(Func func) {
    return [func](HybridEPBuffer& self,
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
            record_hybrid_ep_buffer_init();

            std::vector<c10::IValue> inputValues;
            inputValues.reserve(11);
            inputValues.push_back(to_ivalue(config));
            inputValues.push_back(to_ivalue(hidden));
            inputValues.push_back(to_ivalue(probs));
            inputValues.push_back(to_ivalue(scaling_factor));
            inputValues.push_back(to_ivalue(sparse_to_dense_map));
            inputValues.push_back(to_ivalue(rdma_to_attn_map));
            inputValues.push_back(to_ivalue(attn_to_rdma_map));
            inputValues.push_back(to_ivalue(num_dispatched_tokens_tensor));
            inputValues.push_back(to_ivalue(num_dispatched_tokens));
            inputValues.push_back(to_ivalue(num_of_tokens_per_rank));
            inputValues.push_back(to_ivalue(with_probs));

            std::vector<c10::IValue> outputValues;
            record_result(outputValues, result);
            std::vector<c10::Argument> args;
            args.emplace_back("config", c10::AnyType::get());
            args.emplace_back("hidden", c10::TensorType::get());
            args.emplace_back("probs", c10::OptionalType::create(c10::TensorType::get()));
            args.emplace_back("scaling_factor", c10::OptionalType::create(c10::TensorType::get()));
            args.emplace_back("sparse_to_dense_map", c10::TensorType::get());
            args.emplace_back("rdma_to_attn_map", c10::TensorType::get());
            args.emplace_back("attn_to_rdma_map", c10::TensorType::get());
            args.emplace_back("num_dispatched_tokens_tensor", c10::OptionalType::create(c10::TensorType::get()));
            args.emplace_back("num_dispatched_tokens", c10::OptionalType::create(c10::IntType::get()));
            args.emplace_back("num_of_tokens_per_rank", c10::IntType::get());
            args.emplace_back("with_probs", c10::BoolType::get());
            std::vector<c10::Argument> returns;
            returns.emplace_back("", c10::TensorType::get());
            returns.emplace_back("", c10::TensorType::get());
            returns.emplace_back("", c10::TensorType::get());
            c10::FunctionSchema schema(
                "HybridEPBuffer::dispatch", "",
                std::move(args), std::move(returns), false, false);
            RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(schema, &inputValues, outputValues);
        }
        return result;
    };
}

// Wrapper for dispatch_with_permute (returns 6-tuple)
template<typename Func>
auto hybrid_ep_buffer_dispatch_with_permute(Func func) {
    return [func](HybridEPBuffer& self,
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
            record_hybrid_ep_buffer_init();

            std::vector<c10::IValue> inputValues;
            inputValues.reserve(15);
            inputValues.push_back(to_ivalue(config));
            inputValues.push_back(to_ivalue(hidden));
            inputValues.push_back(to_ivalue(probs));
            inputValues.push_back(to_ivalue(scaling_factor));
            inputValues.push_back(to_ivalue(sparse_to_dense_map));
            inputValues.push_back(to_ivalue(rdma_to_attn_map));
            inputValues.push_back(to_ivalue(attn_to_rdma_map));
            inputValues.push_back(to_ivalue(num_dispatched_tokens_tensor));
            inputValues.push_back(to_ivalue(local_expert_routing_map));
            inputValues.push_back(to_ivalue(row_id_map));
            inputValues.push_back(to_ivalue(num_permuted_tokens));
            inputValues.push_back(to_ivalue(num_of_tokens_per_rank));
            inputValues.push_back(to_ivalue(pad_multiple));
            inputValues.push_back(to_ivalue(non_blocking));
            inputValues.push_back(to_ivalue(with_probs));

            std::vector<c10::IValue> outputValues;
            record_result(outputValues, result);

            std::vector<c10::Argument> args;
            args.emplace_back("config", c10::AnyType::get());
            args.emplace_back("hidden", c10::TensorType::get());
            args.emplace_back("probs", c10::OptionalType::create(c10::TensorType::get()));
            args.emplace_back("scaling_factor", c10::OptionalType::create(c10::TensorType::get()));
            args.emplace_back("sparse_to_dense_map", c10::TensorType::get());
            args.emplace_back("rdma_to_attn_map", c10::TensorType::get());
            args.emplace_back("attn_to_rdma_map", c10::TensorType::get());
            args.emplace_back("num_dispatched_tokens_tensor", c10::OptionalType::create(c10::TensorType::get()));
            args.emplace_back("local_expert_routing_map", c10::OptionalType::create(c10::TensorType::get()));
            args.emplace_back("row_id_map", c10::OptionalType::create(c10::TensorType::get()));
            args.emplace_back("num_permuted_tokens", c10::OptionalType::create(c10::IntType::get()));
            args.emplace_back("num_of_tokens_per_rank", c10::IntType::get());
            args.emplace_back("pad_multiple", c10::OptionalType::create(c10::IntType::get()));
            args.emplace_back("non_blocking", c10::BoolType::get());
            args.emplace_back("with_probs", c10::BoolType::get());
            std::vector<c10::Argument> returns;
            returns.emplace_back("", c10::TensorType::get());
            returns.emplace_back("", c10::TensorType::get());
            returns.emplace_back("", c10::TensorType::get());
            returns.emplace_back("", c10::TensorType::get());
            returns.emplace_back("", c10::TensorType::get());
            returns.emplace_back("", c10::TensorType::get());
            c10::FunctionSchema schema(
                "HybridEPBuffer::dispatch_with_permute", "",
                std::move(args), std::move(returns), false, false);
            RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(schema, &inputValues, outputValues);
        }
        return result;
    };
}

// Wrapper for combine_with_unpermute (returns 2-tuple)
template<typename Func>
auto hybrid_ep_buffer_combine_with_unpermute(Func func) {
    return [func](HybridEPBuffer& self,
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
            record_hybrid_ep_buffer_init();

            std::vector<c10::IValue> inputValues;
            inputValues.reserve(10);
            inputValues.emplace_back(to_ivalue(config));
            inputValues.emplace_back(to_ivalue(hidden));
            inputValues.emplace_back(to_ivalue(probs));
            inputValues.emplace_back(to_ivalue(sparse_to_dense_map));
            inputValues.emplace_back(to_ivalue(rdma_to_attn_map));
            inputValues.emplace_back(to_ivalue(attn_to_rdma_map));
            inputValues.emplace_back(to_ivalue(num_dispatched_tokens_tensor));
            inputValues.emplace_back(to_ivalue(row_id_map));
            inputValues.emplace_back(to_ivalue(num_of_tokens_per_rank));
            inputValues.emplace_back(to_ivalue(pad_multiple));
            inputValues.emplace_back(to_ivalue(with_probs));

            std::vector<c10::IValue> outputValues;
            record_result(outputValues, result);

            std::vector<c10::Argument> args;
            args.emplace_back("config", c10::AnyType::get());
            args.emplace_back("hidden", c10::TensorType::get());
            args.emplace_back("probs", c10::OptionalType::create(c10::TensorType::get()));
            args.emplace_back("sparse_to_dense_map", c10::TensorType::get());
            args.emplace_back("rdma_to_attn_map", c10::TensorType::get());
            args.emplace_back("attn_to_rdma_map", c10::TensorType::get());
            args.emplace_back("num_dispatched_tokens_tensor", c10::OptionalType::create(c10::TensorType::get()));
            args.emplace_back("row_id_map", c10::OptionalType::create(c10::TensorType::get()));
            args.emplace_back("num_of_tokens_per_rank", c10::IntType::get());
            args.emplace_back("pad_multiple", c10::OptionalType::create(c10::IntType::get()));
            args.emplace_back("with_probs", c10::BoolType::get());
            std::vector<c10::Argument> returns;
            returns.emplace_back("", c10::TensorType::get());
            returns.emplace_back("", c10::TensorType::get());
            c10::FunctionSchema schema(
                "HybridEPBuffer::combine_with_unpermute", "",
                std::move(args), std::move(returns), false, false);
            RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(schema, &inputValues, outputValues);
        }
        return result;
    };
}

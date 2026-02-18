// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include <cuda_runtime.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "allocator/allocator.cuh"
#include "hybrid_ep.cuh"
#include "utils.cuh"
#include "config.cuh"
#include <iostream>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>
#include <ATen/core/ivalue.h>
#include <ATen/record_function.h>

namespace py = pybind11;

// Convert a single argument to c10::IValue for tracing
template<typename T>
c10::IValue to_ivalue(T&& arg) {
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
static void record_result(c10::impl::GenericList& outputList, Result&& result) {
    std::apply([&outputList](const auto&... elem) {
        (outputList.emplace_back(c10::IValue(elem)), ...);
    }, result);
}

// Wrapper for combine
template<typename Func>
auto wrap_with_tracing_combine(Func func, const std::string& name) {
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
auto wrap_with_tracing_dispatch(Func func, const std::string& name) {
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
auto wrap_with_tracing_dispatch_with_permute(Func func, const std::string& name) {
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
auto wrap_with_tracing_combine_with_unpermute(Func func, const std::string& name) {
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

// Wrapper for HybridEPBuffer constructor with tracing
static HybridEPBuffer* wrap_hybrid_ep_buffer_init(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HybridEP, efficiently enable the expert-parallel communication in "
              "the Hopper+ architectures";
    
    pybind11::class_<ExtendedMemoryAllocator>(m, "ExtendedMemoryAllocator")
        .def(py::init<>())
        .def("detect_accessible_ranks", &ExtendedMemoryAllocator::detect_accessible_ranks, py::arg("process_group"));
      
    pybind11::enum_<APP_TOKEN_DATA_TYPE>(m, "APP_TOKEN_DATA_TYPE")
        .value("UINT16", APP_TOKEN_DATA_TYPE::UINT16)
        .value("UINT8", APP_TOKEN_DATA_TYPE::UINT8)
        .export_values() // So we can use hybrid_ep_cpp.TYPE instead of the
                         // hybrid_ep_cpp.APP_TOKEN_DATA_TYPE.TYPE
        .def("__str__",
             [](const APP_TOKEN_DATA_TYPE &type) { return type_to_string(type); });
  
    pybind11::class_<BufferConfig>(m, "BufferConfig")
        .def(py::init<>())
        .def_readwrite("hidden_dim", &BufferConfig::hidden_dim)
        .def_readwrite("max_num_of_tokens_per_rank", &BufferConfig::max_num_of_tokens_per_rank)
        .def_readwrite("num_of_experts_per_rank", &BufferConfig::num_of_experts_per_rank)
        .def_readwrite("num_of_ranks_per_node", &BufferConfig::num_of_ranks_per_node)
        .def_readwrite("num_of_nodes", &BufferConfig::num_of_nodes)
        .def_readwrite("token_data_type", &BufferConfig::token_data_type)
        .def_readwrite("num_of_blocks_preprocessing_api", &BufferConfig::num_of_blocks_preprocessing_api)
        .def_readwrite("num_of_blocks_dispatch_api", &BufferConfig::num_of_blocks_dispatch_api)
        .def_readwrite("num_of_blocks_combine_api", &BufferConfig::num_of_blocks_combine_api)
        .def_readwrite("num_of_blocks_permute_api", &BufferConfig::num_of_blocks_permute_api)
        .def_readwrite("num_of_tokens_per_chunk_dispatch_api", &BufferConfig::num_of_tokens_per_chunk_dispatch_api)
        .def_readwrite("num_of_tokens_per_chunk_combine_api", &BufferConfig::num_of_tokens_per_chunk_combine_api)
        .def("is_valid", &BufferConfig::is_valid)
        .def("__repr__", [](const BufferConfig &config) {
          return "<BufferConfig hidden_dim=" +
                 std::to_string(config.hidden_dim) + " max_num_of_tokens_per_rank=" +
                 std::to_string(config.max_num_of_tokens_per_rank) +
                 " num_of_experts_per_rank=" + std::to_string(config.num_of_experts_per_rank) +
                 " num_of_ranks_per_node=" + std::to_string(config.num_of_ranks_per_node) +
                 " num_of_nodes=" + std::to_string(config.num_of_nodes) +
                 " token_data_type=" + type_to_string(config.token_data_type) +
                 " num_of_blocks_preprocessing_api=" + std::to_string(config.num_of_blocks_preprocessing_api) + 
                 " num_of_blocks_dispatch_api=" + std::to_string(config.num_of_blocks_dispatch_api) + 
                 " num_of_blocks_combine_api=" + std::to_string(config.num_of_blocks_combine_api) + 
                 " num_of_blocks_permute_api=" + std::to_string(config.num_of_blocks_permute_api) + 
                 " num_of_tokens_per_chunk_dispatch_api=" + std::to_string(config.num_of_tokens_per_chunk_dispatch_api) + 
                 " num_of_tokens_per_chunk_combine_api=" + std::to_string(config.num_of_tokens_per_chunk_combine_api) + 
                 ">";
        });

    pybind11::class_<HybridEpConfigInstance>(m, "HybridEpConfigInstance")
        .def(py::init<>())
        // Hybrid-ep Config
        .def_readwrite("hidden_dim", &HybridEpConfigInstance::hidden_dim)
        .def_readwrite("max_num_of_tokens_per_rank",
                       &HybridEpConfigInstance::max_num_of_tokens_per_rank)
        .def_readwrite("num_of_experts_per_rank",
                       &HybridEpConfigInstance::num_of_experts_per_rank)
        .def_readwrite("num_of_ranks_per_node",
                       &HybridEpConfigInstance::num_of_ranks_per_node)
        .def_readwrite("num_of_nodes", &HybridEpConfigInstance::num_of_nodes)
        // Metadata-preprocessing API Config
        .def_readwrite(
            "num_of_threads_per_block_preprocessing_api",
            &HybridEpConfigInstance::num_of_threads_per_block_preprocessing_api)
        .def_readwrite("num_of_blocks_preprocessing_api",
                       &HybridEpConfigInstance::num_of_blocks_preprocessing_api)
        .def_readwrite("num_of_blocks_permute_api",
                       &HybridEpConfigInstance::num_of_blocks_permute_api)
        // Dispatch API Config
        .def_readwrite("token_data_type", &HybridEpConfigInstance::token_data_type)
        .def_readwrite("num_of_stages_dispatch_api",
                       &HybridEpConfigInstance::num_of_stages_dispatch_api)
        .def_readwrite("num_of_in_flight_s2g_dispatch_api",
                       &HybridEpConfigInstance::num_of_in_flight_s2g_dispatch_api)
        .def_readwrite("num_of_tokens_per_chunk_dispatch_api",
                       &HybridEpConfigInstance::num_of_tokens_per_chunk_dispatch_api)
        .def_readwrite("num_of_blocks_dispatch_api",
                       &HybridEpConfigInstance::num_of_blocks_dispatch_api)
        .def_readwrite("forward_dispatch_api",
                       &HybridEpConfigInstance::forward_dispatch_api)
        .def_readwrite("device_side_sync_dispatch_api",
                       &HybridEpConfigInstance::device_side_sync_dispatch_api)
        // Combine API Config
        .def_readwrite("num_of_stages_g2s_combine_api",
                       &HybridEpConfigInstance::num_of_stages_g2s_combine_api)
        .def_readwrite("num_of_stages_s2g_combine_api",
                       &HybridEpConfigInstance::num_of_stages_s2g_combine_api)
        .def_readwrite("num_of_tokens_per_chunk_combine_api",
                       &HybridEpConfigInstance::num_of_tokens_per_chunk_combine_api)
        .def_readwrite("num_of_tokens_per_group_combine_api",
                       &HybridEpConfigInstance::num_of_tokens_per_group_combine_api)
        .def_readwrite("num_of_blocks_combine_api",
                       &HybridEpConfigInstance::num_of_blocks_combine_api)
        .def_readwrite(
            "num_of_additional_in_flight_s2g_combine_api",
            &HybridEpConfigInstance::num_of_additional_in_flight_s2g_combine_api)
        .def_readwrite("backward_combine_api",
                       &HybridEpConfigInstance::backward_combine_api)
        .def_readwrite("device_side_sync_combine_api",
                       &HybridEpConfigInstance::device_side_sync_combine_api)
        .def("is_valid", &HybridEpConfigInstance::is_valid)
        .def("__repr__", [](const HybridEpConfigInstance &config) {
          return "<HybridEpConfigInstance hidden_dim=" +
                 std::to_string(config.hidden_dim) + " max_num_of_tokens_per_rank=" +
                 std::to_string(config.max_num_of_tokens_per_rank) +
                 " token_data_type=" + type_to_string(config.token_data_type) +
                 ">";
        });
  
    pybind11::class_<HybridEPBuffer>(m, "HybridEPBuffer")
        .def(py::init(&wrap_hybrid_ep_buffer_init),
            py::arg("process_group"),
            py::arg("config"),
            py::arg("local_rank"),
            py::arg("node_rank"),
            py::arg("group_size"),
            py::arg("base_path"),
            py::arg("load_cached_kernels") = false,
            py::arg("use_shared_buffer") = true,
            py::arg("enable_custom_allgather") = true)
        .def("update_buffer", &HybridEPBuffer::update_buffer, py::arg("config"))
        .def("metadata_preprocessing", &HybridEPBuffer::metadata_preprocessing,
             py::kw_only(), py::arg("config"), py::arg("routing_map"), py::arg("num_of_tokens_per_rank"), py::arg("non_blocking") = false)
        .def("dispatch",
             wrap_with_tracing_dispatch(&HybridEPBuffer::dispatch, "HybridEPBuffer::dispatch"),
             py::kw_only(),
             py::arg("config"), py::arg("hidden"),
             py::arg("probs") = c10::nullopt,
             py::arg("scaling_factor") = c10::nullopt,
             py::arg("sparse_to_dense_map"), py::arg("rdma_to_attn_map"),
             py::arg("attn_to_rdma_map"), py::arg("num_dispatched_tokens_tensor"),
             py::arg("num_dispatched_tokens") = std::nullopt, py::arg("num_of_tokens_per_rank"),
             py::arg("with_probs"))
        .def("combine", 
             wrap_with_tracing_combine(&HybridEPBuffer::combine, "HybridEPBuffer::combine"),
             py::kw_only(), 
             py::arg("config"), py::arg("hidden"),
             py::arg("probs") = c10::nullopt, py::arg("sparse_to_dense_map"),
             py::arg("rdma_to_attn_map"), py::arg("attn_to_rdma_map"),
             py::arg("num_of_tokens_per_rank"),
             py::arg("with_probs"))
        .def("dispatch_with_permute",
             wrap_with_tracing_dispatch_with_permute(&HybridEPBuffer::dispatch_with_permute, "HybridEPBuffer::dispatch_with_permute"),
             py::kw_only(),
             py::arg("config"), py::arg("hidden"),
             py::arg("probs") = c10::nullopt,
             py::arg("scaling_factor") = c10::nullopt,
             py::arg("sparse_to_dense_map"), py::arg("rdma_to_attn_map"),
             py::arg("attn_to_rdma_map"), py::arg("num_dispatched_tokens_tensor"),
             py::arg("local_expert_routing_map"), py::arg("row_id_map"),
             py::arg("num_permuted_tokens") = std::nullopt,
             py::arg("num_of_tokens_per_rank"), py::arg("pad_multiple") = std::nullopt, py::arg("non_blocking") = false,
             py::arg("with_probs") = false)
        .def("combine_with_unpermute",
             wrap_with_tracing_combine_with_unpermute(&HybridEPBuffer::combine_with_unpermute, "HybridEPBuffer::combine_with_unpermute"),
             py::kw_only(),
             py::arg("config"), py::arg("hidden"),
             py::arg("probs") = c10::nullopt,
             py::arg("sparse_to_dense_map"), py::arg("rdma_to_attn_map"),
             py::arg("attn_to_rdma_map"), py::arg("num_dispatched_tokens_tensor"),
             py::arg("row_id_map"),
             py::arg("num_of_tokens_per_rank"), py::arg("pad_multiple") = std::nullopt,
             py::arg("with_probs") = false);    
    
  }
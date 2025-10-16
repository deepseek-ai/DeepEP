// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hybrid_ep.cuh"
#include "utils.cuh"
#include "config.cuh"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HybridEP, efficiently enable the expert-parallel communication in "
              "the Hopper+ architectures";
  
    pybind11::enum_<TOKEN_DATA_TYPE>(m, "TokenDataType")
        .value("UINT16", TOKEN_DATA_TYPE::UINT16)
        .value("UINT8", TOKEN_DATA_TYPE::UINT8)
        .export_values() // So we can use hybrid_ep_cpp.TYPE instead of the
                         // hybrid_ep_cpp.TOKEN_DATA_TYPE.TYPE
        .def("__str__",
             [](const TOKEN_DATA_TYPE &type) { return type_to_string(type); });
  
    pybind11::class_<BufferConfig>(m, "BufferConfig")
        .def(py::init<>())
        .def_readwrite("hidden_dim", &BufferConfig::hidden_dim)
        .def_readwrite("max_num_of_tokens_per_rank", &BufferConfig::max_num_of_tokens_per_rank)
        .def_readwrite("num_of_experts_per_rank", &BufferConfig::num_of_experts_per_rank)
        .def_readwrite("num_of_ranks_per_node", &BufferConfig::num_of_ranks_per_node)
        .def_readwrite("num_of_nodes", &BufferConfig::num_of_nodes)
        .def_readwrite("token_data_type", &BufferConfig::token_data_type)
        .def_readwrite("num_of_blocks_preprocessing_api", &BufferConfig::num_of_blocks_preprocessing_api)
        .def_readwrite("num_of_tokens_per_chunk_dispatch_api", &BufferConfig::num_of_tokens_per_chunk_dispatch_api)
        .def_readwrite("num_of_tokens_per_chunk_combine_api", &BufferConfig::num_of_tokens_per_chunk_combine_api)
        .def("__repr__", [](const BufferConfig &config) {
          return "<BufferConfig hidden_dim=" +
                 std::to_string(config.hidden_dim) + " max_num_of_tokens_per_rank=" +
                 std::to_string(config.max_num_of_tokens_per_rank) +
                 " num_of_experts_per_rank=" + std::to_string(config.num_of_experts_per_rank) +
                 " num_of_ranks_per_node=" + std::to_string(config.num_of_ranks_per_node) +
                 " num_of_nodes=" + std::to_string(config.num_of_nodes) +
                 " token_data_type=" + type_to_string(config.token_data_type) +
                 " num_of_blocks_preprocessing_api=" + std::to_string(config.num_of_blocks_preprocessing_api) +
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
        // Dispatch API Config
        .def_readwrite("token_data_type", &HybridEpConfigInstance::token_data_type)
        .def_readwrite("num_of_stages_dispatch_api",
                       &HybridEpConfigInstance::num_of_stages_dispatch_api)
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
        .def("__repr__", [](const HybridEpConfigInstance &config) {
          return "<HybridEpConfigInstance hidden_dim=" +
                 std::to_string(config.hidden_dim) + " max_num_of_tokens_per_rank=" +
                 std::to_string(config.max_num_of_tokens_per_rank) +
                 " token_data_type=" + type_to_string(config.token_data_type) +
                 ">";
        });
  
    pybind11::class_<HybridEPBuffer>(m, "HybridEPBuffer")
        .def(py::init<BufferConfig, int, int, int, std::string>())
        .def("update_buffer", &HybridEPBuffer::update_buffer, py::arg("config"))
        .def("exchange_ipc_address", &HybridEPBuffer::exchange_ipc_address)
        .def("metadata_preprocessing", &HybridEPBuffer::metadata_preprocessing,
             py::kw_only(), py::arg("config"), py::arg("routing_map"), py::arg("num_of_tokens_per_rank"))
        .def("dispatch", &HybridEPBuffer::dispatch, py::kw_only(), 
             py::arg("config"), py::arg("hidden"),
             py::arg("probs") = c10::nullopt,
             py::arg("scaling_factor") = c10::nullopt,
             py::arg("sparse_to_dense_map"), py::arg("rdma_to_attn_map"),
             py::arg("attn_to_rdma_map"), py::arg("num_dispatched_tokens_tensor"),
             py::arg("num_dispatched_tokens") = -1, py::arg("num_of_tokens_per_rank"),
             py::arg("with_probs"))
        .def("combine", &HybridEPBuffer::combine, py::kw_only(), 
             py::arg("config"), py::arg("hidden"),
             py::arg("probs") = c10::nullopt, py::arg("sparse_to_dense_map"),
             py::arg("rdma_to_attn_map"), py::arg("attn_to_rdma_map"),
             py::arg("num_of_tokens_per_rank"),
             py::arg("with_probs"))
        .def("dispatch_with_permute", &HybridEPBuffer::dispatch_with_permute, py::kw_only(), 
             py::arg("config"), py::arg("hidden"),
             py::arg("probs") = c10::nullopt,
             py::arg("scaling_factor") = c10::nullopt,
             py::arg("sparse_to_dense_map"), py::arg("rdma_to_attn_map"),
             py::arg("attn_to_rdma_map"), py::arg("num_dispatched_tokens_tensor"),
             py::arg("local_expert_routing_map"), py::arg("row_id_map"), py::arg("num_dispatched_tokens") = -1,
             py::arg("num_permuted_tokens") = -1,
             py::arg("num_of_tokens_per_rank"), py::arg("pad_multiple") = 0, py::arg("use_host_meta") = false,
             py::arg("with_probs") = false)
        .def("combine_with_unpermute", &HybridEPBuffer::combine_with_unpermute, py::kw_only(), 
             py::arg("config"), py::arg("hidden"),
             py::arg("probs") = c10::nullopt,
             py::arg("sparse_to_dense_map"), py::arg("rdma_to_attn_map"),
             py::arg("attn_to_rdma_map"), py::arg("num_dispatched_tokens_tensor"),
             py::arg("row_id_map"), py::arg("num_dispatched_tokens") = -1,
             py::arg("num_of_tokens_per_rank"), py::arg("pad_multiple") = 0,
             py::arg("with_probs") = false);    
  }
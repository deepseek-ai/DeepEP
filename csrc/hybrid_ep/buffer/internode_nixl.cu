// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#ifdef USE_NIXL

#include "buffer/internode.cuh"
#include <pybind11/pybind11.h>

NIXLCoordinator::~NIXLCoordinator() {
    destroy();
}

void NIXLCoordinator::init(
    pybind11::object process_group,
    int node_rank,
    int local_rank,
    BufferConfig config
) {
    this->process_group = process_group;
    this->node_rank = node_rank;
    this->local_rank = local_rank;
    this->buffer_config = config;
    assert(buffer_config.num_of_nodes > 1);
}

bool NIXLCoordinator::grow_buffer_config(const HybridEpConfigInstance& config, BufferConfig& buf_config) {
    bool changed = false;
    changed |= grow_to(buf_config.max_num_of_tokens_per_rank, config.max_num_of_tokens_per_rank);
    changed |= grow_to(buf_config.hidden_dim, config.hidden_dim);
    changed |= grow_to(buf_config.num_of_experts_per_rank, config.num_of_experts_per_rank);
    changed |= grow_to(buf_config.num_of_ranks_per_node, config.num_of_ranks_per_node);
    changed |= grow_to(buf_config.num_of_nodes, config.num_of_nodes);
    changed |= grow_to(buf_config.num_of_blocks_dispatch_api, config.num_of_blocks_dispatch_api);
    changed |= grow_to(buf_config.num_of_blocks_combine_api, config.num_of_blocks_combine_api);
    if (buf_config.num_of_tokens_per_chunk_dispatch_api != config.num_of_tokens_per_chunk_dispatch_api) {
        changed = true;
        buf_config.num_of_tokens_per_chunk_dispatch_api = config.num_of_tokens_per_chunk_dispatch_api;
    }
    if (buf_config.num_of_tokens_per_chunk_combine_api != config.num_of_tokens_per_chunk_combine_api) {
        changed = true;
        buf_config.num_of_tokens_per_chunk_combine_api = config.num_of_tokens_per_chunk_combine_api;
    }
    return changed;
}

void NIXLCoordinator::update_config(BufferConfig config) {
    this->buffer_config = config;
}

void NIXLCoordinator::destroy() {
    if (!buffer_allocated) return;

    CUDA_CHECK(cudaDeviceSynchronize());

    nixl_connector.reset();

    free_buffers();
    buffer_allocated = false;

    CUDA_CHECK(cudaDeviceSynchronize());
}

void NIXLCoordinator::free_buffers() {
    auto free_ptr = [](auto*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };

    free_ptr(dispatch_buffers.attn_input_token);
    free_ptr(dispatch_buffers.attn_input_prob);
    free_ptr(dispatch_buffers.attn_input_flags);
    free_ptr(dispatch_buffers.attn_input_scaling_factor);
    free_ptr(dispatch_buffers.rdma_inter_node_group_token);
    free_ptr(dispatch_buffers.rdma_inter_node_group_prob);
    free_ptr(dispatch_buffers.rdma_inter_node_group_scaling_factor);
    free_ptr(dispatch_buffers.rdma_inter_node_group_flags);
    free_ptr(dispatch_buffers.expected_rdma_flag_value);

    free_ptr(combine_buffers.attn_output_flags);
    free_ptr(combine_buffers.rdma_intra_node_red_token);
    free_ptr(combine_buffers.rdma_intra_node_red_prob);
    free_ptr(combine_buffers.rdma_inter_node_group_token);
    free_ptr(combine_buffers.rdma_inter_node_group_prob);
    free_ptr(combine_buffers.rdma_inter_node_group_flags);
    free_ptr(combine_buffers.expected_rdma_flag_value);
}

void NIXLCoordinator::allocate_dispatch_buffers() {
    dispatch_buffers.data_type = buffer_config.token_data_type;
    size_t sizeof_token_data_type = get_token_data_type_size(dispatch_buffers.data_type);

    auto attn_input_token_elts = buffer_config.max_num_of_tokens_per_rank * buffer_config.hidden_dim;
    auto attn_input_prob_elts = buffer_config.max_num_of_tokens_per_rank
        * (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node * buffer_config.num_of_nodes);
    auto attn_input_token_scaling_factor_elts = buffer_config.max_num_of_tokens_per_rank * (buffer_config.hidden_dim / 128);
    auto rdma_inter_node_group_token_elts = buffer_config.max_num_of_tokens_per_rank *
        (buffer_config.num_of_nodes - 1) * buffer_config.hidden_dim;
    auto rdma_inter_node_group_prob_elts = buffer_config.max_num_of_tokens_per_rank
        * (buffer_config.num_of_nodes - 1)
        * (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
    auto rdma_inter_node_group_scaling_factor_elts = buffer_config.max_num_of_tokens_per_rank *
        (buffer_config.num_of_nodes - 1) * (buffer_config.hidden_dim / 128);
    auto rdma_inter_node_group_flags_elts = ((buffer_config.max_num_of_tokens_per_rank - 1) /
        buffer_config.num_of_tokens_per_chunk_dispatch_api + 1) * (buffer_config.num_of_nodes - 1);

    dispatch_buffers.attn_input_token_sz = attn_input_token_elts * sizeof_token_data_type;
    dispatch_buffers.attn_input_prob_sz = attn_input_prob_elts * sizeof(float);
    dispatch_buffers.attn_input_scaling_factor_sz = attn_input_token_scaling_factor_elts * sizeof(float);
    dispatch_buffers.rdma_inter_node_group_token_sz = rdma_inter_node_group_token_elts * sizeof_token_data_type;
    dispatch_buffers.rdma_inter_node_group_prob_sz = rdma_inter_node_group_prob_elts * sizeof(float);
    dispatch_buffers.rdma_inter_node_group_scaling_factor_sz = rdma_inter_node_group_scaling_factor_elts * sizeof(float);
    dispatch_buffers.rdma_inter_node_group_flags_sz = rdma_inter_node_group_flags_elts * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_token, dispatch_buffers.attn_input_token_sz));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_prob, dispatch_buffers.attn_input_prob_sz));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_scaling_factor, dispatch_buffers.attn_input_scaling_factor_sz));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_token, dispatch_buffers.rdma_inter_node_group_token_sz));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_prob, dispatch_buffers.rdma_inter_node_group_prob_sz));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_scaling_factor, dispatch_buffers.rdma_inter_node_group_scaling_factor_sz));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_flags, dispatch_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMemset(dispatch_buffers.rdma_inter_node_group_flags, 0, dispatch_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_flags, dispatch_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMemset(dispatch_buffers.attn_input_flags, 0, dispatch_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(dispatch_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));
}

void NIXLCoordinator::allocate_combine_buffers() {
    auto rdma_intra_node_red_token_elts = buffer_config.max_num_of_tokens_per_rank *
        (buffer_config.num_of_nodes - 1) * buffer_config.hidden_dim;
    auto rdma_intra_node_red_prob_elts = buffer_config.max_num_of_tokens_per_rank * (buffer_config.num_of_nodes - 1) *
        (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
    auto rdma_inter_node_group_token_elts = buffer_config.max_num_of_tokens_per_rank *
        (buffer_config.num_of_nodes - 1) * buffer_config.hidden_dim;
    auto rdma_inter_node_group_prob_elts = buffer_config.max_num_of_tokens_per_rank * (buffer_config.num_of_nodes - 1) *
        (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
    auto rdma_inter_node_group_flags_elts = ((buffer_config.max_num_of_tokens_per_rank - 1) /
        buffer_config.num_of_tokens_per_chunk_combine_api + 1) * (buffer_config.num_of_nodes - 1);

    combine_buffers.rdma_intra_node_red_token_sz = rdma_intra_node_red_token_elts * sizeof(uint16_t);
    combine_buffers.rdma_intra_node_red_prob_sz = rdma_intra_node_red_prob_elts * sizeof(float);
    combine_buffers.rdma_inter_node_group_token_sz = rdma_inter_node_group_token_elts * sizeof(uint16_t);
    combine_buffers.rdma_inter_node_group_prob_sz = rdma_inter_node_group_prob_elts * sizeof(float);
    combine_buffers.rdma_inter_node_group_flags_sz = rdma_inter_node_group_flags_elts * sizeof(uint64_t);

    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_intra_node_red_token, combine_buffers.rdma_intra_node_red_token_sz));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_intra_node_red_prob, combine_buffers.rdma_intra_node_red_prob_sz));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_token, combine_buffers.rdma_inter_node_group_token_sz));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_prob, combine_buffers.rdma_inter_node_group_prob_sz));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_flags, combine_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMemset(combine_buffers.rdma_inter_node_group_flags, 0, combine_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.attn_output_flags, combine_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMemset(combine_buffers.attn_output_flags, 0, combine_buffers.rdma_inter_node_group_flags_sz));
    CUDA_CHECK(cudaMalloc((void**)&combine_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(combine_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));
}

void NIXLCoordinator::allocate_buffers() {
    allocate_combine_buffers();
    allocate_dispatch_buffers();

    int rank_uuid = node_rank * buffer_config.num_of_ranks_per_node + local_rank;
    int num_ranks = buffer_config.num_of_ranks_per_node * buffer_config.num_of_nodes;

    nixl_connector = std::make_unique<hybrid_ep::HybridEP_NIXLConnector>(rank_uuid, local_rank);

    nixl_connector->updateMemoryBuffers(
        num_ranks,
        buffer_config.num_of_experts_per_rank,
        buffer_config.num_of_nodes,
        buffer_config.num_of_ranks_per_node,
        buffer_config.num_of_blocks_dispatch_api,
        buffer_config.num_of_blocks_combine_api,
        dispatch_buffers,
        combine_buffers);

    auto torch_distributed = pybind11::module_::import("torch.distributed");
    torch_distributed.attr("barrier")(this->process_group);

    // sendLocalMD is async — metadata publication to etcd happens in a background
    // thread.  The barrier above ensures all ranks have *called* sendLocalMD, but
    // the etcd writes may not yet be visible.  A brief sleep reduces spurious
    // invalidate+refetch cycles in _nixl_agents_connect.
    {
        const char* env = std::getenv("DEEPEP_NIXL_POST_BARRIER_MS");
        int delay_ms = env ? std::atoi(env) : 2000;
        if (delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
    }

    std::vector<int> remote_rank_uuids;
    for (int node_idx = 0; node_idx < buffer_config.num_of_nodes; ++node_idx) {
        if (node_idx != node_rank) {
            remote_rank_uuids.push_back(node_idx * buffer_config.num_of_ranks_per_node + local_rank);
        }
    }
    nixl_connector->connectRanks(remote_rank_uuids);

    dispatch_buffers.nixl_gpu_ctx = nixl_connector->get_dispatch_gpu_ctx();
    combine_buffers.nixl_gpu_ctx = nixl_connector->get_combine_gpu_ctx();

    buffer_allocated = true;
}

#endif  // USE_NIXL

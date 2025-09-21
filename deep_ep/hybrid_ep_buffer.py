# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
import torch
import hybrid_ep_cpp

class HybridEpBuffer:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        hidden_dim: int,
        max_num_of_tokens_per_rank: int,
        num_local_experts: int,
        num_of_experts: int,
        use_fp8: bool = False,
        num_of_ranks_per_node: int = 32,
        num_sms_preprocessing_api: int = 32,
        num_sms_dispatch_api: int = 32,
        num_sms_combine_api: int = 32,
    ):
        self.group = group
        self.rank = self.group.rank()
        self.group_size = self.group.size()

        assert (
            self.group_size % num_of_ranks_per_node == 0
        ), f"The number of ranks {self.group_size} should be divisible by the number of ranks per node {num_of_ranks_per_node}."
        assert (
            self.group_size > 1
        ), f"The hybrid-ep kernel should be used with at least 2 ranks, but got {self.group_size}."
        self.num_of_ranks_per_node = num_of_ranks_per_node

        # Local rank: the active rank in the nvlink domain.
        self.local_rank = self.rank % self.num_of_ranks_per_node
        # Node rank: the active rank between the nvlink domains.
        self.node_rank = self.rank // self.num_of_ranks_per_node
        # The number of nodes.
        self.num_of_nodes = self.group_size // self.num_of_ranks_per_node

        self.hidden_dim = hidden_dim
        self.max_num_of_tokens_per_rank = max_num_of_tokens_per_rank
        self.num_local_experts = num_local_experts
        self.num_of_experts = num_of_experts
        self.use_fp8 = use_fp8
        self.num_sms_preprocessing_api = num_sms_preprocessing_api
        self.num_sms_dispatch_api = num_sms_dispatch_api
        self.num_sms_combine_api = num_sms_combine_api
        self.init_config()
        self.init_buffer()

    def init_config(
        self,
        num_of_threads_per_block_preprocessing_api: int = 512,
        num_of_stages_dispatch_api: int = 12,
        num_of_tokens_per_chunk_dispatch_api: int = 128,
        device_side_sync_dispatch_api: bool = True,
        num_of_stages_g2s_combine_api: int = 12,
        num_of_stages_s2g_combine_api: int = 2,
        num_of_tokens_per_chunk_combine_api: int = 128,
        num_of_tokens_per_group_combine_api: int = 4,
        num_of_additional_in_flight_s2g_combine_api: int = 2,
        device_side_sync_combine_api: bool = True,
    ):
        """
        Initialize the HybridEpConfigInstance for the hybrid-ep kernel.
        We can contoal the detailed setting of the hybrid-ep kernel.
        In common case, no need to change the default setting.
        """
        config = hybrid_ep_cpp.HybridEpConfigInstance()

        # Initialize the HybridEpConfigInstance
        # Hybrid-ep Config
        config.hidden_dim = self.hidden_dim
        config.max_num_of_tokens_per_rank = self.max_num_of_tokens_per_rank
        config.num_of_tokens_per_rank = self.max_num_of_tokens_per_rank # init to max_num_of_tokens_per_rank, will be updated in dispatch
        config.num_of_experts_per_rank = self.num_local_experts
        config.num_of_ranks_per_node = self.num_of_ranks_per_node
        config.num_of_nodes = self.num_of_nodes
        # Metadata-preprocessing API Config
        config.num_of_threads_per_block_preprocessing_api = (
            num_of_threads_per_block_preprocessing_api
        )
        config.num_of_blocks_preprocessing_api = self.num_sms_preprocessing_api
        # Dispatch API Config
        if self.use_fp8:
            # The fp8 data is communicated in the uint8 format.
            config.token_data_type = hybrid_ep_cpp.UINT8
        else:
            # The bf16 data is communicated in the uint16 format.
            config.token_data_type = hybrid_ep_cpp.UINT16
        config.num_of_stages_dispatch_api = num_of_stages_dispatch_api
        config.num_of_tokens_per_chunk_dispatch_api = (
            num_of_tokens_per_chunk_dispatch_api
        )
        config.num_of_blocks_dispatch_api = self.num_sms_dispatch_api
        config.device_side_sync_dispatch_api = device_side_sync_dispatch_api
        # Combine API Config
        config.num_of_stages_g2s_combine_api = num_of_stages_g2s_combine_api
        config.num_of_stages_s2g_combine_api = num_of_stages_s2g_combine_api
        config.num_of_tokens_per_chunk_combine_api = num_of_tokens_per_chunk_combine_api
        config.num_of_tokens_per_group_combine_api = num_of_tokens_per_group_combine_api
        config.num_of_blocks_combine_api = self.num_sms_combine_api
        config.num_of_additional_in_flight_s2g_combine_api = (
            num_of_additional_in_flight_s2g_combine_api
        )
        config.device_side_sync_combine_api = device_side_sync_combine_api

        self.config = config

    def init_buffer(self):
        """
        Initialize the buffer for the hybrid-ep kernel.
        Creates the C++ buffer (which allocates buffers) and exchanges IPC addresses.
        """
        assert self.config is not None, "Please initialize the config first."
        # Create C++ buffer - this will allocate all buffers during construction
        self.runtime = hybrid_ep_cpp.HybridEpBuffer(
            self.config, self.rank, self.group_size, self.num_of_ranks_per_node
        )
        
        # Exchange IPC addresses using C++ distributed communication
        self.runtime.exchange_ipc_address(self.group)

    def dispatch(
        self,
        tensor: torch.Tensor,
        scaling_factor: torch.Tensor = None,
        topk_idx: torch.Tensor = None,
        topk_weights: torch.Tensor = None,
        routing_map: torch.Tensor = None,
        num_of_tokens_for_experts: int = -1,
        handle: tuple = None,
        async_mode: bool = False,
    ):
        """
        Dispatch the data to the experts.

        Forward direction:
        dispatch_in_forward -> local_permute -> epxert_mlp -> local_unpermute -> combine_in_forward

        Backward direction:
        combine_in_backward <- local_unpermute -> expert_mlp -> local_permute -> dispatch_in_backward
        """
        num_of_tokens = tensor.shape[0]
        # Update the num_of_tokens_per_rank, both dispatch and combine will use this value
        self.runtime.update_num_of_tokens_per_rank(num_of_tokens)
        routing_map_as_input_and_probs_as_output = routing_map is not None
        if routing_map is not None:
            assert routing_map.dtype == torch.bool
        else:
            # Generate the routing map and the probs according to the topk_idx and topk_weights.
            assert topk_idx is not None
            routing_map = torch.zeros(num_of_tokens, self.num_of_experts, device="cuda", dtype=torch.bool)
            routing_map = routing_map.scatter(1, topk_idx.to(torch.int64), 1).bool()
            if topk_weights is not None:
                probs = torch.zeros(num_of_tokens, self.num_of_experts, device="cuda", dtype=torch.float32)
                probs = probs.scatter(1, topk_idx.to(torch.int64), topk_weights)
            else:
                probs = None

        global_routing_map = torch.empty(
            num_of_tokens * self.group_size,
            self.num_of_experts,
            device="cuda",
            dtype=torch.bool,
        )
        assert (
            handle is not None or routing_map is not None
        ), "The handle and routing_map should be both None"
        # If the handle is not provided, we need to generate the handle using the preprocessing kernel.
        if handle is None:
            torch.distributed.all_gather_into_tensor(
                global_routing_map, routing_map, self.group
            )
            (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_of_tokens_for_experts_tensor,
                local_expert_routing_map,
            ) = self.runtime.metadata_preprocessing(
                routing_map=global_routing_map,
                node_rank=self.node_rank,
                local_rank=self.local_rank,
            )
            # Create the handle using the data generated by the preprocessing kernel.
            handle = (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
            )
            if not async_mode:
                num_of_tokens_for_experts = num_of_tokens_for_experts_tensor.item()
        else:
            (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
            ) = handle
            num_of_tokens_for_experts_tensor = None
            local_expert_routing_map = None
            if not async_mode:
                assert (
                    num_of_tokens_for_experts >= 0
                ), "The num_of_tokens_for_experts should be provided."

        dispatched_token, dispatched_probs, dispatched_scaling_factor = (
            self.runtime.dispatch(
                hidden=tensor,
                probs=probs,
                scaling_factor=scaling_factor,
                sparse_to_dense_map=sparse_to_dense_map,
                rdma_to_attn_map=rdma_to_attn_map,
                attn_to_rdma_map=attn_to_rdma_map,
                num_of_tokens_for_experts=(
                    num_of_tokens_for_experts if not async_mode else -1
                ),
                with_probs=probs is not None,
            )
        )

        return (
            dispatched_token,
            dispatched_probs,
            dispatched_scaling_factor,
            num_of_tokens_for_experts_tensor,
            local_expert_routing_map,
            handle,
        )

    def combine(
        self, tensor: torch.Tensor, probs: torch.Tensor = None, handle: tuple = None
    ):
        """
        Combine the data from the experts.
        Do not require preprocessing, but the handle is necessary.
        """
        assert handle is not None, "The handle is necessary for combine."
        sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map = handle
        combined_token, combined_probs = self.runtime.combine(
            hidden=tensor,
            probs=probs,
            sparse_to_dense_map=sparse_to_dense_map,
            rdma_to_attn_map=rdma_to_attn_map,
            attn_to_rdma_map=attn_to_rdma_map,
            with_probs=probs is not None,
        )
        return combined_token, combined_probs

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
import torch
import os
import hybrid_ep_cpp

class HybridEpBuffer:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
         # Basic tensor setting
        hidden_dim: int,
        max_num_of_tokens_per_rank: int,
        num_local_experts: int,
        num_of_experts: int,
        # Use fp8 in dispatch or not.
        use_fp8: bool = False,
        # Device-SM occupancy setting
        num_sms_dispatch_api: int = 32,
        num_sms_combine_api: int = 32,
        num_sms_preprocessing_api: int = 128,
        nvlink_domain_size: int = None,
    ):
        self.group = group
        self.rank = self.group.rank()
        self.group_size = self.group.size()
        assert (
            self.group_size > 1
        ), f"The hybrid-ep kernel should be used with at least 2 ranks, but got {self.group_size}."

        # Compute the number of the involved ranks in the nvlink domain.
        global_ranks = torch.distributed.get_process_group_ranks(self.group)
        rank_stride = global_ranks[1] - global_ranks[0]
        # Number of ranks in the first nvlink domain.
        if nvlink_domain_size is None:
            nvlink_domain_size = int(os.getenv("NVLINK_DOMAIN_SIZE", "8"))
        assert (
            rank_stride <= nvlink_domain_size
        ), "The rank stride should be less than or equal to the nvlink domain size."
        num_of_ranks_per_node = min(nvlink_domain_size // rank_stride, self.group_size)
        self.nvlink_domain_size = nvlink_domain_size

        assert (
            self.group_size % num_of_ranks_per_node == 0
        ), "The number of ranks should be divisible by the number of ranks per node."
        self.rank = self.group.rank()
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

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        sm_count = props.multi_processor_count
        assert (
            sm_count >= num_sms_preprocessing_api
            and sm_count >= num_sms_dispatch_api
            and sm_count >= num_sms_combine_api
        ), "check the sms occupancy setting"
        self.num_sms_preprocessing_api = num_sms_preprocessing_api
        self.num_sms_dispatch_api = num_sms_dispatch_api
        self.num_sms_combine_api = num_sms_combine_api

        self.init_config()
        self.init_buffer()

    def init_config(
        self,
        # Metadata-preprocessing API Config
        num_of_threads_per_block_preprocessing_api: int = None,
        # Dispatch API Config
        num_of_stages_dispatch_api: int = None,
        num_of_tokens_per_chunk_dispatch_api: int = None,
        device_side_sync_dispatch_api: bool = True,
        # Combine API Config
        num_of_stages_g2s_combine_api: int = None,
        num_of_stages_s2g_combine_api: int = None,
        num_of_tokens_per_chunk_combine_api: int = None,
        num_of_tokens_per_group_combine_api: int = None,
        num_of_additional_in_flight_s2g_combine_api: int = None,
        device_side_sync_combine_api: bool = True,

    ):
        """
        Initialize the HybridEpConfigInstance for the hybrid-ep kernel.
        We can contoal the detailed setting of the hybrid-ep kernel.
        In common case, no need to change the default setting.
        """
        config = hybrid_ep_cpp.HybridEpConfigInstance()

        # Initialize the ConfigInstance
        # Hybrid-ep Config
        config.hidden_dim = self.hidden_dim
        config.max_num_of_tokens_per_rank = self.max_num_of_tokens_per_rank
        config.num_of_experts_per_rank = self.num_local_experts
        config.num_of_ranks_per_node = self.num_of_ranks_per_node
        config.num_of_nodes = self.num_of_nodes

        # Metadata-preprocessing API Config
        config.num_of_blocks_preprocessing_api = self.num_sms_preprocessing_api
        # 1. Try to get the value from the environment variable, Default value: 512
        # 2. If the value is provided, use the provided value.
        config.num_of_threads_per_block_preprocessing_api = int(
            os.getenv("NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API", "512")
        )
        if num_of_threads_per_block_preprocessing_api is not None:
            config.num_of_threads_per_block_preprocessing_api = (
                num_of_threads_per_block_preprocessing_api
            )

        # Dispatch API Config
        if self.use_fp8:
            # The fp8 data is communicated in the uint8 format.
            config.token_data_type = hybrid_ep_cpp.UINT8
        else:
            # The bf16 data is communicated in the uint16 format.
            config.token_data_type = hybrid_ep_cpp.UINT16
        config.num_of_blocks_dispatch_api = self.num_sms_dispatch_api
        config.device_side_sync_dispatch_api = device_side_sync_dispatch_api
        # Dispatch stages config:
        # 1. Try to get the value from the environment variable
        config.num_of_stages_dispatch_api = int(
            os.getenv("NUM_OF_STAGES_DISPATCH_API", "10")
        )
        config.num_of_tokens_per_chunk_dispatch_api = int(
            os.getenv("NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API", "128")
        )
        # 2. If the value is provided, use the provided value.
        if num_of_stages_dispatch_api is not None:
            config.num_of_stages_dispatch_api = num_of_stages_dispatch_api
        if num_of_tokens_per_chunk_dispatch_api is not None:
            config.num_of_tokens_per_chunk_dispatch_api = (
                num_of_tokens_per_chunk_dispatch_api
            )

        # Combine API Config
        config.num_of_blocks_combine_api = self.num_sms_combine_api
        config.device_side_sync_combine_api = device_side_sync_combine_api
        # Combine stages config:
        # 1. Try to get the value from the environment variable
        config.num_of_stages_g2s_combine_api = int(
            os.getenv("NUM_OF_STAGES_G2S_COMBINE_API", "10")
        )
        config.num_of_stages_s2g_combine_api = int(
            os.getenv("NUM_OF_STAGES_S2G_COMBINE_API", "2")
        )
        config.num_of_tokens_per_chunk_combine_api = int(
            os.getenv("NUM_OF_TOKENS_PER_CHUNK_COMBINE_API", "128")
        )
        config.num_of_tokens_per_group_combine_api = int(
            os.getenv("NUM_OF_TOKENS_PER_GROUP_COMBINE_API", "4")
        )
        config.num_of_additional_in_flight_s2g_combine_api = int(
            os.getenv("NUM_OF_ADDITIONAL_IN_FLIGHT_S2G_COMBINE_API", "2")
        )
        # 2. If the value is provided, use the provided value.
        if num_of_stages_g2s_combine_api is not None:
            config.num_of_stages_g2s_combine_api = num_of_stages_g2s_combine_api
        if num_of_stages_s2g_combine_api is not None:
            config.num_of_stages_s2g_combine_api = num_of_stages_s2g_combine_api
        if num_of_tokens_per_chunk_combine_api is not None:
            config.num_of_tokens_per_chunk_combine_api = (
                num_of_tokens_per_chunk_combine_api
            )
        if num_of_tokens_per_group_combine_api is not None:
            config.num_of_tokens_per_group_combine_api = (
                num_of_tokens_per_group_combine_api
            )
        if num_of_additional_in_flight_s2g_combine_api is not None:
            config.num_of_additional_in_flight_s2g_combine_api = (
                num_of_additional_in_flight_s2g_combine_api
            )

        self.config = config

    def init_buffer(self):
        """
        Initialize the buffer for the hybrid-ep kernel.
        Creates the C++ buffer (which allocates buffers) and exchanges IPC addresses.
        """
        assert self.config is not None, "Please initialize the config first."
        # Create C++ buffer - this will allocate all buffers during construction
        self.runtime = hybrid_ep_cpp.HybridEpBuffer(
            self.config, self.local_rank, self.node_rank, self.group_size, self.num_of_ranks_per_node, self.nvlink_domain_size
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
    ):
        """
        Dispatch the data to the experts.

        Forward direction:
        dispatch_in_forward -> local_permute -> epxert_mlp -> local_unpermute -> combine_in_forward

        Backward direction:
        combine_in_backward <- local_unpermute -> expert_mlp -> local_permute -> dispatch_in_backward
        """
        num_of_tokens = tensor.shape[0]
        assert num_of_tokens <= self.max_num_of_tokens_per_rank, "The number of tokens should be less than or equal to the max number of tokens per rank."
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
                num_of_tokens_per_rank=num_of_tokens,
            )
            # Create the handle using the data generated by the preprocessing kernel.
            handle = (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_of_tokens_for_experts_tensor,
                local_expert_routing_map,
                num_of_tokens,
            )
        else:
            (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_of_tokens_for_experts_tensor,
                local_expert_routing_map,
                num_of_tokens,
            ) = handle

        if num_of_tokens_for_experts < 0:
            num_of_tokens_for_experts = num_of_tokens_for_experts_tensor.item()

        dispatched_token, dispatched_probs, dispatched_scaling_factor = (
            self.runtime.dispatch(
                hidden=tensor,
                probs=probs,
                scaling_factor=scaling_factor,
                sparse_to_dense_map=sparse_to_dense_map,
                rdma_to_attn_map=rdma_to_attn_map,
                attn_to_rdma_map=attn_to_rdma_map,
                num_of_tokens_for_experts=num_of_tokens_for_experts,
                num_of_tokens_per_rank=num_of_tokens,
                with_probs=probs is not None,
            )
        )

        return (
            dispatched_token,
            dispatched_probs,
            dispatched_scaling_factor,
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
        sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map, num_of_tokens_for_experts_tensor, local_expert_routing_map, num_of_tokens = handle
        combined_token, combined_probs = self.runtime.combine(
            hidden=tensor,
            probs=probs,
            sparse_to_dense_map=sparse_to_dense_map,
            rdma_to_attn_map=rdma_to_attn_map,
            attn_to_rdma_map=attn_to_rdma_map,
            num_of_tokens_per_rank=num_of_tokens,
            with_probs=probs is not None,
        )
        return combined_token, combined_probs

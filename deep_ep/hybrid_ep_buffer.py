# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
import torch
import os
import hybrid_ep_cpp


def indices_to_map(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_of_tokens: int,
    num_of_experts: int,
):
    """
    Map the map to the indices.
    """
    # Generate the routing map and the probs according to the topk_idx and topk_weights.
    assert topk_idx is not None
    routing_map = torch.zeros(
        num_of_tokens, num_of_experts, device="cuda", dtype=torch.bool
    )
    routing_map = routing_map.scatter(1, topk_idx.to(torch.int64), 1).bool()
    if topk_weights is not None:
        probs = torch.zeros(
            num_of_tokens, num_of_experts, device="cuda", dtype=torch.float32
        )
        probs = probs.scatter(1, topk_idx.to(torch.int64), topk_weights)
    else:
        probs = None
    return routing_map, probs


class HybridEPBuffer:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        # Parameters for the hybrid-ep buffer allocation
        hidden_dim: int,
        max_num_of_tokens_per_rank: int,
        num_local_experts: int,
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

        # Initialize the BufferConfig for the hybrid-ep buffer allocation.
        self.config = hybrid_ep_cpp.BufferConfig()
        self.config.hidden_dim = hidden_dim
        self.config.max_num_of_tokens_per_rank = max(max_num_of_tokens_per_rank, 1024)
        self.config.num_of_experts_per_rank = num_local_experts
        self.config.num_of_ranks_per_node = self.num_of_ranks_per_node
        self.config.num_of_nodes = self.num_of_nodes
        # The SMs of preprocessing, chunk size of dispatch and combine will affact the size of intermediate buffers.
        self.config.num_of_blocks_preprocessing_api = self.num_sms_preprocessing_api
        # The fp8/bf16/fp16 data is communicated in the uint8/uint16 format.
        self.config.token_data_type = (
            hybrid_ep_cpp.UINT8 if self.use_fp8 else hybrid_ep_cpp.UINT16
        )
        self.config.num_of_tokens_per_chunk_dispatch_api = int(
            os.getenv("NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API", "128")
        )
        self.config.num_of_tokens_per_chunk_combine_api = int(
            os.getenv("NUM_OF_TOKENS_PER_CHUNK_COMBINE_API", "128")
        )

        # Create C++ buffer - this will allocate all buffers during construction
        self.runtime = hybrid_ep_cpp.HybridEPBuffer(
            self.config, self.local_rank, self.node_rank, self.group_size, os.path.dirname(os.path.abspath(__file__))
        )
        # Exchange IPC addresses using C++ distributed communication
        self.runtime.exchange_ipc_address(self.group)

    def empty_jit_cache(self):
        '''
        Clean the cached kernel files.
        '''
        jit_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "jit")
        if os.path.exists(jit_cache_path):
            shutil.rmtree(jit_cache_path)

    def update_template_config(
        self,
        hidden_dim: int = None,
        max_num_of_tokens_per_rank: int = None,
        num_local_experts: int = None,
        use_fp8: bool = None,
    ):
        """
        Initialize the HybridEpConfigInstance which used to control the detailed setting of the hybrid-ep kernel.
        In common case, no need to change the default setting.
        """
        config = hybrid_ep_cpp.HybridEpConfigInstance()

        # Initialize the ConfigInstance
        # Hybrid-ep Config
        config.hidden_dim = (
            hidden_dim if hidden_dim is not None else self.config.hidden_dim
        )
        config.max_num_of_tokens_per_rank = (
            max_num_of_tokens_per_rank
            if max_num_of_tokens_per_rank is not None
            else self.config.max_num_of_tokens_per_rank
        )
        config.max_num_of_tokens_per_rank = max(
            config.max_num_of_tokens_per_rank, self.config.max_num_of_tokens_per_rank
        )
        config.num_of_experts_per_rank = (
            num_local_experts
            if num_local_experts is not None
            else self.config.num_of_experts_per_rank
        )
        config.num_of_ranks_per_node = self.num_of_ranks_per_node
        config.num_of_nodes = self.num_of_nodes

        # Metadata-preprocessing API Config
        config.num_of_blocks_preprocessing_api = self.num_sms_preprocessing_api
        config.num_of_threads_per_block_preprocessing_api = int(
            os.getenv("NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API", "512")
        )

        # Dispatch API Config
        if use_fp8 is None:
            use_fp8 = self.use_fp8
        config.token_data_type = (
            hybrid_ep_cpp.UINT8 if use_fp8 else hybrid_ep_cpp.UINT16
        )
        config.num_of_blocks_dispatch_api = self.num_sms_dispatch_api
        config.device_side_sync_dispatch_api = True
        # Dispatch stages config:
        config.num_of_stages_dispatch_api = int(
            os.getenv("NUM_OF_STAGES_DISPATCH_API", "10")
        )
        config.num_of_tokens_per_chunk_dispatch_api = int(
            os.getenv("NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API", "128")
        )

        # Combine API Config
        config.num_of_blocks_combine_api = self.num_sms_combine_api
        config.device_side_sync_combine_api = True
        # Combine stages config:
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

        # Use the runtime kernel config to update the buffer.
        reallocated = self.runtime.update_buffer(config)
        if reallocated:
            self.runtime.exchange_ipc_address(self.group)
        return config

    def dispatch(
        self,
        hidden: torch.Tensor,
        scaling_factor: torch.Tensor = None,
        topk_idx: torch.Tensor = None,
        topk_weights: torch.Tensor = None,
        num_of_experts: int = None,
        probs: torch.Tensor = None,
        routing_map: torch.Tensor = None,
        num_dispatched_tokens_tensor: torch.Tensor = None,
        num_dispatched_tokens: int = None,
        handle: tuple = None,
    ):
        """
        Dispatch the data to the experts.

        Forward direction:
        dispatch_in_forward -> local_permute -> epxert_mlp -> local_unpermute -> combine_in_forward

        Backward direction:
        combine_in_backward <- local_unpermute -> expert_mlp -> local_permute -> dispatch_in_backward
        """
        num_of_tokens, hidden_dim = hidden.shape

        if routing_map is not None:
            assert routing_map.dtype == torch.bool
            num_of_experts = routing_map.size(-1)
        else:
            # Generate the routing map and the probs according to the topk_idx and topk_weights.
            assert (
                num_of_experts is not None
            ), "The number of experts should be provided on index-based routing."
            if topk_idx is not None:
                routing_map, probs = indices_to_map(
                    topk_idx, topk_weights, num_of_tokens, num_of_experts
                )

        assert (
            handle is not None or routing_map is not None
        ), "The handle and routing_map should be both None"
        # If the handle is not provided, we need to generate the handle using the preprocessing kernel.
        if handle is None:
            config = self.update_template_config(
                hidden_dim=hidden_dim,
                max_num_of_tokens_per_rank=num_of_tokens,
            )
            # The hybrid-ep kernel requires the routing info from all ranks.
            global_routing_map = torch.empty(
                num_of_tokens * self.group_size,
                num_of_experts,
                device="cuda",
                dtype=torch.bool,
            )
            torch.distributed.all_gather_into_tensor(
                global_routing_map, routing_map, self.group
            )
            # Run the metadata preprocessing kernel.
            (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_dispatched_tokens_tensor,
                local_expert_routing_map,
            ) = self.runtime.metadata_preprocessing(
                config=config,
                routing_map=global_routing_map,
                num_of_tokens_per_rank=num_of_tokens,
            )
            # Create the handle using the data generated by the preprocessing kernel.
            handle = (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_dispatched_tokens_tensor,
                local_expert_routing_map,
                num_of_tokens,
                config,
            )
        else:
            (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_dispatched_tokens_tensor,
                local_expert_routing_map,
                num_of_tokens,
                config,
            ) = handle

        if num_dispatched_tokens is None:
            num_dispatched_tokens = num_dispatched_tokens_tensor.item()

        dispatched_token, dispatched_probs, dispatched_scaling_factor = (
            self.runtime.dispatch(
                config=config,
                hidden=hidden,
                probs=probs,
                scaling_factor=scaling_factor,
                sparse_to_dense_map=sparse_to_dense_map,
                rdma_to_attn_map=rdma_to_attn_map,
                attn_to_rdma_map=attn_to_rdma_map,
                num_dispatched_tokens_tensor=num_dispatched_tokens_tensor,
                num_dispatched_tokens=num_dispatched_tokens,
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
        self, hidden: torch.Tensor, probs: torch.Tensor = None, handle: tuple = None
    ):
        """
        Combine the data from the experts.
        Do not require preprocessing, but the handle is necessary.
        """
        assert handle is not None, "The handle is necessary for combine."
        (
            sparse_to_dense_map,
            rdma_to_attn_map,
            attn_to_rdma_map,
            num_dispatched_tokens_tensor,
            local_expert_routing_map,
            num_of_tokens,
            config,
        ) = handle
        combined_token, combined_probs = self.runtime.combine(
            config=config,
            hidden=hidden,
            probs=probs,
            sparse_to_dense_map=sparse_to_dense_map,
            rdma_to_attn_map=rdma_to_attn_map,
            attn_to_rdma_map=attn_to_rdma_map,
            num_of_tokens_per_rank=num_of_tokens,
            with_probs=probs is not None,
        )
        return combined_token, combined_probs

    def dispatch_with_permute(
        self,
        *,
        # Input tensors
        hidden: torch.Tensor,
        topk_idx: torch.Tensor = None,
        topk_weights: torch.Tensor = None,
        num_of_experts_per_rank: int = None,
        num_of_experts: int = None,
        use_fp8: bool = None,
        routing_map: torch.Tensor = None,
        probs: torch.Tensor = None,
        scaling_factor: torch.Tensor = None,
        # Used in the sync-free permute
        num_dispatched_tokens: int = None,
        num_permuted_tokens: int = None,
        # If we use permute kernel, the output tensor will be permuted. the result can be directly used in the gemm.
        pad_multiple: int = None,
        # The handle means the cached info from the first invocation of the dispatch kernel.
        # The handle includes:
        # # Output of Metadata Preprocessing
        # 1. sparse_to_dense_map
        # 2. rdma_to_attn_map
        # 3. attn_to_rdma_map
        # 4. num_of_tokens_for_experts_tensor
        # 5. local_expert_routing_map
        # # Output of Permute Preprocessing
        # 6. row_id_map
        # # Cache for template config
        # 7. template_config: HybridEpConfigInstance
        handle: tuple = None,
        # If enable this, the produced num_dispatched_tokens will be put on the CPU pinned memory, and the tokens_per_expert will be put on the CPU, which may reduce the times of the sync
        use_host_meta: bool = True,
    ):
        """
        Dispatch the data to the experts with permute.
        """
        with torch.cuda.nvtx.range("hybrid-ep dispatch with permute phase"):
            num_of_tokens_per_rank, hidden_dim = hidden.shape
            if routing_map is not None:
                assert routing_map.dtype == torch.bool
                num_of_experts = routing_map.size(-1)
            else:
                # Generate the routing map and the probs according to the topk_idx and topk_weights.
                if topk_idx is not None:
                    assert (
                        num_of_experts is not None
                    ), "The number of experts should be provided on index-based routing."
                    routing_map, probs = indices_to_map(
                        topk_idx, topk_weights, num_of_tokens_per_rank, num_of_experts
                    )

            # If the handle is not provided, we need to generate the handle in the first invocation of the dispatch kernel.
            if handle is None:
                assert hidden.size(0) == routing_map.size(
                    0
                ), "The hidden and the routing_map should have the same row number."
                # Update the template config.
                config = self.update_template_config(
                    hidden_dim=hidden_dim,
                    max_num_of_tokens_per_rank=num_of_tokens_per_rank,
                    num_local_experts=num_of_experts_per_rank,
                    use_fp8=use_fp8,
                )
                # Global routing map: the routing map for all tokens to all experts.
                global_routing_map = torch.empty(
                    num_of_tokens_per_rank * self.group_size,
                    num_of_experts,
                    device="cuda",
                    dtype=torch.bool,
                )
                torch.distributed.all_gather_into_tensor(
                    global_routing_map, routing_map, self.group
                )
                # Run the metadata preprocessing kernel.
                row_id_map = None
                (
                    sparse_to_dense_map,
                    rdma_to_attn_map,
                    attn_to_rdma_map,
                    num_dispatched_tokens_tensor,
                    local_expert_routing_map,
                ) = self.runtime.metadata_preprocessing(
                    config=config,
                    routing_map=global_routing_map,
                    num_of_tokens_per_rank=num_of_tokens_per_rank,
                )
                if use_host_meta:
                    # Put the num_dispatched_tokens_tensor on the CPU pinned memory, because this tensor also will be used in the GPU kernel
                    num_dispatched_tokens_tensor = (
                        num_dispatched_tokens_tensor.cpu().pin_memory()
                    )
            else:
                (
                    sparse_to_dense_map,
                    rdma_to_attn_map,
                    attn_to_rdma_map,
                    num_dispatched_tokens_tensor,
                    local_expert_routing_map,
                    row_id_map,
                    num_of_tokens_per_rank,
                    config,
                ) = handle

            # Dispatch phase
            (
                dispatched_token,
                dispatched_probs,
                dispatched_scaling_factor,
                row_id_map,
                tokens_per_expert,
            ) = self.runtime.dispatch_with_permute(
                config=config,
                hidden=hidden,
                probs=probs,
                scaling_factor=scaling_factor,
                sparse_to_dense_map=sparse_to_dense_map,
                rdma_to_attn_map=rdma_to_attn_map,
                attn_to_rdma_map=attn_to_rdma_map,
                num_dispatched_tokens_tensor=num_dispatched_tokens_tensor,
                local_expert_routing_map=local_expert_routing_map,
                row_id_map=row_id_map,
                num_dispatched_tokens=num_dispatched_tokens,
                num_permuted_tokens=num_permuted_tokens,
                num_of_tokens_per_rank=num_of_tokens_per_rank,
                pad_multiple=pad_multiple,
                use_host_meta=use_host_meta,
                with_probs=probs is not None,
            )

            handle = (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_dispatched_tokens_tensor,
                local_expert_routing_map,
                row_id_map,
                num_of_tokens_per_rank,
                config,
            )
        return (
            dispatched_token,
            dispatched_probs,
            dispatched_scaling_factor,
            tokens_per_expert,
            handle,
        )

    def combine_with_unpermute(
        self,
        *,
        # Input tensors
        hidden: torch.Tensor,
        probs: torch.Tensor = None,
        num_dispatched_tokens: int = None,
        handle: tuple = None,
        pad_multiple: int = None,
    ):
        """
        Combine the data from the experts with unpermute.
        Do not require the routing_map, but the handle is necessary.
        """
        with torch.cuda.nvtx.range("hybrid-ep combine with unpermute phase"):
            assert self.config is not None, "Please initialize the config first."
            assert handle is not None, "The handle is necessary in the combine pass."

            (
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_dispatched_tokens_tensor,
                _,
                row_id_map,
                num_of_tokens_per_rank,
                config,
            ) = handle

            combined_token, combined_probs = self.runtime.combine_with_unpermute(
                config=config,
                hidden=hidden,
                probs=probs,
                sparse_to_dense_map=sparse_to_dense_map,
                rdma_to_attn_map=rdma_to_attn_map,
                attn_to_rdma_map=attn_to_rdma_map,
                num_dispatched_tokens_tensor=num_dispatched_tokens_tensor,
                row_id_map=row_id_map,
                num_dispatched_tokens=num_dispatched_tokens,
                num_of_tokens_per_rank=num_of_tokens_per_rank,
                pad_multiple=pad_multiple,
                with_probs=probs is not None,
            )
        return combined_token, combined_probs

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
import torch
import os
import shutil
import hybrid_ep_cpp
import warnings

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
        num_sms_dispatch_api: int = None,
        num_sms_combine_api: int = None,
        num_sms_preprocessing_api: int = None,
        # Deprecated parameters
        num_of_hybrid_ep_ranks_per_nvlink_domain: int = None,
        use_mnnvl: bool = None
    ):
        self.group = group
        self.rank = self.group.rank()
        self.group_size = self.group.size()
        assert (
            self.group_size > 1
        ), f"The hybrid-ep kernel should be used with at least 2 ranks, but got {self.group_size}."

        allocator = hybrid_ep_cpp.ExtendedMemoryAllocator()
        self.num_of_hybrid_ep_ranks_per_nvlink_domain = allocator.detect_accessible_ranks(self.group)
        assert (
            self.group_size % self.num_of_hybrid_ep_ranks_per_nvlink_domain == 0
        ), f"The number of ranks {self.group_size} should be divisible by the number of ranks per node {self.num_of_hybrid_ep_ranks_per_nvlink_domain}."

        # Local rank: the active rank in the nvlink domain.
        self.local_rank = self.rank % self.num_of_hybrid_ep_ranks_per_nvlink_domain
        # Node rank: the active rank between the nvlink domains.
        self.node_rank = self.rank // self.num_of_hybrid_ep_ranks_per_nvlink_domain
        # The number of nodes.
        self.num_of_nodes = self.group_size // self.num_of_hybrid_ep_ranks_per_nvlink_domain
        self.use_fp8 = use_fp8

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        sm_count = props.multi_processor_count
        if num_sms_preprocessing_api is None:
            num_sms_preprocessing_api = 108
        num_blocks_permute_api = sm_count * 16
        # Inter-node case should use less SMs for the dispatch and combine APIs.
        if num_sms_dispatch_api is None:
            num_sms_dispatch_api = 32 if self.num_of_nodes == 1 else 8
        if num_sms_combine_api is None:
            num_sms_combine_api = 32 if self.num_of_nodes == 1 else 8
        assert (
            sm_count >= num_sms_preprocessing_api
            and sm_count >= num_sms_dispatch_api
            and sm_count >= num_sms_combine_api
        ), "check the sms occupancy setting"
        # Used SMs for preprocessing of dispatch and permute.
        self.num_sms_preprocessing_api = num_sms_preprocessing_api
        self.num_sms_dispatch_api = num_sms_dispatch_api
        self.num_sms_combine_api = num_sms_combine_api
        self.num_blocks_permute_api = num_blocks_permute_api
        
        # Initialize the BufferConfig for the hybrid-ep buffer allocation.
        self.config = hybrid_ep_cpp.BufferConfig()
        self.config.hidden_dim = hidden_dim
        self.config.max_num_of_tokens_per_rank = max(max_num_of_tokens_per_rank, 512)
        self.config.num_of_experts_per_rank = num_local_experts
        self.config.num_of_ranks_per_node = self.num_of_hybrid_ep_ranks_per_nvlink_domain
        self.config.num_of_nodes = self.num_of_nodes
        self.config.num_of_blocks_dispatch_api = self.num_sms_dispatch_api
        self.config.num_of_blocks_combine_api = self.num_sms_combine_api
        # The SMs of preprocessing, chunk size of dispatch and combine will affact the size of intermediate buffers.
        self.config.num_of_blocks_preprocessing_api = self.num_sms_preprocessing_api
        self.config.num_of_blocks_permute_api = self.num_blocks_permute_api
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
        assert self.config.is_valid(), "The buffer config is not valid."

        # Create C++ buffer - this will allocate all buffers during construction
        self.runtime = hybrid_ep_cpp.HybridEPBuffer(
            self.group, 
            self.config, 
            self.local_rank, 
            self.node_rank, 
            self.group_size, 
            os.path.dirname(os.path.abspath(__file__)), 
            load_cached_kernels = False, 
            use_shared_buffer = True
        )

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
        if max_num_of_tokens_per_rank is None:
            max_num_of_tokens_per_rank = self.config.max_num_of_tokens_per_rank
        else:
            config.max_num_of_tokens_per_rank = max(
                max_num_of_tokens_per_rank, self.config.max_num_of_tokens_per_rank
            )
            self.config.max_num_of_tokens_per_rank = config.max_num_of_tokens_per_rank
        
        config.num_of_experts_per_rank = (
            num_local_experts
            if num_local_experts is not None
            else self.config.num_of_experts_per_rank
        )
        config.num_of_ranks_per_node = self.num_of_hybrid_ep_ranks_per_nvlink_domain
        config.num_of_nodes = self.num_of_nodes

        # Metadata-preprocessing API Config
        config.num_of_blocks_preprocessing_api = self.num_sms_preprocessing_api
        if config.num_of_ranks_per_node >= 64:
            # Use less threads to avoid the register overflow in large EP size.
            config.num_of_threads_per_block_preprocessing_api = int(
                os.getenv("NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API", "256")
            )
        else:
            config.num_of_threads_per_block_preprocessing_api = int(
                os.getenv("NUM_OF_THREADS_PER_BLOCK_PREPROCESSING_API", "512")
            )
        config.num_of_blocks_permute_api = self.num_blocks_permute_api

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
        config.num_of_in_flight_s2g_dispatch_api = int(
            os.getenv("NUM_OF_IN_FLIGHT_S2G_DISPATCH_API", "8")
        )
        config.num_of_tokens_per_chunk_dispatch_api = int(
            os.getenv("NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API", "128")
        )

        # Combine API Config
        config.num_of_blocks_combine_api = self.num_sms_combine_api
        config.device_side_sync_combine_api = True
        # Combine stages config:
        if self.config.num_of_nodes > 1:
            config.num_of_stages_g2s_combine_api = int(
                os.getenv("NUM_OF_STAGES_G2S_COMBINE_API", "5")
            )
        else:
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

        assert config.is_valid(), "The config is not valid."

        # Use the runtime kernel config to update the buffer.
        self.runtime.update_buffer(config)
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
            # Synchronize the stream to make sure the data in the pinned_memory_buffer: num_dispatched_tokens_tensor is ready.
            torch.cuda.current_stream().synchronize()
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
            _,
            _,
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
        # There are 2 tensors are put on the CPU pinned memory
        # 1. num_dispatched_tokens in handle
        # 2. tokens_per_expert
        # If non_blocking is True, no stream synchronization will be used, the all output are on the GPU.
        # Otherwise, num_dispatched_tokens_tensor and tokens_per_expert are on the CPU pinned memory, the stream synchronization will be used to wait for the data in pinned memory.
        non_blocking: bool = False,
        # Deprecated parameters
        num_dispatched_tokens: int = None,
        use_host_meta: bool = None,
    ):
        """
        Dispatch the data to the experts with permute.
        """
        if num_dispatched_tokens is not None:
            warnings.warn("The num_dispatched_tokens is deprecated, it will be removed in the future.")
        if use_host_meta is not None:
            warnings.warn("The use_host_meta is deprecated, it will be removed in the future.")
            non_blocking = not use_host_meta

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
            if non_blocking:
                assert num_permuted_tokens >= 0, "The num_permuted_tokens is required for non-blocking mode."

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
                    non_blocking=non_blocking,
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
                    overflow_flag,
                ) = handle

            # Dispatch phase
            (
                dispatched_token,
                dispatched_probs,
                dispatched_scaling_factor,
                overflow_flag,
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
                num_permuted_tokens=num_permuted_tokens,
                num_of_tokens_per_rank=num_of_tokens_per_rank,
                pad_multiple=pad_multiple,
                non_blocking=non_blocking,
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
                overflow_flag,
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
        handle: tuple = None,
        pad_multiple: int = None,
        # Deprecated parameters
        num_dispatched_tokens: int = None,
    ):
        """
        Combine the data from the experts with unpermute.
        Do not require the routing_map, but the handle is necessary.
        """
        if num_dispatched_tokens is not None:
            warnings.warn("The num_dispatched_tokens is deprecated, it will be removed in the future.")

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
                overflow_flag,
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
                num_of_tokens_per_rank=num_of_tokens_per_rank,
                pad_multiple=pad_multiple,
                with_probs=probs is not None,
            )
        return combined_token, combined_probs

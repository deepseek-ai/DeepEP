// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved
#include "internode.cuh"

// Functions realted to get RDMA context.
static ibv_device *ctx_find_dev(const char *ib_devname) {
  int num_of_device;
  struct ibv_device **dev_list;
  struct ibv_device *ib_dev = NULL;
  dev_list = ibv_get_device_list(&num_of_device);
  // coverity[uninit_use]
  if (num_of_device <= 0) {
    fprintf(stderr, " Did not detect devices \n");
    fprintf(stderr, " If device exists, check if driver is up\n");
    return NULL;
  }
  for (; (ib_dev = *dev_list); ++dev_list) {
    if (!strcmp(ibv_get_device_name(ib_dev), ib_devname))
      break;
  }
  if (!ib_dev) {
    fprintf(stderr, "IB device %s not found\n", ib_devname);
    return NULL;
  }
  return ib_dev;
}

static const char *link_layer_str(int8_t link_layer) {
  switch (link_layer) {
  case IBV_LINK_LAYER_UNSPECIFIED:
  case IBV_LINK_LAYER_INFINIBAND:
    return "IB";
  case IBV_LINK_LAYER_ETHERNET:
    return "Ethernet";
  default:
    return "Unknown";
  }
}

// Functions related to initialization of gverbs_context.
static int get_gpu_handler(struct doca_gpu *handler,
                           struct ibv_context *ib_context, int local_rank) {
  char pciBusId[256];
  int compute_cap_major;
  unsigned int flag = 1;
  cudaError_t cuda_error;
  CUresult cu_error;
  CUdeviceptr dev_ptr = 0UL;
  CUdeviceptr db_gpu_ptr = 0UL;
  CUdeviceptr bf_gpu_ptr = 0UL;
  struct mlx5dv_devx_uar *db_uar = nullptr;
  struct mlx5dv_devx_uar *bf_uar = nullptr;
  // Getting BDF of GPU and setting dev_id.
  cuda_error = cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), local_rank);
  assert(cuda_error == cudaSuccess);
  handler->cuda_dev = local_rank;
  // Determining whether GPU supports ASYNC_STORE_RELEASE.
  cuda_error = cudaDeviceGetAttribute(
      &compute_cap_major, cudaDevAttrComputeCapabilityMajor, local_rank);
  assert(cuda_error == cudaSuccess);
  handler->support_async_store_release =
      compute_cap_major >=
      GPU_FULL_ASYNC_STORE_RELEASE_SUPPORT_COMPUTE_CAP_MAJOR;
  // Determining whether GPU supports DMABUF.
  int support_dmabuf = 0;
  cu_error = cuDeviceGetAttribute(
      &support_dmabuf, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, local_rank);
  assert(cu_error == CUDA_SUCCESS);
  handler->support_dmabuf = support_dmabuf;
  // Determining whether GPU supports wq_gpumem and cq_gpumem.
  cu_error = cuMemAlloc(&dev_ptr, 1 << 11);
  assert(cu_error == CUDA_SUCCESS);
  cu_error =
      cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dev_ptr);
  assert(cu_error == CUDA_SUCCESS);
  handler->support_wq_gpumem = true;
  handler->support_cq_gpumem = true;
  // Determining whether uar_gpumem is supported.
  db_uar =
      mlx5dv_devx_alloc_uar(ib_context, MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED);
#if CUDA_VERSION >= 12020
  if (!db_uar)
    db_uar = mlx5dv_devx_alloc_uar(ib_context, MLX5DV_UAR_ALLOC_TYPE_NC);
#endif
  assert(db_uar);
  cu_error = cuMemHostRegister(db_uar->reg_addr, DOCA_VERBS_DB_UAR_SIZE,
                               CU_MEMHOSTREGISTER_DEVICEMAP |
                                   CU_MEMHOSTREGISTER_PORTABLE |
                                   CU_MEMHOSTREGISTER_IOMEMORY);
  assert(cu_error == CUDA_SUCCESS);
  cu_error = cuMemHostGetDevicePointer(&db_gpu_ptr, db_uar->reg_addr, 0);
  assert(cu_error == CUDA_SUCCESS);
  handler->support_uar_gpumem = true;
  // Determining whether bf_uar is supported.
  bf_uar = mlx5dv_devx_alloc_uar(ib_context, MLX5DV_UAR_ALLOC_TYPE_BF);
  if (bf_uar) {
    cu_error = cuMemHostRegister(bf_uar->reg_addr, DOCA_VERBS_DB_UAR_SIZE,
                                 CU_MEMHOSTREGISTER_DEVICEMAP |
                                     CU_MEMHOSTREGISTER_PORTABLE |
                                     CU_MEMHOSTREGISTER_IOMEMORY);
    assert(cu_error == CUDA_SUCCESS);
    cu_error = cuMemHostGetDevicePointer(&bf_gpu_ptr, bf_uar->reg_addr, 0);
    assert(cu_error == CUDA_SUCCESS);
    handler->support_bf_uar = true;
  } else {
    handler->support_bf_uar = false;
  }
  // Setting support_gdrcopy to false.
  handler->support_gdrcopy = false;
  // Creating mtable.
  try {
    handler->mtable =
        new std::unordered_map<uintptr_t, struct doca_gpu_mtable *>();
  } catch (...) {
    printf("mtable map allocation failed\n");
    assert(0);
  }
  // Debug info.
  printf("cuda_dev:%d, GPU pciBusId:%s, support_gdrcopy:%d, support_dmabuf:%d, "
         "support_wq_gpumem:%d, "
         "support_cq_gpumem:%d, support_uar_gpumem:%d, "
         "support_async_store_release:%d, support_bf_uar:%d\n",
         handler->cuda_dev, pciBusId, handler->support_gdrcopy,
         handler->support_dmabuf, handler->support_wq_gpumem,
         handler->support_cq_gpumem, handler->support_uar_gpumem,
         handler->support_async_store_release, handler->support_bf_uar);
  cuMemFree(dev_ptr);
  if (db_uar) {
    cuMemHostUnregister(db_uar->reg_addr);
    mlx5dv_devx_free_uar(db_uar);
  }
  if (bf_uar) {
    cuMemHostUnregister(bf_uar->reg_addr);
    mlx5dv_devx_free_uar(bf_uar);
  }
  return 0;
}

void setup_qp_init_attr(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                        struct doca_gpu *gpu_handler, struct ibv_pd *ib_pd,
                        int tx_depth) {
  qp_init_attr->gpu_dev = gpu_handler;
  qp_init_attr->ibpd = ib_pd;
  qp_init_attr->sq_nwqe = tx_depth;
  qp_init_attr->nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
  qp_init_attr->mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
}

int create_and_place_qps(struct gverbs_context *g_ctx,
                         struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                         int num_qps) {
  int status = 0;
  for (int i = 0; i < num_qps; i++) {
    struct doca_gpu_verbs_qp_hl *qp_hl = NULL;
    status = doca_gpu_verbs_create_qp_hl(qp_init_attr, &qp_hl);
    if (status) {
      fprintf(stderr, "Failed to create QP\n");
      assert(0);
    }
    g_ctx->qp_hls[i] = qp_hl;
  }
  return 0;
}

static int setup_qp_attr_for_modify(struct doca_verbs_qp_attr *qp_attr,
                                    struct remote_info *rem_dest,
                                    struct ibv_context *ib_context) {
  int status = 0;
  status = doca_verbs_qp_attr_set_dest_qp_num(qp_attr, rem_dest->qpn);
  assert(status == 0);
  struct doca_verbs_ah *ah = nullptr;
  status = doca_verbs_ah_create(ib_context, &ah);
  assert(status == 0);
  status = doca_verbs_ah_set_dlid(ah, rem_dest->lid);
  assert(status == 0);
  status = doca_verbs_ah_set_sl(ah, 0);
  assert(status == 0);
  status = doca_verbs_ah_set_sgid_index(ah, rem_dest->gid_index);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_ah_attr(qp_attr, ah);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_pkey_index(qp_attr, PKEY_INDEX);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_rq_psn(qp_attr, 0);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_sq_psn(qp_attr, 0);
  assert(status == 0);
  status =
      doca_verbs_qp_attr_set_path_mtu(qp_attr, DOCA_VERBS_MTU_SIZE_1K_BYTES);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_min_rnr_timer(qp_attr, 1);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_ack_timeout(qp_attr, 22);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_retry_cnt(qp_attr, 7);
  assert(status == 0);
  status = doca_verbs_qp_attr_set_rnr_retry(qp_attr, 1);
  assert(status == 0);
  return 0;
}

int doca_gpunetio_test_change_qp_state(struct doca_gpu_verbs_qp_hl *qp,
                                       struct doca_verbs_qp_attr *qp_attr,
                                       int attr_mask) {
  int status;
  int init_mask = attr_mask;
  status = doca_verbs_qp_attr_set_next_state(qp_attr, DOCA_VERBS_QP_STATE_INIT);
  if (status) {
    fprintf(stderr, "Failed to set QP next_state to INIT: %d\n", status);
    return status;
  }
  attr_mask = init_mask;
  status = doca_verbs_qp_modify(qp->qp, qp_attr, attr_mask);
  if (status != 0) {
    fprintf(stderr, "Failed to modify QP state to INIT\n");
    return status;
  }
  attr_mask = DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
              DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_AH_ATTR |
              DOCA_VERBS_QP_ATTR_PATH_MTU | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER;
  status = doca_verbs_qp_attr_set_next_state(qp_attr, DOCA_VERBS_QP_STATE_RTR);
  if (status) {
    fprintf(stderr, "Failed to set QP next_state to RTR: %d\n", status);
    return status;
  }
  status = doca_verbs_qp_modify(qp->qp, qp_attr, attr_mask);
  if (status != 0) {
    fprintf(stderr, "Failed to modify QP state to RTR\n");
    return status;
  }
  status = doca_verbs_qp_attr_set_next_state(qp_attr, DOCA_VERBS_QP_STATE_RTS);
  if (status) {
    fprintf(stderr, "Failed to set QP next_state to RTS: %d\n", status);
    return status;
  }
  attr_mask = DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
              DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
              DOCA_VERBS_QP_ATTR_RNR_RETRY;
  status = doca_verbs_qp_modify(qp->qp, qp_attr, attr_mask);
  if (status != 0) {
    fprintf(stderr, "Failed to modify QP state to RTS\n");
    return status;
  }
  return 0;
}

static int setup_qp_attr_and_set_qp(struct gverbs_context *g_ctx,
                                    struct ibv_context *ib_context,
                                    struct remote_info *rem_dest,
                                    struct doca_verbs_qp_attr *qp_attr,
                                    int num_of_blocks, int num_of_nodes,
                                    int node_rank, uint32_t qp_cnt) {
  int status = 0;
  int attr_mask = DOCA_VERBS_QP_ATTR_NEXT_STATE |
                  DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                  DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ |
                  DOCA_VERBS_QP_ATTR_PORT_NUM | DOCA_VERBS_QP_ATTR_PKEY_INDEX;
  for (int qp_idx = 0; qp_idx < num_of_blocks; ++qp_idx) {
    for (int peer_idx = 0; peer_idx < num_of_nodes - 1; ++peer_idx) {
      int actual_node_idx = peer_idx < node_rank ? peer_idx : (peer_idx + 1);
      int actual_idx_in_node =
          peer_idx < node_rank ? (node_rank - 1) : node_rank;
      int my_idx = peer_idx + qp_idx * (num_of_nodes - 1);
      int rem_idx = actual_node_idx * qp_cnt + qp_idx * (num_of_nodes - 1) +
                    actual_idx_in_node;
      struct remote_info *r_info = &rem_dest[rem_idx];
      struct doca_gpu_verbs_qp_hl *qp = g_ctx->qp_hls[my_idx];
      setup_qp_attr_for_modify(qp_attr, r_info, ib_context);
      doca_gpunetio_test_change_qp_state(qp, qp_attr, attr_mask);
    }
  }
  return 0;
}

void set_IB_device_list(std::vector<std::string> ib_dev_name_list) {
  assert(ib_dev_name_list.size() <= MAX_NUM_OF_RANKS_PER_NODE);
  for (int i = 0; i < ib_dev_name_list.size(); ++i) {
    IBV_DEV_NAME_LIST[i] = ib_dev_name_list[i].c_str();
  }
}

RDMACoordinator::RDMACoordinator(
  pybind11::object process_group,
  int node_rank,
  int local_rank, 
  BufferConfig config, 
  ExtendedMemoryAllocator allocator
) : process_group(process_group), node_rank(node_rank), local_rank(local_rank), buffer_config(config), allocator(allocator){
    assert(buffer_config.num_of_nodes > 1);
    // Get name of ibv device.
    char *ib_devname = IBV_DEV_NAME_LIST[local_rank];
    // Find ib device and get ibv_context.
    struct ibv_device *ib_dev = ctx_find_dev(ib_devname);

    ib_context = ibv_open_device(ib_dev);;
    ibv_query_port(ib_context, IB_PORT, &port_attr);
    auto transport_type = ib_context->device->transport_type;
    assert(transport_type == IBV_TRANSPORT_IB);
    // Alloc protect domain.
    ib_pd = ibv_alloc_pd(ib_context);
    gpu_handler = (struct doca_gpu *)calloc(1, sizeof(struct doca_gpu));
    get_gpu_handler(gpu_handler, ib_context, local_rank);
    mr_access_flag = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                       IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_RELAXED_ORDERING;
}

RDMACoordinator::~RDMACoordinator() {
   destory();
}

void RDMACoordinator::allocate_dispatch_rdma_buffers(DispatchBuffers &dispatch_buffers) {
  size_t sizeof_token_data_type = get_token_data_type_size(dispatch_buffers.data_type);

  // Calculate rdma buffers sizes
  auto attn_input_token_elts = buffer_config.max_num_of_tokens_per_rank * buffer_config.hidden_dim;
  auto attn_input_prob_elts = buffer_config.max_num_of_tokens_per_rank 
                            * (buffer_config.num_of_experts_per_rank 
                            * buffer_config.num_of_ranks_per_node 
                            * buffer_config.num_of_nodes);
  auto attn_input_token_scaling_factor_elts = buffer_config.max_num_of_tokens_per_rank 
                                            * (buffer_config.hidden_dim / 128);
  auto rdma_inter_node_group_token_elts = buffer_config.max_num_of_tokens_per_rank * 
                                          (buffer_config.num_of_nodes - 1) * 
                                          buffer_config.hidden_dim;
  auto rdma_inter_node_group_prob_elts = buffer_config.max_num_of_tokens_per_rank 
                                        * (buffer_config.num_of_nodes - 1) 
                                        * (buffer_config.num_of_experts_per_rank 
                                        * buffer_config.num_of_ranks_per_node);
  auto rdma_inter_node_group_scaling_factor_elts = buffer_config.max_num_of_tokens_per_rank * 
                                                    (buffer_config.num_of_nodes - 1) * (buffer_config.hidden_dim / 128);
  auto rdma_inter_node_group_flags_elts = (buffer_config.max_num_of_tokens_per_rank /
                                           buffer_config.num_of_tokens_per_chunk_dispatch_api) *
                                          (buffer_config.num_of_nodes - 1);
                                          
  // Allocate RDMA buffers
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_token,
                        attn_input_token_elts * sizeof_token_data_type));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_prob,
                        attn_input_prob_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_scaling_factor,
                        attn_input_token_scaling_factor_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_token,
                        rdma_inter_node_group_token_elts * sizeof_token_data_type));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_prob,
                        rdma_inter_node_group_prob_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_scaling_factor,
                        rdma_inter_node_group_scaling_factor_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.rdma_inter_node_group_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(dispatch_buffers.rdma_inter_node_group_flags, 0, 
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.attn_input_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(dispatch_buffers.attn_input_flags, 0, 
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
                    
  // Allocate RDMA flags
  CUDA_CHECK(cudaMalloc((void**)&dispatch_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(dispatch_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));

  // Allocate memory region
  attn_input_token_mr = ibv_reg_mr(ib_pd, dispatch_buffers.attn_input_token,
                        attn_input_token_elts * sizeof_token_data_type, mr_access_flag);
  rdma_inter_node_group_token_mr = ibv_reg_mr(ib_pd, dispatch_buffers.rdma_inter_node_group_token,
                        rdma_inter_node_group_token_elts * sizeof_token_data_type, mr_access_flag);
  attn_input_flags_mr = ibv_reg_mr(ib_pd, dispatch_buffers.attn_input_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t), mr_access_flag);
  rdma_inter_node_group_flags_mr = ibv_reg_mr(ib_pd, dispatch_buffers.rdma_inter_node_group_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t), mr_access_flag);
  attn_input_prob_mr = ibv_reg_mr(ib_pd, dispatch_buffers.attn_input_prob,
                        attn_input_prob_elts * sizeof(float), mr_access_flag);
  rdma_inter_node_group_prob_mr = ibv_reg_mr(ib_pd, dispatch_buffers.rdma_inter_node_group_prob,
                        rdma_inter_node_group_prob_elts * sizeof(float), mr_access_flag);
  attn_input_token_scaling_factor_mr = ibv_reg_mr(ib_pd, dispatch_buffers.attn_input_scaling_factor,
                        attn_input_token_scaling_factor_elts * sizeof(float), mr_access_flag);
  rdma_inter_node_group_scaling_factor_mr = ibv_reg_mr(ib_pd, dispatch_buffers.rdma_inter_node_group_scaling_factor,
                        rdma_inter_node_group_scaling_factor_elts * sizeof(float), mr_access_flag);

  // Set dispatch queue pair attributes.
  int num_of_dispatch_qps = (buffer_config.num_of_nodes - 1) * buffer_config.num_of_blocks_dispatch_api;
  memset(&dispatch_gverbs_ctx, 0, sizeof(gverbs_context));
  if (GID_INDEX != -1) {
    ibv_query_gid(ib_context, IB_PORT, GID_INDEX, &dispatch_gverbs_ctx.gid);
  }
  dispatch_gverbs_ctx.qp_init_attr = (struct doca_gpu_verbs_qp_init_attr_hl *)calloc(1, sizeof(struct doca_gpu_verbs_qp_init_attr_hl));
  setup_qp_init_attr(dispatch_gverbs_ctx.qp_init_attr, gpu_handler, ib_pd, 3 * buffer_config.max_num_of_tokens_per_rank + 1);
  dispatch_gverbs_ctx.qp_hls = (struct doca_gpu_verbs_qp_hl **)calloc(sizeof(struct doca_gpu_verbs_qp_hl *), num_of_dispatch_qps);
  create_and_place_qps(&dispatch_gverbs_ctx, dispatch_gverbs_ctx.qp_init_attr, num_of_dispatch_qps);
  doca_verbs_qp_attr_create(&dispatch_gverbs_ctx.qp_attr);
  doca_verbs_qp_attr_set_port_num(dispatch_gverbs_ctx.qp_attr, IB_PORT);
  doca_verbs_qp_attr_set_allow_remote_write(dispatch_gverbs_ctx.qp_attr, 1);
  doca_verbs_qp_attr_set_allow_remote_read(dispatch_gverbs_ctx.qp_attr, 1);
  doca_verbs_qp_attr_set_allow_remote_atomic(dispatch_gverbs_ctx.qp_attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC);
  
  // Construct dispatch remote_info
  dispatch_remote_info_vec = static_cast<remote_info *>(calloc(buffer_config.num_of_nodes * num_of_dispatch_qps, sizeof(remote_info)));
  remote_info *my_dispatch_info = static_cast<remote_info *>(calloc(num_of_dispatch_qps, sizeof(remote_info)));
  int token_stride = buffer_config.max_num_of_tokens_per_rank * buffer_config.hidden_dim;
  int flag_stride = (buffer_config.max_num_of_tokens_per_rank - 1) / buffer_config.num_of_tokens_per_chunk_dispatch_api + 1;
  int prob_stride = buffer_config.max_num_of_tokens_per_rank * buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node;
  int scaling_factor_stride = buffer_config.max_num_of_tokens_per_rank * buffer_config.hidden_dim / 128;
  // For each queue pair to the same remote. 
  for (int qp_idx = 0; qp_idx < buffer_config.num_of_blocks_dispatch_api; ++qp_idx) {
    // For each remote.
    for (int peer_idx = 0; peer_idx < buffer_config.num_of_nodes - 1; ++peer_idx) {
      // Fill rkeys and raddrs into remote_info.
      int idx = qp_idx * (buffer_config.num_of_nodes - 1) + peer_idx;
      struct remote_info *curr_info = my_dispatch_info + idx;
      curr_info->lid = port_attr.lid;
      curr_info->qpn = doca_verbs_qp_get_qpn(dispatch_gverbs_ctx.qp_hls[idx]->qp);
      curr_info->gid_index = GID_INDEX;
      memset(&curr_info->gid, 0, sizeof(curr_info->gid));;
      memcpy(curr_info->gid.raw, dispatch_gverbs_ctx.gid.raw, 16);
      curr_info->token_rkey = rdma_inter_node_group_token_mr->rkey;
      curr_info->token_vaddr = (uintptr_t)((TOKEN_DATA_TYPE *)rdma_inter_node_group_token_mr->addr +
                                            peer_idx * token_stride);
      curr_info->flag_rkey = rdma_inter_node_group_flags_mr->rkey;
      curr_info->flag_vaddr = (uintptr_t)((uint64_t *)rdma_inter_node_group_flags_mr->addr +
                                          peer_idx * flag_stride);
      curr_info->prob_rkey = rdma_inter_node_group_prob_mr->rkey;
      curr_info->prob_vaddr = (uintptr_t)((float *)rdma_inter_node_group_prob_mr->addr +
                                          peer_idx * prob_stride);
      curr_info->scaling_factor_rkey = rdma_inter_node_group_scaling_factor_mr->rkey;
      curr_info->scaling_factor_vaddr = (uintptr_t)((float *)rdma_inter_node_group_scaling_factor_mr->addr +
                                                    peer_idx * scaling_factor_stride);
    }
  }
  exchange_remote_rmda_info(dispatch_remote_info_vec, my_dispatch_info, num_of_dispatch_qps);

  // Init queue pairs.
  setup_qp_attr_and_set_qp(&dispatch_gverbs_ctx, ib_context, dispatch_remote_info_vec, dispatch_gverbs_ctx.qp_attr,
    buffer_config.num_of_blocks_dispatch_api, buffer_config.num_of_nodes, node_rank, num_of_dispatch_qps);
  // Move queue pairs to GPU.
  doca_gpu_dev_verbs_qp **h_qps_gpu = (doca_gpu_dev_verbs_qp**)calloc(sizeof(*h_qps_gpu), num_of_dispatch_qps);
  for (int idx = 0; idx <  num_of_dispatch_qps; ++idx) {
    doca_gpu_verbs_get_qp_dev(dispatch_gverbs_ctx.qp_hls[idx]->qp_gverbs, &h_qps_gpu[idx]);
  }
  CUDA_CHECK(cudaMalloc(&dispatch_gverbs_ctx.d_qps_gpu, num_of_dispatch_qps * sizeof(doca_gpu_dev_verbs_qp*)));
  CUDA_CHECK(cudaMemcpy(dispatch_gverbs_ctx.d_qps_gpu, h_qps_gpu, num_of_dispatch_qps * sizeof(doca_gpu_dev_verbs_qp*), cudaMemcpyHostToDevice));
  // Move Memory regions to GPU.
  dispatch_mr_info_h = (hybrid_ep::dispatch_memory_region_info_t *)calloc(sizeof(hybrid_ep::dispatch_memory_region_info_t), num_of_dispatch_qps);
  CUDA_CHECK(cudaMalloc((void**)&dispatch_mr_info_d, num_of_dispatch_qps * sizeof(hybrid_ep::dispatch_memory_region_info_t)));
  for (int qp_idx = 0; qp_idx < buffer_config.num_of_blocks_dispatch_api; ++qp_idx) {
    for (int peer_idx = 0; peer_idx < buffer_config.num_of_nodes - 1; ++peer_idx) {
      int actual_node_idx = peer_idx < node_rank ? peer_idx : (peer_idx + 1);
      int actual_idx_in_node = peer_idx < node_rank ? (node_rank - 1) : node_rank;
      int my_idx = qp_idx * (buffer_config.num_of_nodes - 1) + peer_idx;
      int rem_idx = actual_node_idx * num_of_dispatch_qps + qp_idx * (buffer_config.num_of_nodes - 1) + actual_idx_in_node;
      struct hybrid_ep::dispatch_memory_region_info_t *data = dispatch_mr_info_h + my_idx;
      data->token_laddr = (uint64_t)attn_input_token_mr->addr;
      data->token_lkey = htobe32(attn_input_token_mr->lkey);
      data->token_raddr = dispatch_remote_info_vec[rem_idx].token_vaddr;
      data->token_rkey = htobe32(dispatch_remote_info_vec[rem_idx].token_rkey);
      data->flag_laddr = (uint64_t)((uint64_t *)attn_input_flags_mr->addr + peer_idx * flag_stride);
      data->flag_lkey = htobe32(attn_input_flags_mr->lkey);
      data->flag_raddr = dispatch_remote_info_vec[rem_idx].flag_vaddr;
      data->flag_rkey = htobe32(dispatch_remote_info_vec[rem_idx].flag_rkey);
      data->prob_laddr = (uint64_t)attn_input_prob_mr->addr;
      data->prob_lkey = htobe32(attn_input_prob_mr->lkey);
      data->prob_raddr = dispatch_remote_info_vec[rem_idx].prob_vaddr;
      data->prob_rkey = htobe32(dispatch_remote_info_vec[rem_idx].prob_rkey);
    }
  }
  CUDA_CHECK(cudaMemcpy(dispatch_mr_info_d, dispatch_mr_info_h, num_of_dispatch_qps * sizeof(hybrid_ep::dispatch_memory_region_info_t), cudaMemcpyHostToDevice));

  // Free temporary resources.
  free(my_dispatch_info);
  free(h_qps_gpu);
}

void RDMACoordinator::allocate_combine_rdma_buffers(CombineBuffers &combine_buffers) {
  // Calculate rdma buffers sizes
  auto rdma_intra_node_red_token_elts = buffer_config.max_num_of_tokens_per_rank *
                                        (buffer_config.num_of_nodes - 1) * buffer_config.hidden_dim;
  auto rdma_intra_node_red_prob_elts = buffer_config.max_num_of_tokens_per_rank * (buffer_config.num_of_nodes - 1) *
                                       (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
  auto rdma_inter_node_group_token_elts = buffer_config.max_num_of_tokens_per_rank *
                                          (buffer_config.num_of_nodes - 1) * buffer_config.hidden_dim;
  auto rdma_inter_node_group_prob_elts = buffer_config.max_num_of_tokens_per_rank * (buffer_config.num_of_nodes - 1) *
                                         (buffer_config.num_of_experts_per_rank * buffer_config.num_of_ranks_per_node);
  auto rdma_inter_node_group_flags_elts = (buffer_config.max_num_of_tokens_per_rank /
                                           buffer_config.num_of_tokens_per_chunk_combine_api) *
                                          (buffer_config.num_of_nodes - 1);
                                    
  // Allocate RDMA buffers
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_intra_node_red_token,
                        rdma_intra_node_red_token_elts * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_intra_node_red_prob,
                        rdma_intra_node_red_prob_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_token,
                        rdma_inter_node_group_token_elts * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_prob,
                        rdma_inter_node_group_prob_elts * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.rdma_inter_node_group_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(combine_buffers.rdma_inter_node_group_flags, 0, 
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.attn_output_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(combine_buffers.attn_output_flags, 0, 
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t)));
  
  // Allocate RDMA flags
  CUDA_CHECK(cudaMalloc((void**)&combine_buffers.expected_rdma_flag_value, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(combine_buffers.expected_rdma_flag_value, 0, sizeof(uint64_t)));

  rdma_intra_node_red_token_mr = ibv_reg_mr(ib_pd, combine_buffers.rdma_intra_node_red_token,
                        rdma_intra_node_red_token_elts * sizeof(uint16_t), mr_access_flag);
  rdma_inter_node_group_token_mr = ibv_reg_mr(ib_pd, combine_buffers.rdma_inter_node_group_token,
                        rdma_inter_node_group_token_elts * sizeof(uint16_t), mr_access_flag);
  attn_output_flags_mr = ibv_reg_mr(ib_pd, combine_buffers.attn_output_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t), mr_access_flag);
  rdma_inter_node_group_flags_mr = ibv_reg_mr(ib_pd, combine_buffers.rdma_inter_node_group_flags,
                        rdma_inter_node_group_flags_elts * sizeof(uint64_t), mr_access_flag);
  rdma_intra_node_red_prob_mr = ibv_reg_mr(ib_pd, combine_buffers.rdma_intra_node_red_prob,
                        rdma_intra_node_red_prob_elts * sizeof(float), mr_access_flag);
  rdma_inter_node_group_prob_mr = ibv_reg_mr(ib_pd, combine_buffers.rdma_inter_node_group_prob,
                        rdma_inter_node_group_prob_elts * sizeof(float), mr_access_flag);

  // Set combine queue pair attributes.
  int num_of_combine_qps = (buffer_config.num_of_nodes - 1) * buffer_config.num_of_blocks_combine_api;
  memset(&combine_gverbs_ctx, 0, sizeof(gverbs_context));
  if (GID_INDEX != -1) {
    ibv_query_gid(ib_context, IB_PORT, GID_INDEX, &combine_gverbs_ctx.gid);
  }
  combine_gverbs_ctx.qp_init_attr = (struct doca_gpu_verbs_qp_init_attr_hl *)calloc(1, sizeof(struct doca_gpu_verbs_qp_init_attr_hl));
  setup_qp_init_attr(combine_gverbs_ctx.qp_init_attr, gpu_handler, ib_pd, 2 * buffer_config.max_num_of_tokens_per_rank + 1);
  combine_gverbs_ctx.qp_hls = (struct doca_gpu_verbs_qp_hl **)calloc(sizeof(struct doca_gpu_verbs_qp_hl *), num_of_combine_qps);
  create_and_place_qps(&combine_gverbs_ctx, combine_gverbs_ctx.qp_init_attr, num_of_combine_qps);
  doca_verbs_qp_attr_create(&combine_gverbs_ctx.qp_attr);
  doca_verbs_qp_attr_set_port_num(combine_gverbs_ctx.qp_attr, IB_PORT);
  doca_verbs_qp_attr_set_allow_remote_write(combine_gverbs_ctx.qp_attr, 1);
  doca_verbs_qp_attr_set_allow_remote_read(combine_gverbs_ctx.qp_attr, 1);
  doca_verbs_qp_attr_set_allow_remote_atomic(combine_gverbs_ctx.qp_attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC);

  // Construct combine remote_info
  combine_remote_info_vec = static_cast<remote_info *>(calloc(buffer_config.num_of_nodes * num_of_combine_qps, sizeof(remote_info)));
  remote_info *my_combine_info = static_cast<remote_info *>(calloc(num_of_combine_qps, sizeof(remote_info)));
  // For each queue pair to the same remote. 
  for (int qp_idx = 0; qp_idx < buffer_config.num_of_blocks_combine_api; ++qp_idx) {
    // For each remote.
    for (int peer_idx = 0; peer_idx < buffer_config.num_of_nodes - 1; ++peer_idx) {
      // Fill rkeys and raddrs into remote_info.
      int idx = qp_idx * (buffer_config.num_of_nodes - 1) + peer_idx;
      struct remote_info *curr_info = my_combine_info + idx;
      curr_info->lid = port_attr.lid;
      curr_info->qpn = doca_verbs_qp_get_qpn(combine_gverbs_ctx.qp_hls[idx]->qp);
      curr_info->gid_index = GID_INDEX;
      memset(&curr_info->gid, 0, sizeof(curr_info->gid));;
      memcpy(curr_info->gid.raw, combine_gverbs_ctx.gid.raw, 16);
      curr_info->token_rkey = combine_rdma_inter_node_group_token_mr->rkey;
      curr_info->token_vaddr = (uintptr_t)((uint16_t *)combine_rdma_inter_node_group_token_mr->addr +
                                           peer_idx * token_stride);
      curr_info->flag_rkey = combine_rdma_inter_node_group_flags_mr->rkey;
      curr_info->flag_vaddr = (uintptr_t)((uint64_t *)combine_rdma_inter_node_group_flags_mr->addr +
                                          peer_idx * flag_stride);
      curr_info->prob_rkey = combine_rdma_inter_node_group_prob_mr->rkey;
      curr_info->prob_vaddr = (uintptr_t)((float *)combine_rdma_inter_node_group_prob_mr->addr +
                                          peer_idx * prob_stride);
    }
  }
  exchange_remote_rmda_info(combine_remote_info_vec, my_combine_info, num_of_combine_qps);
  // Init queue pairs.
  setup_qp_attr_and_set_qp(&combine_gverbs_ctx, ib_context, combine_remote_info_vec, combine_gverbs_ctx.qp_attr,
    buffer_config.num_of_blocks_combine_api, buffer_config.num_of_nodes, node_rank, num_of_combine_qps);
  // Move queue pairs to GPU.
  doca_gpu_dev_verbs_qp **h_qps_gpu = (doca_gpu_dev_verbs_qp**)calloc(sizeof(*h_qps_gpu), num_of_combine_qps);
  for (int idx = 0; idx <  num_of_combine_qps; ++idx) {
    doca_gpu_verbs_get_qp_dev(combine_gverbs_ctx.qp_hls[idx]->qp_gverbs, &h_qps_gpu[idx]);
  }
  CUDA_CHECK(cudaMalloc(&combine_gverbs_ctx.d_qps_gpu, num_of_combine_qps * sizeof(doca_gpu_dev_verbs_qp*)));
  CUDA_CHECK(cudaMemcpy(combine_gverbs_ctx.d_qps_gpu, h_qps_gpu, num_of_combine_qps * sizeof(doca_gpu_dev_verbs_qp*), cudaMemcpyHostToDevice));
  // Move Memory regions to GPU.
  combine_mr_info_h = (hybrid_ep::combine_memory_region_info_t *)calloc(sizeof(hybrid_ep::combine_memory_region_info_t), num_of_combine_qps);
  CUDA_CHECK(cudaMalloc((void**)&combine_mr_info_d, num_of_combine_qps * sizeof(hybrid_ep::combine_memory_region_info_t)));
  for (int qp_idx = 0; qp_idx < buffer_config.num_of_blocks_combine_api; ++qp_idx) {
    for (int peer_idx = 0; peer_idx < buffer_config.num_of_nodes - 1; ++peer_idx) {
      int actual_node_idx = peer_idx < node_rank ? peer_idx : (peer_idx + 1);
      int actual_idx_in_node = peer_idx < node_rank ? (node_rank - 1) : node_rank;
      int my_idx = qp_idx * (buffer_config.num_of_nodes - 1) + peer_idx;
      int rem_idx = actual_node_idx * num_of_combine_qps + qp_idx * (buffer_config.num_of_nodes - 1) + actual_idx_in_node;
      struct hybrid_ep::combine_memory_region_info_t *data = combine_mr_info_h + my_idx;
      data->token_laddr = (uint64_t)((uint16_t *)rdma_intra_node_red_token_mr->addr);
      data->token_lkey = htobe32(rdma_intra_node_red_token_mr->lkey);
      data->token_raddr = combine_remote_info_vec[rem_idx].token_vaddr;
      data->token_rkey = htobe32(combine_remote_info_vec[rem_idx].token_rkey);
      data->flag_laddr = (uint64_t)((uint64_t *)combine_attn_input_flags_mr->addr + peer_idx * flag_stride);
      data->flag_lkey = htobe32(combine_attn_input_flags_mr->lkey);
      data->flag_raddr = combine_remote_info_vec[rem_idx].flag_vaddr;
      data->flag_rkey = htobe32(combine_remote_info_vec[rem_idx].flag_rkey);
      data->prob_laddr = (uint64_t)rdma_intra_node_red_prob_mr->addr;
      data->prob_lkey = htobe32(rdma_intra_node_red_prob_mr->lkey);
      data->prob_raddr = combine_remote_info_vec[rem_idx].prob_vaddr;
      data->prob_rkey = htobe32(combine_remote_info_vec[rem_idx].prob_rkey);
    }
  }
  CUDA_CHECK(cudaMemcpy(combine_mr_info_d, combine_mr_info_h, num_of_combine_qps * sizeof(hybrid_ep::combine_memory_region_info_t), cudaMemcpyHostToDevice));

  // Free temporary resources.
  free(my_combine_info);
  free(h_qps_gpu);
}    

void RDMACoordinator::destory() {
  CUDA_CHECK(cudaDeviceSynchronize());

  // Close memory regions
  ibv_dereg_mr(attn_input_token_mr);
  ibv_dereg_mr(rdma_inter_node_group_token_mr);
  ibv_dereg_mr(attn_input_flags_mr);
  ibv_dereg_mr(rdma_inter_node_group_flags_mr);
  ibv_dereg_mr(attn_input_prob_mr);
  ibv_dereg_mr(rdma_inter_node_group_prob_mr);
  ibv_dereg_mr(attn_input_token_scaling_factor_mr);
  ibv_dereg_mr(rdma_inter_node_group_scaling_factor_mr);
  ibv_dereg_mr(rdma_intra_node_red_token_mr);
  ibv_dereg_mr(rdma_inter_node_group_token_mr);
  ibv_dereg_mr(rdma_intra_node_red_prob_mr);
  ibv_dereg_mr(rdma_inter_node_group_prob_mr);
  ibv_dereg_mr(attn_output_flags_mr);
  ibv_dereg_mr(rdma_inter_node_group_flags_mr);

  // Free misc resources.
  free(dispatch_remote_info_vec);
  free(dispatch_mr_info_h);
  CUDA_CHECK(cudaFree(dispatch_gverbs_ctx.d_qps_gpu));
  CUDA_CHECK(cudaFree(dispatch_mr_info_d));
  free(combine_remote_info_vec);
  free(combine_mr_info_h);
  CUDA_CHECK(cudaFree(combine_gverbs_ctx.d_qps_gpu));
  CUDA_CHECK(cudaFree(combine_mr_info_d));

  int num_of_dispatch_qps = (buffer_config.num_of_nodes - 1) * buffer_config.num_of_blocks_dispatch_api;
  int num_of_combine_qps = (buffer_config.num_of_nodes - 1) * buffer_config.num_of_blocks_combine_api;
  for (int idx = 0; idx < num_of_dispatch_qps; ++idx) {
    doca_gpu_verbs_destroy_qp_hl(dispatch_gverbs_ctx.qp_hls[idx]);
  }
  for (int idx = 0; idx < num_of_combine_qps; ++idx) {
    doca_gpu_verbs_destroy_qp_hl(combine_gverbs_ctx.qp_hls[idx]);
  }
  free(dispatch_gverbs_ctx.qp_hls);
  free(dispatch_gverbs_ctx.qp_init_attr);
  free(combine_gverbs_ctx.qp_hls);
  free(combine_gverbs_ctx.qp_init_attr);
  doca_verbs_qp_attr_destroy(dispatch_gverbs_ctx.qp_attr);
  doca_verbs_qp_attr_destroy(combine_gverbs_ctx.qp_attr);
  // Dealloc protect domain.
  ibv_dealloc_pd(ib_pd);
  // Close device.
  ibv_close_device(ib_context);
  delete gpu_handler->mtable;
  free(gpu_handler);
}

void RDMACoordinator::exchange_remote_rmda_info(remote_info* dst, remote_info *src, int num_of_qps) {
  // Exchange between ranks.
  try {
    auto torch_distributed = py::module_::import("torch.distributed");

    torch::Tensor buffer = torch::empty({num_of_qps * sizeof(remote_info)}, torch::kInt8, at::device(at::kCUDA));
    CUDA_CHECK(cudaMemcpy(buffer.data_ptr<int8_t>(), reinterpret_cast<int8_t *>(src), num_of_qps * sizeof(remote_info), cudaMemcpyHostToDevice));
    
    // Get world size from process group
    int world_size = process_group.attr("size")().cast<int>();
    // Create empty tensors for allgather output
    py::list output_list;
    for (int i = 0; i < world_size; i++) {
      output_list.append(torch::empty_like(buffer));
    }
    torch_distributed.attr("all_gather")(output_list, buffer, process_group);
    
    // Move the gathered remote info to CPU.
    for(int i = local_rank; i < world_size; i += buffer_config.num_of_nodes) {
      CUDA_CHECK(cudaMemcpy(dst + i * num_of_qps, output_list[i].cast<torch::Tensor>().data_ptr<int8_t>(), num_of_qps * sizeof(remote_info), cudaMemcpyDeviceToHost));
    }
    
  } catch (const std::exception& e) {
    throw std::runtime_error(
      "C++ distributed communication failed: " + std::string(e.what())
    );
  }
}
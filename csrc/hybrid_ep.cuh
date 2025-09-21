// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once
#include "kernels/hybrid_ep_backend_configs.hpp"
#include "kernels/hybrid_ep_backend.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

inline std::string type_to_string(TOKEN_DATA_TYPE token_data_type) {
  switch (token_data_type) {
  case TOKEN_DATA_TYPE::UINT16:
    return "uint16_t";
  case TOKEN_DATA_TYPE::UINT8:
    return "uint8_t";
  default:
    return "unknown";
  }
}

union MemHandleInner{
  cudaIpcMemHandle_t cuda_ipc_mem_handle;
  CUmemFabricHandle cu_mem_fabric_handle;
};

struct MemHandle{
  MemHandleInner inner;
  size_t size;
};

// Utility function to get token data type size
inline size_t get_token_data_type_size(TOKEN_DATA_TYPE data_type) {
  switch (data_type) {
    case TOKEN_DATA_TYPE::UINT8:
      return sizeof(uint8_t);
    case TOKEN_DATA_TYPE::UINT16:
      return sizeof(uint16_t);
    default:
      throw std::runtime_error("Invalid token data type:" + std::to_string(static_cast<int>(data_type)));
  }
}

// Round-up allocation size to fabric granularity.
inline size_t get_size_align_to_granularity(size_t size_raw, size_t granularity){
  size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
  if(size == 0) size = granularity;
  return size;
}

// Device memory allocator, allocate local device memory. Support both normal cudaMalloc and fabric allocator.
inline void device_mem_malloc(void** ptr, size_t size_raw, bool enable_fabric){
  if(enable_fabric){
    CUdevice device;
    CU_CHECK(cuCtxGetDevice(&device));

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.location.id = device;

    size_t granularity = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    size_t size = get_size_align_to_granularity(size_raw, granularity);

    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemCreate(&handle, size, &prop, 0));

    CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
    CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &access_desc, 1));
  }else{
    CUDA_CHECK(cudaMalloc(ptr, size_raw));
  }
}

// Get sharable memory handle of local device memory for remote ranks to access. Support both IPC handle and fabric handle.
inline void get_device_mem_handle(MemHandle* mem_handle, void* ptr, bool enable_fabric){
  size_t size = 0;
  CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

  mem_handle->size = size;

  if(enable_fabric){
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));
    CU_CHECK(cuMemExportToShareableHandle(&mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
  }else{
    CUDA_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
  }
}

// Open sharable memory handle from other remote ranks and map it for local device to access. Support both IPC handle and fabric handle.
inline void open_device_mem_handle(void** ptr, MemHandle* mem_handle, bool enable_fabric){
  if(enable_fabric){
    CUdevice device;
    CU_CHECK(cuCtxGetDevice(&device));
    size_t size = mem_handle->size;

    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));

    CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, 0, 0, 0));
    CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &access_desc, 1));
  }else{
    CUDA_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));
  }
}

// Close and unmap sharable memory handle from other remote ranks. Support both IPC handle and fabric handle.
inline void close_device_mem_handle(void* ptr, bool enable_fabric){
  if(enable_fabric){
    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  }else{
    CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
  }
}

// Free local device memory allocated by device_mem_malloc.
inline void device_mem_free(void* ptr, bool enable_fabric){
  if(enable_fabric){
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemRelease(handle));
  }else{
    CUDA_CHECK(cudaFree(ptr));
  }
}

class HybridEpBuffer {
public:
  HybridEpBuffer(HybridEpConfigInstance config, int local_rank, int node_rank,
         int num_of_ranks_per_node);
  ~HybridEpBuffer();

  // Exchange IPC addresses using C++ distributed communication
  void exchange_ipc_address(pybind11::object process_group);

  void update_num_of_tokens_per_rank(int num_of_tokens_per_rank);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor>
  metadata_preprocessing(torch::Tensor routing_map, int64_t node_rank,
                         int64_t local_rank);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  dispatch(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
           c10::optional<torch::Tensor> scaling_factor,
           torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
           torch::Tensor attn_to_rdma_map, int64_t num_of_tokens_for_experts,
           bool with_probs);

  std::tuple<torch::Tensor, torch::Tensor>
  combine(torch::Tensor hidden, c10::optional<torch::Tensor> probs,
          torch::Tensor sparse_to_dense_map, torch::Tensor rdma_to_attn_map,
          torch::Tensor attn_to_rdma_map, bool with_probs);

private:
  void allocate_buffer();
  void allocate_buffer_for_preprocessing();
  void allocate_buffer_for_dispatch();
  void allocate_buffer_for_combine();
  void open_handles_from_other_ranks(std::vector<torch::Tensor> dispatch_handles,
                                     std::vector<torch::Tensor> combine_handles);

  HybridEpConfigInstance config;
  int rank;
  int group_size;
  int local_rank;
  int node_rank;
  int num_of_ranks_per_node;

  int64_t max_num_of_tokens_for_experts; 

  hybrid_ep::tmp_state_t *preprocessing_tmp;

  struct DispatchBuffers {
    TOKEN_DATA_TYPE data_type; 
    
    void *expert_output_token;

    void **expert_output_token_all_ranks;

    float *expert_output_prob;

    float **expert_output_prob_all_ranks;

    float *expert_output_scaling_factor;

    float **expert_output_scaling_factor_all_ranks;

    void *rdma_inter_node_group_token;

    float *rdma_inter_node_group_prob;

    float *rdma_inter_node_group_scaling_factor;

    uint64_t *rdma_inter_node_group_flags;

    uint32_t *intra_node_write_completion_flags;

    uint64_t *expected_rdma_flag_value;

    uint32_t *expected_intra_node_flag_value;

  } dispatch_buffers;

  torch::Tensor
      dispatch_memory_handles; 
      
  struct CombineBuffers {

    uint16_t *expert_input_token;

    uint16_t **expert_input_token_all_ranks;

    float *expert_input_prob;

    float **expert_input_prob_all_ranks;

    uint16_t *rdma_intra_node_red_token; 

    float *rdma_intra_node_red_prob;
    
    uint16_t *rdma_inter_node_group_token;
    
    float
        *rdma_inter_node_group_prob; 
        
    uint64_t
        *rdma_inter_node_group_flags; 
        
    uint32_t *intra_node_write_completion_flags;

    uint64_t *expected_rdma_flag_value;

    uint32_t *expected_intra_node_flag_value;

    
  } combine_buffers;

  torch::Tensor
      combine_memory_handles;

};
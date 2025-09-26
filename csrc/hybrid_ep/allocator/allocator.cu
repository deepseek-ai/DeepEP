// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "allocator.cuh"

// Check if the current device supports fabric.
bool ExtendedMemoryAllocator::support_fabric() {
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  for (int device = 0; device < device_count; ++device) {
    int support = 0;
    CU_CHECK(
        cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
    if (!support) {
      return false;
    }
  }
  return true;
}

// Round-up allocation size to fabric granularity.
size_t inline get_size_align_to_granularity(size_t size_raw, size_t granularity) {
  size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
  if (size == 0)
    size = granularity;
  return size;
}

void ExtendedMemoryAllocator::init(bool enable_fabric) {
  this->support_fabric_ = this->support_fabric();
  enable_fabric_ = enable_fabric;

  if (support_fabric_ && enable_fabric_) {
    int device_id = -1;
    // It seems a dummy call to set the device. but it is useful to prevent the invalid device context error in gb..
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaSetDevice(device_id));
    // Get the device context.
    CU_CHECK(cuCtxGetDevice(&device_));
    fabric_prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    fabric_prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    fabric_prop_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    fabric_prop_.location.id = device_;
    CU_CHECK(cuMemGetAllocationGranularity(&fabric_granularity_, &fabric_prop_,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }
  if (!support_fabric_) {
    enable_fabric_ = false;
  }
}

void ExtendedMemoryAllocator::allocate(void** ptr, size_t size_raw) {
  if (enable_fabric_) {
    size_t size = get_size_align_to_granularity(size_raw, fabric_granularity_);
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemCreate(&handle, size, &fabric_prop_, 0));
    CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, fabric_granularity_, 0, 0));
    CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
    CU_CHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &access_desc, 1));
  } else {
    CUDA_CHECK(cudaMalloc(ptr, size_raw));
  }
}

void ExtendedMemoryAllocator::free(void* ptr) {
  if (enable_fabric_) {
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));
    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
    CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemRelease(handle));
  } else {
    CUDA_CHECK(cudaFree(ptr));
  }
}

void ExtendedMemoryAllocator::get_handle(MemHandle* mem_handle, void* ptr) {
  size_t size = 0;
  CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  
  mem_handle->size = size;
  if (enable_fabric_) {
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));
    CU_CHECK(cuMemExportToShareableHandle(&mem_handle->inner.cu_mem_fabric_handle, handle,
                                          CU_MEM_HANDLE_TYPE_FABRIC, 0));
  } else {
    CUDA_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
  }
}

void ExtendedMemoryAllocator::open_handle(void** ptr, MemHandle* mem_handle) {
  if (enable_fabric_) {
    size_t size = mem_handle->size;
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle,
                                            CU_MEM_HANDLE_TYPE_FABRIC));
    CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, 0, 0, 0));
    CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
    CU_CHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &access_desc, 1));
  } else {
    CUDA_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle,
                                    cudaIpcMemLazyEnablePeerAccess));
  }
}

void ExtendedMemoryAllocator::close_handle(void* ptr) {
  if (enable_fabric_) {
    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
    CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  } else {
    CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
  }
}

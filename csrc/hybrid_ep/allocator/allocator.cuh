// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved

#pragma once
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "utils.cuh"

struct MemHandle {
  union MemHandleInner {
    cudaIpcMemHandle_t cuda_ipc_mem_handle;
    CUmemFabricHandle cu_mem_fabric_handle;
  } inner;
  size_t size;
};

// Remote memory allocator, allocate memory which can be accessed by remote devices.
class ExtendedMemoryAllocator {
 public:
  // @param enable_fabric Whether to enable fabric.
  // The fabric handle is used on the GB200 case currently.
  void init(bool enable_fabric);

  void allocate(void** ptr, size_t size_raw);
  
  void free(void* ptr);
  
  void get_handle(MemHandle* mem_handle, void* ptr);
  
  void open_handle(void** ptr, MemHandle* mem_handle);
  
  void close_handle(void* ptr);

  bool get_fabric_status() { return enable_fabric_; }

 private:
  bool support_fabric_;
  bool enable_fabric_;
  size_t fabric_granularity_;
  CUdevice device_;
  CUmemAllocationProp fabric_prop_ = {};
  CUmemAccessDesc access_desc = {};

  bool support_fabric();
};

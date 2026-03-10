// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#pragma once

#ifdef USE_NIXL

#include "nixl.h"
#include "nixl_device.cuh"

namespace hybrid_ep {

// GPU-side context for NIXL RDMA transfers.
// Each memory view is indexed by remote_idx.
// num_channels maps to UCX_RC_GDA_NUM_CHANNELS for QP channel distribution.
struct dispatch_gpu_nixl_ctx {
  nixlMemViewH local_mvh;
  nixlMemViewH remote_data_mvh;
  nixlMemViewH remote_signal_mvh;
  uint64_t *local_flag_counters;
  int num_remote_nodes;
  int num_channels;
  int rank;
};

struct combine_gpu_nixl_ctx {
  nixlMemViewH local_mvh;
  nixlMemViewH remote_data_mvh;
  nixlMemViewH remote_signal_mvh;
  uint64_t *local_flag_counters;
  int num_remote_nodes;
  int num_channels;
  int rank;
};

}  // namespace hybrid_ep

#endif  // USE_NIXL

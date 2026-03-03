// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once
#include "config.cuh"

class HybridEPCoordinator {
public:
    virtual ~HybridEPCoordinator() = default;
    virtual bool grow_buffer_config(const HybridEpConfigInstance& config, BufferConfig& buf_config) = 0;
    virtual void update_config(BufferConfig config) = 0;
    virtual void allocate_buffers() = 0;
    virtual void destroy() = 0;
};

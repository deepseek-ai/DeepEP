// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved
#pragma once

class NIXLCoordinator : public HybridEPCoordinator {
public:
    NIXLCoordinator() = default;
    ~NIXLCoordinator() override;

    void init(pybind11::object process_group, int node_rank, int local_rank, BufferConfig config);
    bool grow_buffer_config(const HybridEpConfigInstance& config, BufferConfig& buf_config) override;
    void update_config(BufferConfig config) override;
    void allocate_buffers() override;
    void destroy() override;
};

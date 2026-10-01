#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/layout/ep/token.cuh>

#include "../../runtime/jit.hpp"

namespace deep_ep::ep {

constexpr int kNumNotifyWarps = 4;

static int get_num_notify_smem_bytes(const int& num_ranks, const int& num_experts) {
    return math::align(num_ranks + num_experts, kNumNotifyWarps * 32) * sizeof(int);
}

static layout::TokenLayout get_dispatch_token_layout(
    const int& hidden, const int& elem_size, const int& num_sf_packs, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
}

static void launch_dispatch(void* x, void* sf,
                            topk_idx_t* topk_idx, float* topk_weights,
                            int* cumulative_local_expert_recv_stats,
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            int* psum_num_recv_tokens_per_expert,
                            int* num_unaligned_recv_tokens_per_expert,
                            int* dst_buffer_slot_idx,
                            int* token_metadata_at_forward,
                            const int& num_tokens, const int& num_max_tokens_per_rank,
                            const int& hidden, const int& elem_size,
                            const int& num_sf_packs, const int& sf_token_stride, const int& sf_hidden_stride,
                            const int& num_experts, const int& num_topk, const int& expert_alignment,
                            const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window,
                            void* buffer,
                            void* workspace, void* mapped_host_workspace,
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                            const bool& is_scaleup_nvlink,
                            const int& num_sms, const int& num_channels_per_sm,
                            const int& num_smem_bytes,
                            const int& num_qps, const int64_t& num_timeout_cycles,
                            const bool& cached_mode,
                            const bool& do_cpu_sync,
                            const at::cuda::CUDAStream& stream) {
    // Cached mode does not support expert token counting
    if (cached_mode)
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats == nullptr);

    // Utils
    const auto num_ranks = num_scaleout_ranks * num_scaleup_ranks;

    // Notify warps
    // TODO: why don't we use 4 notify warps?
    const int num_notify_warps = cached_mode ? 0 : kNumNotifyWarps;
    const bool reuse_slot_indices = cached_mode;
    const int num_notify_smem_bytes = cached_mode ? 0 : get_num_notify_smem_bytes(num_ranks, num_experts);
    EP_HOST_ASSERT(num_notify_warps % 4 == 0);

    // Other warps
    int num_dispatch_warps = 0;
    int num_scaleout_warps = 0, num_forward_warps = 0;
    int num_threads = 0;

    // Maximize shared memory utilization
    if (num_scaleout_ranks == 1) {
        const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);
        num_dispatch_warps = std::min<int>(
            (num_smem_bytes - num_notify_smem_bytes) / token_layout.get_num_bytes<true>(), 32 - num_notify_warps);
        num_threads = (num_notify_warps + num_dispatch_warps) * 32;
    } else {
        // Hybrid kernels
        num_scaleout_warps = num_channels_per_sm;
        num_forward_warps = num_channels_per_sm;
        num_threads = (num_notify_warps + num_scaleout_warps + num_forward_warps) * 32;
    }

    // Compile
    std::string header_name, func_name;
    if (num_scaleout_ranks == 1) {
        header_name = "dispatch";
        func_name = std::format("dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
            is_scaleup_nvlink,
            do_cpu_sync,
            reuse_slot_indices,
            num_sms,
            num_notify_warps, num_dispatch_warps,
            num_scaleup_ranks,
            hidden * elem_size, num_sf_packs,
            num_max_tokens_per_rank,
            num_experts, num_topk, expert_alignment,
            num_qps, num_timeout_cycles);
    } else {
        header_name = "hybrid_dispatch";
        func_name = std::format("hybrid_dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
            do_cpu_sync,
            reuse_slot_indices,
            num_sms,
            num_notify_warps, num_scaleout_warps, num_forward_warps,
            num_scaleout_ranks, num_scaleup_ranks,
            hidden * elem_size, num_sf_packs,
            num_max_tokens_per_rank,
            num_experts, num_topk, expert_alignment,
            num_qps, num_timeout_cycles);
    }
    const auto kernel = jit->compile("dispatch", std::format(R"(
#include <deep_ep/impls/ep/{}.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::{});
}}
)", header_name, func_name));

    // Launch
    const auto options = deep_jit::cuda::LaunchOptions {
        .stream = stream.stream(),
        .num_smem_bytes = num_smem_bytes,
        .grid_dim = dim3(num_sms, 1, 1),
        .block_dim = dim3(num_threads, 1, 1),
        .cluster_dim = dim3(2 - (num_sms % 2), 1, 1),
        .cooperative = true,
    };
    if (num_scaleout_ranks == 1) {
        jit->launch(
            kernel, options,
            x, static_cast<sf_pack_t*>(sf), topk_idx, topk_weights,
            cumulative_local_expert_recv_stats,
            psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert,
            num_unaligned_recv_tokens_per_expert,
            dst_buffer_slot_idx,
            num_tokens,
            sf_token_stride, sf_hidden_stride,
            nccl_dev_comm, nccl_window,
            buffer,
            workspace, mapped_host_workspace,
            scaleup_rank_idx
        );
    } else {
        jit->launch(
            kernel, options,
            x, static_cast<sf_pack_t*>(sf), topk_idx, topk_weights,
            cumulative_local_expert_recv_stats,
            psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert,
            num_unaligned_recv_tokens_per_expert,
            dst_buffer_slot_idx,
            token_metadata_at_forward,
            num_tokens,
            sf_token_stride, sf_hidden_stride,
            nccl_dev_comm, nccl_window,
            buffer,
            workspace, mapped_host_workspace,
            scaleout_rank_idx, scaleup_rank_idx
        );
    }
}

static void launch_dispatch_copy_epilogue(void* buffer, void* workspace,
                                          int* psum_num_recv_tokens_per_scaleup_rank,
                                          int* psum_num_recv_tokens_per_expert,
                                          void* recv_x, void* recv_sf,
                                          topk_idx_t* recv_topk_idx, float* recv_topk_weights,
                                          int* recv_src_metadata,
                                          int* channel_linked_list,
                                          int* num_unaligned_recv_tokens_per_expert,
                                          const int& num_recv_tokens, const int& num_max_tokens_per_rank,
                                          const int& num_hidden_bytes,
                                          const int& num_sf_packs, const int& recv_sf_token_stride, const int& recv_sf_hidden_stride,
                                          const int& num_experts, const int& num_topk, const int& expert_alignment,
                                          const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                          const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                          const int& num_sms, const int& num_smem_bytes,
                                          const int& num_channels,
                                          const bool& do_expand, const bool& cached_mode,
                                          const bool& do_zero_padding,
                                          const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    const auto token_layout = layout::TokenLayout(num_hidden_bytes, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
    const auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32);
    const auto num_threads = num_warps * 32;

    // Compile
    const auto kernel = jit->compile("dispatch_copy_epilogue", std::format(R"(
#include <deep_ep/impls/ep/dispatch_copy_epilogue.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::dispatch_copy_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", do_expand, cached_mode, do_zero_padding,
        num_sms, num_channels, num_warps,
        num_scaleout_ranks, num_scaleup_ranks,
        num_hidden_bytes, num_sf_packs,
        num_max_tokens_per_rank,
        num_experts, num_topk, expert_alignment));

    // Launch
    jit->launch(
        kernel, {
            .stream = stream.stream(),
            .num_smem_bytes = num_smem_bytes,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
            .enable_pdl = true,
        },
        buffer, workspace,
        psum_num_recv_tokens_per_scaleup_rank,
        psum_num_recv_tokens_per_expert,
        recv_x, recv_sf, recv_topk_idx, recv_topk_weights,
        recv_src_metadata,
        channel_linked_list,
        num_unaligned_recv_tokens_per_expert,
        num_recv_tokens,
        recv_sf_token_stride, recv_sf_hidden_stride,
        scaleout_rank_idx, scaleup_rank_idx
    );
}

}  // namespace deep_ep::ep

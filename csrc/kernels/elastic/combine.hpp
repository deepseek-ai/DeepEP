#pragma once

// MIT License
//
// Copyright (c) 2025 DeepSeek
// Changes and additions copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <nccl.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/impls/proxy_ring.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"
#include "kernel_select.hpp"

namespace deep_ep::elastic {

class CombineRuntime final : public jit::LaunchRuntime<CombineRuntime> {
public:
    struct Args {
        // Templated arguments
        bool is_scaleup_nvlink;
        bool use_expanded_layout, allow_multiple_reduction;
        // Use the ordered (upstream) hybrid kernel instead of the unordered one.
        // Resolved once from `EP_HYBRID_KERNEL`; only affects the hybrid path.
        bool use_ordered_kernel;
        int num_scaleup_warps, num_forward_warps;
        int num_scaleout_ranks, num_scaleup_ranks;
        int hidden;
        int num_max_tokens_per_rank;
        int num_experts;
        int num_topk;
        int num_qps;
        int64_t num_timeout_cycles;

        // Parameters
        nv_bfloat16* x;
        float* topk_weights;
        int* src_metadata;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* token_metadata_at_forward;
        int* channel_linked_list;
        int* token_map_at_dispatch;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* buffer;
        void* workspace;
        int scaleout_rank_idx, scaleup_rank_idx;
        int num_reduced_tokens;
        int num_combined_tokens;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        std::string header_name, func_name;
        if (args.num_scaleout_ranks == 1) {
            header_name = "combine";
            func_name = fmt::format("combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                    args.is_scaleup_nvlink,
                                    args.use_expanded_layout, args.allow_multiple_reduction,
                                    args.launch_args.grid_dim.first,
                                    args.launch_args.num_threads / 32,
                                    args.num_scaleup_ranks * args.num_scaleout_ranks,
                                    args.hidden,
                                    args.num_max_tokens_per_rank,
                                    args.num_experts,
                                    args.num_topk,
                                    args.num_qps, args.num_timeout_cycles);
        } else {
            header_name = args.use_ordered_kernel ? "hybrid_combine" : "hybrid_combine_unordered";
            func_name = fmt::format("{}<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                    args.use_ordered_kernel ? "hybrid_combine_impl" : "hybrid_unordered_combine_impl",
                                    args.use_expanded_layout, args.allow_multiple_reduction,
                                    args.launch_args.grid_dim.first,
                                    args.num_scaleup_warps, args.num_forward_warps,
                                    args.num_scaleout_ranks, args.num_scaleup_ranks,
                                    args.hidden,
                                    args.num_max_tokens_per_rank,
                                    args.num_experts,
                                    args.num_topk,
                                    args.num_qps,
                                    args.num_timeout_cycles);
        }
        return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", header_name, func_name);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        if (args.num_scaleout_ranks == 1) {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                     args.x, args.topk_weights,
                                                     args.src_metadata, args.psum_num_recv_tokens_per_scaleup_rank,
                                                     args.nccl_dev_comm, args.nccl_window,
                                                     args.buffer, args.workspace,
                                                     args.scaleup_rank_idx,
                                                     args.num_reduced_tokens));
        } else if (args.use_ordered_kernel) {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                     args.x, args.topk_weights,
                                                     args.src_metadata,
                                                     args.psum_num_recv_tokens_per_scaleup_rank,
                                                     args.token_metadata_at_forward,
                                                     args.channel_linked_list,
                                                     args.nccl_dev_comm, args.nccl_window,
                                                     args.buffer, args.workspace,
                                                     args.scaleout_rank_idx, args.scaleup_rank_idx,
                                                     args.num_reduced_tokens));
        } else {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                     args.x, args.topk_weights,
                                                     args.src_metadata,
                                                     args.psum_num_recv_tokens_per_scaleup_rank,
                                                     args.token_metadata_at_forward,
                                                     args.channel_linked_list,
                                                     args.token_map_at_dispatch,
                                                     args.nccl_dev_comm, args.nccl_window,
                                                     args.buffer, args.workspace,
                                                     args.scaleout_rank_idx, args.scaleup_rank_idx,
                                                     args.num_reduced_tokens,
                                                     args.num_combined_tokens));
        }
    }
};

static layout::TokenLayout get_combine_token_layout(
    const int& hidden, const int& elem_size, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, 0, num_topk, false);
}

static void* launch_combine(void* x,
                            void* topk_weights,
                            int* src_metadata,
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            int* token_metadata_at_forward,
                            int* channel_linked_list,
                            int* token_map_at_dispatch,
                            const jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window,
                            void* buffer, void* workspace,
                            const int& num_reduced_tokens, const int& num_combined_tokens,
                            const int& num_max_tokens_per_rank,
                            const int& hidden,
                            const int& num_experts, const int& num_topk,
                            const int& num_qps, const int64_t& num_timeout_cycles,
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                            const bool& is_scaleup_nvlink,
                            const int& num_sms, const int& num_smem_bytes,
                            const int& num_channels,
                            const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                            const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
    auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32);

    // Decide warps
    const bool use_ordered_kernel = use_ordered_hybrid_kernel();
    int num_scaleup_warps = 0, num_forward_warps = 0;
    if (num_scaleout_ranks > 1) {
        EP_HOST_ASSERT(num_channels % num_sms == 0 and
                       "Invalid number of channels or SMs, you may use a different SM count than dispatch");
        EP_HOST_ASSERT(num_channels / num_sms <= 16);

        num_scaleup_warps = num_forward_warps = num_channels / num_sms;

        if (use_ordered_kernel) {
            // The ordered kernel has no proxy warp and only carves TMA buffers out of shmem.
            num_warps = num_scaleup_warps + num_forward_warps;
            EP_HOST_ASSERT(num_warps * token_layout.get_num_bytes<true>() <= num_smem_bytes and
                           "Invalid combine SM count, please try to match your dispatch config");
        } else {
            const auto num_data_warps = num_scaleup_warps + num_forward_warps;
            num_warps = num_data_warps + 1;
            EP_HOST_ASSERT(num_warps * 32 <= 1024 and
                           "combine warp count (scale-up + forward + proxy) exceeds the "
                           "1024-thread block limit; use at least num_channels / 15 SMs");

            // TMA buffers and the proxy hand-off rings both live in the dynamic shmem arena; the
            // rings are placed after the TMA region (see hybrid_combine_unordered.cuh) so no alignment pad is
            // needed. Bound by the same total the channel auto-tuner sized against.
            const int64_t tma_smem_bytes = static_cast<int64_t>(num_data_warps) * token_layout.get_num_bytes<true>();
            const int64_t proxy_ring_bytes = deep_ep::elastic::ProxyRingLayout::get_num_bytes(
                num_forward_warps, deep_ep::elastic::kProxyRingDepthDefault);
            // The channel auto-tuner should prevent this assert from firing; leaving it as a sanity check.
            EP_HOST_ASSERT(tma_smem_bytes + proxy_ring_bytes <= num_smem_bytes and
                           "Combine TMA buffers + proxy rings exceed per-block shared memory");
        }
    }

    // Generate, build and launch
    const auto num_threads = num_warps * 32;
    const CombineRuntime::Args args = {
        .is_scaleup_nvlink = is_scaleup_nvlink,
        .use_expanded_layout = use_expanded_layout,
        .allow_multiple_reduction = allow_multiple_reduction,
        .use_ordered_kernel = use_ordered_kernel,
        .num_scaleup_warps = num_scaleup_warps, .num_forward_warps = num_forward_warps,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts,
        .num_topk = num_topk,
        .num_qps = num_qps, .num_timeout_cycles = num_timeout_cycles,
        .x = static_cast<nv_bfloat16*>(x),
        .topk_weights = static_cast<float*>(topk_weights),
        .src_metadata = src_metadata,
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        .token_metadata_at_forward = token_metadata_at_forward,
        .channel_linked_list = channel_linked_list,
        .token_map_at_dispatch = token_map_at_dispatch,
        .nccl_dev_comm = nccl_dev_comm, .nccl_window = nccl_window,
        .buffer = buffer, .workspace = workspace,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .num_reduced_tokens = num_reduced_tokens,
        .num_combined_tokens = num_combined_tokens,
        // NOTES: make cluster dim 2 to overlap with clustered computation kernels
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 2 - (num_sms % 2), true)
    };
    const auto code = CombineRuntime::generate(args);
    const auto runtime = jit::compiler->build("combine", code);
    CombineRuntime::launch(runtime, args, stream);

    // Return the buffer to be reduced
    if (num_scaleout_ranks == 1)
        return buffer;

    // For hybrid mode, we have to skip the scale-up buffer
    const bool is_scaleup_buffer_rank_layout =
        allow_multiple_reduction ? (num_scaleup_ranks <= num_topk) : false;
    const auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, 
        is_scaleup_buffer_rank_layout ? num_scaleup_ranks : num_topk,
        num_scaleout_ranks * num_max_tokens_per_rank,
        buffer);
    return scaleup_buffer.get_buffer_end_ptr();
}

class CombineReduceEpilogueRuntime final : public jit::LaunchRuntime<CombineReduceEpilogueRuntime> {
public:
    struct Args {
        // Templated arguments
        bool use_expanded_layout, allow_multiple_reduction;
        // Use the ordered (upstream) epilogue lookup instead of the unordered recv-map one.
        // Only meaningful for the hybrid path (`num_scaleout_ranks > 1`).
        bool use_ordered_kernel;
        int num_scaleout_ranks, num_scaleup_ranks;
        int hidden;
        int num_max_tokens_per_rank;
        int num_experts, num_topk;
        int num_channels;

        // Parameters
        nv_bfloat16* combined_x;
        float* combined_topk_weights;
        topk_idx_t* combined_topk_idx;
        void* reduce_buffer;
        void* bias_0;
        void* bias_1;
        int* token_map_at_dispatch;
        int num_combined_tokens;
        int scaleout_rank_idx, scaleup_rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/combine_reduce_epilogue.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",                        args.use_expanded_layout, args.allow_multiple_reduction,
                           args.launch_args.grid_dim.first,
                           args.launch_args.num_threads / 32,
                           args.num_scaleout_ranks, args.num_scaleup_ranks,
                           args.hidden,
                           args.num_max_tokens_per_rank,
                           args.num_experts, args.num_topk,
                           args.num_channels,
                           args.use_ordered_kernel);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.combined_x,
                                                 args.combined_topk_weights,
                                                 args.combined_topk_idx,
                                                 args.reduce_buffer,
                                                 args.bias_0, args.bias_1,
                                                 args.token_map_at_dispatch,
                                                 args.num_combined_tokens,
                                                 args.scaleout_rank_idx, args.scaleup_rank_idx));
    }
};

static void launch_combine_reduce_epilogue(void* combined_x,
                                           float* combined_topk_weights,
                                           topk_idx_t* combined_topk_idx,
                                           const int& num_combined_tokens, const int& num_max_tokens_per_rank,
                                           const int& hidden,
                                           const int& num_experts, const int& num_topk,
                                           const int& num_channels,
                                           void* reduce_buffer,
                                           void* bias_0, void* bias_1,
                                           int* token_map_at_dispatch,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                           const int& num_sms, const int& num_smem_bytes,
                                           const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                                           const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    // Too many warps may cause performance degrade, so we limit into 1024
    const auto token_layout = layout::TokenLayout(hidden * sizeof(nv_bfloat16), 0, 0, false);
    const auto num_warps = std::min<int>(num_smem_bytes / token_layout.get_num_bytes<false>(), 32);
    const auto num_threads = num_warps * 32;

    // Generate, build and launch
    const CombineReduceEpilogueRuntime::Args args = {
        .use_expanded_layout = use_expanded_layout,
        .allow_multiple_reduction = allow_multiple_reduction,
        .use_ordered_kernel = use_ordered_hybrid_kernel(),
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_channels = num_channels,
        .combined_x = static_cast<nv_bfloat16*>(combined_x),
        .combined_topk_weights = combined_topk_weights,
        .combined_topk_idx = combined_topk_idx,
        .reduce_buffer = reduce_buffer,
        .bias_0 = bias_0, .bias_1 = bias_1,
        .token_map_at_dispatch = token_map_at_dispatch,
        .num_combined_tokens = num_combined_tokens,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, false, true)
    };
    const auto code = CombineReduceEpilogueRuntime::generate(args);
    const auto runtime = jit::compiler->build("combine_reduce_epilogue", code);
    CombineReduceEpilogueRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic

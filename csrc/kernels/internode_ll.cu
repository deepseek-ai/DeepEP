#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "ibgda_device.cuh"

namespace deep_ep {

namespace internode_ll {

template <int kNumThreads> __launch_bounds__(kNumThreads, 1)
__global__ void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                                         int* clean_1, int num_clean_int_1) {
    // Barrier before cleaning (in case of unfinished chunked EP)
    nvshmemx_barrier_all_block();

    // Clean
    auto thread_id = static_cast<int>(threadIdx.x);
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
        clean_0[i] = 0;
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
        clean_1[i] = 0;

    // Barrier after cleaning (make sure the low-latency mode works fine)
    nvshmemx_barrier_all_block();
}

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                              int* clean_1, int num_clean_int_1,
                              cudaStream_t stream) {
    constexpr int kNumThreads = 256;

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, clean_low_latency_buffer<kNumThreads>,
                  clean_0, num_clean_int_0, clean_1, num_clean_int_1);
}

template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void
dispatch(void* packed_recv_x, void* packed_recv_x_scales,
         int* packed_recv_src_info, int64_t* packed_recv_layout_range,
         int* packed_recv_count,
         int* cumulative_local_expert_recv_stats,
         void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
         const void* x, const int64_t* topk_idx,
         int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
         int* next_clean, int num_next_clean_int,
         int num_tokens, int num_max_dispatch_tokens_per_rank,
         int num_topk, int num_experts, int rank, int num_ranks,
         int num_warp_groups, int num_warps_per_group,
         bool round_scale, int phases) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    // May extract UE8M0 from the scales
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
    const int num_scales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: hidden data, FP8 scales, index at source
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
    const size_t num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16)));
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

    // Expert counts
    constexpr int kNumMaxWarpGroups = 32;
    __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warp_id < num_warps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        const auto num_threads = (num_warps - 1) * 32;
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = static_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(static_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // Overlap top-k index read and source token index writes
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            // FP8 cast
            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                // Read
                auto int4_value = __ldg(x_int4 + i);

                if constexpr (kUseFP8) {
                    // Calculate local amax
                    auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
                    float fp32_values[kNumElemsPerRead];
                    float amax = kFP8Margin, scale, scale_inv;
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++ j) {
                        fp32_values[j] = static_cast<float>(bf16_values[j]);
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                    }

                    // Reduce amax and scale
                    EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
                    amax = half_warp_reduce_max(amax);
                    calculate_fp8_scales(amax, scale, scale_inv, round_scale);
                    if (lane_id == 0 or lane_id == 16)
                        rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

                    // Cast into send buffer
                    vec_t int2_value;
                    auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; j += 2) {
                        float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                        fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                    }
                    rdma_x_vec[i] = int2_value;
                } else {
                    // Reinterpret-cast is for C++14 compatibility
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                }
            }
            asm volatile("bar.sync 1, %0;" :: "r"(num_threads));

            // Issue IBGDA sends
            if (dst_expert_idx >= 0) {
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
                const auto dst_rank = dst_expert_idx / num_local_experts;
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                                     dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     slot_idx * num_bytes_per_msg;
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                if (dst_p2p_ptr == 0) {
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
                } else {
                    // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                    const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                    const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
                    UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                }

                // Increase counter after finishing
                __syncwarp();
                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
            }
        }
    } else if (warp_id == num_warps - 1) {
        EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            // The first SM is also responsible for checking QPs
            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);

            // The first SM is also responsible for cleaning the next buffer
            #pragma unroll
            for (int i = lane_id; i < num_next_clean_int; i += 32)
                next_clean[i] = 0;

            // Notify before executing `int_p`
            __syncwarp();
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += 32)
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
        }

        // This SM should be responsible for some destination experts, read `topk_idx` for them
        int expert_count[kNumMaxWarpGroups] = {0};
        const auto expert_begin_idx = sm_id * num_warp_groups;
        const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);

        // Per lane count
        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx] ++;
        }

        // Warp reduce
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++ i) {
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
            }
        }
    }
    __syncthreads();

    // Issue count sends
    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * num_warp_groups];

        // Wait local sends issued and send expert counts
        while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != FINISHED_SUM_TAG * 2);
        auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
        auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
        if (dst_p2p_ptr == 0) {
            nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), -num_tokens_sent - 1, dst_rank, dst_expert_local_idx);
        } else {
            st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), -num_tokens_sent - 1);
        }

        // Clean workspace for next use
        atomic_counter_per_expert[responsible_expert_idx] = 0;
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

        // Clean `packed_recv_count`
        if (dst_rank == 0)
            packed_recv_count[dst_expert_local_idx] = 0;
    }
    __syncwarp();

    // Receiving phase
    LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    if (phases & LOW_LATENCY_SEND_PHASE)
        cg::this_grid().sync();

    // Receiving and packing
    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        const auto recv_x_int4 = static_cast<int4*>(packed_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        const auto num_aligned_scales = align<int>(num_scales, sizeof(float) / sizeof(scale_t));
        const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_aligned_scales;

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups], shared_recv_token_begin_idx[kNumMaxWarpGroups];

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens, recv_token_begin_idx;
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
        if (sub_warp_id == 1 and lane_id == 0) {
            while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) == 0);
            num_recv_tokens = -num_recv_tokens - 1;
            recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
            if (cumulative_local_expert_recv_stats != nullptr)
                atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 2), "r"(num_warps_per_group * 32));
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Copy tokens
        EP_DEVICE_ASSERT(num_scales <= 64);
        for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
            // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            if (lane_id == 0)
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            __syncwarp();

            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);

            // Copy scales
            if constexpr (kUseFP8) {
                // Equivalent CuTe layout:
                //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                const auto token_idx = recv_token_begin_idx + i;
                const auto token_stride = num_elems_per_pack;
                const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;
                if (lane_id < num_scales) {
                    const auto pack_idx = lane_id / num_elems_per_pack;
                    const auto elem_idx = lane_id % num_elems_per_pack;
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id));
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
                if (lane_id + 32 < num_scales) {
                    const auto pack_idx = (lane_id + 32) / num_elems_per_pack;
                    const auto elem_idx = (lane_id + 32) % num_elems_per_pack;
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id + 32));
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
            }
        }
    }
}

void dispatch(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* cumulative_local_expert_recv_stats,
              void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
              const void* x, const int64_t* topk_idx,
              int* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0,
              void* workspace, int num_device_sms,
              cudaStream_t stream, int phases) {
    constexpr int kNumMaxTopK = 9;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = static_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

    // FP8 checks
    if (use_ue8m0)
        EP_HOST_ASSERT(round_scale and "UE8M0 SF requires `round_scale=True`");

#define DISPATCH_LAUNCH_CASE(hidden) { \
auto dispatch_func = dispatch<false, false, hidden>; \
if (use_fp8 and not use_ue8m0) \
    dispatch_func = dispatch<true, false, hidden>; \
if (use_fp8 and use_ue8m0) \
    dispatch_func = dispatch<true, true, hidden>; \
LAUNCH_KERNEL(&cfg, dispatch_func, \
              packed_recv_x, packed_recv_x_scales, \
              packed_recv_src_info, packed_recv_layout_range, \
              packed_recv_count, \
              cumulative_local_expert_recv_stats, \
              rdma_recv_x, rdma_recv_count, rdma_x, \
              x, topk_idx, \
              atomic_counter_per_expert, atomic_finish_counter_per_expert, \
              next_clean, num_next_clean_int, \
              num_tokens, num_max_dispatch_tokens_per_rank, \
              num_topk, num_experts, rank, num_ranks, \
              num_warp_groups, num_warps_per_group, \
              round_scale, phases); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kHidden>
__global__ __launch_bounds__(1024, 1) void
compress_logfmt(void* rdma_recv_x, void* rdma_send_x,
                const void* x,
                const int32_t* packed_recv_count, const int* src_info, const int64_t* layout_range,
                int num_max_dispatch_tokens_per_rank,
                int num_experts, int rank, int num_ranks) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = __shfl_sync(0xffffffff, thread_id / 32, 0), lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    constexpr int kNumWarpsPerGroup = 2;
    EP_STATIC_ASSERT(kNumWarpsPerGroup == 2, "Invalid kNumWarpsPerGroup");
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    int token_idx = warp_group_id * num_sms + sm_id;
    const auto lane_group_id = lane_id / 16, sub_lane_id = lane_id % 16;

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Message package
    constexpr size_t half_log_amax_amin_bytes_per_combine_msg = kHidden / kNumWarpsPerGroup / 128 * (2 * sizeof(nv_bfloat16));
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16) + sizeof(int4) + half_log_amax_amin_bytes_per_combine_msg * kNumWarpsPerGroup;
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    __shared__ int32_t shared_packed_recv_count[32];
    __shared__ int64_t shared_layout_range[288];
    if (thread_id < num_local_experts * num_ranks) {
        shared_layout_range[thread_id] = __ldg(layout_range + thread_id);
    }
    if (thread_id - (num_threads - 32) >= 0 && thread_id - (num_threads - 32) < num_local_experts) {
        shared_packed_recv_count[thread_id - (num_threads - 32)] = __ldg(packed_recv_count + (thread_id - (num_threads - 32)));
    }
    __syncthreads();

    int local_expert_idx = 0;
    int new_token_idx = token_idx - shared_packed_recv_count[local_expert_idx];
    while (true) {
        while (local_expert_idx + 1 < num_local_experts && new_token_idx >= 0) {
            token_idx = new_token_idx;
            new_token_idx = token_idx - shared_packed_recv_count[++local_expert_idx];
        }
        if (new_token_idx >= 0) {
            break;
        } else {
            const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
            int src_idx;

            int64_t layout;
            int offset, num_tokens_to_send;
            int dst_rank = 0;
            #pragma unroll
            for (int i = lane_id; i < num_ranks; i += 32) {
                layout = shared_layout_range[local_expert_idx * num_ranks + i];
                // Unpack layout
                unpack2(layout, num_tokens_to_send, offset);
                if (offset <= token_idx && token_idx < offset + num_tokens_to_send) {
                    dst_rank = i;
                }
            }
            /*
            if (lane_id < num_ranks) {
                layout = __ldg(layout_range + local_expert_idx * num_ranks + lane_id);
                // Unpack layout
                unpack2(layout, num_tokens_to_send, offset);
            }
            */
            if (lane_id == 32 - 1) {
                // Copy directly to local rank, or copy to buffer
                src_idx = __ldg(local_src_info + token_idx);
            }
            dst_rank = __reduce_add_sync(0xffffffff, dst_rank);
            /*
            unsigned int flag = __ballot_sync(0xffffffff, lane_id < num_ranks && offset <= token_idx && token_idx < offset + num_tokens_to_send);
            const int dst_rank = __ffs(flag) - 1;
            */
            src_idx = __shfl_sync(0xffffffff, src_idx, 32 - 1);

            const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
            const auto local_x = static_cast<const int4*>(x) +
                    local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
            const auto rdma_send_x_vec = static_cast<uint8_t*>(rdma_send_x) +
                    local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

            const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

            const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
            const auto cpy_src_int4_ptr = x_int4 + hidden_bf16_int4 / kNumWarpsPerGroup * sub_warp_id;
            const auto cpy_dst_int64_ptr = reinterpret_cast<int64_t*>(rank == dst_rank ? dst_ptr : buf_ptr);

            const int initial_dst_int64_ptr_offset = sizeof(int4) / sizeof(int64_t) + half_log_amax_amin_bytes_per_combine_msg / sizeof(int64_t) + (half_log_amax_amin_bytes_per_combine_msg / sizeof(int64_t) + hidden_bf16_int4 / kNumWarpsPerGroup * (sizeof(int4) / sizeof(int64_t))) * sub_warp_id;

            constexpr int kPrefetchCount = 2;
            constexpr float kThreshold = 1.0f;
            constexpr float kLogThreshold = 10.0f; // __logf(kThreshold)
            constexpr float kMinClip = 22.1807097779182499013514278f; // `== log(2 ^ (2 ^ 5))`
            constexpr int kNumBits = 10;
            constexpr unsigned int kNumValues = 1 << (kNumBits - 1);
            EP_STATIC_ASSERT(kHidden % (kNumWarpsPerGroup * kNumElemsPerInt4 * 32) == 0 and kNumElemsPerInt4 == 8 and kNumElemsPerInt4 % 2 == 0, "Invalid hidden");
            EP_STATIC_ASSERT(kHidden / (kNumWarpsPerGroup * kNumElemsPerInt4 * 32) >= kPrefetchCount, "Invalid hidden");
            EP_STATIC_ASSERT(kHidden <= 128 * 64, "Invalid hidden");
            EP_STATIC_ASSERT(kNumElemsPerInt4 * kNumBits == (sizeof(int64_t) + sizeof(int16_t)) * 8, "Invalid log format");

            if constexpr (kPrefetchCount > 1) {
                #pragma unroll
                for (int k = 0; k < kPrefetchCount; k++) {
                    asm volatile("prefetch.global.L1 [%0];" :: "l"(cpy_src_int4_ptr + (lane_id + k * 32)));
                }
            }
            int prefetch_i = lane_id + kPrefetchCount * 32;

            int half_warp_flag = 0;
            int log_a_store_idx = 0;
            int my_store_idx = -1;
            nv_bfloat16 my_amax, my_amin;

            #pragma unroll
            for (int k = 0, channel_group_id_in_segment = lane_group_id; k * 32 < hidden_bf16_int4 / kNumWarpsPerGroup; k++, channel_group_id_in_segment += 2) {
                // Read
                auto int4_value = ld_nc_global(cpy_src_int4_ptr + (lane_id + k * 32));
                if (prefetch_i < hidden_bf16_int4 / kNumWarpsPerGroup) {
                    asm volatile("prefetch.global.L1 [%0];" :: "l"(cpy_src_int4_ptr + (prefetch_i)));
                    prefetch_i += 32;
                }
                auto bf162_values = reinterpret_cast<const nv_bfloat162*>(&int4_value);

                // Local log amax
                nv_bfloat162 bf16_abs_values[kNumElemsPerInt4 / 2], amaxs, amins;
                #pragma unroll
                for (int j = 0; j < kNumElemsPerInt4 / 2; ++ j) {
                    bf16_abs_values[j] = __habs2(bf162_values[j]);
                    amaxs = j == 0 ? bf16_abs_values[j] : __hmax2(amaxs, bf16_abs_values[j]);
                    amins = j == 0 ? bf16_abs_values[j] : __hmin2(amins, bf16_abs_values[j]);
                }
                nv_bfloat16 amax = __hmax(amaxs.x, amaxs.y), amin = __hmin(amins.x, amins.y);

                // Reduce per 128 channels
                amax = half_warp_reduce_max(amax);
                amin = half_warp_reduce_min(amin);
                float log_amax = __logf(static_cast<float>(amax));
                float log_amin = fmaxf(__logf(static_cast<float>(amin)), log_amax - kMinClip);

                int is_peer_half_warp_compressed = __shfl_xor_sync(0xffffffff, log_amax <= kLogThreshold, 16);
                // Use LogFMT only with `amax <= kThreshold` (maybe not all half-warps)
                if (log_amax <= kLogThreshold) {
                    //const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
                    const auto inv_step = __fdividef(static_cast<float>(kNumValues - 2), log_amax - log_amin);// / (log_amax - log_amin);
                    const auto rounding = 1.5f;//2.0f - __logf((1.0f + __expf(__fdividef(1.0f, inv_step))) / 2.0f) * inv_step;

                    // Encode
                    auto encode = [=](const nv_bfloat16& value, const float& log_abs_value) -> unsigned int {
                        const auto encoded = (static_cast<unsigned int>(fmaxf((log_abs_value - log_amin) * inv_step + rounding, 0.0f))) & (kNumValues - 1);
                        return value < CUDART_ZERO_BF16 ? kNumValues + encoded : encoded;
                    };
                    unsigned int out[kNumElemsPerInt4];
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerInt4 / 2; ++ j) {
                        float2 abs_values = __bfloat1622float2(bf16_abs_values[j]);
                        out[j * 2] = encode(bf162_values[j].x, __logf(abs_values.x));
                        out[j * 2 + 1] = encode(bf162_values[j].y, __logf(abs_values.y));
                    }
                    uint32_t out32[2];
                    out32[0] = (out[0] << (kNumBits * 2)) + (out[1] << (kNumBits * 1)) + (out[3] << (kNumBits * 3)) + out[2];
                    out32[1] = (out[4] << (32 - kNumBits * 1)) + (out[5] << (32 - kNumBits * 2)) + (out[3] >> (10 - (32 - kNumBits * 3))) + (out[6] << (32 - kNumBits * 3));
                    uint16_t out16 = ((out[3] & 0b0011111100u) << (10 - (32 - kNumBits * 3))) + out[7];

                    log_a_store_idx += lane_group_id * is_peer_half_warp_compressed;
                    int dst_int64_ptr_offset = initial_dst_int64_ptr_offset + channel_group_id_in_segment * (16 * 2) - log_a_store_idx * (16 * 2 - 10 * 2);
                    st_na_global(cpy_dst_int64_ptr + (dst_int64_ptr_offset + sub_lane_id), *reinterpret_cast<const int64_t*>(out32));
                    st_na_global(reinterpret_cast<uint16_t*>(cpy_dst_int64_ptr) + ((dst_int64_ptr_offset + 16) * (sizeof(int64_t) / sizeof(uint16_t)) + sub_lane_id), out16);
                    if (sub_lane_id == k) {
                        my_store_idx = channel_group_id_in_segment;
                        my_amax = amax;
                        my_amin = amin;
                    }
                    log_a_store_idx += 1 + (1 - lane_group_id) * is_peer_half_warp_compressed;
                } else {
                    log_a_store_idx += lane_group_id * is_peer_half_warp_compressed;
                    int dst_int64_ptr_offset = initial_dst_int64_ptr_offset + channel_group_id_in_segment * (16 * 2) - log_a_store_idx * (16 * 2 - 10 * 2);
                    st_na_global(reinterpret_cast<int4*>(cpy_dst_int64_ptr) + (dst_int64_ptr_offset / (sizeof(int4) / sizeof(int64_t)) + sub_lane_id), int4_value);
                    log_a_store_idx += 0 + (1 - lane_group_id) * is_peer_half_warp_compressed;
                    half_warp_flag += 1 << channel_group_id_in_segment;
                }
            }
            int first_half_warp_flag = __shfl_sync(0xffffffff, half_warp_flag, 0);
            if (my_store_idx >= 0) {
                int dst_int64_ptr_offset = initial_dst_int64_ptr_offset - half_log_amax_amin_bytes_per_combine_msg / sizeof(int64_t);
                st_na_global(reinterpret_cast<int*>(cpy_dst_int64_ptr) + (dst_int64_ptr_offset * (sizeof(int64_t) / sizeof(int)) + my_store_idx), pack2<nv_bfloat16, int>(my_amax, my_amin));
            }
            if (lane_id == 32 - 1) {
                st_na_global(reinterpret_cast<int*>(cpy_dst_int64_ptr) + sub_warp_id, half_warp_flag + first_half_warp_flag);
            }

            token_idx += (1024 / 32 / kNumWarpsPerGroup) * num_sms;
            new_token_idx += (1024 / 32 / kNumWarpsPerGroup) * num_sms;
        }
    }
}

void compress_logfmt(void* rdma_recv_x, void* rdma_send_x,
                     const void* x,
                     const int32_t* packed_recv_count, const int* src_info, const int64_t* layout_range,
                     int hidden, int num_max_dispatch_tokens_per_rank,
                     int num_experts, int rank, int num_ranks,
                     int num_device_sms,
                     cudaStream_t stream) {
    EP_HOST_ASSERT(num_experts <= 1024);
    EP_HOST_ASSERT(num_experts / num_ranks <= 32);

#define COMPRESS_LOGFMT_LAUNCH_CASE(hidden) { \
auto compress_logfmt_func = compress_logfmt<hidden>; \
LAUNCH_KERNEL(&cfg, compress_logfmt_func, \
              rdma_recv_x, rdma_send_x, \
              x, \
              packed_recv_count, src_info, layout_range, \
              num_max_dispatch_tokens_per_rank, \
              num_experts, rank, num_ranks); } break

    SETUP_LAUNCH_CONFIG(num_device_sms, 1024, stream);
    SWITCH_HIDDEN(COMPRESS_LOGFMT_LAUNCH_CASE);
#undef COMPRESS_LOGFMT_LAUNCH_CASE
}

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(1024, 1) void
combine(void* combined_x,
        void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
        const void* x, const int64_t* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int* next_clean, int num_next_clean_int,
        int* atomic_clean_flag,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int num_warp_groups, int num_warps_per_group,
        int phases, bool zero_copy) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    constexpr float kMinClip = 22.1807097779182499013514278f; // `== log(2 ^ (2 ^ 5))`
    constexpr int kNumBits = 10;
    constexpr unsigned int kNumValues = 1 << (kNumBits - 1);

    // Message package
    constexpr size_t half_log_amax_amin_bytes_per_combine_msg = kHidden / 2 / 128 * (2 * sizeof(nv_bfloat16));
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16) + sizeof(int4) + half_log_amax_amin_bytes_per_combine_msg * 2;
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        __syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x = static_cast<const int4*>(x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec = static_cast<uint8_t*>(rdma_send_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        // Unpack layout
        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

        // Issue IBGDA send
        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
            const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

            // Copy directly to local rank, or copy to buffer and issue RDMA
            auto src_idx = __ldg(local_src_info + token_idx);
            const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
            const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            size_t buf_bytes[2];
            if constexpr (kUseLogFMT) {
                int uncompressed_cnt;
                if (lane_id < 2) {
                    unsigned int flag = ld_nc_global(reinterpret_cast<const int*>(rank == dst_rank ? dst_ptr : buf_ptr) + lane_id);
                    uncompressed_cnt = __popc(flag);
                }
                buf_bytes[0] = __shfl_sync(0xffffffff, uncompressed_cnt, 0) * (128 * sizeof(nv_bfloat16) - (128 * kNumBits / 8)) + (kHidden / 2 * kNumBits / 8 + sizeof(int4) + half_log_amax_amin_bytes_per_combine_msg);
                buf_bytes[1] = __shfl_sync(0xffffffff, uncompressed_cnt, 1) * (128 * sizeof(nv_bfloat16) - (128 * kNumBits / 8)) + (kHidden / 2 * kNumBits / 8 + half_log_amax_amin_bytes_per_combine_msg);
            }
            if (dst_p2p_ptr == 0) {
                const auto buf_int4_ptr = reinterpret_cast<int4*>(buf_ptr);
                if constexpr (not kUseLogFMT) {
                    if (not zero_copy)
                        UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, buf_int4_ptr, x_int4, ld_nc_global, st_na_global);
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, hidden * sizeof(nv_bfloat16), dst_rank, local_expert_idx, lane_id, token_idx - offset);
                } else {
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, buf_bytes[0], dst_rank, local_expert_idx, lane_id, 0);
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr + (kHidden / 2 * sizeof(nv_bfloat16) + sizeof(int4) + half_log_amax_amin_bytes_per_combine_msg), buf_ptr + (kHidden / 2 * sizeof(nv_bfloat16) + sizeof(int4) + half_log_amax_amin_bytes_per_combine_msg), buf_bytes[1], dst_rank, local_expert_idx, lane_id, token_idx - offset);
                }
            } else {
                const auto dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
                if constexpr (not kUseLogFMT) {
                    UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, dst_int4_ptr, x_int4, ld_nc_global, st_na_global);
                } else {
                    if (rank != dst_rank) {
                        UNROLLED_WARP_COPY(7, lane_id, buf_bytes[0] / sizeof(int4), dst_int4_ptr, reinterpret_cast<int4*>(buf_ptr), ld_nc_global, st_na_global);
                        UNROLLED_WARP_COPY(7, lane_id, buf_bytes[1] / sizeof(int4), dst_int4_ptr + (kHidden / 2 * sizeof(nv_bfloat16) + sizeof(int4) + half_log_amax_amin_bytes_per_combine_msg) / sizeof(int4), reinterpret_cast<int4*>(buf_ptr + (kHidden / 2 * sizeof(nv_bfloat16) + sizeof(int4) + half_log_amax_amin_bytes_per_combine_msg)), ld_nc_global, st_na_global);
                    }
                }
            }
        }

        // Put the finishing flag
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
        if (sub_warp_id == 1 and lane_id == 0) {
            while (ld_acquire_global(atomic_clean_flag) == 0);
            auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_flag + global_expert_idx);
            auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            if (dst_p2p_ptr == 0) {
                nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
            } else {
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), 1);
            }
            atomic_add_release_global(atomic_clean_flag, -1);
        }
        __syncwarp();
    }

    // Receiving phase
    LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    const auto thread_group_id = __shfl_sync(0xffffffff, thread_id / static_cast<int>(hidden_bf16_int4 / 2), 0);
    const auto sub_thread_id = thread_id % static_cast<int>(hidden_bf16_int4 / 2);
    const auto channel_group_id_in_segment = sub_thread_id / (128 / kNumElemsPerInt4);
    const auto thread_id_in_channel_group = sub_thread_id % (128 / kNumElemsPerInt4);
    const unsigned int sync_mask = lane_id < 16 ? 0x0000ffff : 0xffff0000;
    const int amax_amin_bf162_ptr_base_offset = sizeof(int4) / sizeof(nv_bfloat162) + ((half_log_amax_amin_bytes_per_combine_msg + kHidden / 2 * sizeof(nv_bfloat16)) / sizeof(nv_bfloat162)) * thread_group_id;
    const int amax_amin_bf162_ptr_offset = amax_amin_bf162_ptr_base_offset + channel_group_id_in_segment;
    const int data_int64_ptr_base_offset = amax_amin_bf162_ptr_base_offset / (sizeof(int64_t) / sizeof(nv_bfloat162)) + half_log_amax_amin_bytes_per_combine_msg / sizeof(int64_t);
    const int data_int64_ptr_channel_group_offset_unfixed = data_int64_ptr_base_offset + channel_group_id_in_segment * (10 * 2);
    const unsigned int channel_group_mask = (1u << channel_group_id_in_segment);
    const unsigned int prefix_channel_group_mask = channel_group_mask - 1u;

    // Wait all ranks to arrive
    if (responsible_expert_idx < num_experts) {
        EP_DEVICE_ASSERT(num_warps_per_group > 1);
        if (sub_warp_id == 0 and lane_id == 0) {
            while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0);
        }
    }
    cg::this_grid().sync();

    // Reduce tokens
    EP_DEVICE_ASSERT(num_topk <= 16 and hidden_bf16_int4 <= num_threads);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    if (thread_id < hidden_bf16_int4) {
        for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
            // Read top-k indices and weights
            int reg_topk_idx[kNumMaxTopk];
            float reg_topk_weights[kNumMaxTopk];
            int local_reg_topk_idx;
            int local_flag;
            int local_amax_amin_info;
            if (thread_id_in_channel_group < num_topk) {
                local_reg_topk_idx = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + thread_id_in_channel_group));
                if constexpr (kUseLogFMT) {
                    if (local_reg_topk_idx >= 0) {
                        auto rdma_buffer_type = (static_cast<uint8_t*>(rdma_recv_x) + (local_reg_topk_idx * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot);
                        local_flag = ld_nc_global(reinterpret_cast<const int*>(rdma_buffer_type) + thread_group_id);
                        local_amax_amin_info = ld_nc_global(reinterpret_cast<const int*>(rdma_buffer_type) + amax_amin_bf162_ptr_offset);
                        //asm volatile("prefetch.global.L1 [%0];" :: "l"(reinterpret_cast<const int64_t*>(rdma_buffer_type) + (data_int64_ptr_channel_group_offset_unfixed)));
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) {
                //reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
                reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
            }

            float combined_values[kNumElemsPerInt4] = {0.0f};
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) if (int reg_topk_idx_i = __shfl_sync(sync_mask, local_reg_topk_idx, i, 16); reg_topk_idx_i >= 0) {
                // Read from sources
                auto rdma_buffer_type = reinterpret_cast<const int64_t*>(static_cast<uint8_t*>(rdma_recv_x) + (reg_topk_idx_i * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot);
                int flag;
                int data_int64_ptr_offset;
                if constexpr (kUseLogFMT) {
                    flag = __shfl_sync(sync_mask, local_flag, i, 16);
                    int prefix_uncompressed_cnt = __popc(flag & prefix_channel_group_mask);
                    data_int64_ptr_offset = data_int64_ptr_channel_group_offset_unfixed + prefix_uncompressed_cnt * (16 * 2 - 10 * 2);
                }
                float log_amax, log_amin;
                if ((not kUseLogFMT) or ((flag & channel_group_mask))) {
                    auto rdma_buffer_row = reinterpret_cast<const int4*>(rdma_buffer_type);

                    // Reduce
                    auto x_vec = ld_nc_global(rdma_buffer_row + (not kUseLogFMT ? thread_id : data_int64_ptr_offset / (sizeof(int4) / sizeof(int64_t)) + thread_id_in_channel_group));
                    const auto x_bf16 = reinterpret_cast<nv_bfloat16*>(&x_vec);
                    /*if constexpr (kUseLogFMT) {
                        log_amax = __shfl_sync(sync_mask, log_amax, i, 16);
                        log_amin = __shfl_sync(sync_mask, log_amin, i, 16);
                    }*/
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerInt4; ++ j)
                        combined_values[j] += static_cast<float>(x_bf16[j]) * reg_topk_weights[i];
                } else {
                    // Reduce
                    auto x_vec = ld_nc_global(reinterpret_cast<const int64_t*>(rdma_buffer_type) + (data_int64_ptr_offset + thread_id_in_channel_group));
                    auto x_uint16 = ld_nc_global(reinterpret_cast<const uint16_t*>(rdma_buffer_type) + (data_int64_ptr_offset * (sizeof(int64_t) / sizeof(uint16_t)) + 16 * (sizeof(int64_t) / sizeof(uint16_t)) + thread_id_in_channel_group));

                    float2 float_amax_amin_info;
                    if (thread_id_in_channel_group == i) {
                        float_amax_amin_info = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&local_amax_amin_info));

                        log_amax = __logf(float_amax_amin_info.x);
                        log_amin = fmaxf(__logf(float_amax_amin_info.y), log_amax - kMinClip);
                    }
                    log_amax = __shfl_sync(sync_mask, log_amax, i, 16);
                    log_amin = __shfl_sync(sync_mask, log_amin, i, 16);

                    const auto step = __fdividef(log_amax - log_amin, static_cast<float>(kNumValues - 2));
                    //const auto inv_step = static_cast<float>(kNumValues - 2) / (log_amax - log_amin);
                    const auto rounding = 1.5f;//2.0f - __logf((1.0f + __expf(__fdividef(1.0f, inv_step))) / 2.0f) * inv_step;

                    // Decode
                    auto decode = [=](const uint32_t& out) -> float {
                        uint32_t y = out & (kNumValues - 1);
                        float abs_value;
                        if (log_amax == log_amin) {
                            return 0.0f; // for speed
                        } else if (y == 0) {
                            return 0.0f;
                        } else {
                            abs_value = __expf((y - 1) * step + log_amin);
                            return (out & kNumValues) ? -abs_value : abs_value;
                        }
                    };

                    const auto x_uint32 = reinterpret_cast<uint32_t*>(&x_vec);
                    unsigned int out[kNumElemsPerInt4];
                    out[0] = (x_uint32[0] >> (kNumBits * 2));// & ((1u << kNumBits) - 1);
                    combined_values[0] += decode(out[0]) * reg_topk_weights[i];
                    out[1] = (x_uint32[0] >> kNumBits);// & ((1u << kNumBits) - 1);
                    combined_values[1] += decode(out[1]) * reg_topk_weights[i];
                    out[2] = x_uint32[0];// & ((1u << kNumBits) - 1);
                    combined_values[2] += decode(out[2]) * reg_topk_weights[i];
                    out[4] = x_uint32[1] >> (32 - kNumBits * 1);
                    combined_values[4] += decode(out[4]) * reg_topk_weights[i];
                    out[5] = (x_uint32[1] >> (32 - kNumBits * 2));// & ((1u << kNumBits) - 1);
                    combined_values[5] += decode(out[5]) * reg_topk_weights[i];
                    out[6] = (x_uint32[1] >> (32 - kNumBits * 3));// & ((1u << kNumBits) - 1);
                    combined_values[6] += decode(out[6]) * reg_topk_weights[i];

                    out[3] = ((x_uint32[0] >> (kNumBits * 3)) + (x_uint32[1] << (10 - (32 - kNumBits * 3))) + ((x_uint16 & 0b001111110000000000u) >> 8));// & ((1u << kNumBits) - 1);
                    combined_values[3] += decode(out[3]) * reg_topk_weights[i];
                    out[7] = x_uint16;// & ((1u << kNumBits) - 1);
                    combined_values[7] += decode(out[7]) * reg_topk_weights[i];
                }
            }

            // Write results
            int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
            auto combined_bf16 = reinterpret_cast<nv_bfloat16*>(&combined_values);
            #pragma unroll
            for (int j = 0; j < kNumElemsPerInt4; ++ j)
                combined_bf16[j] = static_cast<nv_bfloat16>(combined_values[j]);
            (static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4)[thread_id] = combined_int4;
        }
    }
}

void combine(void* combined_x,
             void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
             const void* x, const int64_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             bool use_logfmt,
             void* workspace, int num_device_sms,
             cudaStream_t stream, int phases, bool zero_copy) {
    constexpr int kNumMaxTopk = 9;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);

    // Check workspace
    auto atomic_clean_flag = static_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

    // Online cast cannot use zero-copy
    EP_HOST_ASSERT(not (zero_copy and use_logfmt));

#define COMBINE_LAUNCH_CASE(hidden) { \
auto combine_func = use_logfmt ? \
    combine<true, hidden, kNumMaxTopk>: \
    combine<false, hidden, kNumMaxTopk>; \
LAUNCH_KERNEL(&cfg, combine_func, \
              combined_x, \
              rdma_recv_x, rdma_recv_flag, rdma_send_x, \
              x, topk_idx, topk_weights, src_info, layout_range, \
              next_clean, num_next_clean_int, \
              atomic_clean_flag, \
              num_combined_tokens, hidden, num_topk, \
              num_max_dispatch_tokens_per_rank, \
              num_experts, rank, num_ranks, \
              num_warp_groups, num_warps_per_group, \
              phases, zero_copy); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace internode_ll

} // namespace deep_ep

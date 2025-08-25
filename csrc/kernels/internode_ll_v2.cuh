#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "ibgda_device.cuh"

namespace deep_ep {
namespace internode_ll {

template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void
dispatch_v2(void* packed_recv_x, void* packed_recv_x_scales,
         int* packed_recv_src_info, int64_t* packed_recv_layout_range,
         int* packed_recv_count,
         int* cumulative_local_expert_recv_stats,
         int64_t* dispatch_wait_recv_cost_stats,
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
    using vec_t = std::conditional_t<kUseFP8, int2, int4>;
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
        EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerRead) == 0, "Invalid hidden");
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
            EP_STATIC_ASSERT(hidden_bf16_int4 % 32 == 0, "Must use the full warp to reduce");
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
                    amax = warp_reduce_max<16>(amax);
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
            auto start_time = clock64();
            while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) == 0);
            auto wait_recv_cost = clock64() - start_time;
            num_recv_tokens = -num_recv_tokens - 1;
            recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);

            // Add stats for diagnosis
            if (cumulative_local_expert_recv_stats != nullptr)
                atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
            if (dispatch_wait_recv_cost_stats != nullptr)
                atomicAdd(reinterpret_cast<unsigned long long*>(dispatch_wait_recv_cost_stats + src_rank), wait_recv_cost);
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

void dispatch_v2(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
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
              dispatch_wait_recv_cost_stats, \
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

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk, int kNumMaxUnrolls>
__global__ __launch_bounds__(1024, 1) void
combine_v2(void* combined_x,
        void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
        const void* x, const int64_t* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int64_t* combine_wait_recv_cost_stats,
        int* next_clean, int num_next_clean_int,
        int* atomic_clean_flag,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int num_warp_groups, int num_warps_per_group,
        int phases, bool zero_copy) {
    const auto sm_id = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    const auto num_sms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warp_id = __shfl_sync(0xffffffff, thread_id / 32, 0), lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int64_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Use different unroll factors for send and recv phases
    constexpr int kNumSendUnrolls = kHidden % (32 * 4 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 ? 4 : 2;
    constexpr int kNumRecvUnrolls = 2;
    constexpr int hidden_bf16_int4_pad = align(static_cast<int>(hidden_bf16_int4), 32 * kNumSendUnrolls);
    EP_STATIC_ASSERT(kHidden % (32 * 2 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hidden_bf16_int4 % kNumSendUnrolls == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");

    // Message package
    EP_STATIC_ASSERT(kHidden % 128 == 0, "Invalid hidden");
    constexpr int kNumDivisions = kHidden / 128;
    constexpr int kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16) + kNumMetaBytes;
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
        // NOTE move tma-related to outside local_expert_idx loop
        // ------------------------------------------ START tma-related -------------------------------------------------
        // TMA stuffs
        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;
        constexpr int kNumStages = 3;
        constexpr int kNumPrefetch = 1;
        EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

        auto smem_ptr = smem_buffer + warp_id * (kNumStages * (kNumTMABufferBytes + 16) + kNumMetaBytes);
        uint32_t tma_phase = 0;
        auto tma_buffers   = PatternVisitor([=](const int& i) { return reinterpret_cast<int4*>(smem_ptr + i * (kNumTMABufferBytes + 16)); });
        auto full_barriers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_ptr + i * (kNumTMABufferBytes + 16) + kNumTMABufferBytes); });
        auto meta_buffers  = kUseLogFMT ? reinterpret_cast<nv_bfloat162*>(smem_ptr + kNumStages * (kNumTMABufferBytes + 16)) : nullptr;
        EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");

        // Initialize m-barriers
        if (lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            fence_view_async_shared();
            fence_barrier_init();
        }
        __syncwarp();

        constexpr int kNumIters = hidden_bf16_int4_pad / (32 * kNumSendUnrolls);
        auto tma_load_and_arrive = [&](const int& stage_idx, const int4* gmem_ptr, const int& num_bytes) {
            tma_load_1d(tma_buffers[stage_idx], gmem_ptr, full_barriers[stage_idx], num_bytes);
            mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_bytes);
        };
        auto get_num_tma_bytes = [&](const int& offset_int4) {
            return min(kNumTMABufferBytes, static_cast<int>((hidden_bf16_int4 - offset_int4) * sizeof(int4)));
        };
        // -------------------------------------------- END tma-related -----------------------------------------------

        // NOTE
        // before: "one warp group --- all tokens for one (dsk_rank, local_expert_idx)"
        // after: "multiple warp groups --- cooperate on tokens for one (dsk_rank, local_expert_idx)"
        for (int local_expert_idx = 0; local_expert_idx < num_local_experts; ++local_expert_idx) {
            const auto dst_rank = responsible_expert_idx / num_local_experts;

            // NOTE changed
            // const auto local_expert_idx = responsible_expert_idx % num_local_experts;
            const auto token_cooperate_part_idx = responsible_expert_idx % num_local_experts;
            const auto num_token_cooperate_parts = num_local_experts;

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

            // NOTE added
            if (src_signals != nullptr) {
                // TODO shall we let 1st expert be separately computed and then do *not* wait for it
                // if ((threadIdx.x == 0) and (local_expert_idx > 0)) {
                if (threadIdx.x == 0) {
                    wait_signal(src_signals + local_expert_idx, src_signal_expect_value);
                }

                // TODO original code uses NamedBarrier, better than this?
                __syncthreads();
            }

            // Issue IBGDA send
            // NOTE changed
            // for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
            const int num_tokens_to_send_per_cooperate_part = ceil_div(num_tokens_to_send, num_token_cooperate_parts);
            const int token_idx_part_end = offset + min(num_tokens_to_send, num_tokens_to_send_per_cooperate_part * (token_cooperate_part_idx + 1));
            for (
                int token_idx = offset + num_tokens_to_send_per_cooperate_part * token_cooperate_part_idx + sub_warp_id;
                token_idx < token_idx_part_end;
                token_idx += num_warps_per_group
            ) {
                const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
                const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
                const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

                // Copy directly to local rank, or copy to buffer and issue RDMA
                const auto src_idx = __shfl_sync(0xffffffff, __ldg(local_src_info + token_idx), 0);
                const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                int num_send_bytes = hidden * sizeof(nv_bfloat16);

                if (not zero_copy or dst_p2p_ptr != 0) {
                    // Read from `cpy_src_int4_ptr` and copy into `cpy_dst_int4_ptr`
                    const auto cpy_src_int4_ptr = zero_copy ? reinterpret_cast<int4*>(buf_ptr) : x_int4;
                    const auto cpy_dst_int4_ptr = dst_p2p_ptr == 0 ? reinterpret_cast<int4*>(buf_ptr) : reinterpret_cast<int4*>(dst_p2p_ptr);

                    // Prefetch
                    if (elect_one_sync(lane_id))
                        tma_load_and_arrive(0, cpy_src_int4_ptr, get_num_tma_bytes(0));
                    __syncwarp();

                    int tma_offset_bytes = kNumMetaBytes;
                    #pragma unroll
                    for (int i = lane_id * kNumSendUnrolls, iter_idx = 0; i < hidden_bf16_int4_pad; i += 32 * kNumSendUnrolls, ++ iter_idx) {
                        // Load the next iteration
                        const int& stage_idx = iter_idx % kNumStages;
                        const int& next_stage_idx = (iter_idx + 1) % kNumStages;
                        if (iter_idx + 1 < kNumIters and elect_one_sync(lane_id)) {
                            tma_store_wait<kNumStages - kNumPrefetch - 1>();
                            const auto& offset_int4 = i + 32 * kNumSendUnrolls;
                            tma_load_and_arrive(next_stage_idx, cpy_src_int4_ptr + offset_int4, get_num_tma_bytes(offset_int4));
                        }
                        __syncwarp();

                        // Wait the current TMA arrival
                        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
                        mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                        if constexpr (kUseLogFMT) {
                            // Cast if possible
                            constexpr int kNumInt4PerDivision = 128 / kNumElemsPerInt4;
                            int num_tma_bytes = logfmt_encode<kNumSendUnrolls>(
                                tma_buffers[stage_idx],
                                // NOTES: only the leader lane will write the result
                                (i % kNumInt4PerDivision == 0) ? meta_buffers + i / kNumInt4PerDivision : nullptr,
                                lane_id);
                            if (elect_one_sync(lane_id))
                                tma_store_1d(tma_buffers[stage_idx], reinterpret_cast<uint8_t*>(cpy_dst_int4_ptr) + tma_offset_bytes, num_tma_bytes);
                            tma_offset_bytes += num_tma_bytes;
                        } else {
                            // BF16 original values
                            if (elect_one_sync(lane_id))
                                tma_store_1d(tma_buffers[stage_idx], cpy_dst_int4_ptr + i, get_num_tma_bytes(i));
                        }
                        __syncwarp();
                    }

                    // Store metadata (min/max values) for LogFMT
                    if constexpr (kUseLogFMT) {
                        num_send_bytes = tma_offset_bytes;
                        if (elect_one_sync(lane_id))
                            tma_store_1d(meta_buffers, cpy_dst_int4_ptr, kNumMetaBytes);
                    }

                    // Flush all stores
                    tma_store_wait();
                    __syncwarp();
                }

                // Issue RDMA
                // NOTES: for zero-copy mode, we assume the data is already in the send buffer
                if (dst_p2p_ptr == 0)
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, num_send_bytes, dst_rank, local_expert_idx, lane_id, token_idx - offset);
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

        // Destroy m-barriers
        if (lane_id < kNumStages) {
            mbarrier_inval(full_barriers[lane_id]);
            fence_view_async_shared();
            fence_barrier_init();
        }
        __syncwarp();
    } else {
        // NOTE add
        for (int local_expert_idx = 0; local_expert_idx < num_local_experts; ++local_expert_idx) {
            if (src_signals != nullptr) {
              // TODO original code uses NamedBarrier, better than this?
              __syncthreads();
            }
        }
    }

    // Receiving phase
    LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive
    if (responsible_expert_idx < num_experts) {
        EP_DEVICE_ASSERT(num_warps_per_group > 1);
        if (sub_warp_id == 0 and lane_id == 0) {
            auto start_time = clock64();
            while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0);
            auto wait_recv_cost = clock64() - start_time;
            if (combine_wait_recv_cost_stats != nullptr) {
                const auto& src_rank = responsible_expert_idx / num_local_experts;
                atomicAdd(reinterpret_cast<unsigned long long*>(combine_wait_recv_cost_stats + src_rank), wait_recv_cost);
            }
        }
    }
    cg::this_grid().sync();

    // Reassign warp groups
    constexpr int kMaxNumGroups = 2;
    const int num_decode_warps = hidden_bf16_int4_pad / (kNumRecvUnrolls * 32);
    const int num_groups = min(kMaxNumGroups, (num_threads / 32) / (num_decode_warps + 1));
    const int decode_warp_idx = __shfl_sync(0xffffffff, warp_id % (num_decode_warps + 1), 0);
    const int group_idx = __shfl_sync(0xffffffff, warp_id / (num_decode_warps + 1), 0);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(num_groups > 0);

    if (group_idx < num_groups) {
        constexpr int kNumStages = 3;
        constexpr int kNumTMABufferBytes = 16 * 2 + kHidden * 2;
        constexpr int kNumBF16PerWarpBytes = 32 * kNumRecvUnrolls * kNumElemsPerInt4 * 2;
        constexpr int kNumLogFMTPerWarpBytes = kNumBF16PerWarpBytes / 16 * 10;
        constexpr int kNumDivisionBytes = kNumDivisions * sizeof(uint32_t);
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kHidden * 2 + kNumStages * kNumDivisionBytes * 3;

        // Reallocate shared memory
        const auto smem_group_buffer = smem_buffer + kNumBytesPerGroup * group_idx;
        auto full_barriers  = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes); });
        auto empty_barriers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes + 8); });
        auto tma_ld_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint8_t* >(smem_group_buffer + i * kNumTMABufferBytes + 16); });
        auto tma_st_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint32_t*>(smem_group_buffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes); });

        // Redundant when logfmt is disabled
        const auto smem_group_ptr = smem_group_buffer + kNumStages * kNumTMABufferBytes + kHidden * 2;
        auto log_amax_buffers  = PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smem_group_ptr + i * kNumDivisionBytes); });
        auto log_amin_buffers  = PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smem_group_ptr + kNumStages * kNumDivisionBytes + i * kNumDivisionBytes); });
        auto cast_info_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<int*>  (smem_group_ptr + kNumStages * kNumDivisionBytes * 2 + i * kNumDivisionBytes); });

        uint32_t tma_phase = 0;
        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
        if (decode_warp_idx == num_decode_warps)
            tma_phase = (1 << kNumStages) - 1;

        // Initialize m-barriers
        if (decode_warp_idx == num_decode_warps and lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            mbarrier_init(empty_barriers[lane_id], num_decode_warps);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(group_idx + 1), "r"((num_decode_warps + 1) * 32));

        int stage_idx = 0, topk_idx_by_lane = 0;
        EP_STATIC_ASSERT(kNumMaxTopk <= 32, "Invalid number of topks");
        if (decode_warp_idx == num_decode_warps) {
            // TMA load warp
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk)
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                for (int i = 0; i < num_topk; ++ i) {
                    int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (topk_idx_reg < 0)
                        continue;

                    mbarrier_wait<true>(empty_barriers[stage_idx], tma_phase, stage_idx);
                    auto buffer = static_cast<uint8_t*>(rdma_recv_x) + (topk_idx_reg * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot;
                    if constexpr (kUseLogFMT) {
                        logfmt_check_amaxmin<kNumDivisions / 2, kNumSendUnrolls, kNumRecvUnrolls>(
                            buffer, reinterpret_cast<float2*>(log_amax_buffers[stage_idx]),
                            reinterpret_cast<float2*>(log_amin_buffers[stage_idx]), cast_info_buffers[stage_idx], lane_id);
                    }
                    if (elect_one_sync(lane_id)) {
                        int num_casted = 0;
                        if constexpr (kUseLogFMT) {
                            const auto& info = cast_info_buffers[stage_idx][num_decode_warps - 1];
                            num_casted = (info >> 1) + (info & 1);
                        }
                        int num_tma_bytes = num_casted * kNumLogFMTPerWarpBytes + (num_decode_warps - num_casted) * kNumBF16PerWarpBytes;
                        tma_load_1d(tma_ld_buffers[stage_idx], buffer + (kUseLogFMT ? kNumMetaBytes : 0), full_barriers[stage_idx], num_tma_bytes);
                        mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_tma_bytes);
                    }
                    __syncwarp();
                    stage_idx = (stage_idx + 1) % kNumStages;
                }
            }
        } else {
            // Reduction warps
            float topk_weights_by_lane;
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk) {
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                    topk_weights_by_lane = __ldg(topk_weights + token_idx * num_topk + lane_id);
                }
                __syncwarp();

                float combined_values[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
                for (int i = 0; i < num_topk; ++ i) {
                    if (__shfl_sync(0xffffffff, topk_idx_by_lane, i) < 0)
                        continue;
                    const auto& topk_weight = __shfl_sync(0xffffffff, topk_weights_by_lane, i);

                    mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                    if constexpr (kUseLogFMT) {
                        const auto& info = cast_info_buffers[stage_idx][decode_warp_idx];
                        bool enable_cast = info & 1;
                        int num_casted_prefix = info >> 1;
                        int tma_offset = kNumLogFMTPerWarpBytes * num_casted_prefix + kNumBF16PerWarpBytes * (decode_warp_idx - num_casted_prefix);
                        int division_idx = decode_warp_idx * (kNumRecvUnrolls * 2) + lane_id * kNumRecvUnrolls / 16;
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset + (enable_cast ? kNumLogFMTPerWarpBytes : kNumBF16PerWarpBytes) / 32 * lane_id),
                            combined_values, log_amax_buffers[stage_idx][division_idx], log_amin_buffers[stage_idx][division_idx], enable_cast, topk_weight
                        );
                    } else {
                        int tma_offset = kNumBF16PerWarpBytes * decode_warp_idx;
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset + kNumBF16PerWarpBytes / 32 * lane_id),
                            combined_values, 0, 0, false, topk_weight
                        );
                    }

                    if (elect_one_sync(lane_id))
                        mbarrier_arrive(empty_barriers[stage_idx]);
                    stage_idx = (stage_idx + 1) % kNumStages;
                }
                tma_store_wait<0>();

                #pragma unroll
                for (int k = 0; k < kNumRecvUnrolls * 4; ++ k) {
                    auto combined_pack = __nv_bfloat162(combined_values[k * 2], combined_values[k * 2 + 1]);
                    tma_st_buffers[decode_warp_idx][kNumRecvUnrolls * 4 * lane_id + k] = *reinterpret_cast<uint32_t*>(&combined_pack);
                }
                tma_store_fence();
                if (elect_one_sync(lane_id)) {
                    tma_store_1d(tma_st_buffers[decode_warp_idx],
                                 static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4 + decode_warp_idx * kNumRecvUnrolls * 32,
                                 kNumBF16PerWarpBytes);
                }
                __syncwarp();
            }
        }

        // Flush all stores
        tma_store_wait<0>();
    }
}

void combine_v2(void* combined_x,
             void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
             const void* x, const int64_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             bool use_logfmt,
             void* workspace, int num_device_sms,
             cudaStream_t stream, int phases, bool zero_copy) {
    constexpr int kNumMaxTopk = 9;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    const int num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and ((num_combined_tokens == 0) or (num_recv_per_sm > 0)));

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = max(ceil_div(num_experts, num_warp_groups), ceil_div(num_combined_tokens, num_recv_per_sm));

    // Check workspace
    auto atomic_clean_flag = static_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

    // Online cast cannot use zero-copy
    EP_HOST_ASSERT(not (zero_copy and use_logfmt));

    constexpr int kNumStages = 3;
    constexpr int kNumMaxUnrolls = 4;
    constexpr int kMaxNumGroups = 2;

    // Send buffer size
    const int num_meta_bytes = hidden / 128 * 4;
    const int num_send_tma_bytes = 32 * sizeof(int4) * kNumMaxUnrolls + 16;
    const int smem_send_size = num_warps * (kNumStages * num_send_tma_bytes + num_meta_bytes);

    // Receive buffer size
    const int num_recv_tma_bytes = 16 + hidden * 2;
    const int smem_recv_size = kMaxNumGroups * (kNumStages * num_recv_tma_bytes + hidden * 2 + kNumStages * num_meta_bytes * 3);

    // Total requirement
    const int smem_size = max(smem_send_size, smem_recv_size);

#define COMBINE_LAUNCH_CASE(hidden) { \
auto combine_func = use_logfmt ? \
    combine_v2<true, hidden, kNumMaxTopk, kNumMaxUnrolls> : \
    combine_v2<false, hidden, kNumMaxTopk, kNumMaxUnrolls>; \
SET_SHARED_MEMORY_FOR_TMA(combine_func); \
LAUNCH_KERNEL(&cfg, combine_func, \
              combined_x, \
              rdma_recv_x, rdma_recv_flag, rdma_send_x, \
              x, topk_idx, topk_weights, src_info, layout_range, \
              combine_wait_recv_cost_stats, \
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

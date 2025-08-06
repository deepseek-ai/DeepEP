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

template <int kNumSendUnrolls>
__forceinline__ __device__ int logfmt_encode(uint32_t* ld_buffer, uint32_t* st_buffer, uint32_t *shared_amaxmin) {
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32; // `== log_2(2 ^ (2 ^ 5))`
    constexpr int kNumBits = 10;
    constexpr int kNumValues = 1 << (kNumBits - 1);

    int4 int4_values[kNumSendUnrolls];
    const auto& uint32_values = reinterpret_cast<uint32_t*>(int4_values);
    EP_STATIC_ASSERT(sizeof(uint32_t) == sizeof(nv_bfloat162), "Invalid length");

    // Local log amax
    float log_abs[kNumElemsPerInt4 * kNumSendUnrolls];
    auto bf162_amax = __nv_bfloat162(CUDART_ZERO_BF16, CUDART_ZERO_BF16);
    auto bf162_amin = __nv_bfloat162(CUDART_INF_BF16, CUDART_INF_BF16);
    uint32_t local_signs = 0;
    #pragma unroll
    for (int k = 0; k < kNumSendUnrolls * sizeof(int4) / sizeof(nv_bfloat162); ++ k) {
        // TODO: eliminate bank conflicts
        uint32_values[k] = ld_buffer[k];
        auto uint32_abs = uint32_values[k] & 0x7fff7fff;
        local_signs |= ((uint32_values[k] >> 15) & 1) << (k * 2);
        local_signs |= ((uint32_values[k] >> 31) & 1) << (k * 2 + 1);

        const auto& bf162_abs = *reinterpret_cast<__nv_bfloat162*>(&uint32_abs);
        bf162_amax = __hmax2(bf162_amax, bf162_abs);
        bf162_amin = __hmin2(bf162_amin, bf162_abs);

        const auto& float2_abs = __bfloat1622float2(bf162_abs);
        log_abs[k * 2 + 0] = log2f_approx(float2_abs.x);
        log_abs[k * 2 + 1] = log2f_approx(float2_abs.y);
    }

    // Reduce per 128 channels
    // TODO: figure out how hardware do 2-byte min/max
    auto bf16_amax = __hmax(bf162_amax.x, bf162_amax.y);
    auto bf16_amin = __hmin(bf162_amin.x, bf162_amin.y);
    constexpr static int kNumLanesToReduce = 128 * sizeof(nv_bfloat16) / (kNumSendUnrolls * sizeof(int4));
    bf16_amax = warp_reduce_max<kNumLanesToReduce>(bf16_amax);
    bf16_amin = warp_reduce_min<kNumLanesToReduce>(bf16_amin);

    // Write min/max into the shared memory
    auto bf162_amaxmin = __nv_bfloat162(bf16_amax, bf16_amin);
    *shared_amaxmin = *reinterpret_cast<uint32_t*>(&bf162_amaxmin);

    // Calculate log amin/amax float
    const auto& amax = static_cast<float>(bf16_amax);
    const auto& amin = static_cast<float>(bf16_amin);
    const auto& log_amax = log2f_approx(amax);
    const auto& log_amin = amin == 0 ? log_amax - kMinClip : fmaxf(log2f_approx(amin), log_amax - kMinClip);
    const bool& enable_cast = warp_reduce_and<kNumLanesToReduce, true>(log_amax < kLogThreshold and log_amin < log_amax);

    // Case into LogFMT-10 if satisfied
    if (enable_cast) {
        const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        const auto step_inv = 1.0f / step;
        const auto rounding = 2.0f - log2f_approx((1.0f + exp2f_approx(step)) * 0.5f) * step_inv;
        const auto fused_rounding = rounding - log_amin * step_inv;

        auto encode = [=](const float& x) {
            return __float2uint_rd(x < log_amax - kMinClip ? 0.0f : x * step_inv + fused_rounding);
        };

        // Pack every 256 bits into 160 bits
        EP_STATIC_ASSERT(kNumSendUnrolls == 2 or kNumSendUnrolls == 4, "kNumSendUnrolls == 2 or 4 only");
        #pragma unroll
        for (int i = 0; i < kNumSendUnrolls / 2; ++ i) {
            uint32_t concat[6];
            #pragma unroll
            for (int k = 0; k < 5; ++ k)
                concat[k] = encode(log_abs[i * 16 + k * 3]) | (encode(log_abs[i * 16 + k * 3 + 1]) << 9) | (encode(log_abs[i * 16 + k * 3 + 2]) << 18);
            concat[5] = encode(log_abs[i * 16 + 15]);
            #pragma unroll
            for (int k = 0; k < 5; ++ k)
                st_buffer[i * 5 + k] = (concat[k] >> (k * 5)) | (concat[k + 1] << (27 - k * 5));
            st_buffer[i * 5 + 4] |= (local_signs >> 16 * i) << 16;
        }
    }
    tma_store_fence();
    __syncwarp();
    // TODO: make `enable_cast` as a template?
    return 32 * (kNumSendUnrolls * sizeof(int4) * 8 * (enable_cast ? 10 : 16) / 16 / 8);
}

template <int kNumLanes, int kNumSendUnrolls, int kNumRecvUnrolls>
__forceinline__ __device__ void logfmt_check_amaxmin(uint8_t* meta_buffer, float2* shared_log_amax,
                                                     float2* shared_log_amin, int* shared_cast_info,
                                                     const int lane_id) {
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32; // `== log_2(2 ^ (2 ^ 5))`

    bool enable_cast = true;
    if (lane_id < kNumLanes) {
        // Calculate log amin/amax float
        auto amaxmin2 = reinterpret_cast<uint64_t*>(meta_buffer)[lane_id];
        const auto& bf162_amaxmin = reinterpret_cast<__nv_bfloat162*>(&amaxmin2);
        float log_amax[2], log_amin[2];
        #pragma unroll
        for (int i = 0; i < 2; ++ i) { 
            auto amax = static_cast<float>(bf162_amaxmin[i].x);
            auto amin = static_cast<float>(bf162_amaxmin[i].y);
            log_amax[i] = log2f_approx(amax);
            log_amin[i] = amin == 0 ? log_amax[i] - kMinClip : fmaxf(log2f_approx(amin), log_amax[i] - kMinClip);
            enable_cast = enable_cast and log_amax[i] < kLogThreshold and log_amin[i] < log_amax[i];
        }
        shared_log_amax[lane_id] = make_float2(log_amax[0], log_amax[1]);
        shared_log_amin[lane_id] = make_float2(log_amin[0], log_amin[1]);
    }

    const auto& cast_bit = warp_reduce_and<kNumSendUnrolls>(enable_cast) ? 1u << (lane_id / kNumRecvUnrolls): 0u;
    const auto& cast_prefix_count = std::__popcount(warp_reduce_or<kNumRecvUnrolls, true>(cast_bit) & ((1u << (lane_id / kNumRecvUnrolls)) - 1));

    if (lane_id < kNumLanes)
        shared_cast_info[lane_id / kNumRecvUnrolls] = (cast_prefix_count << 1) | (cast_bit ? 1u : 0u);
}

template <int kHidden, int kNumRecvUnrolls>
__forceinline__ __device__ void logfmt_decode_and_accumulate(uint32_t* ld_buffer, float* accum,
                                                             const float& log_amax, const float& log_amin,
                                                             const bool& enable_cast, const float& weight) {
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int kNumBits = 10;
    constexpr int kNumValues = 1 << (kNumBits - 1);
    EP_STATIC_ASSERT(kHidden % (kNumElemsPerInt4 * 32) == 0 and kNumElemsPerInt4 == 8, "Invalid hidden");

    if (enable_cast) {
        const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        auto decode = [=](const uint32_t &encoded, const uint32_t &op) {
            const auto decoded = encoded == 0 ? .0f : exp2f_approx((encoded - 1) * step + log_amin);
            return op ? -decoded : decoded;
        };

        EP_STATIC_ASSERT(kNumRecvUnrolls == 2 or kNumRecvUnrolls == 4, "kNumRecvUnrolls == 2 or 4 only");
        #pragma unroll
        for (int i = 0; i < kNumRecvUnrolls / 2; ++ i) {
            uint32_t concat[6];
            concat[0] = ld_buffer[i * 5];
            #pragma unroll
            for (int k = 1; k < 5; ++ k)
                concat[k] = (ld_buffer[i * 5 + k - 1] >> (32 - k * 5)) | (ld_buffer[i * 5 + k] << (k * 5));
            concat[5] = ld_buffer[i * 5 + 4] >> 7;

            const uint32_t& local_signs = ld_buffer[i * 5 + 4] >> 16;
            #pragma unroll
            for (int k = 0; k < 5; ++ k) {
                accum[i * 16 + k * 3 + 0] += decode((concat[k] >>  0) & 0x1ff, (local_signs >> (k * 3 + 0)) & 1) * weight;
                accum[i * 16 + k * 3 + 1] += decode((concat[k] >>  9) & 0x1ff, (local_signs >> (k * 3 + 1)) & 1) * weight;
                accum[i * 16 + k * 3 + 2] += decode((concat[k] >> 18) & 0x1ff, (local_signs >> (k * 3 + 2)) & 1) * weight;
            }
            accum[i * 16 + 15] += decode(concat[5] & 0x1ff, (local_signs >> 15) & 1) * weight;
        }
    } else {
        #pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++ k) {
            auto bf16_pack = *reinterpret_cast<__nv_bfloat162*>(ld_buffer + k);
            accum[k * 2 + 0] += static_cast<float>(bf16_pack.x) * weight;
            accum[k * 2 + 1] += static_cast<float>(bf16_pack.y) * weight;
        }
    }
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

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int64_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;
    // Use different unroll factors for send & recv phase
    constexpr int kNumSendUnrolls = 4;
    constexpr int kNumRecvUnrolls = 2;
    constexpr int hidden_bf16_int4_pad = align(static_cast<int>(hidden_bf16_int4), 32 * kNumSendUnrolls);
    EP_STATIC_ASSERT(hidden_bf16_int4 % kNumSendUnrolls == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");

    // Message package
    EP_STATIC_ASSERT(kHidden % 128 == 0, "Invalid hidden");
    constexpr int kNumDivisions = kHidden / 128;
    constexpr int kNumMetaBytes = kNumDivisions * sizeof(uint32_t);
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

        // TMA stuffs
        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;
        constexpr int kNumStages = 3;
        constexpr int kNumPrefetch = 1;
        EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

        auto smem_ptr = smem_buffer + warp_id * (kNumStages * (kNumTMABufferBytes + 16) + kNumMetaBytes);
        uint32_t tma_phase[kNumStages] = {0};
        auto tma_buffer   = PatternVisitor([=](const int& i) { return reinterpret_cast<int4*>(smem_ptr + i * (kNumTMABufferBytes + 16)); });
        auto tma_mbarrier = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_ptr + i * (kNumTMABufferBytes + 16) + kNumTMABufferBytes); });
        auto meta_buffer  = kUseLogFMT ? reinterpret_cast<uint32_t*>(smem_ptr + kNumStages * (kNumTMABufferBytes + 16)) : nullptr;
        EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");

        // Initialize m-barriers
        if (lane_id < kNumStages) {
            mbarrier_init(tma_mbarrier[lane_id], 1);
            fence_view_async_shared();
            fence_barrier_init();
        }
        __syncwarp();

        constexpr int kNumIters = hidden_bf16_int4_pad / (32 * kNumSendUnrolls);
        auto tma_load_and_arrive = [&](const int& stage_idx, const int4* gmem_ptr, const int& num_bytes) {
            tma_load_1d(tma_buffer[stage_idx], gmem_ptr, tma_mbarrier[stage_idx], num_bytes);
            mbarrier_arrive_and_expect_tx(tma_mbarrier[stage_idx], num_bytes);
        };
        auto get_num_tma_bytes = [&](const int& offset_int4) {
            return min(kNumTMABufferBytes, static_cast<int>((hidden_bf16_int4 - offset_int4) * sizeof(int4)));
        };

        // Issue IBGDA send
        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
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
                    mbarrier_wait(tma_mbarrier[stage_idx], tma_phase[stage_idx]);
                    const auto& ld_buffer = reinterpret_cast<uint32_t*>(tma_buffer[stage_idx] + lane_id * kNumSendUnrolls);

                    if constexpr (kUseLogFMT) {
                        // Cast if possible
                        auto st_buffer = reinterpret_cast<uint32_t*>(tma_buffer[stage_idx]) + lane_id * kNumSendUnrolls * kNumElemsPerInt4 * 10 / 32;
                        int num_tma_bytes = logfmt_encode<kNumSendUnrolls>(ld_buffer, st_buffer, meta_buffer + i * kNumElemsPerInt4 / 128);
                        if (elect_one_sync(lane_id))
                            tma_store_1d(tma_buffer[stage_idx], reinterpret_cast<uint8_t*>(cpy_dst_int4_ptr) + tma_offset_bytes, num_tma_bytes);
                        tma_offset_bytes += num_tma_bytes;
                    } else {
                        // BF16 original values
                        if (elect_one_sync(lane_id))
                            tma_store_1d(tma_buffer[stage_idx], cpy_dst_int4_ptr + i, get_num_tma_bytes(i));
                    }
                }

                // Store metadata (min/max values) for LogFMT
                if constexpr (kUseLogFMT) {
                    num_send_bytes = tma_offset_bytes;
                    if (elect_one_sync(lane_id))
                        tma_store_1d(meta_buffer, cpy_dst_int4_ptr, kNumMetaBytes);
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
            mbarrier_inval(tma_mbarrier[lane_id]);
            fence_view_async_shared();
            fence_barrier_init();
        }
        __syncwarp();
    }

    // Receiving phase
    LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive
    if (responsible_expert_idx < num_experts) {
        EP_DEVICE_ASSERT(num_warps_per_group > 1);
        if (sub_warp_id == 0 and lane_id == 0) {
            while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0);
        }
    }
    cg::this_grid().sync();

    // Reduce tokens
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    if constexpr (kUseLogFMT) {
        // TODO: reduce registers usage
        // Reassign warp groups
        constexpr int kMaxNumGroups = 2;
        const int num_decode_warps = hidden_bf16_int4_pad / (kNumRecvUnrolls * 32);
        const int num_groups = min(kMaxNumGroups, (num_threads / 32) / (num_decode_warps + 1));
        const int decode_warp_idx = __shfl_sync(0xffffffff, warp_id % (num_decode_warps + 1), 0);
        const int group_idx = __shfl_sync(0xffffffff, warp_id / (num_decode_warps + 1), 0);
        EP_DEVICE_ASSERT(num_groups > 0);
        if (group_idx >= num_groups)
            return;

        // Reallocate shared memory
        constexpr int kNumStages = 3;
        constexpr int kNumTMABufferBytes = 16 * 2 + kHidden * 2;
        constexpr int kNumBF16PerWarpBytes = 32 * kNumRecvUnrolls * kNumElemsPerInt4 * 2;
        constexpr int kNumLogFMTPerWarpBytes = kNumBF16PerWarpBytes / 16 * 10;
        constexpr int kNumDivisionBytes = kNumDivisions * sizeof(uint32_t);
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kHidden * 2 + kNumStages * kNumDivisionBytes * 3;

        uint32_t tma_phase[kNumStages];
        #pragma unroll
        for (int i = 0; i < kNumStages; ++ i)
            tma_phase[i] = (decode_warp_idx == num_decode_warps ? 1 : 0);

        const auto smem_group_buffer = smem_buffer + kNumBytesPerGroup * group_idx;
        auto full_barriers  = PatternVisitor([=](const int& k) { return reinterpret_cast<uint64_t*>(smem_group_buffer + k * kNumTMABufferBytes); });
        auto empty_barriers = PatternVisitor([=](const int& k) { return reinterpret_cast<uint64_t*>(smem_group_buffer + k * kNumTMABufferBytes + 16); });
        auto tma_ld_buffer  = PatternVisitor([=](const int& k) { return reinterpret_cast<uint8_t* >(smem_group_buffer + k * kNumTMABufferBytes + 32); });
        auto tma_st_buffer  = PatternVisitor([=](const int& i) { return reinterpret_cast<uint32_t*>(smem_group_buffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes); });

        const auto smem_group_ptr = smem_group_buffer + kNumStages * kNumTMABufferBytes + kHidden * 2;
        auto log_amax  = PatternVisitor([=](const int& k) { return reinterpret_cast<float*>(smem_group_ptr + k * kNumDivisionBytes); });
        auto log_amin  = PatternVisitor([=](const int& k) { return reinterpret_cast<float*>(smem_group_ptr + kNumStages * kNumDivisionBytes + k * kNumDivisionBytes); });
        auto cast_info = PatternVisitor([=](const int& k) { return reinterpret_cast<int*>  (smem_group_ptr + kNumStages * kNumDivisionBytes * 2 + k * kNumDivisionBytes); });

        // Initialize m-barriers
        if (decode_warp_idx == num_decode_warps and lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);
            mbarrier_init(empty_barriers[lane_id], num_decode_warps);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(group_idx + 1), "r"((num_decode_warps + 1) * 32));

        int stage_idx = 0;
        for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
            if (decode_warp_idx == num_decode_warps) {
                // TMA load warp
                #pragma unroll
                for (int i = 0; i < num_topk; ++ i) {
                    int reg_topk_idx = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
                    if (reg_topk_idx < 0)
                        continue;

                    mbarrier_wait(empty_barriers[stage_idx], tma_phase[stage_idx]);
                    auto buffer = static_cast<uint8_t*>(rdma_recv_x) + (reg_topk_idx * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot;
                    logfmt_check_amaxmin<kNumDivisions / 2, kNumSendUnrolls, kNumRecvUnrolls>(buffer, reinterpret_cast<float2*>(log_amax[stage_idx]),
                                                                                              reinterpret_cast<float2*>(log_amin[stage_idx]), cast_info[stage_idx], lane_id);
                    if (elect_one_sync(lane_id)) {
                        int cast_count = (cast_info[stage_idx][num_decode_warps - 1] >> 1) + (cast_info[stage_idx][num_decode_warps - 1] & 1);
                        int num_tma_bytes = cast_count * kNumLogFMTPerWarpBytes + (num_decode_warps - cast_count) * kNumBF16PerWarpBytes;
                        tma_load_1d(tma_ld_buffer[stage_idx], buffer + kNumMetaBytes, full_barriers[stage_idx], num_tma_bytes);
                        mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_tma_bytes);
                    }
                    __syncwarp();
                    stage_idx = (stage_idx + 1) % kNumStages;
                }
            } else if (decode_warp_idx < num_decode_warps) {
                // Reduction warps
                float combined_values[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
                #pragma unroll
                for (int i = 0; i < num_topk; ++ i) {
                    if (__ldg(topk_idx + token_idx * num_topk + i) < 0)
                        continue;

                    float topk_weight = __ldg(topk_weights + token_idx * num_topk + i);
                    mbarrier_wait(full_barriers[stage_idx], tma_phase[stage_idx]);

                    int cast_prefix_count = cast_info[stage_idx][decode_warp_idx] >> 1;
                    bool enable_cast = cast_info[stage_idx][decode_warp_idx] & 1;
                    int tma_offset = kNumLogFMTPerWarpBytes * cast_prefix_count + kNumBF16PerWarpBytes * (decode_warp_idx - cast_prefix_count);
                    int division_idx = decode_warp_idx * kNumRecvUnrolls * 2 + lane_id * kNumRecvUnrolls / 16;

                    logfmt_decode_and_accumulate<kHidden, kNumRecvUnrolls>(
                        reinterpret_cast<uint32_t*>(tma_ld_buffer[stage_idx] + tma_offset + (enable_cast ? kNumLogFMTPerWarpBytes : kNumBF16PerWarpBytes) / 32 * lane_id),
                        combined_values, log_amax[stage_idx][division_idx], log_amin[stage_idx][division_idx], enable_cast, topk_weight
                    );

                    if (elect_one_sync(lane_id))
                        mbarrier_arrive(empty_barriers[stage_idx]);
                    stage_idx = (stage_idx + 1) % kNumStages;
                }
                tma_store_wait<0>();

                #pragma unroll
                for (int k = 0; k < kNumRecvUnrolls * 4; ++ k) {
                    auto combined_pack = __nv_bfloat162(combined_values[k * 2], combined_values[k * 2 + 1]);
                    tma_st_buffer[decode_warp_idx][kNumRecvUnrolls * 4 * lane_id + k] = *reinterpret_cast<uint32_t*>(&combined_pack);
                }
                tma_store_fence();
                if (elect_one_sync(lane_id))
                    tma_store_1d(tma_st_buffer[decode_warp_idx], static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4 + decode_warp_idx * kNumRecvUnrolls * 32, kNumBF16PerWarpBytes);
            }
        }
        tma_store_wait<0>();
    } else {
        for (int hidden_idx = thread_id; hidden_idx < hidden_bf16_int4; hidden_idx += num_threads) {
            for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
                // Read top-k indices and weights
                int reg_topk_idx[kNumMaxTopk];
                float reg_topk_weights[kNumMaxTopk];
                #pragma unroll
                for (int i = 0; i < num_topk; ++ i) {
                    reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
                    reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
                }

                float combined_values[kNumElemsPerInt4] = {0.0f};
                #pragma unroll
                for (int i = 0; i < num_topk; ++ i) if (reg_topk_idx[i] >= 0) {
                    // Read from sources
                    auto rdma_buffer_type = reinterpret_cast<const int*>(static_cast<uint8_t*>(rdma_recv_x) + (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot);
                    auto rdma_buffer_row = reinterpret_cast<const uint8_t*>(rdma_buffer_type);

                    // Reduce
                    auto x_vec = ld_nc_global(reinterpret_cast<const int4*>(rdma_buffer_row) + hidden_idx);
                    const auto x_bf16 = reinterpret_cast<nv_bfloat16*>(&x_vec);
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerInt4; ++ j)
                        combined_values[j] += static_cast<float>(x_bf16[j]) * reg_topk_weights[i];
                }

                // Write results
                int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
                auto combined_bf16 = reinterpret_cast<nv_bfloat16*>(&combined_values);
                #pragma unroll
                for (int j = 0; j < kNumElemsPerInt4; ++ j)
                    combined_bf16[j] = static_cast<nv_bfloat16>(combined_values[j]);
                (static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4)[hidden_idx] = combined_int4;
            }
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
    const int num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and num_recv_per_sm > 0);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = max(ceil_div(num_experts, num_warp_groups), ceil_div(num_combined_tokens, num_recv_per_sm));

    // Check workspace
    auto atomic_clean_flag = static_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

    // Online cast cannot use zero-copy
    EP_HOST_ASSERT(not (zero_copy and use_logfmt));

    constexpr int kNumStages = 3;
    constexpr int kNumUnrolls = 4;
    constexpr int kMaxNumGroups = 2;
    const int num_meta_bytes = hidden / 128 * 4;
    const int num_send_tma_bytes = 32 * sizeof(int4) * kNumUnrolls + 16;
    const int smem_send_size = num_warps * (kNumStages * num_send_tma_bytes + num_meta_bytes);
    const int num_recv_tma_bytes = 16 * 2 + hidden * 2;
    const int smem_recv_size = kMaxNumGroups * (kNumStages * num_recv_tma_bytes + hidden * 2 + kNumStages * num_meta_bytes * 3);
    const int smem_size = max(smem_send_size, smem_recv_size);

#define COMBINE_LAUNCH_CASE(hidden) { \
auto combine_func = use_logfmt ? \
    combine<true, hidden, kNumMaxTopk> : \
    combine<false, hidden, kNumMaxTopk>; \
SET_SHARED_MEMORY_FOR_TMA(combine_func); \
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

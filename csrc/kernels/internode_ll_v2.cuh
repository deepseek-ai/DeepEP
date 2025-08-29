#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "ibgda_device.cuh"

#include "internode_ll_v2_inc.cuh"

constexpr int DST_SIGNAL_EXPECT_VALUE = 1000000;

namespace deep_ep {
namespace internode_ll {

constexpr int kNumMaxWarpGroups = 32;

template <bool kUseFP8, bool kUseUE8M0, bool kUseNVFP4, int kHidden>
__forceinline__ __device__ void dispatch_send(
    int subroutine_thread_id, int num_warp_groups,

    // copied args
    void* packed_recv_x, void* packed_recv_x_scales,
    int* packed_recv_src_info, int64_t* packed_recv_layout_range,
    int* packed_recv_count,
    int* cumulative_local_expert_recv_stats,
    int64_t* dispatch_wait_recv_cost_stats,
    void* rdma_recv_x,
    // int* rdma_recv_count, // NOTE removed
    // void* rdma_x, // NOTE removed
    void* x, const int64_t* topk_idx, // NOTE rm `const` of x
    int* atomic_counter_per_expert,
    // int* atomic_finish_counter_per_expert, // NOTE removed
    int* next_clean, int num_next_clean_int,
    int num_tokens, int num_max_dispatch_tokens_per_rank,
    int num_topk, int num_experts, int rank, int num_ranks,
    // int num_send_warp_groups, int num_recv_warp_groups, // NOTE removed
    int num_warps_per_group,
    bool round_scale, int phases,
    uint32_t* dst_signals,
    uint32_t* count_per_expert, int64_t* token_idx_and_dst_expert_flat_list,
    int64_t* layout_range_buffer, int* negotiate_offset_of_expert_buffer, int* remote_start_offset_of_dst_rank_buffer
) {
    using Consts = DispatchConstsTemplate<kUseFP8, kUseNVFP4, kHidden>;
    EP_DEVICE_ASSERT(Consts::num_bytes_per_msg % sizeof(int4) == 0);

    // NOTE copied from dispatch body
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto warp_id = subroutine_thread_id / 32, lane_id = get_lane_id();
    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_local_experts = num_experts / num_ranks;
    // unused
    // const auto warp_group_id = warp_id / num_warps_per_group;
    // const auto sub_warp_id = warp_id % num_warps_per_group;

    // NOTE removed
    // const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;
    // Expert counts
    // __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

//     if (subroutine_thread_id % 32 == 0) { printf("[R%d,S%d,T%d] dispatch_send START\n", rank, sm_id, subroutine_thread_id); }

    if ((sm_id == 0) and (warp_id == 0)) {
        // The first SM is also responsible for cleaning the next buffer
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        // TODO do we really need this? since `next_clean` will be used only in the next round of kernels
        // not needed in per-token signal approach
//         // Notify before executing `int_p`
//         __syncwarp();
//         #pragma unroll
//         for (int i = lane_id; i < num_experts; i += 32)
//             atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
    }

    // Reserve remote locations
    {
        EP_DEVICE_ASSERT(num_ranks <= num_sms);
        EP_DEVICE_ASSERT(num_warps * 32 <= num_local_experts);
        const int dst_rank = sm_id;
        const int dst_expert_local_idx = subroutine_thread_id;

        if ((dst_rank < num_ranks) and (dst_expert_local_idx < num_local_experts)) {
            const auto dst_global_expert_idx = dst_rank * num_local_experts + dst_expert_local_idx;

            const int num_tokens_to_send = count_per_expert[dst_global_expert_idx];

            // 1. Compete to get a range of locations to set data to
            // TODO maybe do not need `release` (but yes need `sys`)
            int remote_start_offset_of_dst_rank;
            {
                const auto dst_ptr = reinterpret_cast<uint64_t>(negotiate_offset_of_expert_buffer);
                const auto dst_p2p_ptr = reinterpret_cast<int*>(nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank));
                remote_start_offset_of_dst_rank = atomic_add_release_sys_global(dst_p2p_ptr + dst_expert_local_idx, num_tokens_to_send);
            }

            // 2. Write metadata to remote
            // TODO is this strong enough
            {
                const auto dst_ptr = reinterpret_cast<uint64_t>(layout_range_buffer);
                const auto dst_p2p_ptr = reinterpret_cast<int64_t*>(nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank));
                const auto val = pack2<int, int64_t>(num_tokens_to_send, remote_start_offset_of_dst_rank);
                dst_p2p_ptr[dst_expert_local_idx * num_ranks + rank] = -val-1;
            }

            // 2. Write metadata to local
            // TODO is this strong enough
            remote_start_offset_of_dst_rank_buffer[dst_global_expert_idx] = -remote_start_offset_of_dst_rank-1;
        }
    }

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information

    // NOTE remove the last warp (and thus the if)
    // if (warp_id < num_warps - 1) {

    constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerRead) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumElemsPerRead * 32 % Consts::kNumPerChannels == 0, "Invalid vectorization");

    // NOTE no need "-1" b/c we do not reserve one warp for counting anymore
    // const auto num_threads = (num_warps - 1) * 32;
    // const auto num_threads = num_warps * 32; // not used

    // unused
    // const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

    // NOTE
    // before: one SM = one token, one warp = one dst rank of that token, only use first 8 warps of the SM (?)
    // after: flatten all (warp_id, sm_id),
    //        then one warp = one pseudo_token_idx (i.e. one dst rank of one token)
    //
    // NOTE: deliberately be (warp_id, sm_id) instead of (sm_id, warp_id)
    //       to allow work be distributed to all SMs when few work
    // TODO is these ordering suboptimal for nvlink write or gmem read?
    // TODO may use multi warp to send one token
    const int flat_worker_id = warp_id * num_sms + sm_id;
    const int flat_worker_num = num_warps * num_sms;
    for (
        // "tefl" := "token_idx_and_dst_expert_flat_list"
        int tefl_idx = flat_worker_id;
        tefl_idx < num_tokens * num_topk;
        tefl_idx += flat_worker_num
    ) {
//         if (subroutine_thread_id % 32 == 0) { printf("[R%d,S%d,T%d] dispatch_send local_expert_idx=%d START \n", rank, sm_id, subroutine_thread_id, local_expert_idx); }

        // TODO do prefetching if needed
        // NOTE ldg is for read-only data cache, if token_idx_and_dst_expert_flat_list is somehow overlapped in the future we should change it
        const auto token_idx_and_dst_expert = __ldg(token_idx_and_dst_expert_flat_list + tefl_idx);
        int token_idx, dst_expert_idx;
        unpack2(token_idx_and_dst_expert, token_idx, dst_expert_idx);
        const auto dst_rank = dst_expert_idx / num_local_experts;

        // TODO can speedup by prefetching, delayed checking, etc
        // TODO is this load strong enough?
        int remote_start_offset_of_dst_rank;
        while ((remote_start_offset_of_dst_rank = ld_volatile_global(remote_start_offset_of_dst_rank_buffer + dst_rank)) == 0);
        remote_start_offset_of_dst_rank = -remote_start_offset_of_dst_rank - 1;

        // NOTE changed, see "before-after" above
        // for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {

        // const auto x_int4 = static_cast<const int4*>(x) + token_idx * hidden_bf16_int4;

        // NOTE do not use `rdma_x` but use `x`
        // const auto rdma_x_src_idx = reinterpret_cast<int*>(static_cast<uint8_t*>(rdma_x) + token_idx * Consts::num_bytes_per_msg);
        const auto x_src_idx = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(x) + token_idx * Consts::num_bytes_per_msg);

        // const auto rdma_x_vec = reinterpret_cast<Consts::vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
        // const auto rdma_x_scales = reinterpret_cast<Consts::rdma_x_scale_t*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + Consts::hidden_bytes);

        // Overlap top-k index read and source token index writes
        // NOTE the parallel strategy is changed
        // auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;

        // NOTE (0828) require users to set this value
        // NOTE do not use `rdma_x` but use `x`
        // NOTE use lane_id instead of local_thread id
        // NOTE and the new code will write `x_src_idx` *MULTIPLE* times w/ same value, thus wasting but correct
        // subroutine_thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;
        // lane_id == 0 ? (*x_src_idx = token_idx) : 0;

        // NOTE no read or cast in fp4
        // FP8 cast
//             EP_STATIC_ASSERT(hidden_bf16_int4 % 32 == 0, "Must use the full warp to reduce");
//             #pragma unroll
//             for (int i = subroutine_thread_id; i < hidden_bf16_int4; i += num_threads) {
//                 // Read
//                 auto int4_value = __ldg(x_int4 + i);
//
//                 if constexpr (kUseFP8) {
//                     // Calculate local amax
//                     auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
//                     float fp32_values[kNumElemsPerRead];
//                     float amax = kFP8Margin, scale, scale_inv;
//                     #pragma unroll
//                     for (int j = 0; j < kNumElemsPerRead; ++ j) {
//                         fp32_values[j] = static_cast<float>(bf16_values[j]);
//                         amax = fmaxf(amax, fabsf(fp32_values[j]));
//                     }
//
//                     // Reduce amax and scale
//                     EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
//                     amax = warp_reduce_max<16>(amax);
//                     calculate_fp8_scales(amax, scale, scale_inv, round_scale);
//                     if (lane_id == 0 or lane_id == 16)
//                         rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;
//
//                     // Cast into send buffer
//                     vec_t int2_value;
//                     auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
//                     #pragma unroll
//                     for (int j = 0; j < kNumElemsPerRead; j += 2) {
//                         float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
//                         fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
//                     }
//                     rdma_x_vec[i] = int2_value;
//                 } else {
//                     // Reinterpret-cast is for C++14 compatibility
//                     rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
//                 }
//             }

        // NOTE this cannot be removed even if we do not do casting
        // b/c we need to write to `rdma_x_src_idx`
        // (but we may optimize it later)
        // asm volatile("bar.sync 1, %0;" :: "r"(num_threads));

        // Issue IBGDA sends
        if (dst_expert_idx >= 0) {
            int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
            slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
            const auto dst_rank = dst_expert_idx / num_local_experts;
            const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
            // NOTE do not use `rdma_x` but use `x`
            // const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
            const auto src_ptr = reinterpret_cast<uint64_t>(x_src_idx);
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                                 dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * Consts::num_bytes_per_msg +
                                 // NOTE modified rm
                                 // rank * num_max_dispatch_tokens_per_rank * Consts::num_bytes_per_msg +
                                 remote_start_offset_of_dst_rank * Consts::num_bytes_per_msg +
                                 slot_idx * Consts::num_bytes_per_msg;
            const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            if (dst_p2p_ptr == 0) {
                // NOTE remove to simplify code (and it does not handle signals etc)
                EP_DEVICE_ASSERT(false);
                // nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, Consts::num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
            } else {
                // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);

                // NOTE do *not* send the first int4, which is handled via the signal
                // UNROLLED_WARP_COPY(8, lane_id, Consts::num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                UNROLLED_WARP_COPY(
                    8, lane_id,
                    Consts::num_int4_per_msg - sizeof(int4),
                    dst_int4_ptr + 1,
                    src_int4_ptr + 1,
                    ld_nc_global, st_na_global
                );

                // Send per-token signal
                // NOTE only first 4B of 16B has value, the other 12B is not needed
                __syncwarp();
                if (lane_id == 0) {
                    st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), -token_idx - 1);
                }
            }

            // not needed in per-token signal approach
//             // Increase counter after finishing
//             __syncwarp();
//             lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
        }

        // NOTE mv from do-once to do-per-local-expert
        // TODO what does this do? do we break something, b/c we let multi SM cooperate?
        // (seems it is safe, b/c our next step will check gmem?)
        // __syncthreads();

        // not needed in per-token signal approach
//         // NOTE mv from do-once to do-per-local-expert
//         //
//         // NOTE
//         // before: one (sm_id, warp_group_id) = one responsible_expert_idx = send counter to that (dst rank, dst local expert)
//         //         thus use one thread per warp_group
//         // after: reuse the (cooperate_idx, dst_rank) assignment and send counter to that (dsk_rank, const local_expert_idx)
//         //         thus use one thread per SM
//         //
//         // Issue count sends
//         EP_DEBUG_DEVICE_ASSERT(num_sms >= num_ranks);
//         // NOTE changed
//         // if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
//         if ((cooperate_idx == 0) and (lane_id == 0)) {
//             // NOTE changed
//             // const auto dst_rank = responsible_expert_idx / num_local_experts;
//             // const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
//             // const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * num_warp_groups];
//             const auto dst_expert_local_idx = local_expert_idx;
//             const auto responsible_expert_idx = dst_expert_idx;
//             const int num_tokens_sent = num_tokens_of_dst_expert;
//
//             // Wait local sends issued and send expert counts
//             while (
//                 ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) !=
//                 // NOTE changed
//                 // FINISHED_SUM_TAG * 2
//                 FINISHED_SUM_TAG + num_tokens_sent
//             );
//             auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
//             auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
//             if (dst_p2p_ptr == 0) {
//                 nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), -num_tokens_sent - 1, dst_rank, dst_expert_local_idx);
//             } else {
//                 st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), -num_tokens_sent - 1);
//             }
//
//             // Clean workspace for next use
//             atomic_counter_per_expert[responsible_expert_idx] = 0;
//             atomic_finish_counter_per_expert[responsible_expert_idx] = 0;
//
//             // NOTE packed_recv_count zeroing is removed
//             // // Clean `packed_recv_count`
//             // if (dst_rank == 0)
//             //     packed_recv_count[dst_expert_local_idx] = 0;
//         }
//         // TODO what does this do?
//         __syncwarp();
    }

//     if (subroutine_thread_id % 32 == 0) { printf("[R%d,S%d,T%d] dispatch_send END\n", rank, sm_id, subroutine_thread_id); }

//     } else if (warp_id == num_warps - 1) {
//         EP_DEVICE_ASSERT(num_sms > 1);
//         if (sm_id == 0) {
//             // The first SM is also responsible for checking QPs
//             EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);
//
//             // NOTE (the `next_clean` + notify part is moved)
//         }
//
//         // This SM should be responsible for some destination experts, read `topk_idx` for them
//         int expert_count[kNumMaxWarpGroups] = {0};
//         const auto expert_begin_idx = sm_id * num_warp_groups;
//         const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);
//
//         // Per lane count
//         #pragma unroll 8
//         for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
//             auto idx = static_cast<int>(__ldg(topk_idx + i));
//             if (idx >= expert_begin_idx and idx < expert_end_idx)
//                 expert_count[idx - expert_begin_idx] ++;
//         }
//
//         // Warp reduce
//         #pragma unroll
//         for (int i = expert_begin_idx; i < expert_end_idx; ++ i) {
//             auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
//             if (lane_id == 0) {
//                 shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
//                 atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
//             }
//         }
//     }
}

template <bool kUseFP8, bool kUseUE8M0, bool kUseNVFP4, int kHidden>
__forceinline__ __device__ void dispatch_recv(
    int subroutine_thread_id, int num_warp_groups,

    // copied args
    void* packed_recv_x, void* packed_recv_x_scales,
    int* packed_recv_src_info, int64_t* packed_recv_layout_range,
    int* packed_recv_count,
    int* cumulative_local_expert_recv_stats,
    int64_t* dispatch_wait_recv_cost_stats,
    void* rdma_recv_x,
    // int* rdma_recv_count, // NOTE removed
    // void* rdma_x, // NOTE removed
    const void* x, const int64_t* topk_idx,
    int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
    int* next_clean, int num_next_clean_int,
    int num_tokens, int num_max_dispatch_tokens_per_rank,
    int num_topk, int num_experts, int rank, int num_ranks,
    // int num_send_warp_groups, int num_recv_warp_groups, // NOTE removed
    int num_warps_per_group,
    bool round_scale, int phases,
    uint32_t* dst_signals,
    uint32_t* count_per_expert, int64_t* token_idx_and_dst_expert_flat_list,
    int64_t* layout_range_buffer, int* negotiate_offset_of_expert_buffer, int* remote_start_offset_of_dst_rank_buffer
) {
    using Consts = DispatchConstsTemplate<kUseFP8, kUseNVFP4, kHidden>;

    // NOTE copied from dispatch body
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x); // unused
    const auto warp_id = subroutine_thread_id / 32, lane_id = get_lane_id();
    const auto num_warps = num_warp_groups * num_warps_per_group; // unused
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;

    // NOTE rm
    // const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    // May extract UE8M0 from the scales
    using scale_t = std::conditional_t<kUseUE8M0 || kUseNVFP4, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0 || kUseNVFP4, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");
    EP_STATIC_ASSERT(!(kUseFP8 && kUseNVFP4), "FP8 and NVFP4 cannot be used together");

//     if (subroutine_thread_id % 32 == 0) { printf("[R%d,S%d,T%d] dispatch_recv START\n", rank, sm_id, subroutine_thread_id); }

// NOTE packed_recv_count zeroing is removed
//     // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
//     if (phases & LOW_LATENCY_SEND_PHASE)
//         cg::this_grid().sync();

    // TODO a lot of SM is wasted, optimize it later
    // TODO at least make dispatch_recv have 16 instead of 8 warps
    //
    // NOTE
    // before: one (sm_id, warp_group_id) = one responsible_expert_idx = handle all tokens for one (src_rank, local_expert_idx)
    // after: reshape (warp_id, sm_id) into (cooperate_idx, src_rank)
    //        then all num_cooperate warps handle tokens from one src_rank
    const int num_cooperate_parts = num_sms * num_warps / num_ranks;
    EP_DEVICE_ASSERT(num_sms * num_warps == num_cooperate_parts * num_ranks); // even division
    const int flatten_id = warp_id * num_sms + sm_id;
    const int cooperate_idx = flatten_id / num_ranks;
    const int src_rank = flatten_id % num_ranks;

    // Receiving and packing
    // NOTE if -> for
    // if (responsible_expert_idx < num_experts) {
    EP_DEVICE_ASSERT(num_warp_groups == 1); // not consider multi warp_group case below
    for (int local_expert_idx = 0; local_expert_idx < num_local_experts; ++local_expert_idx) {
//         if (subroutine_thread_id % 32 == 0) { printf("[R%d,S%d,T%d] dispatch_recv local_expert_idx=%d START \n", rank, sm_id, subroutine_thread_id, local_expert_idx); }

        // NOTE modified
        // const auto src_rank = responsible_expert_idx / num_local_experts;
        // const auto local_expert_idx = responsible_expert_idx % num_local_experts;

        // NOTE MODIFIED
        const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * Consts::num_bytes_per_msg;
                // this is removed
                // + src_rank * num_max_dispatch_tokens_per_rank * Consts::num_bytes_per_msg;
//         const auto recv_x_int4 = static_cast<int4*>(packed_recv_x) +
//                 local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * Consts::hidden_int4;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        const auto num_aligned_scales = align<int>(Consts::num_scales, sizeof(float) / sizeof(scale_t));
        const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_aligned_scales;

        int num_recv_tokens, token_start_offset;
        if (lane_id == 0) {
            int64_t layout;
            while((layout = ld_volatile_global(layout_range_buffer + local_expert_idx * num_ranks + src_rank)) == 0);
            layout = -layout - 1;
            unpack2(layout, num_recv_tokens, token_start_offset);

            if ((dst_signals != nullptr) and (cooperate_idx == 0)) {
                atomic_add_release_global(dst_signals + local_expert_idx, DST_SIGNAL_EXPECT_VALUE - num_recv_tokens);
            }
        }
        num_recv_tokens = __shfl_sync(0xffffffff, num_recv_tokens, 0);
        token_start_offset = __shfl_sync(0xffffffff, token_start_offset, 0);

        // NOTE no longer have per-expert signals
//         // Shared between sub-warps in warp groups
//         __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups], shared_recv_token_begin_idx[kNumMaxWarpGroups];
//
//         // Wait tokens to arrive
//         // NOTES: using sub-warp 1 to overlap with sub-warp 0
//         int num_recv_tokens, recv_token_begin_idx;
//         EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
//         if (sub_warp_id == 1 and lane_id == 0) {
//             // auto start_time = clock64(); // not used
//             while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) == 0);
//             // auto wait_recv_cost = clock64() - start_time; // not used
//             num_recv_tokens = -num_recv_tokens - 1;
//             recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
//             shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
//             shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
//             recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
//
//             // not handled
// //                 // Add stats for diagnosis
// //                 if (cumulative_local_expert_recv_stats != nullptr)
// //                     atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
// //                 if (dispatch_wait_recv_cost_stats != nullptr)
// //                     atomicAdd(reinterpret_cast<unsigned long long*>(dispatch_wait_recv_cost_stats + src_rank), wait_recv_cost);
//         }
//         asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 2), "r"(num_warps_per_group * 32));
//         num_recv_tokens = shared_num_recv_tokens[warp_group_id];
//         recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Copy tokens
        // for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
        for (
            int i_raw = cooperate_idx;
            i_raw < num_recv_tokens;
            i_raw += num_cooperate_parts
        ) {
            const int i = i_raw + token_start_offset;

//             // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * Consts::num_bytes_per_msg);
//             if (lane_id == 0)
//                 recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);

            // Read signal + Copy source info
            if (lane_id == 0) {
                int recv_src_idx;
                while ((recv_src_idx = ld_acquire_sys_global(src_src_idx)) == 0);
                recv_src_idx = -recv_src_idx-1;

                // cleanup (will be used in the next round)
                *src_src_idx = 0;

                recv_src_info[recv_token_begin_idx + i] = recv_src_idx;
            }
            __syncwarp();

            // do not need to copy real data now
//             // Copy data
//             // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
//             const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * Consts::hidden_int4;
//             UNROLLED_WARP_COPY(7, lane_id, Consts::hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);

            // Copy scales
            if constexpr (kUseFP8) {
                // NOTE simply remove to simplify code
                EP_DEVICE_ASSERT(false);
//                 EP_DEVICE_ASSERT(Consts::num_scales <= 64);
//                 // Equivalent CuTe layout:
//                 //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
//                 const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + Consts::hidden_bytes);
//                 const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
//                 const auto token_idx = recv_token_begin_idx + i;
//                 const auto token_stride = num_elems_per_pack;
//                 const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;
//                 if (lane_id < Consts::num_scales) {
//                     const auto pack_idx = lane_id / num_elems_per_pack;
//                     const auto elem_idx = lane_id % num_elems_per_pack;
//                     auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id));
//                     recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
//                 }
//                 if (lane_id + 32 < Consts::num_scales) {
//                     const auto pack_idx = (lane_id + 32) / num_elems_per_pack;
//                     const auto elem_idx = (lane_id + 32) % num_elems_per_pack;
//                     auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id + 32));
//                     recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
//                 }
            } else if constexpr (kUseNVFP4) {
                // TODO wait for new swizzle layout
                // Equivalent CuTe layout:
                //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
                const auto src_scales = reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(src_data) + Consts::hidden_bytes);
                const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                const auto token_idx = recv_token_begin_idx + i;
                const auto token_stride = num_elems_per_pack;
                const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;
                #pragma unroll
                for (int j = lane_id; j < Consts::num_scales; j += 32) {
                    const auto pack_idx = j / num_elems_per_pack;
                    const auto elem_idx = j % num_elems_per_pack;
                    auto scale = ld_nc_global(src_scales + j);
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
            }

            if (dst_signals != nullptr) {
                __syncwarp();
                if (lane_id == 0) {
                    atomic_add_release_global(dst_signals + local_expert_idx, 1);
                }
            }
        }
    }

//     if (subroutine_thread_id % 32 == 0) { printf("[R%d,S%d,T%d] dispatch_recv END\n", rank, sm_id, subroutine_thread_id); }
}

template <bool kUseFP8, bool kUseUE8M0, bool kUseNVFP4, int kHidden>
__global__ __launch_bounds__(1024, 1) void
dispatch_v2(void* packed_recv_x, void* packed_recv_x_scales,
         int* packed_recv_src_info, int64_t* packed_recv_layout_range,
         int* packed_recv_count,
         int* cumulative_local_expert_recv_stats,
         int64_t* dispatch_wait_recv_cost_stats,
         void* rdma_recv_x,
         int* rdma_general_signal, // NOTE renamed from `rdma_recv_count`
         // void* rdma_x, // NOTE removed
         void* x, const int64_t* topk_idx, // NOTE rm `const` of x
         int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
         int* next_clean, int num_next_clean_int,
         int num_tokens, int num_max_dispatch_tokens_per_rank,
         int num_topk, int num_experts, int rank, int num_ranks,
         // NOTE split num_warp_groups
         int num_send_warp_groups, int num_recv_warp_groups,
         int num_warps_per_group,
         bool round_scale, int phases,
         uint32_t* dst_signals,
         uint32_t* count_per_expert, int64_t* token_idx_and_dst_expert_flat_list,
         int* remote_start_offset_of_dst_rank_buffer) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_send_threads = num_send_warp_groups * num_warps_per_group * 32;
    const auto raw_thread_id = static_cast<int>(threadIdx.x);

    // NOTE Please keep in sync: Config.get_nvl_buffer_size_hint, LowLatencyLayout.constructor, internode_ll_v2
    //
    // (num_local_experts, num_ranks). written by REMOTE gpus, read by curr gpu.
    // arr[local_expert_idx, src_rank] := the (num_tokens, start_offset) layout information of that src_rank
    // similar to `packed_recv_layout_range`, but written remotely
    int64_t* layout_range_buffer = (int64_t*) rdma_general_signal;
    // (num_local_experts,). use by REMOTE gpus. all gpus atomic-add on it to get a slice of locations to send data to
    int* negotiate_offset_of_expert_buffer = (int*) (((uint8_t*)rdma_general_signal) + num_experts * sizeof(int64_t));

    if ((sm_id == 0) and (raw_thread_id == 0)) {
        // assert alignment
        EP_DEVICE_ASSERT(((int64_t)layout_range_buffer) % 16 == 0);
        EP_DEVICE_ASSERT(((int64_t)negotiate_offset_of_expert_buffer) % 16 == 0);
        EP_DEVICE_ASSERT(zero);
    }

    if (raw_thread_id < num_send_threads) {
        if (phases & LOW_LATENCY_SEND_PHASE) {
            const auto send_thread_id = raw_thread_id;
            dispatch_send<kUseFP8, kUseUE8M0, kUseNVFP4, kHidden>(
                send_thread_id, num_send_warp_groups,

                // forward args
                packed_recv_x, packed_recv_x_scales,
                packed_recv_src_info, packed_recv_layout_range,
                packed_recv_count,
                cumulative_local_expert_recv_stats,
                dispatch_wait_recv_cost_stats,
                rdma_recv_x,
                x, topk_idx,
                atomic_counter_per_expert, atomic_finish_counter_per_expert,
                next_clean, num_next_clean_int,
                num_tokens, num_max_dispatch_tokens_per_rank,
                num_topk, num_experts, rank, num_ranks,
                num_warps_per_group,
                round_scale, phases,
                dst_signals,
                count_per_expert, token_idx_and_dst_expert_flat_list,
                layout_range_buffer, negotiate_offset_of_expert_buffer, remote_start_offset_of_dst_rank_buffer
            );
        }
    } else {
        if (phases & LOW_LATENCY_RECV_PHASE) {
            const auto recv_thread_id = raw_thread_id - num_send_threads;
            dispatch_recv<kUseFP8, kUseUE8M0, kUseNVFP4, kHidden>(
                recv_thread_id, num_recv_warp_groups,

                // forward args
                packed_recv_x, packed_recv_x_scales,
                packed_recv_src_info, packed_recv_layout_range,
                packed_recv_count,
                cumulative_local_expert_recv_stats,
                dispatch_wait_recv_cost_stats,
                rdma_recv_x,
                x, topk_idx,
                atomic_counter_per_expert, atomic_finish_counter_per_expert,
                next_clean, num_next_clean_int,
                num_tokens, num_max_dispatch_tokens_per_rank,
                num_topk, num_experts, rank, num_ranks,
                num_warps_per_group,
                round_scale, phases,
                dst_signals,
                count_per_expert, token_idx_and_dst_expert_flat_list,
                layout_range_buffer, negotiate_offset_of_expert_buffer, remote_start_offset_of_dst_rank_buffer
            );
        }
    }

// NOTE removed
//     // Sending phase
//     if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
//         goto LOW_LATENCY_DISPATCH_RECV;
//
//     // Receiving phase
//     LOW_LATENCY_DISPATCH_RECV:
//     if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
//         return;
}

void dispatch_v2(void* packed_recv_x, void* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x, int* rdma_recv_count,
              // void* rdma_x, // NOTE removed
              void* x, const int64_t* topk_idx, // NOTE rm `const` of x
              int* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks,
              bool use_fp8, bool round_scale, bool use_ue8m0,
              void* workspace, int num_device_sms,
              cudaStream_t stream, int phases,
              bool use_nvfp4, uint32_t* dst_signals,
              uint32_t* count_per_expert, int64_t* token_idx_and_dst_expert_flat_list,
              int* remote_start_offset_of_dst_rank_buffer) {
    constexpr int kNumMaxTopK = 9;

    // NOTE simple renaming
    int* rdma_general_signal = rdma_recv_count;

    // NOTE MODIFIED
    // const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warp_groups = 2;

    // NOTE temporarily reduce num warps per group to avoid workload imbalance in dispatch_send
    // TODO may increase it later e.g. for dispatch_recv
    const int num_warps_per_group = 8;
    // const int num_warps_per_group = 32 / num_warp_groups;

    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);

    // NOTE no longer need one SM to send all topk destinations
    // EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = static_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

    // TODO inefficient, may change it
    // NOTE add
    EP_HOST_ASSERT(num_warp_groups >= 2);
    const int num_send_warp_groups = num_warp_groups - 1;
    const int num_recv_warp_groups = 1;

    // FP8 checks
    if (use_ue8m0)
        EP_HOST_ASSERT(round_scale and "UE8M0 SF requires `round_scale=True`");

#define DISPATCH_LAUNCH_CASE(hidden) { \
auto dispatch_func = dispatch_v2<false, false, false, hidden>; \
if (use_fp8 and not use_ue8m0) \
    dispatch_func = dispatch_v2<true, false, false, hidden>; \
if (use_fp8 and use_ue8m0) \
    dispatch_func = dispatch_v2<true, true, false, hidden>; \
if (use_nvfp4) \
    dispatch_func = dispatch_v2<false, false, true, hidden>; \
LAUNCH_KERNEL(&cfg, dispatch_func, \
              packed_recv_x, packed_recv_x_scales, \
              packed_recv_src_info, packed_recv_layout_range, \
              packed_recv_count, \
              cumulative_local_expert_recv_stats, \
              dispatch_wait_recv_cost_stats, \
              rdma_recv_x, \
              x, topk_idx, \
              atomic_counter_per_expert, atomic_finish_counter_per_expert, \
              next_clean, num_next_clean_int, \
              num_tokens, num_max_dispatch_tokens_per_rank, \
              num_topk, num_experts, rank, num_ranks, \
              num_send_warp_groups, num_recv_warp_groups, num_warps_per_group, \
              round_scale, phases, \
              dst_signals, \
              count_per_expert, token_idx_and_dst_expert_flat_list); } break

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
        int phases, bool zero_copy,
        uint32_t* src_signals, uint32_t src_signal_expect_value) {
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

        const auto dst_rank = responsible_expert_idx / num_local_experts;

        // NOTE
        // before: "one warp group --- all tokens for one (dsk_rank, local_expert_idx)"
        // after: "multiple warp groups --- cooperate on tokens for one (dsk_rank, local_expert_idx)"
        for (int local_expert_idx = 0; local_expert_idx < num_local_experts; ++local_expert_idx) {
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

        // TODO maybe move to above?
        // Put the finishing flag
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
        if (sub_warp_id == 1 and lane_id == 0) {
            // copied from global to this part
            const auto local_expert_idx_for_signal = responsible_expert_idx % num_local_experts;
            const auto global_expert_idx_for_signal = rank * num_local_experts + local_expert_idx_for_signal;
            // =============================================

            while (ld_acquire_global(atomic_clean_flag) == 0);
            auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_flag + global_expert_idx_for_signal);
            auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            if (dst_p2p_ptr == 0) {
                // will not visit this branch
                // nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
                EP_DEVICE_ASSERT(0);
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
             cudaStream_t stream, int phases, bool zero_copy,
             uint32_t* src_signals, uint32_t src_signal_expect_value) {
    // NOTE reduce combine_send num sm
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0) {
        num_device_sms = 32;
    }

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
              phases, zero_copy, \
              src_signals, src_signal_expect_value); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace internode_ll
} // namespace deep_ep

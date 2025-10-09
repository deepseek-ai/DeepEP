#include "configs.cuh"
#include "api.cuh"
#include "buffer.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"
#include "ibgda_device.cuh"

namespace deep_ep {

namespace internode {

extern nvshmem_team_t cpu_rdma_team;

}

namespace internode_zcopy {

// TODO: Zero-copy: Eliminate duplicate definitions
struct SourceMeta {
    int src_rdma_rank, is_token_in_nvl_rank_bits;
    int token_idx; // Used in local token dispatch as an indirect reference to the original token
    int dummy;

    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

    __forceinline__ SourceMeta() = default;

    // TODO: faster encoding
    __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks, int token_idx = 0) {
        src_rdma_rank = rdma_rank;
        is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
        this->token_idx = token_idx;
        #pragma unroll
        for (int i = 1; i < NUM_MAX_NVL_PEERS; ++ i)
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
    }

    __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const {
        return (is_token_in_nvl_rank_bits >> nvl_rank) & 1;
    }
};

EP_STATIC_ASSERT(sizeof(SourceMeta) == get_source_meta_bytes(), "Invalid size of `SourceMeta`");

template<int SourceMetaBytes = sizeof(SourceMeta)>
__host__ __device__ __forceinline__
int get_num_bytes_per_token(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
    return static_cast<int>(align_up(hidden_int4 * sizeof(int4) + SourceMetaBytes + num_scales * sizeof(float) + num_topk_idx * sizeof(topk_idx_t) + num_topk_weights * sizeof(float), sizeof(int4)));
}

__host__ __device__
std::pair<int, int> get_rdma_clean_meta(int hidden_int4, int num_scales, int num_topk_idx,
                                        int num_topk_weights, int num_rdma_ranks, int num_rdma_recv_buffer_tokens,
                                        int num_channels, bool is_dispatch) {
    if (is_dispatch) {
        return {
            get_num_bytes_per_token<get_source_meta_bytes()>(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_rdma_recv_buffer_tokens * num_rdma_ranks * 1 * num_channels / sizeof(int) + // recv buffer
            get_source_meta_bytes() * ceil_div(NUM_MAX_ZCOPY_DISPATCH_TOKENS, num_channels) * num_rdma_ranks * 1 * num_channels / sizeof(int), // send buffer for SourceMeta
            (NUM_MAX_NVL_PEERS * 2 + 2) * num_rdma_ranks * 2 * num_channels + // meta
            3 * num_rdma_ranks * 1 * num_channels * sizeof(uint64_t) / sizeof(int) // head & tail & recv finish signal
        };
    }
    return {
        get_num_bytes_per_token<get_source_meta_bytes()>(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_rdma_recv_buffer_tokens * num_rdma_ranks * 2 * num_channels / sizeof(int), // recv buffer
        2 * num_rdma_ranks * 1 * num_channels * sizeof(uint64_t) / sizeof(int) // head & tail
    };
}

template <bool kLowLatencyMode>
__forceinline__ __device__ static int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

// At most 8 RDMA ranks to be sent
constexpr int get_num_topk_rdma_ranks(int num_rdma_ranks) {
    return num_rdma_ranks < 8 ? num_rdma_ranks : 8;
}

template <bool kLowLatencyMode, int kNumRDMARanks, bool kCachedMode,
          size_t kNvlFwdTMASMemLen,
          int kNumDispatchRDMASenderWarps, int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
__global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 2 + NUM_MAX_NVL_PEERS) * 32), 1)  // 2 + 1 + 1 + 8 = 12
dispatch(int4* recv_x, float* recv_x_scales, topk_idx_t* recv_topk_idx, float* recv_topk_weights, SourceMeta* recv_src_meta,
         const int4* x, const float* x_scales, const topk_idx_t* topk_idx, const float* topk_weights,
         int* send_rdma_head, int* send_nvl_head,
         int* recv_rdma_channel_prefix_matrix, int* recv_gbl_channel_prefix_matrix,
         const int* rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
         const int* gbl_channel_prefix_matrix, const int* recv_gbl_rank_prefix_sum,
         const int* recv_gbl_rank_prefix_sum_fwd,
         const bool* is_token_in_rank,
         int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,
         void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
         void** buffer_fused_ptrs, void** buffer_ptrs, int num_zcopy_buffers, int buffer_id, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
         int rank, int num_ranks) {
    enum class WarpRole {
        kRDMASender,
        kRDMAAndNVLForwarder,
        kForwarderCoordinator,
        kNVLReceivers
    };

    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_channels = static_cast<int>(gridDim.x), channel_id = sm_id;
    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;

    const auto role_meta = [=]() -> std::pair<WarpRole, int> {
        if (warp_id < kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASender, -1};
        } else if (warp_id == kNumDispatchRDMASenderWarps) {
            return {WarpRole::kNVLReceivers, -1};
        } else if (warp_id == kNumDispatchRDMASenderWarps + 1) {
            return {WarpRole::kForwarderCoordinator, -1};
        } else {
            return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id - kNumDispatchRDMASenderWarps - 2) % NUM_MAX_NVL_PEERS};
        }
    }();
    auto warp_role = role_meta.first;
    auto target_rank = role_meta.second; // Not applicable for RDMA senders and NVL receivers
    EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 2 + NUM_MAX_NVL_PEERS);

    // Data checks
    EP_DEVICE_ASSERT(num_topk <= 32);

    // RDMA symmetric layout
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    auto hidden_bytes = hidden_int4 * sizeof(int4);
    auto num_bytes_per_rdma_token = get_num_bytes_per_token(hidden_int4, num_scales, num_topk, num_topk);

    auto rdma_channel_data = SymBuffer<int8_t, false>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token, kNumRDMARanks, channel_id, num_channels);
    EP_DEVICE_ASSERT(num_tokens <= NUM_MAX_ZCOPY_DISPATCH_TOKENS);
    auto rdma_channel_src_meta = SymBuffer<SourceMeta, false>(rdma_buffer_ptr, ceil_div(NUM_MAX_ZCOPY_DISPATCH_TOKENS, num_channels), kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_meta = SymBuffer<int>(rdma_buffer_ptr, NUM_MAX_NVL_PEERS * 2 + 2, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    // Each element indicates the RDMA rank has finished receiving (1) or not (0)
    // TODO: Zero-copy: Replace this with polling CQ
    auto rdma_channel_recv_finish_signal = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

    // NVL buffer layouts
    // NOTES: `rs_wr_buffer_ptr` means "Read for Senders, Write for Receivers", `ws_rr_buffer_ptr` means "Write for Senders, Read for Receivers"
    void *ws_rr_buffer_ptr = nullptr;
    void *ws_rr_fused_buffer_ptr = nullptr;
    if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
        ws_rr_fused_buffer_ptr = shift_ptr(buffer_fused_ptrs[target_rank], NUM_COMBINE_INPUT_BYTES_PER_ZCOPY_BUFFER * num_zcopy_buffers + NUM_DISPATCH_OUTPUT_BYTES_PER_ZCOPY_BUFFER * buffer_id);
        ws_rr_buffer_ptr = buffer_ptrs[target_rank];
    }
    if (warp_role == WarpRole::kNVLReceivers) {
        ws_rr_buffer_ptr = buffer_ptrs[nvl_rank];
    }

    ws_rr_buffer_ptr = reinterpret_cast<void *>(reinterpret_cast<int *>(ws_rr_buffer_ptr) + ZCOPY_NOTIFY_NVL_METADATA_OFFSET_INTS);
    auto nvl_channel_prefix_start = AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels);
    auto nvl_channel_prefix_end = AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels);
    auto nvl_channel_finish_signal = AsymBuffer<int>(ws_rr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels);

    // RDMA sender warp synchronization
    __shared__ volatile int rdma_send_channel_next_tail[kNumRDMARanks];
    auto sync_rdma_sender_smem = []() { asm volatile("barrier.sync 0, %0;" :: "r"((kNumDispatchRDMASenderWarps) * 32)); };

    // Forward warp synchronization
    __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
    __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
    auto sync_forwarder_smem = []() { asm volatile("barrier.sync 1, %0;" :: "r"((NUM_MAX_NVL_PEERS + 1) * 32)); };

    // RDMA multi-sge list
    __shared__ __align__(16) ibgda_sge_t sge_list_buf[kNumDispatchRDMASenderWarps * NUM_MAX_SGE_PER_WQE];
    ibgda_sge_t *sge_list = sge_list_buf + warp_id * NUM_MAX_SGE_PER_WQE;

    const size_t scale_bytes = num_scales * sizeof(float);

    if (warp_role == WarpRole::kRDMASender) {
        // NOTES: in case of splitting the issued put at the end of the buffer
        EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);

        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // Clean shared memory
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");
        (warp_id == 0 and lane_id < kNumRDMARanks) ? (rdma_send_channel_next_tail[lane_id] = 0) : 0;

        // Send number of tokens in this channel by `-value - 1`
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");

        for (int i = warp_id; i < kNumRDMARanks; i += kNumDispatchRDMASenderWarps) {
            int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;

            auto dst_ptr = dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
            if (lane_id < NUM_MAX_NVL_PEERS) {
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_channels + channel_id - 1]) - 1;
            } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = -gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id - NUM_MAX_NVL_PEERS) * num_channels + channel_id] - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1]) - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
                dst_ptr[lane_id] = -rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] - 1;
            }
            __syncwarp();

            // Issue RDMA for non-local ranks
            if (dst_rdma_rank != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                  channel_id, lane_id, 0);
            }
        }
        sync_rdma_sender_smem();

        // Iterate over tokens and copy into buffer
        int64_t token_idx;
        int num_sge_per_rdma_token = 2;
        if (num_scales > 0) num_sge_per_rdma_token ++;
        if (num_topk > 0) num_sge_per_rdma_token += 2;

        for (int i = warp_id; i < kNumRDMARanks; i += kNumDispatchRDMASenderWarps) {
            int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;

            int last_issued_tail = 0;
            int num_tokens_to_send = 0;
            num_tokens_to_send = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            if (channel_id > 0) {
                num_tokens_to_send -= rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            }

            int issue_token_idx = 0;
            int num_tokens_to_issue = min(num_tokens_to_send, num_max_rdma_chunked_send_tokens);
            int num_sge_per_msg = num_sge_per_rdma_token * num_tokens_to_issue;
            EP_DEVICE_ASSERT(num_sge_per_msg <= NUM_MAX_SGE_PER_WQE);
            for (token_idx = token_start_idx; token_idx < token_end_idx; token_idx ++) {
                // Read RDMA rank existence
                uint64_t is_token_in_rank_uint64 = 0;
                int cached_rdma_channel_head = 0, rdma_tail_idx = -1;
                if (lane_id == 0) {
                    is_token_in_rank_uint64 = *reinterpret_cast<const uint64_t*>(is_token_in_rank + token_idx * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS);
                    // Acquire next tail
                    if (is_token_in_rank_uint64 != 0) {
                        rdma_tail_idx = rdma_send_channel_next_tail[dst_rdma_rank] ++;
                        // Since in the zcopy case we use the SGE buffer exclusively, we do not
                        // wait for the head here and directly proceed to fill the SGE instead,
                        // without risk of overwriting the src buffer of an ongoing RDMA WRITE.
                    }

                    // Store RDMA head for combine
                    if (not kCachedMode)
                        send_rdma_head[token_idx * kNumRDMARanks + dst_rdma_rank] = rdma_tail_idx;
                }
                __syncwarp();

                auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, 0);
                if (recv_is_token_in_rank_uint64 == 0) {
                    continue;
                }

                auto token_x = x + token_idx * hidden_int4;
                auto token_x_scales = x_scales + token_idx * num_scales;
                auto token_topk_idx = topk_idx + token_idx * num_topk;
                auto token_topk_weights = topk_weights + token_idx * num_topk;
                auto token_source_meta_rdma_buf = rdma_channel_src_meta.buffer(dst_rdma_rank) + rdma_tail_idx;

                SourceMeta src_meta;
                // Construct source meta
                if (lane_id == 0) {
                    auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&is_token_in_rank_uint64);
                    src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values, token_idx);
                    st_na_global(token_source_meta_rdma_buf, src_meta);
                }
                __syncwarp();

                // Prepare SGE
                if (dst_rdma_rank != rdma_rank) {
                    int sge_idx = 0;
                    if (lane_id == 0) {
                        sge_list[issue_token_idx * num_sge_per_rdma_token + sge_idx].addr = reinterpret_cast<uint64_t>(token_x);
                        sge_list[issue_token_idx * num_sge_per_rdma_token + sge_idx].length = hidden_bytes;
                        sge_idx++;
                        sge_list[issue_token_idx * num_sge_per_rdma_token + sge_idx].addr = reinterpret_cast<uint64_t>(token_source_meta_rdma_buf);
                        sge_list[issue_token_idx * num_sge_per_rdma_token + sge_idx].length = sizeof(SourceMeta);
                        sge_idx++;
                        if (num_scales > 0) {
                            sge_list[issue_token_idx * num_sge_per_rdma_token + sge_idx].addr = reinterpret_cast<uint64_t>(token_x_scales);
                            sge_list[issue_token_idx * num_sge_per_rdma_token + sge_idx].length = scale_bytes;
                            sge_idx++;
                        }
                        if (num_topk > 0) {
                            sge_list[issue_token_idx * num_sge_per_rdma_token + sge_idx].addr = reinterpret_cast<uint64_t>(token_topk_idx);
                            sge_list[issue_token_idx * num_sge_per_rdma_token + sge_idx].length = num_topk * sizeof(topk_idx_t);
                            sge_idx++;
                            sge_list[issue_token_idx * num_sge_per_rdma_token + sge_idx].addr = reinterpret_cast<uint64_t>(token_topk_weights);
                            sge_list[issue_token_idx * num_sge_per_rdma_token + sge_idx].length = num_topk * sizeof(float);
                            sge_idx++;
                        }
                    }
                    __syncwarp();
                }

                issue_token_idx ++;
                if (issue_token_idx < num_tokens_to_issue) {
                    continue;
                }

                // Actually wait for the tail now
                auto start_time = clock64();
                while (rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
                    cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank)));

                    // Timeout check
                    if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP dispatch RDMA sender timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA lane: %d, head: %d, tail: %d, num_max_rdma_chunked_recv_tokens: %d \n",
                        channel_id, rdma_rank, nvl_rank, lane_id, cached_rdma_channel_head, rdma_tail_idx, num_max_rdma_chunked_recv_tokens);
                        trap();
                    }
                }

                // Issue the WRITE operation
                if (dst_rdma_rank != rdma_rank) {
                    size_t num_bytes_per_msg = num_bytes_per_rdma_token * num_tokens_to_issue;
                    int dst_slot_idx = last_issued_tail % num_max_rdma_chunked_recv_tokens;
                    auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.buffer(rdma_rank) + dst_slot_idx * num_bytes_per_rdma_token);
                    EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
                    nvshmemi_ibgda_put_nbi_warp_multi_sge_parallel<true>(sge_list, num_sge_per_msg, dst_ptr, num_bytes_per_msg,
                                                                         translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, lane_id);
                } else {
                    // The local forwarder will fetch the SourceMeta from the SourceMeta RDMA
                    // source buffer. Simply updating the tail suffices.
                    memory_fence();
                }

                last_issued_tail += num_tokens_to_issue;
                __syncwarp();

                issue_token_idx = 0;
                num_tokens_to_send -= num_tokens_to_issue;
                if (lane_id == 0) {
                    nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank), num_tokens_to_issue,
                                                    translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, dst_rdma_rank == rdma_rank);
                }
                __syncwarp();
                num_tokens_to_issue = min(num_tokens_to_send, num_max_rdma_chunked_send_tokens);
                num_sge_per_msg = num_sge_per_rdma_token * num_tokens_to_issue;
            }
        }
        // Wait for all RDMA ranks to finish receiving
        if (warp_id == 0 and lane_id < kNumRDMARanks) {
            auto start_time = clock64();
            constexpr uint64_t expected = 1;
            while (true) {
                auto signal = ld_volatile_global(rdma_channel_recv_finish_signal.buffer());
                if (signal == expected) {
                    break;
                }
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES or signal > expected) {
                    printf("DeepEP zcopy dispatch recv completion signal corruption or timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, signal: %#lx, expected: %#lx\n",
                        channel_id, rdma_rank, nvl_rank, lane_id, signal, expected);
                    trap();
                }
            }
        }
    } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
        // RDMA consumers and NVL producers
        const auto dst_nvl_rank = target_rank;
        const auto dst_rank = rdma_rank * NUM_MAX_NVL_PEERS + dst_nvl_rank;
        const auto dst_rank_expert_begin = dst_rank * (num_experts / num_ranks);
        const auto dst_rank_expert_end = dst_rank_expert_begin + (num_experts / num_ranks);

        // Dynamic TMA shared memory layout
        const size_t kNumHiddenTMABytesPerWarp = 16384;
        extern __shared__ int4 tma_smem[];
        char *tma_smem_aligned = reinterpret_cast<char *>(align_up<uintptr_t>(reinterpret_cast<uintptr_t>(tma_smem), ZCOPY_TMA_SMEM_ALIGNMENT));
        __shared__ uint64_t tma_mbarrier[NUM_MAX_NVL_PEERS];

        // Dedicated TMA shared memory for scales
        const size_t kNumScalesTMABytesPerWarp = 512;
        __shared__ __align__(kNumScalesTMABytesPerWarp) char tma_smem_scales[NUM_MAX_NVL_PEERS][kNumScalesTMABytesPerWarp];
        __shared__ uint64_t tma_mbarrier_scales[NUM_MAX_NVL_PEERS];

        EP_DEVICE_ASSERT(hidden_bytes <= kNumHiddenTMABytesPerWarp);
        EP_DEVICE_ASSERT(scale_bytes <= kNumScalesTMABytesPerWarp);

        char *smem_ptrs[NUM_MAX_NVL_PEERS];
        #pragma unroll
        for (size_t i = 0; i < NUM_MAX_NVL_PEERS; ++ i) {
            smem_ptrs[i] = tma_smem_aligned + kNumHiddenTMABytesPerWarp * i;
        }
        EP_DEVICE_ASSERT(
            reinterpret_cast<uintptr_t>(smem_ptrs[NUM_MAX_NVL_PEERS - 1]) + kNumHiddenTMABytesPerWarp <=
            reinterpret_cast<uintptr_t>(tma_smem) + kNvlFwdTMASMemLen);

        // Wait counters to arrive
        int num_tokens_to_recv_from_rdma = 0, num_tokens_to_recv_from_rdma_saved = 0, src_rdma_channel_prefix = 0;
        bool finish_signaled = false;
        auto rdma_signal_finish =
            [&rdma_channel_recv_finish_signal, &finish_signaled, rdma_rank, nvl_rank, channel_id](int src_rdma_rank, int qp_id) {
            nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_recv_finish_signal.buffer(rdma_rank), 1,
                                            translate_dst_rdma_rank<kLowLatencyMode>(src_rdma_rank, nvl_rank), qp_id, src_rdma_rank == rdma_rank);
            finish_signaled = true;
        };
        EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
        auto start_time = clock64();
        int start_sum = -1, end_sum = -1;
        if (lane_id < kNumRDMARanks) {
            while (true) {
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
                auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
                auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
                if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
                    // Notify NVL ranks
                    start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + nvl_rank * kNumRDMARanks + lane_id, -start_sum - 1);
                    st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + nvl_rank * kNumRDMARanks + lane_id, -end_sum - 1);

                    // Save RDMA channel received token count
                    src_rdma_channel_prefix = -meta_2 - 1;
                    auto src_rdma_channel_prefix_1 = -meta_3 - 1;
                    num_tokens_to_recv_from_rdma = num_tokens_to_recv_from_rdma_saved = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
                    if (not kCachedMode)
                        recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = src_rdma_channel_prefix_1;
                    src_rdma_channel_prefix += lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];
                    EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);
                    // TODO: Zero-copy: We can check if the recv count is 0 on the sender side instead
                    if (num_tokens_to_recv_from_rdma == 0 and dst_nvl_rank == lane_id % NUM_MAX_NVL_PEERS) {
                        // Need to immediately send finish signal here, as we won't receive any
                        // tokens and thus will not trigger the logic upon token reception.
                        // As of the second condition, see comments on the other atomic add
                        rdma_signal_finish(lane_id, channel_id);
                    }
                    break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP zcopy dispatch forwarder timeout (RDMA meta), channel: %d, RDMA: %d, nvl: %d, src RDMA lane: %d, dst NVL: %d, meta: %d, %d, %d, %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, dst_nvl_rank, meta_0, meta_1, meta_2, meta_3);
                    trap();
                }
            }
        }
        __syncwarp();

        // Shift cached head
        send_nvl_head += src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank;

        // Wait shared memory to be cleaned
        sync_forwarder_smem();

        // Forward tokens from RDMA buffer
        // NOTES: always start from the local rank
        int src_rdma_rank = sm_id % kNumRDMARanks;
        int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
        const uint64_t output_buffer_size = NUM_DISPATCH_OUTPUT_BYTES_PER_ZCOPY_BUFFER / (hidden_bytes + num_topk * sizeof(int64_t) + num_topk * sizeof(float) + scale_bytes);
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
            // Find next source RDMA rank (round-robin)
            start_time = clock64();
            while (true) {
                src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
                if (__shfl_sync(0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
                    if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
                        cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank)));
                    if (__shfl_sync(0xffffffff, cached_rdma_channel_tail > cached_rdma_channel_head, src_rdma_rank))
                        break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf("DeepEP zcopy dispatch forwarder timeout (RDMA check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, src RDMA lane: %d, head: %d, tail: %d, expected: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id, cached_rdma_channel_head, cached_rdma_channel_tail, num_tokens_to_recv_from_rdma);
                    trap();
                }
            }
            auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);
            auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);

            int total_offset = 0;
            if (src_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank > 0) {
                total_offset = recv_gbl_rank_prefix_sum_fwd[dst_nvl_rank * num_ranks + src_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank - 1];
            }

            int num_recv_tokens = recv_gbl_rank_prefix_sum_fwd[dst_nvl_rank * num_ranks + num_ranks - 1];
            auto target_nvl_rank_x_bytes = num_recv_tokens * hidden_bytes;
            auto target_nvl_rank_topk_idx_bytes = num_recv_tokens * num_topk * sizeof(topk_idx_t);
            auto target_nvl_rank_topk_weights_bytes = num_recv_tokens * num_topk * sizeof(float);
            auto target_nvl_rank_x_scales_bytes = num_recv_tokens * scale_bytes;

            if (num_recv_tokens > output_buffer_size) {
                if (lane_id == 0) {
                    printf("DeepEP dispatch output buffer overflow: RDMA %d, dst_nvl_rank %d, %d > %lu\n",
                        rdma_rank, dst_nvl_rank, num_recv_tokens, output_buffer_size);
                }
                trap();
            }

            auto target_nvl_rank_x = ws_rr_fused_buffer_ptr;
            auto target_nvl_rank_topk_idx = reinterpret_cast<topk_idx_t*>(reinterpret_cast<uint8_t*>(ws_rr_fused_buffer_ptr) + target_nvl_rank_x_bytes);
            auto target_nvl_rank_topk_weights = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ws_rr_fused_buffer_ptr) + target_nvl_rank_x_bytes + target_nvl_rank_topk_idx_bytes);
            auto target_nvl_rank_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ws_rr_fused_buffer_ptr) + target_nvl_rank_x_bytes + target_nvl_rank_topk_idx_bytes + target_nvl_rank_topk_weights_bytes);
            auto target_nvl_rank_src_meta = reinterpret_cast<SourceMeta*>(reinterpret_cast<uint8_t*>(ws_rr_fused_buffer_ptr) + target_nvl_rank_x_bytes + target_nvl_rank_topk_idx_bytes + target_nvl_rank_topk_weights_bytes + target_nvl_rank_x_scales_bytes);

            int start_offset = __shfl_sync(0xffffffff, start_sum, src_rdma_rank);
            int end_offset = __shfl_sync(0xffffffff, end_sum, src_rdma_rank);
            total_offset += start_offset;

            if (lane_id == src_rdma_rank and not finish_signaled) {
                // There will be NUM_MAX_NVL_PEERS threads in this <channel, rank> that learn
                // about the first condition, each belonging to a different forwarder warp. The
                // second condition guarantees only 1 warp (and thus 1 thread) will issue the
                // atomic add.
                if (src_rdma_tail == num_tokens_to_recv_from_rdma_saved and dst_nvl_rank == src_rdma_rank % NUM_MAX_NVL_PEERS) {
                    rdma_signal_finish(lane_id, channel_id);
                }
            }

            // Iterate over every token from the RDMA buffer
            for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++ i) {
                // Wait for previous TMA transfers to finish, if any
                tma_store_wait<0>();
                __syncwarp();

                const bool is_local_token = src_rdma_rank == rdma_rank;
                void *shifted;
                if (is_local_token) {
                    shifted = rdma_channel_src_meta.buffer(src_rdma_rank) + i;
                } else {
                    auto rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;
                    shifted = rdma_channel_data.buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_rdma_token;
                }
                auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(reinterpret_cast<int8_t*>(shifted) +
                    (is_local_token ? 0 : hidden_bytes)));
                __syncwarp();
                bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);
                if (lane_id == src_rdma_rank) {
                    --num_tokens_to_recv_from_rdma;
                    auto cached_head = is_in_dst_nvl_rank ? total_offset : -1;
                    if (not kCachedMode) {
                        send_nvl_head[i * NUM_MAX_NVL_PEERS] = cached_head;
                    }
                    if (src_meta.src_rdma_rank != src_rdma_rank or src_meta.is_token_in_nvl_rank_bits == 0 or src_meta.is_token_in_nvl_rank_bits >= (1<<NUM_MAX_NVL_PEERS)) {
                        printf("DeepEP zcopy dispatch bad source meta: rank %d, i = %d, src_rdma_rank = %d, dst_nvl_rank = %d, meta = (%d, %#x)\n", rank, i, src_rdma_rank, dst_nvl_rank, src_meta.src_rdma_rank, src_meta.is_token_in_nvl_rank_bits);
                        trap();
                    }
                }
                __syncwarp();
                if (not is_in_dst_nvl_rank)
                    continue;

                const void *src_x, *src_scales, *src_topk_idx, *src_topk_weights;
                if (is_local_token) {
                    src_x = x + src_meta.token_idx * hidden_int4;
                    src_scales = x_scales + src_meta.token_idx * num_scales;
                    src_topk_idx = topk_idx + src_meta.token_idx * num_topk;
                    src_topk_weights = topk_weights + src_meta.token_idx * num_topk;
                } else {
                    src_x = shifted;
                    shifted = shift_ptr(shifted, hidden_bytes);
                    shifted = shift_ptr(shifted, sizeof(SourceMeta));
                    src_scales = shifted;
                    shifted = shift_ptr(shifted, scale_bytes);
                    src_topk_idx = shifted;
                    shifted = shift_ptr(shifted, sizeof(topk_idx_t) * num_topk);
                    src_topk_weights = shifted;
                    shifted = shift_ptr(shifted, sizeof(float) * num_topk);
                }

                (lane_id == src_rdma_rank) ? (start_sum += 1) : 0;

                if (elect_one_sync()) {
                    // Load hidden into shared memory
                    tma_load_1d_launch(smem_ptrs[target_rank], src_x, tma_mbarrier + target_rank, hidden_bytes);
                    // Load scales into shared memory
                    tma_load_1d_launch(tma_smem_scales[target_rank], src_scales, tma_mbarrier_scales + target_rank, scale_bytes);
                }

                // Copy source meta
                if (not kCachedMode and elect_one_sync())
                    st_na_global(target_nvl_rank_src_meta + total_offset, src_meta);

                // Copy `topk_idx` and `topk_weights`
                if (lane_id < num_topk) {
                    // Read
                    auto idx_value = ld_nc_global(reinterpret_cast<const topk_idx_t *>(src_topk_idx) + lane_id);
                    auto weight_value = ld_nc_global(reinterpret_cast<const float *>(src_topk_weights) + lane_id);

                    // Transform and write
                    idx_value = (idx_value >= dst_rank_expert_begin and idx_value < dst_rank_expert_end) ? idx_value - dst_rank_expert_begin : -1;
                    st_na_global(target_nvl_rank_topk_idx + total_offset * num_topk + lane_id, idx_value);
                    weight_value = idx_value >= 0 ? weight_value : 0.0f;
                    st_na_global(target_nvl_rank_topk_weights + total_offset * num_topk + lane_id, weight_value);
                }

                if (elect_one_sync()) {
                    // Store scales into target
                    tma_store_1d_launch(tma_smem_scales[target_rank], reinterpret_cast<void *>(target_nvl_rank_x_scales + total_offset * num_scales), tma_mbarrier_scales + target_rank, scale_bytes);
                    // Store hidden into target
                    tma_store_1d_launch(smem_ptrs[target_rank], shift_ptr(target_nvl_rank_x, total_offset * hidden_bytes), tma_mbarrier + target_rank, hidden_bytes);
                }

                total_offset++;

                // In case of insufficient NVL buffers, early stopping
                if ((++ num_tokens_sent) == num_max_nvl_chunked_send_tokens)
                    src_rdma_tail = i + 1;
            }

            // Sync head index
            // With the token data landed in SMEM, RDMA recv buffer can be safely overwritten without waiting for TMA store
            if (lane_id == src_rdma_rank)
                forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);
            tma_store_wait<0>();
            __syncwarp();
        }

        // Retired
        __syncwarp();
        if (elect_one_sync()) {
            st_relaxed_sys_global(nvl_channel_finish_signal.buffer() + nvl_rank, 1);
            forward_channel_retired[dst_nvl_rank] = true;
        };
    } else if (warp_role == WarpRole::kForwarderCoordinator) {
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

        // Clean shared memory
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        #pragma unroll
        for (int i = lane_id; i < kNumRDMARanks * NUM_MAX_NVL_PEERS; i += 32)
            forward_channel_head[i % NUM_MAX_NVL_PEERS][i / NUM_MAX_NVL_PEERS] = 0;
        if (lane_id < NUM_MAX_NVL_PEERS)
            forward_channel_retired[lane_id] = false;
        sync_forwarder_smem();

        int last_head = 0, target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;
        while (true) {
            // Find minimum head
            int min_head = std::numeric_limits<int>::max();
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i) if (not forward_channel_retired[i])
                min_head = min(min_head, forward_channel_head[i][target_rdma]);
            if (__all_sync(0xffffffff, min_head == std::numeric_limits<int>::max()))
                break;

            // Update remote head
            if (min_head != std::numeric_limits<int>::max() and min_head >= last_head + num_max_rdma_chunked_send_tokens and lane_id < kNumRDMARanks) {
                nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank), min_head - last_head,
                                                translate_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank), channel_id, lane_id == rdma_rank);
                last_head = min_head;
            }

            // Nanosleep and let other warps work
            __nanosleep(NUM_WAIT_NANOSECONDS);
        }
    } else {
        // NVL consumers
        // Retrieve rank offset from barrier results (each lane's register stores an RDMA rank)

        for (int i = 0; i < NUM_MAX_NVL_PEERS; i++) {
            int src_nvl_rank = (i + channel_id) % NUM_MAX_NVL_PEERS;
            int total_offset = 0;
            EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
            if (lane_id < kNumRDMARanks and lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
                total_offset = recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank - 1];

            // Receive channel offsets
            int start_offset = 0, end_offset = 0;
            auto start_time = clock64();
            while (lane_id < kNumRDMARanks) {
                start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + src_nvl_rank * kNumRDMARanks + lane_id);
                end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + src_nvl_rank * kNumRDMARanks + lane_id);
                if (start_offset < 0 and end_offset < 0) {
                    start_offset = -start_offset - 1, end_offset = -end_offset - 1;
                    total_offset += start_offset;
                    break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP zcopy dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, src nvl: %d, start: %d, end: %d\n",
                        channel_id, rdma_rank, nvl_rank, lane_id, src_nvl_rank, start_offset, end_offset);
                    trap();
                }
            }

            // Save for combine usage
            if (lane_id < kNumRDMARanks and not kCachedMode)
                recv_gbl_channel_prefix_matrix[(lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] = total_offset;
            __syncwarp();
        }

        // Wait for all local ranks to finish forwarding, so that the output buffer contains the complete dispatch results
        for (int i = lane_id; i < NUM_MAX_NVL_PEERS; i += 32) {
            auto start_time = clock64();
            while (not ld_volatile_global(nvl_channel_finish_signal.buffer() + i)) {
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP zcopy dispatch forwarding wait timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d\n",
                        channel_id, rdma_rank, nvl_rank, i);
                    trap();
                }
            }
        }
    }
}

void dispatch(void* recv_x, float* recv_x_scales, topk_idx_t* recv_topk_idx, float* recv_topk_weights, void* recv_src_meta,
              const void* x, const float* x_scales, const topk_idx_t* topk_idx, const float* topk_weights,
              int* send_rdma_head, int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix, int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix, const int* recv_gbl_rank_prefix_sum,
              const int* recv_gbl_rank_prefix_sum_fwd,
              const bool* is_token_in_rank,
              int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,
              void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
              void** buffer_fused_ptrs, void** buffer_ptrs, int num_zcopy_buffers, int buffer_id, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
              int rank, int num_ranks, bool is_cached_dispatch,
              cudaStream_t stream, int num_channels, bool low_latency_mode) {
    constexpr int smem_size = 16384 * NUM_MAX_NVL_PEERS + ZCOPY_TMA_SMEM_ALIGNMENT;
#define DISPATCH_LAUNCH_CASE(num_rdma_ranks) { \
    const int kNumDispatchRDMASenderWarps = num_rdma_ranks; \
    SETUP_LAUNCH_CONFIG(num_channels, (kNumDispatchRDMASenderWarps + 2 + NUM_MAX_NVL_PEERS) * 32, stream); \
    auto dispatch_func = low_latency_mode ? \
        (is_cached_dispatch ? dispatch<true, num_rdma_ranks, true, smem_size, kNumDispatchRDMASenderWarps> : dispatch<true, num_rdma_ranks, false, smem_size, kNumDispatchRDMASenderWarps>) : \
        (is_cached_dispatch ? dispatch<false, num_rdma_ranks, true, smem_size, kNumDispatchRDMASenderWarps> : dispatch<false, num_rdma_ranks, false, smem_size, kNumDispatchRDMASenderWarps>); \
    SET_SHARED_MEMORY_FOR_TMA(dispatch_func); \
    LAUNCH_KERNEL(&cfg, dispatch_func, \
                  reinterpret_cast<int4*>(recv_x), recv_x_scales, recv_topk_idx, recv_topk_weights, reinterpret_cast<SourceMeta*>(recv_src_meta), \
                  reinterpret_cast<const int4*>(x), x_scales, topk_idx, topk_weights, \
                  send_rdma_head, send_nvl_head, \
                  recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix, \
                  rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
                  gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                  recv_gbl_rank_prefix_sum_fwd, \
                  is_token_in_rank, \
                  num_tokens, hidden_int4, num_scales, num_topk, num_experts, \
                  rdma_buffer_ptr, num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, \
                  buffer_fused_ptrs, buffer_ptrs, num_zcopy_buffers, buffer_id, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, \
                  rank, num_ranks); } break

    EP_HOST_ASSERT((topk_idx == nullptr)  == (topk_weights == nullptr));
    EP_HOST_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

    SWITCH_RDMA_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

__device__ __forceinline__ static void
reduce_add_tma_warp(void* combined_row, void* combined_row_topk_weights, void** src_ptr, void** src_tw_ptr,
                    int num_topk_ranks, int num_topk, size_t size, int lane_id,
                    char* smem, size_t smem_len, uint64_t& tma_mbarrier) {
    EP_DEVICE_ASSERT(size <= smem_len);
    float topk_weight_value = 0;
    for (int i = 0; i < num_topk_ranks; i++) {
        if (elect_one_sync()) {
            mbarrier_init(&tma_mbarrier, 1);
            mbarrier_arrive_and_expect_tx(&tma_mbarrier, size);
            tma_load_1d(smem, reinterpret_cast<char *>(src_ptr[i]), &tma_mbarrier, size);
        }
        __syncwarp();

        if (lane_id < num_topk) {
            topk_weight_value += ld_nc_global(reinterpret_cast<float*>(src_tw_ptr[i]) + lane_id);
        }

        uint32_t tma_phase = 0;
        mbarrier_wait(&tma_mbarrier, tma_phase);

        if (elect_one_sync()) {
            if (i == 0) {
                tma_store_1d(smem, reinterpret_cast<char *>(combined_row), size);
            } else {
                tma_reduce_add_bf16(smem, reinterpret_cast<char *>(combined_row), size);
            }
        }
        tma_store_wait<0>();
        __syncwarp();
    }

    if (lane_id < num_topk) {
        st_na_global(reinterpret_cast<float*>(combined_row_topk_weights) + lane_id, topk_weight_value);
    }
}

template <int kNumRanks, typename dtype_t, int kMaxNumRanks, typename ReceiveFn, typename ReceiveTWFn>
__device__ static int combine_token(bool is_token_in_rank, int head_idx,
                                    int lane_id, int hidden_int4, int num_topk,
                                    int4* combined_row, float* combined_topk_weights,
                                    int num_max_recv_tokens, const ReceiveFn& recv_fn, const ReceiveTWFn& recv_tw_fn) {
    constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

    // Broadcast current heads
    // Lane `i` holds the head of rank `i` and `is_token_in_rank`
    EP_STATIC_ASSERT(kMaxNumRanks <= 32, "Too many ranks");
    int num_topk_ranks = 0, topk_ranks[kMaxNumRanks], slot_indices[kMaxNumRanks];
    #pragma unroll
    for (int i = 0; i < kNumRanks; ++ i) if (__shfl_sync(0xffffffff, is_token_in_rank, i)) {
        slot_indices[num_topk_ranks] = __shfl_sync(0xffffffff, head_idx, i) % num_max_recv_tokens;
        topk_ranks[num_topk_ranks ++] = i;
    }
    EP_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);

    // Reduce data
    #pragma unroll
    for (int i = lane_id; i < hidden_int4; i += 32) {
        // Read buffers
        // TODO: maybe too many registers here
        int4 recv_value_int4[kMaxNumRanks];
        #pragma unroll
        for (int j = 0; j < num_topk_ranks; ++ j)
            recv_value_int4[j] = recv_fn(topk_ranks[j], slot_indices[j], i);

        // Reduce all-to-all results
        float values[kDtypePerInt4] = {0};
        #pragma unroll
        for (int j = 0; j < num_topk_ranks; ++ j) {
            auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
            #pragma unroll
            for (int k = 0; k < kDtypePerInt4; ++ k)
                values[k] += static_cast<float>(recv_value_dtypes[k]);
        }

        // Cast back to `dtype_t` and write
        int4 out_int4;
        auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
        #pragma unroll
        for (int j = 0; j < kDtypePerInt4; ++ j)
            out_dtypes[j] = static_cast<dtype_t>(values[j]);
        st_na_global(combined_row + i, out_int4);
    }

    // Reduce `topk_weights`
    if (lane_id < num_topk) {
        float value = 0;
        #pragma unroll
        for (int i = 0; i < num_topk_ranks; ++ i)
            value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);
        st_na_global(combined_topk_weights + lane_id, value);
    }

    // Return the minimum top-k rank
    return topk_ranks[0];
}

template<bool kLowLatencyMode,
         int kNumRDMARanks, typename dtype_t,
         int kNumCombineForwarderWarps,
         int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks),
         int kNumWarpsPerForwarder = (kNumCombineForwarderWarps / kNumRDMARanks > 0) ? kNumCombineForwarderWarps / kNumRDMARanks : 1,
         int kNumForwarders = kNumRDMARanks * kNumWarpsPerForwarder,
         int kNumRDMAReceivers = 32 - 1 - 1 - kNumForwarders>
__global__ void __launch_bounds__(32 * 32, 1)
combine(int4* combined_x, float* combined_topk_weights,
        const bool* is_combined_token_in_rank,
        const int4* x, const float* topk_weights,
        const int* combined_rdma_head, const int* combined_nvl_head,
        const SourceMeta* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, const int* gbl_channel_prefix_matrix,
        const int* recv_gbl_rank_prefix_sum_fwd,
        int num_tokens, int num_combined_tokens, int hidden, int num_topk,
        void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
        void** buffer_fused_ptrs, void** buffer_ptrs, int buffer_id, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
        int rank, int num_ranks) {
    enum class WarpRole {
        kNVLSender,
        kNVLAndRDMAForwarder,
        kRDMAReceiver,
        KCoordinator
    };

    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const auto num_channels = static_cast<int>(gridDim.x), channel_id = sm_id;

    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);
    const auto hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));

    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto role_meta = [=]() -> std::pair<WarpRole, int> {
        auto warp_id = thread_id / 32;
        if (warp_id == 0) {
            return {WarpRole::kNVLSender, 0};
        } else if (warp_id < 1 + kNumForwarders) {
            auto shuffled_warp_id = warp_id - 1;
            return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};
        } else if (warp_id < 1 + kNumForwarders + kNumRDMAReceivers) {
            auto shuffled_warp_id = warp_id - 1 - kNumForwarders;
            return {WarpRole::kRDMAReceiver, shuffled_warp_id};
        } else {
            return {WarpRole::KCoordinator, 0};
        }
    }();
    auto warp_role = role_meta.first;
    auto warp_id = role_meta.second;

    EP_DEVICE_ASSERT(num_warps == 1 + kNumForwarders + kNumRDMAReceivers + 1);

    if (warp_role == WarpRole::kNVLSender) {
        // NVL layouts
        void *rs_wr_buffer_ptr = buffer_ptrs[nvl_rank];
        auto nvl_channel_finish_signal = AsymBuffer<int>(rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels);

        // In zero-copy combine, an NVLAndRDMAForwarder actively pulls data from local NVL ranks.
        // A rank only exits combine after all local NVL ranks have finished pulling (and thus the input buffer can be reused).
        if (lane_id < NUM_MAX_NVL_PEERS) {
            auto start_time = clock64();
            while (not ld_volatile_global(nvl_channel_finish_signal.buffer() + lane_id)) {
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP zcopy combine forwarding wait timeout, channel: %d, RDMA: %d, nvl: %d, dst NVL: %d\n",
                        channel_id, rdma_rank, nvl_rank, lane_id);
                    trap();
                }
            }
        }
    } else {
        // Combiners and coordinators
        // RDMA symmetric layout
        auto hidden_bytes = hidden_int4 * sizeof(int4);
        auto num_bytes_per_rdma_token = get_num_bytes_per_token(hidden_int4, 0, 0, num_topk);
        auto rdma_channel_data = SymBuffer<int8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

        // NVL layouts
        void* ws_rr_fused_buffer_ptr[NUM_MAX_NVL_PEERS], *rs_wr_buffer_ptr[NUM_MAX_NVL_PEERS];
        #pragma unroll
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i) {
            uint64_t offset = NUM_COMBINE_INPUT_BYTES_PER_ZCOPY_BUFFER * buffer_id;
            ws_rr_fused_buffer_ptr[i] = reinterpret_cast<char *>(buffer_fused_ptrs[i]) + offset;
            rs_wr_buffer_ptr[i] = buffer_ptrs[i];
        }
        auto nvl_channel_finish_signal = AsymBuffer<int, NUM_MAX_NVL_PEERS>(rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels);

        // Combiner warp synchronization
        __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers][kNumRDMARanks];
        __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers];
        auto sync_forwarder_smem = [=]() { asm volatile("barrier.sync 1, %0;" :: "r"((kNumForwarders) * 32)); };
        auto sync_rdma_receiver_and_coordinator_smem = [=]() { asm volatile("barrier.sync 2, %0;" :: "r"((kNumRDMAReceivers + 1) * 32)); };

        if (warp_role == WarpRole::kNVLAndRDMAForwarder) {
            // Dynamic TMA shared memory layout
            const size_t kNumTMABytesPerWarp = 16384;
            extern __shared__ int4 tma_smem[];
            char *tma_smem_aligned = reinterpret_cast<char *>(align_up<uintptr_t>(reinterpret_cast<uintptr_t>(tma_smem), ZCOPY_TMA_SMEM_ALIGNMENT));
            __shared__ uint64_t tma_mbarrier[kNumForwarders];

            char *smem_ptrs[kNumForwarders];
            for (size_t i = 0; i < kNumForwarders; ++ i) {
                smem_ptrs[i] = tma_smem_aligned + kNumTMABytesPerWarp * i;
            }

            // Receive from NVL ranks and forward to RDMA ranks
            // NOTES: this part is using "large warps" for each RDMA ranks
            const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder;
            const auto sub_warp_id = warp_id % kNumWarpsPerForwarder;
            auto send_buffer = dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank) : rdma_channel_data.send_buffer(dst_rdma_rank);
            auto sync_large_warp = [=]() {
                if (kNumWarpsPerForwarder == 1) {
                    __syncwarp();
                } else {
                    asm volatile("barrier.sync %0, %1;" :: "r"(dst_rdma_rank + 3), "r"(kNumWarpsPerForwarder * 32));
                }
            };
            EP_STATIC_ASSERT(kNumWarpsPerForwarder == 1 or kNumRDMARanks + 2 <= 16, "Barriers are not enough");

            // Get count and cached head
            int num_tokens_to_combine = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            int num_tokens_prefix = channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            num_tokens_to_combine -= num_tokens_prefix;
            num_tokens_prefix += dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
            combined_nvl_head += num_tokens_prefix * NUM_MAX_NVL_PEERS;

            // Iterate over all tokens and combine by chunks
            for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine; token_start_idx += num_max_rdma_chunked_send_tokens) {
                // Check destination queue emptiness, or wait a buffer to be released
                auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
                auto num_chunked_tokens = token_end_idx - token_start_idx;
                auto start_time = clock64();
                while (sub_warp_id == 0 and lane_id == 0) {
                    // Inequality: `num_max_rdma_chunked_recv_tokens - (tail - head) >= num_chunked_tokens`
                    // Here, `token_start_idx` is the actual tail
                    int num_used_slots = token_start_idx - ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                    if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens)
                        break;

                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP zcopy combine forwarder (RDMA check) timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA: %d, head: %ld, tail: %d, chunked: %d\n",
                               channel_id, rdma_rank, nvl_rank, dst_rdma_rank, ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank)), token_start_idx, num_chunked_tokens);
                        trap();
                    }
                }
                sync_large_warp();

                // Combine and write to the RDMA buffer
                for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder) {
                    // Read expected head
                    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                    int expected_head = -1;
                    if (lane_id < NUM_MAX_NVL_PEERS)
                        expected_head = ld_nc_global(combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);

                    // Combine current token
                    auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
                    void* shifted = send_buffer + rdma_slot_idx * num_bytes_per_rdma_token;

                    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Too many ranks");
                    int num_topk_ranks = 0, slot_idx;
                    void* src_ptr[NUM_MAX_NVL_PEERS];
                    void* src_tw_ptr[NUM_MAX_NVL_PEERS];
                    bool is_token_in_rank = expected_head >= 0;

                    #pragma unroll
                    for (int i = 0; i < NUM_MAX_NVL_PEERS; ++ i) {
                        if (__shfl_sync(0xffffffff, is_token_in_rank, i)) {
                            slot_idx = __shfl_sync(0xffffffff, expected_head, i);
                            int num_dispatch_tokens = recv_gbl_rank_prefix_sum_fwd[i * num_ranks + num_ranks - 1];
                            EP_DEVICE_ASSERT(slot_idx < num_dispatch_tokens);
                            int src_nvl_rank_x_bytes = num_dispatch_tokens * hidden_bytes;
                            auto src_nvl_rank_x = reinterpret_cast<int4*>(ws_rr_fused_buffer_ptr[i]);
                            auto src_nvl_rank_topk_weight = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ws_rr_fused_buffer_ptr[i]) + src_nvl_rank_x_bytes);

                            src_ptr[num_topk_ranks] = src_nvl_rank_x + slot_idx * hidden_int4;
                            src_tw_ptr[num_topk_ranks] = src_nvl_rank_topk_weight + slot_idx * num_topk;
                            num_topk_ranks++;
                        }
                    }
                    EP_DEVICE_ASSERT(num_topk_ranks <= NUM_MAX_NVL_PEERS);
                    EP_DEVICE_ASSERT(warp_id < kNumForwarders);

                    // Reduce `hidden` and `topk_weights`
                    reduce_add_tma_warp(
                        shifted,
                        (void*)(reinterpret_cast<int8_t*>(shifted) + hidden_bytes + sizeof(SourceMeta)),
                        (void**)src_ptr,
                        (void**)src_tw_ptr,
                        num_topk_ranks,
                        num_topk,
                        hidden_bytes,
                        lane_id,
                        smem_ptrs[warp_id],
                        kNumTMABytesPerWarp,
                        tma_mbarrier[warp_id]
                    );
                }
                sync_large_warp();

                // Issue RDMA send
                if (sub_warp_id == kNumWarpsPerForwarder - 1) {
                    if (dst_rdma_rank != rdma_rank) {
                        auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
                        const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_rdma_token;
                        const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_rdma_token);
                        const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * num_bytes_per_rdma_token);
                        nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                          translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, lane_id, 0);
                    } else {
                        memory_fence();
                    }

                    // Write new RDMA tail
                    __syncwarp();
                    if (elect_one_sync()) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank), num_chunked_tokens,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, dst_rdma_rank == rdma_rank);
                    }
                }
            }

            sync_forwarder_smem();
            if (warp_id == 0 and lane_id < NUM_MAX_NVL_PEERS) {
                st_relaxed_sys_global(nvl_channel_finish_signal.buffer_by(lane_id) + nvl_rank, 1);
            }
            __syncwarp();
        } else if (warp_role == WarpRole::kRDMAReceiver) {
            // RDMA Consumer
            // Receive from RDMA ranks and write to the output tensor
            // Clean shared memory and sync
            EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
            lane_id < kNumRDMARanks ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;
            sync_rdma_receiver_and_coordinator_smem();

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over all tokens and combine
            int cached_channel_tail_idx = 0;
            for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers) {
                // Read expected head
                EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                int expected_head = -1;
                if (lane_id < kNumRDMARanks) {
                    expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks + lane_id);
                    (expected_head < 0) ? (rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1) : (rdma_receiver_rdma_head[warp_id][lane_id] = expected_head);
                }

                // Wait lanes to be ready
                auto start_time = clock64();
                while (cached_channel_tail_idx <= expected_head) {
                    cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP zcopy combine RDMA receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, tail: %d, waiting: %ld, expect: %d\n",
                               channel_id, rdma_rank, nvl_rank, lane_id, cached_channel_tail_idx, token_idx, expected_head);
                        trap();
                    }
                }
                __syncwarp();

                // Combine current token
                auto recv_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4 { return ld_nc_global(reinterpret_cast<const int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_rdma_token) + hidden_int4_idx);};
                auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float { return ld_nc_global(reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_rdma_token + hidden_bytes + sizeof(SourceMeta)) + topk_idx);};
                combine_token<kNumRDMARanks, dtype_t, kNumTopkRDMARanks>(expected_head >= 0,
                                                                         expected_head, lane_id,
                                                                         hidden_int4, num_topk,
                                                                         combined_x + token_idx * hidden_int4,
                                                                         combined_topk_weights + token_idx * num_topk,
                                                                         num_max_rdma_chunked_recv_tokens, recv_fn, recv_tw_fn);
            }
            // Retired
            __syncwarp();
            if (elect_one_sync())
                rdma_receiver_retired[warp_id] = true;
        } else {
            // Coordinator
            // Sync shared memory status
            sync_rdma_receiver_and_coordinator_smem();

            int last_rdma_head = 0;
            int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
            EP_STATIC_ASSERT(kNumCombineForwarderWarps <= 32, "Invalid number of forwarder warps");
            while (true) {
                // Retired
                if (__all_sync(0xffffffff, lane_id >= kNumRDMAReceivers or rdma_receiver_retired[lane_id]))
                    break;

                // Find minimum head for RDMA ranks
                int min_head = std::numeric_limits<int>::max();
                #pragma unroll
                for (int i = 0; i < kNumRDMAReceivers; ++ i) if (not rdma_receiver_retired[i])
                    min_head = min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);
                if (min_head != std::numeric_limits<int>::max() and min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and lane_id < kNumRDMARanks) {
                    nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank), min_head - last_rdma_head,
                                                    translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, dst_rdma_rank == rdma_rank);
                    last_rdma_head = min_head;
                }

                // Nanosleep and let other warps work
                __nanosleep(NUM_WAIT_NANOSECONDS);
            }
        }
    }
}

void combine(cudaDataType_t type,
             void* combined_x, float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x, const float* topk_weights,
             const void* bias_0, const void* bias_1,
             const int* combined_rdma_head, const int* combined_nvl_head,
             const void* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, const int* gbl_channel_prefix_matrix,
             const int* recv_gbl_rank_prefix_sum_fwd,
             int num_tokens, int num_combined_tokens, int hidden, int num_topk,
             void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
             void** buffer_fused_ptrs, void** buffer_ptrs, int buffer_id, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
             int rank, int num_ranks, cudaStream_t stream, int num_channels, bool low_latency_mode) {
    // TODO: Zero-copy: Add support for bias
    EP_HOST_ASSERT(bias_0 == nullptr and bias_1 == nullptr);

    constexpr int kNumCombineForwarderWarps = 12;
    int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    auto num_warps_per_forwarder = kNumCombineForwarderWarps / num_rdma_ranks;
    int num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;
    size_t smem_size = 16384 * num_forwarder_warps + ZCOPY_TMA_SMEM_ALIGNMENT;

#define COMBINE_LAUNCH_CASE(num_rdma_ranks) { \
    auto combine_func = low_latency_mode ? \
        combine<true, num_rdma_ranks, nv_bfloat16, kNumCombineForwarderWarps> : combine<false, num_rdma_ranks, nv_bfloat16, kNumCombineForwarderWarps>; \
    SET_SHARED_MEMORY_FOR_TMA(combine_func) \
    LAUNCH_KERNEL(&cfg, combine_func, \
                  reinterpret_cast<int4*>(combined_x), combined_topk_weights, is_combined_token_in_rank, \
                  reinterpret_cast<const int4*>(x), topk_weights, \
                  combined_rdma_head, combined_nvl_head, \
                  reinterpret_cast<const SourceMeta*>(src_meta), rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix, \
                  recv_gbl_rank_prefix_sum_fwd, \
                  num_tokens, num_combined_tokens, hidden, num_topk, \
                  rdma_buffer_ptr, num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, \
                  buffer_fused_ptrs, buffer_ptrs, buffer_id, num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, \
                  rank, num_ranks); } break

    EP_HOST_ASSERT(kNumCombineForwarderWarps / num_rdma_ranks >= 1);
    EP_HOST_ASSERT(num_forwarder_warps > 0 and num_forwarder_warps % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks > std::max(num_max_rdma_chunked_send_tokens, num_max_nvl_chunked_send_tokens));
    EP_HOST_ASSERT(type == CUDA_R_16BF);

    SETUP_LAUNCH_CONFIG(num_channels, 32 * 32, stream);
    SWITCH_RDMA_RANKS(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace internode_zcopy

} // namespace deep_ep

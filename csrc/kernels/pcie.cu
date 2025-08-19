#include "configs.cuh"
#include "buffer.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"
#include "ibgda_device.cuh"

namespace deep_ep {

namespace pcie {

__host__ __device__ __forceinline__
int get_num_bytes_per_pcie_token(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
    return static_cast<int>(align(hidden_int4 * sizeof(int4) + num_scales * sizeof(float) + num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float), sizeof(int4)));
}

__host__ __device__ __forceinline__
std::pair<int, int> get_pcie_clean_meta(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights, int num_ranks, int num_rdma_recv_buffer_tokens, int num_sms) {
    return {
        (get_num_bytes_per_pcie_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_rdma_recv_buffer_tokens * num_ranks * 2 * num_sms) / sizeof(int),
        4 * num_ranks * 2 * num_sms
    };
}


template <int kNumRanks>
__global__ void
notify_dispatch_pcie(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                const bool* is_token_in_rank, int num_tokens, int num_channels, int expert_alignment,
                const int rdma_clean_offset, const int rdma_num_int_clean,
                int* rdma_channel_prefix_matrix, int* recv_rdma_rank_prefix_sum,
                void* rdma_buffer_ptr, int rank) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;

    auto num_local_experts = num_experts / num_ranks;

    if (sm_id == 0) {
        // Communication with others
        EP_DEVICE_ASSERT(num_warps > 1);
        EP_DEVICE_ASSERT(kNumRanks <= num_threads);
        if (thread_id == 32)
            nvshmem_sync_all();

        // Send numbers of tokens per rank/expert to RDMA ranks
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
        auto rdma_recv_num_tokens_mixed = SymBuffer<int>(rdma_buffer_ptr, 1 + num_local_experts, kNumRanks);

        // Clean up for later data dispatch
        EP_DEVICE_ASSERT(rdma_recv_num_tokens_mixed.total_bytes <= rdma_clean_offset * sizeof(int));

        #pragma unroll
        for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
            rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;

        // Copy to send buffer
        #pragma unroll
        for (int i = thread_id; i < num_ranks; i+=num_threads) {
            rdma_recv_num_tokens_mixed.send_buffer(i)[0] = num_tokens_per_rank[i];
        }
        #pragma unroll
        for (int i = thread_id; i < num_experts; i+=num_threads) {
            rdma_recv_num_tokens_mixed.send_buffer(i/num_local_experts)[1 + i % num_local_experts] = num_tokens_per_expert[i];
        }
        __syncthreads();

        for (int i = warp_id; i < num_ranks; i += num_warps) {
            if (i != rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.recv_buffer(rank)),
                                                reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.send_buffer(i)),
                                                (num_local_experts + 1) * sizeof(int),
                                                i, 0, lane_id, 0);
            } else { 
                UNROLLED_WARP_COPY(1, lane_id, num_local_experts + 1, 
                                    rdma_recv_num_tokens_mixed.recv_buffer(rank), 
                                    rdma_recv_num_tokens_mixed.send_buffer(i), 
                                    ld_volatile_global, st_na_global);
            }
        }
        __syncthreads();

        if (thread_id < num_ranks and thread_id != rank)
            nvshmemi_ibgda_quiet(thread_id, 0);
        __syncthreads();
        
        if (thread_id == 0) {
            nvshmem_sync_all();
        }
        __syncthreads();

        // Reduce the number of tokens per rank/expert
        EP_DEVICE_ASSERT(num_local_experts <= num_threads);
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < num_ranks; ++ i) {
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[0];
                recv_rdma_rank_prefix_sum[i] = sum;
            }
            while (ld_volatile_global(moe_recv_counter_mapped) != -1);
            *moe_recv_counter_mapped = sum;
        }
        
        if (thread_id < num_local_experts) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < num_ranks; ++ i)
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[1  + thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            while (ld_volatile_global(moe_recv_expert_counter_mapped + thread_id) != -1);
            moe_recv_expert_counter_mapped[thread_id] = sum;
        }

        // Finally barrier
        if (thread_id == 32)
            nvshmem_sync_all();
    } else {
        // Calculate meta data
        int dst_rank = sm_id - 1;
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over tokens
            int per_rank_count = 0;
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32) {
                per_rank_count += is_token_in_rank[i * kNumRanks + dst_rank];
            }
            per_rank_count = warp_reduce_sum(per_rank_count);
            // Write into channel matrix
            if (lane_id == 0) {
                rdma_channel_prefix_matrix[dst_rank * num_channels + channel_id] = per_rank_count;
            }
        }
        // Calculate prefix sum
        __syncthreads();

        // Pre-compute prefix sum for all channels
        if (thread_id == 0) {
            auto prefix_row = rdma_channel_prefix_matrix + dst_rank * num_channels;
            #pragma unroll
            for (int i = 1; i < num_channels; ++ i)
                prefix_row[i] += prefix_row[i - 1];
        }
    }
}

void notify_dispatch_pcie(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     const bool* is_token_in_rank, int num_tokens, int num_channels,
                     int hidden_int4, int num_scales, int num_topk, int expert_alignment,
                     int* rdma_channel_prefix_matrix, int* recv_rdma_rank_prefix_sum,
                     void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,int rank,
                     cudaStream_t stream, int64_t num_rdma_bytes) {
#define NOTIFY_DISPATCH_PCIE_LAUNCH_CASE(ranks) { \
    LAUNCH_KERNEL(&cfg, notify_dispatch_pcie<ranks>, \
                  num_tokens_per_rank, moe_recv_counter_mapped, num_ranks, \
                  num_tokens_per_expert, moe_recv_expert_counter_mapped, num_experts, \
                  is_token_in_rank, num_tokens, num_channels, expert_alignment, \
                  pcie_clean_meta.first, pcie_clean_meta.second, \
                  rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
                  rdma_buffer_ptr, rank); } break

    constexpr int kNumThreads = 512;

    // Get clean meta
    auto pcie_clean_meta = get_pcie_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
    EP_HOST_ASSERT((pcie_clean_meta.first + pcie_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());

    // Launch kernel
    SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);
    SWITCH_RANKS(NOTIFY_DISPATCH_PCIE_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_PCIE_LAUNCH_CASE
}

// At most 8 RDMA ranks to be sent
constexpr int get_num_topk_rdma_ranks(int num_rdma_ranks) {
    return num_rdma_ranks < 8 ? num_rdma_ranks : 8;
}

template <bool kCachedMode>
__global__ void cached_notify_pcie(const int rdma_clean_offset, const int rdma_num_int_clean,
                              int* combined_rdma_head, int num_combined_tokens, int num_channels,
                              void* rdma_buffer_ptr,
                              int rank, int num_ranks) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);
    auto num_threads = static_cast<int>(blockDim.x);
    auto num_warps = num_threads / 32;
    auto warp_id = thread_id / 32;
    auto lane_id = get_lane_id();

    if (sm_id == 0) {
        // Barrier for RDMA
        if (thread_id == 0)
            nvshmem_sync_all();
        __syncthreads();

        // Clean
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
        #pragma unroll
        for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
            rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;
        __syncthreads();

        // Barrier again
        if (thread_id == 0)
            nvshmem_sync_all();
    } else if (sm_id == 1) {
        if (kCachedMode)
            return;

        // TODO: for now, we only support 32 ranks, and only use lane_id to iterate ranks
        EP_DEVICE_ASSERT(num_warps >= num_channels);
        EP_DEVICE_ASSERT(num_ranks <= 32);

        // Iterate in reverse order
        if (lane_id < num_ranks and warp_id < num_channels) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, warp_id, token_start_idx, token_end_idx);

            // NOTES: `1 << 25` is a heuristic large number
            int last_head = 1 << 25;
            for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; -- token_idx) {
                auto current_head = __ldg(combined_rdma_head + token_idx * num_ranks + lane_id);
                if (current_head < 0) {
                    combined_rdma_head[token_idx * num_ranks + lane_id] = -last_head - 1;
                } else {
                    last_head = current_head;
                }
            }
        }
    }
}

void cached_notify_pcie(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights,
                   int num_ranks, int num_channels, int num_combined_tokens, int* combined_rdma_head,
                   void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                   int rank, cudaStream_t stream,
                   int64_t num_rdma_bytes,
                   bool is_cached_dispatch) {
    const int num_threads = std::max(128, 32 * num_channels);

    // Get clean meta
    auto pcie_clean_meta = get_pcie_clean_meta(hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
    EP_HOST_ASSERT((pcie_clean_meta.first + pcie_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());

    // Launch kernel
    auto cached_notify_pcie_func = is_cached_dispatch ? cached_notify_pcie<true> : cached_notify_pcie<false>;
    SETUP_LAUNCH_CONFIG(2, num_threads, stream);
    LAUNCH_KERNEL(&cfg, cached_notify_pcie_func,
                  pcie_clean_meta.first, pcie_clean_meta.second,
                  combined_rdma_head, num_combined_tokens, num_channels,
                  rdma_buffer_ptr,
                  rank, num_ranks);
}


template <int kNumRanks,bool kCachedMode,
          int kNumDispatchRDMASenderWarps,
          int kNumDispatchReceiverWarps, int kNumDispatchReceiverCoordinatorWarps,int kNumTopkRanks = get_num_topk_rdma_ranks(kNumRanks)>
__global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1) * 32), 1)
dispatch_pcie(int4* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights, 
         const int4* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
	     const int* rdma_channel_prefix_matrix,
         const int* recv_rdma_rank_prefix_sum, 
         int* recv_rdma_channel_prefix_matrix, 
         int* send_rdma_head, 
         const bool* is_token_in_rank,
         int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,int num_local_experts,
         int scale_token_stride, int scale_hidden_stride,
	     void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
         void** buffer_ptrs,
         int rank) {
    enum class WarpRole {
        kRDMASender,
        kRDMASenderCoordinator,
        kRDMAReceiverCoordinator,
        kRDMAReceiver,
        kNone
    };

    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_channels = static_cast<int>(gridDim.x) / 2, channel_id = sm_id / 2;
    const bool is_receiver = sm_id % 2 == 0;

    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_channels);

    WarpRole warp_role;
    if (is_receiver) {
        if (warp_id < kNumDispatchReceiverWarps) {
            warp_role = WarpRole::kRDMAReceiver;
        } else if (warp_id < kNumDispatchReceiverWarps + kNumDispatchReceiverCoordinatorWarps) {
            warp_role = WarpRole::kRDMAReceiverCoordinator;
        } else {
            warp_role = WarpRole::kNone;
        }
    } else {
        if (warp_id < kNumDispatchRDMASenderWarps) {
            warp_role = WarpRole::kRDMASender;
        } else if (warp_id == kNumDispatchRDMASenderWarps) {
            warp_role = WarpRole::kRDMASenderCoordinator;
        } else {
            warp_role = WarpRole::kNone;
        }
    }
    EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 1);

    // Data checks
    EP_DEVICE_ASSERT(num_topk <= 32);

    // RDMA symmetric layout
    auto num_bytes_per_pcie_token = get_num_bytes_per_pcie_token(hidden_int4, num_scales, num_topk, num_topk);
    auto rdma_channel_data = SymBuffer<int8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_pcie_token, kNumRanks, channel_id, num_channels);
    auto rdma_channel_meta = SymBuffer<int>(rdma_buffer_ptr, 2, kNumRanks, channel_id, num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRanks, channel_id, num_channels);

    // RDMA sender warp synchronization
    // NOTES: `rdma_send_channel_tail` means the latest released tail
    // NOTES: `rdma_send_channel_window` means the ongoing 32 transactions' status
    __shared__ int rdma_send_channel_lock[kNumRanks];
    __shared__ int rdma_send_channel_tail[kNumRanks];
    __shared__ uint32_t rdma_send_channel_window[kNumRanks];

    auto sync_rdma_sender_smem = []() { asm volatile("bar.sync 0, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1) * 32)); };

    // Receiver warp synchronization
    __shared__ int rdma_receiver_channel_head[kNumRanks];
    __shared__ int rdma_receiver_channel_start[kNumRanks];
    __shared__ int rdma_receiver_channel_end[kNumRanks];
    auto sync_receiver_smem = []() { asm volatile("bar.sync 1, %0;" :: "r"((kNumDispatchReceiverWarps + kNumDispatchReceiverCoordinatorWarps) * 32)); };

    if (warp_role == WarpRole::kRDMASender) {
        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);
        // Send number of tokens in this channel by `-value - 1`
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            auto dst_ptr = dst_rdma_rank == rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
            if(lane_id == 0) {
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1]) - 1;
            }else if(lane_id == 1) {
                dst_ptr[lane_id] = -rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] - 1;
            }
            __syncwarp();

            // Issue RDMA for non-local ranks
            if (dst_rdma_rank != rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * 2,
                                                  dst_rdma_rank,
                                                  channel_id, lane_id, 0);
            }
        }
        sync_rdma_sender_smem();

        // Iterate over tokens and copy into buffer
        int64_t token_idx;
        int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;
        auto send_buffer = lane_id == rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);
        for (token_idx = token_start_idx; token_idx < token_end_idx; ++ token_idx) {
            // Read RDMA rank existence
            uint64_t is_token_in_rank_uint64 = 0;
            if (lane_id < kNumRanks) {
                bool is_token_in_rank_value = is_token_in_rank[token_idx * kNumRanks + lane_id];
                global_rdma_tail_idx += is_token_in_rank_value ? 1 : 0;    
                is_token_in_rank_uint64 = is_token_in_rank_value ? 1 : 0;
            }            
            __syncwarp();

            // Skip the token which does not belong to this warp
            if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
                continue;
            auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;

            // Wait the remote buffer to be released
            auto start_time = clock64();
            while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
                cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));

                // Timeout check
                if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch RDMA sender timeout, channel: %d, RDMA: %d, dst RDMA lane: %d, head: %d, tail: %d\n",
                           channel_id, rank, lane_id, cached_rdma_channel_head, rdma_tail_idx);
                    trap();
                }
            }
            __syncwarp();

            // Store RDMA head for combine
            if (lane_id < kNumRanks and not kCachedMode)
                send_rdma_head[token_idx * kNumRanks + lane_id] = rdma_tail_idx;

            // Broadcast tails

            int num_topk_ranks = 0;
            void* dst_send_buffers[kNumTopkRanks];
            #pragma unroll
            for (int i = 0, slot_idx; i < kNumRanks; ++ i) if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {
                slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
                auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i);
                dst_send_buffers[num_topk_ranks ++] = reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) + slot_idx * num_bytes_per_pcie_token;
            }
            EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRanks);
            
            // Copy `x` into symmetric send buffer
            auto st_broadcast = [=](const int key, const int4& value) {
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++ j)
                    st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);
            };
            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, ld_nc_global, st_broadcast);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++ i){
                dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;
            }

            // Copy `x_scales` into symmetric send buffer
            #pragma unroll
            for (int i = lane_id; i < num_scales; i += 32) {
                auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                auto value = ld_nc_global(x_scales + offset);
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++ j)
                    st_na_global(reinterpret_cast<float*>(dst_send_buffers[j]) + i, value);
            }
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++ i)
                dst_send_buffers[i] = reinterpret_cast<float*>(dst_send_buffers[i]) + num_scales;

            // Copy `topk_idx` and `topk_weights` into symmetric send buffer
            #pragma unroll
            for (int i = lane_id; i < num_topk * num_topk_ranks; i += 32) {
                auto rank_idx = i / num_topk, copy_idx = i % num_topk;
                auto idx_value = static_cast<int>(ld_nc_global(topk_idx + token_idx * num_topk + copy_idx));
                auto weight_value = ld_nc_global(topk_weights + token_idx * num_topk + copy_idx);
                st_na_global(reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx, idx_value);
                st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) + num_topk + copy_idx, weight_value);
            }
            __syncwarp();

            // Release the transaction in the window
            if (is_token_in_rank_uint64 != 0) {
                // Acquire lock first
                acquire_lock(rdma_send_channel_lock + lane_id);
                auto latest_tail = rdma_send_channel_tail[lane_id];
                auto offset = rdma_tail_idx - latest_tail;
                while (offset >= 32) {
                    release_lock(rdma_send_channel_lock + lane_id);
                    acquire_lock(rdma_send_channel_lock + lane_id);
                    latest_tail = rdma_send_channel_tail[lane_id];
                    offset = rdma_tail_idx - latest_tail;
                }

                // Release the transaction slot
                // Add the bit and move the ones if possible
                auto window = rdma_send_channel_window[lane_id] | (1u << offset);
                if (offset == 0) {
                    auto num_empty_slots = (~window) == 0 ? 32 : __ffs(~window) - 1;
                    st_release_cta(rdma_send_channel_tail + lane_id, latest_tail + num_empty_slots);
                    window >>= num_empty_slots;
                }
                rdma_send_channel_window[lane_id] = window;

                // Release lock
                release_lock(rdma_send_channel_lock + lane_id);
            }
            __syncwarp();
        }
    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
        // NOTES: in case of splitting, the issued put at the end of the buffer
        EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);
  
        int dst_rank = (warp_id - kNumDispatchRDMASenderWarps) * 32 + lane_id;
        int start_rank = (warp_id - kNumDispatchRDMASenderWarps) * 32;
        int end_rank = min(start_rank + 32, kNumRanks);
        int rank_offset = end_rank - start_rank;

        // Clean shared memory 
        (dst_rank < kNumRanks) ? (rdma_send_channel_lock[dst_rank] = 0) : 0;
        (dst_rank < kNumRanks) ? (rdma_send_channel_tail[dst_rank] = 0) : 0;
        (dst_rank < kNumRanks) ? (rdma_send_channel_window[dst_rank] = 0) : 0;  

        // Synchronize shared memory
        sync_rdma_sender_smem();

        int num_tokens_to_send = 0;
        if (dst_rank < kNumRanks) {
            num_tokens_to_send = rdma_channel_prefix_matrix[dst_rank * num_channels + channel_id];
            if (channel_id > 0)
                num_tokens_to_send -= rdma_channel_prefix_matrix[dst_rank * num_channels + channel_id - 1];
        }

        int last_issued_tail = 0;
        auto start_time = clock64();
        while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
            // Timeout check
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES and dst_rank < kNumRanks) {
                printf("DeepEP RDMA sender coordinator timeout, channel: %d, IB: %d, dst IB: %d, tail: %d, remaining: %d\n",
                        channel_id, rank, dst_rank, last_issued_tail, num_tokens_to_send);
                trap();
            }
            for(int i = start_rank,synced_num_tokens_to_send; i < end_rank; ++i) {
                //shuffle dst rank
                int dst_rdma_rank = (i - start_rank + channel_id + rank) % rank_offset + start_rank;
                synced_num_tokens_to_send = __shfl_sync(0xffffffff, num_tokens_to_send, dst_rdma_rank%32);
                if(synced_num_tokens_to_send == 0)
                    continue;
                auto processed_tail = __shfl_sync(0xffffffff, ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank)), 0);
                auto synced_last_issued_tail = __shfl_sync(0xffffffff, last_issued_tail, dst_rdma_rank%32);
                auto num_tokens_processed = processed_tail - synced_last_issued_tail;
                if(num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < num_max_rdma_chunked_send_tokens)
                    continue;
                auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
   
                EP_DEVICE_ASSERT(num_tokens_to_issue >= 0 and num_tokens_to_issue <= synced_num_tokens_to_send);
                if(dst_rdma_rank != rank) {
                    auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
                    EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
                    const size_t num_bytes_per_msg = num_bytes_per_pcie_token * num_tokens_to_issue;
                    const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rank) + dst_slot_idx * num_bytes_per_pcie_token);
                    const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + dst_slot_idx * num_bytes_per_pcie_token);
                    nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                        dst_rdma_rank, channel_id, lane_id, 0);
                }else{
                    memory_fence();
                }
                
                __syncwarp();
                if(lane_id == dst_rdma_rank%32) {
                    last_issued_tail += num_tokens_to_issue;
                    num_tokens_to_send -= num_tokens_to_issue;
                    nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rank), num_tokens_to_issue,
                                                    dst_rdma_rank, channel_id, dst_rdma_rank == rank);
                }
            }  
        }
    } else if (warp_role == WarpRole::kRDMAReceiver) {
        constexpr int ranks_per_warp = (kNumRanks + kNumDispatchReceiverWarps - 1) / kNumDispatchReceiverWarps;
        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x % 32;

        const int start_rank = warp_id * ranks_per_warp;
        const int end_rank = min(start_rank + ranks_per_warp, kNumRanks);
        const int num_target_ranks= end_rank - start_rank;
        EP_DEVICE_ASSERT(num_target_ranks <= 32);

        auto dst_rank_expert_begin = rank * (num_experts / kNumRanks);
        auto dst_rank_expert_end = min(num_experts, dst_rank_expert_begin + (num_experts / kNumRanks));

        int cached_rdma_channel_head = 0;
        int cached_rdma_channel_tail = 0;
        
        int num_tokens_to_recv_from_rank = 0, channel_offset = 0;
        auto start_time = clock64();
        if (lane_id < num_target_ranks) {
            while (true) {
                auto src_rank = start_rank + lane_id;   
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(src_rank));
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(src_rank) + 1);
                if (meta_0 < 0 && meta_1 < 0) {
                    // Meta data ready (using negative value as ready signal)
                    channel_offset = -meta_0 - 1;
                    int channel_offset_1 = -meta_1 - 1;
                    num_tokens_to_recv_from_rank = channel_offset_1 - channel_offset;
                    if (not kCachedMode)
                        recv_rdma_channel_prefix_matrix[src_rank * num_channels + channel_id] = channel_offset_1 + (src_rank == 0 ? 0 : recv_rdma_rank_prefix_sum[src_rank - 1]);   
                    rdma_receiver_channel_start[src_rank] = channel_offset;
                    rdma_receiver_channel_end[src_rank] = channel_offset_1;
                    break;
                }
                
                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP RDMA receiver timeout (meta), rank: %d, meta_status: %d, %d\n",
                           start_rank + lane_id, meta_0, meta_1);
                    trap();
                }
            }
        }
        __syncwarp();
        sync_receiver_smem();

        // Main processing loop
        int src_rank = -1;  // Round-robin starting point
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rank > 0)) {
            start_time = clock64();
      
            while (true) {
                src_rank = (src_rank + 1) % num_target_ranks;
                if (__shfl_sync(0xffffffff, num_tokens_to_recv_from_rank, src_rank) > 0) {
                    if (lane_id == src_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
                        cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(start_rank + src_rank)));
                    if (__shfl_sync(0xffffffff, cached_rdma_channel_tail > cached_rdma_channel_head, src_rank))
                        break;
                }
                // Timeout check  
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP RDMA receiver timeout (polling), src_rank: %d\n", src_rank);
                    trap();
                }
            }
        
            auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rank);
            auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rank);
            auto channel_offset_recv = __shfl_sync(0xffffffff, channel_offset, src_rank);
            int num_recv_tokens = src_rdma_tail - src_rdma_head;
            
            int rank_offset = start_rank + src_rank == 0 ? 0 : recv_rdma_rank_prefix_sum[start_rank + src_rank - 1];
            
            // Process tokens one by one (similar to reference code)
            for (int token_idx = src_rdma_head; token_idx < src_rdma_tail; token_idx++) {
                int src_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
                int dst_slot_idx = rank_offset + channel_offset_recv + token_idx;
                void* shifted = rdma_channel_data.recv_buffer(start_rank + src_rank) + src_slot_idx * num_bytes_per_pcie_token;
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4,
                                    recv_x + dst_slot_idx * hidden_int4,
                                    reinterpret_cast<int4*>(shifted),
                                    ld_nc_global, st_na_global);
                shifted = static_cast<int4*>(shifted) + hidden_int4;

                // Copy `x_scales`
                UNROLLED_WARP_COPY(1, lane_id, num_scales,
                                    recv_x_scales + dst_slot_idx * num_scales,
                                    reinterpret_cast<float*>(shifted),
                                    ld_nc_global, st_na_global);
                shifted = static_cast<float*>(shifted) + num_scales;
                // Copy `topk_idx` and `topk_weights`
                // NOTES: do not use `shifted` after this `if`, because only several lanes are shifted
                if (lane_id < num_topk) {
                    // Read
                    auto idx_value = ld_nc_global(static_cast<int*>(shifted) + lane_id);
                    shifted = static_cast<int*>(shifted) + num_topk;
                    auto weight_value = ld_nc_global(static_cast<float*>(shifted) + lane_id);

                    // Transform and write
                    idx_value = (idx_value >= dst_rank_expert_begin and idx_value < dst_rank_expert_end) ? idx_value - dst_rank_expert_begin : -1;
                    st_na_global(recv_topk_idx + dst_slot_idx * num_topk + lane_id, static_cast<int64_t>(idx_value));
                    weight_value = idx_value >= 0 ? weight_value : 0.0f;
                    st_na_global(recv_topk_weights + dst_slot_idx * num_topk + lane_id, weight_value);
                }
            }
            if (lane_id == src_rank) {
                cached_rdma_channel_head = cached_rdma_channel_tail;
                num_tokens_to_recv_from_rank -= num_recv_tokens;
                // Update progress for Cooridiantor
                st_release_cta(rdma_receiver_channel_head + start_rank + src_rank, cached_rdma_channel_head);
            }
            __syncwarp();
        }
    } else if (warp_role == WarpRole::kRDMAReceiverCoordinator) {
        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x % 32;
        const int sub_warp_id = warp_id - kNumDispatchReceiverWarps;

        constexpr int ranks_per_warp = (kNumRanks + kNumDispatchReceiverCoordinatorWarps - 1) / kNumDispatchReceiverCoordinatorWarps;
        const int start_rank = sub_warp_id * ranks_per_warp;
        const int end_rank = min(start_rank + ranks_per_warp, kNumRanks);
        const int num_target_ranks= end_rank - start_rank;
        EP_DEVICE_ASSERT(num_target_ranks <= 32);

        int last_head = 0, src_rank = lane_id < num_target_ranks ? lane_id + start_rank : -1;
    
        // Initialize shared state
        if (src_rank >= 0) {
            rdma_receiver_channel_head[src_rank] = 0;
        }
        sync_receiver_smem();
        int num_tokens_to_recv = src_rank >= 0 ? rdma_receiver_channel_end[src_rank] - rdma_receiver_channel_start[src_rank] : -1;
        // Iterate all responsible ranks
        while (__any_sync(0xffffffff, num_tokens_to_recv > 0)) {
            for (int i = 0, synced_num_tokens_to_receive = 0; i < num_target_ranks; ++ i) {
                // To mitigate incast congestion between channels
                int src_rank_offset = (i + channel_id + rank) % num_target_ranks;
                synced_num_tokens_to_receive = __shfl_sync(0xffffffff, num_tokens_to_recv, src_rank_offset);
                if (synced_num_tokens_to_receive <= 0)
                    continue;
                auto synced_last_head = __shfl_sync(0xffffffff, last_head, src_rank_offset);
                auto processed_head = __shfl_sync(0xffffffff, ld_acquire_cta(const_cast<const int*>(rdma_receiver_channel_head + start_rank + src_rank_offset)), 0);
                
                if (processed_head >= synced_last_head + num_max_rdma_chunked_send_tokens || processed_head >= synced_last_head + synced_num_tokens_to_receive) {
                    if (src_rank_offset + start_rank == src_rank) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rank), processed_head - last_head,
                                                        src_rank, channel_id, src_rank == rank);
                        num_tokens_to_recv -= (processed_head - last_head);
                        last_head = processed_head;
                    }
                }
            }
            // Nanosleep and let other warps work
            __nanosleep(NUM_WAIT_NANOSECONDS);
        }
    }
}

void dispatch_pcie(void* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights, 
                   const void* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
                   const int* rdma_channel_prefix_matrix,
                   const int* recv_rdma_rank_prefix_sum,
                   int* recv_rdma_channel_prefix_matrix, int* send_rdma_head,
                   const bool* is_token_in_rank,
                   int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,int num_local_experts,
                   int scale_token_stride, int scale_hidden_stride,
                   void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs, int rank, int num_ranks,bool is_cached_dispatch,
                   cudaStream_t stream, int num_channels) {
    constexpr int kNumDispatchRDMASenderWarps = 8;
    constexpr int kNumDispatchReceiverWarps = 8;
    constexpr int kNumDispatchReceiverCoordinatorWarps = 1;

#define DISPATCH_PCIE_LAUNCH_CASE(num_ranks) { \
    auto dispatch_func = is_cached_dispatch ? \
        pcie::dispatch_pcie<num_ranks, true, kNumDispatchRDMASenderWarps, kNumDispatchReceiverWarps, kNumDispatchReceiverCoordinatorWarps> : \
        pcie::dispatch_pcie<num_ranks, false, kNumDispatchRDMASenderWarps, kNumDispatchReceiverWarps, kNumDispatchReceiverCoordinatorWarps>; \
    LAUNCH_KERNEL(&cfg, dispatch_func, \
                  reinterpret_cast<int4*>(recv_x), recv_x_scales, recv_topk_idx, recv_topk_weights,  \
                  reinterpret_cast<const int4*>(x), x_scales, topk_idx, topk_weights, \
                  rdma_channel_prefix_matrix, \
                  recv_rdma_rank_prefix_sum, recv_rdma_channel_prefix_matrix, send_rdma_head, \
                  is_token_in_rank, \
                  num_tokens, hidden_int4, num_scales, num_topk, num_experts,num_local_experts, \
                  scale_token_stride, scale_hidden_stride, \
                  rdma_buffer_ptr, num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, \
                  buffer_ptrs, rank); } break

    SETUP_LAUNCH_CONFIG(num_channels * 2, (kNumDispatchRDMASenderWarps + 1) * 32, stream);
    SWITCH_RANKS(DISPATCH_PCIE_LAUNCH_CASE);
#undef DISPATCH_PCIE_LAUNCH_CASE
}

template <int kNumRanks, bool kMaybeWithBias, typename dtype_t, int kMaxNumRanks, typename ReceiveFn, typename ReceiveTWFn>
__device__ int combine_token(bool is_token_in_rank, int head_idx,
                             int lane_id, int hidden_int4, int num_topk,
                             int4* combined_row, float* combined_topk_weights,
                             const int4* bias_0_int4, const int4* bias_1_int4,
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
        // Read bias
        // TODO: make it as a finer-grained template
        int4 bias_0_value_int4, bias_1_value_int4;
        if (kMaybeWithBias) {
            bias_0_value_int4 = bias_0_int4 != nullptr ? ld_nc_global(bias_0_int4 + i) : make_int4(0, 0, 0, 0);
            bias_1_value_int4 = bias_1_int4 != nullptr ? ld_nc_global(bias_1_int4 + i) : make_int4(0, 0, 0, 0);
        }

        // Read buffers
        // TODO: maybe too many registers here
        int4 recv_value_int4[kMaxNumRanks];
        #pragma unroll
        for (int j = 0; j < num_topk_ranks; ++ j)
            recv_value_int4[j] = recv_fn(topk_ranks[j], slot_indices[j], i);
        
        // Clean
        // Reduce bias
        float values[kDtypePerInt4] = {0};
        if (kMaybeWithBias) {
            auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
            auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++ j)
                values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);
        }

        // Reduce all-to-all results
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

template<int kNumRanks, typename dtype_t,
         int kNumCombineSenderWarps,
         int kNumCombineReceiverWarps,
         int kNumCombineCoordinatorWarps,
         int kNumTopkRanks = get_num_topk_rdma_ranks(kNumRanks)>
__global__ void __launch_bounds__(((kNumCombineSenderWarps + kNumCombineCoordinatorWarps) * 32), 1)
combine_pcie(int4* combined_x, float* combined_topk_weights,
             const int4* recv_x, const float* recv_topk_weights,
             const int4* bias_0, const int4* bias_1,
             const int* combined_rdma_head,
             const int* recv_rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
             int num_recv_tokens, int num_combined_tokens, int hidden_int4, int num_topk,
             void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
             int rank, int num_ranks) {
    
    enum class WarpRole {
        kRDMASender,
        kRDMASenderCoordinator,
        kRDMAReceiver,
        kRDMAReceiverCoordinator,
        kNone
    };

    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_channels = static_cast<int>(gridDim.x) / 2, channel_id = sm_id / 2;
    const bool is_receiver_sm = sm_id % 2 == 1;

    EP_DEVICE_ASSERT(num_topk <= 32);

    // Role assignment (similar to dispatch_pcie)
    WarpRole warp_role;
    if (not is_receiver_sm) {
        if (warp_id < kNumCombineSenderWarps) {
            warp_role = WarpRole::kRDMASender;
        } else if (warp_id == kNumCombineSenderWarps) {
            warp_role = WarpRole::kRDMASenderCoordinator;
        } else {
            warp_role = WarpRole::kNone;
        }
    } else {
        if (warp_id < kNumCombineReceiverWarps) {
            warp_role = WarpRole::kRDMAReceiver;
        } else if (warp_id < kNumCombineReceiverWarps + kNumCombineCoordinatorWarps) {
            warp_role = WarpRole::kRDMAReceiverCoordinator;
        } else {
            warp_role = WarpRole::kNone;
        }
    }

    // RDMA buffer layouts (using PCIe token format)
    auto hidden_bytes = hidden_int4 * sizeof(int4);
    auto num_bytes_per_pcie_token = get_num_bytes_per_pcie_token(hidden_int4, 0, 0, num_topk);
    auto rdma_channel_data = SymBuffer<uint8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_pcie_token, kNumRanks, channel_id, num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRanks, channel_id, num_channels);

    // Shared memory for synchronization
    __shared__ int sender_channel_tail[kNumRanks];
    __shared__ int receiver_channel_head[kNumCombineReceiverWarps][kNumRanks];
    __shared__ bool receiver_retired[kNumCombineReceiverWarps];
    auto sync_sender_smem = [=]() { asm volatile("bar.sync 0, %0;" :: "r"((kNumCombineSenderWarps + 1) * 32)); };
    auto sync_receiver_smem = [=]() { asm volatile("bar.sync 1, %0;" :: "r"((kNumCombineReceiverWarps + kNumCombineCoordinatorWarps) * 32)); };

    if (warp_role == WarpRole::kRDMASender) {
        // Move data from recv_x to RDMA send buffer (following dispatch_pcie sender pattern)
        constexpr int ranks_per_warp = (kNumRanks + kNumCombineSenderWarps - 1) / kNumCombineSenderWarps;
        const int start_rank = warp_id * ranks_per_warp;
        const int end_rank = min(start_rank + ranks_per_warp, kNumRanks);
        const int num_target_ranks = end_rank - start_rank;
        const int target_rank = lane_id < num_target_ranks ? lane_id + start_rank : -1;
        EP_DEVICE_ASSERT(num_target_ranks <= 32);

        // Clean shared memory and sync
        (target_rank >= 0) ? (sender_channel_tail[target_rank] = 0) : 0;
        sync_sender_smem();

        // Get task range for this channel
        int cached_rdma_channel_head = 0;
        int cached_rdma_channel_tail = 0;
        int total_offset = 0;
        if (target_rank > 0 || (target_rank == 0 && channel_id > 0)) {
            total_offset = recv_rdma_channel_prefix_matrix[target_rank * num_channels + channel_id - 1];
        }
        int num_tokens_to_send_to_rank = target_rank >= 0 ? (recv_rdma_channel_prefix_matrix[target_rank * num_channels + channel_id] -  total_offset) : 0;
        // Process tokens using round-robin destination selection
        int src_rank = (channel_id + rank - 1) % num_target_ranks;
        while (__any_sync(0xffffffff, num_tokens_to_send_to_rank > 0)) {
            // Find next available destination rank
            auto start_time = clock64();
            while (true) {
                src_rank = (src_rank + 1) % num_target_ranks;
                if (__shfl_sync(0xffffffff, num_tokens_to_send_to_rank, src_rank) > 0) {
                    int num_tokens_to_issue = min(num_tokens_to_send_to_rank, num_max_rdma_chunked_send_tokens);
                    if (lane_id == src_rank) {
                        if (src_rank == lane_id && num_tokens_to_issue + cached_rdma_channel_tail - cached_rdma_channel_head > num_max_rdma_chunked_recv_tokens)
                            cached_rdma_channel_head = static_cast<int>(ld_acquire_sys_global(rdma_channel_head.buffer(start_rank + src_rank)));
                    }
                    if (__shfl_sync(0xffffffff, cached_rdma_channel_tail - cached_rdma_channel_head + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens, src_rank))
                        break;
                }
                
                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP combine_pcie sender timeout, channel: %d, rank: %d, warp: %d\n", 
                           channel_id, rank, warp_id);
                    trap();
                }
            }

            // Process a token for this destination rank
            auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rank);
            auto num_tokens_to_send = __shfl_sync(0xffffffff, num_tokens_to_send_to_rank, src_rank);
            num_tokens_to_send = min(num_tokens_to_send, num_max_rdma_chunked_send_tokens);
            auto total_offset_send = __shfl_sync(0xffffffff, total_offset, src_rank);
            auto token_idx = src_rdma_tail;
            auto send_buffer = (start_rank + src_rank) == rank ? rdma_channel_data.recv_buffer(rank) : rdma_channel_data.send_buffer(start_rank + src_rank);
            for (; token_idx < src_rdma_tail + num_tokens_to_send; token_idx++) {
                int dst_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;        
                int src_slot_idx = total_offset_send + token_idx;
                
                void* shifted = send_buffer + dst_slot_idx * num_bytes_per_pcie_token;
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4,
                                    reinterpret_cast<int4*>(shifted),
                                    recv_x + src_slot_idx * hidden_int4,
                                    ld_nc_global, st_na_global);
                shifted = static_cast<int4*>(shifted) + hidden_int4;

                // Copy `topk_weights`
                // NOTES: do not use `shifted` after this `if`, because only several lanes are shifted
                if (lane_id < num_topk) {
                    auto weight_value = ld_nc_global(recv_topk_weights + src_slot_idx * num_topk + lane_id);
                    st_na_global(static_cast<float*>(shifted) + lane_id, weight_value);
                }
            }
            __syncwarp();
            if (lane_id == src_rank) {
                cached_rdma_channel_tail = token_idx;
                num_tokens_to_send_to_rank -= num_tokens_to_send;
                // Update progress for Cooridiantor
                st_release_cta(sender_channel_tail + target_rank, cached_rdma_channel_tail);
            }
        }

    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
        // Coordinate RDMA sends and update remote tails (following dispatch_pcie coordinator pattern)
        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x % 32;
        const int sub_warp_id = warp_id - kNumCombineSenderWarps;

        constexpr int ranks_per_warp = (kNumRanks + kNumCombineCoordinatorWarps - 1) / kNumCombineCoordinatorWarps;
        const int start_rank = sub_warp_id * ranks_per_warp;
        const int end_rank = min(start_rank + ranks_per_warp, kNumRanks);
        const int num_target_ranks = end_rank - start_rank;
        EP_DEVICE_ASSERT(num_target_ranks <= 32);

        int last_issued_tail = 0, target_rank = lane_id < num_target_ranks ? lane_id + start_rank : -1;
        sync_sender_smem();

        int total_offset = 0;
        if (target_rank > 0 || (target_rank == 0 && channel_id > 0)) {
            total_offset = recv_rdma_channel_prefix_matrix[target_rank * num_channels + channel_id - 1];
        }
        int num_tokens_to_send_to_rank = target_rank >= 0 ? (recv_rdma_channel_prefix_matrix[target_rank * num_channels + channel_id] -  total_offset) : 0;
        auto start_time = clock64();
        while (__any_sync(0xffffffff, num_tokens_to_send_to_rank > 0)) {
           // Timeout check
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES and target_rank < kNumRanks) {
                printf("DeepEP RDMA sender coordinator timeout, channel: %d, IB: %d, dst IB: %d, tail: %d, remaining: %d\n",
                        channel_id, rank, target_rank, last_issued_tail, num_tokens_to_send_to_rank);
                trap();
            }
            // Process each source rank
            for (int i = 0, synced_num_tokens_to_send; i < num_target_ranks; i++) {
                int src_rank_idx = (i + channel_id + rank) % num_target_ranks;
                synced_num_tokens_to_send = __shfl_sync(0xffffffff, num_tokens_to_send_to_rank, src_rank_idx);
                if (synced_num_tokens_to_send <= 0) continue;
                auto synced_last_issued_tail = __shfl_sync(0xffffffff, last_issued_tail, src_rank_idx);
                auto processed_tail = __shfl_sync(0xffffffff, ld_acquire_cta(const_cast<const int*>(sender_channel_tail + start_rank + src_rank_idx)), 0);
                auto num_tokens_processed = processed_tail - synced_last_issued_tail;
                if(num_tokens_processed < num_max_rdma_chunked_send_tokens and num_tokens_processed != synced_num_tokens_to_send)
                    continue;
                auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
                if (src_rank_idx + start_rank != rank) {
                    auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
                    EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
                    const size_t num_bytes_per_msg = num_bytes_per_pcie_token * num_tokens_to_issue;
                    const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rank) + dst_slot_idx * num_bytes_per_pcie_token);
                    const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(src_rank_idx + start_rank) + dst_slot_idx * num_bytes_per_pcie_token);
                    nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                        src_rank_idx + start_rank, channel_id, lane_id, 0);    
                } else {
                    memory_fence();
                }
                __syncwarp();

                if (src_rank_idx + start_rank == target_rank) {
                    last_issued_tail += num_tokens_to_issue;
                    num_tokens_to_send_to_rank -= num_tokens_to_issue;
                    nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rank), num_tokens_to_issue,
                        target_rank, channel_id, target_rank == rank);
                }
            }

            // Nanosleep and let other warps work
            __nanosleep(NUM_WAIT_NANOSECONDS);
        }
    } else if (warp_role == WarpRole::kRDMAReceiver) {
        // Receive from RDMA buffer and write to combined_x with reduction
        // Clean shared memory and sync
        EP_DEVICE_ASSERT(kNumRanks <= 32);
        for (int i = lane_id; i < kNumRanks; i += 32) {
            receiver_channel_head[warp_id][i] = 0;
        }
        if (lane_id == 0) receiver_retired[warp_id] = false;
        sync_receiver_smem();

        // Get task range for combined tokens
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // Iterate over tokens and combine
        int cached_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumCombineReceiverWarps) {
            int expected_head = -1;
            if (lane_id < kNumRanks) {
                expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRanks + lane_id);
                (expected_head < 0) ? (receiver_channel_head[warp_id][lane_id] = -expected_head - 1) : (receiver_channel_head[warp_id][lane_id] = expected_head);
            }
            
            // Wait lanes to be ready
            auto start_time = clock64();
            while (cached_channel_tail_idx <= expected_head) {
                cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP combine RDMA receiver timeout, channel: %d, RDMA: %d, src RDMA: %d, tail: %d, waiting: %ld, expect: %d\n",
                            channel_id, rank, lane_id, cached_channel_tail_idx, token_idx, expected_head);
                    trap();
                }
            }
            __syncwarp();

            // Combine current token
            auto recv_fn = [&](int src_rank, int slot_idx, int hidden_int4_idx) -> int4 { return ld_nc_global(reinterpret_cast<const int4*>(rdma_channel_data.recv_buffer(src_rank) + slot_idx * num_bytes_per_pcie_token) + hidden_int4_idx);};
            auto recv_tw_fn = [&](int src_rank, int slot_idx, int topk_idx) -> float { return ld_nc_global(reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rank) + slot_idx * num_bytes_per_pcie_token + hidden_bytes) + topk_idx);};
            combine_token<kNumRanks, true, dtype_t, kNumTopkRanks>(expected_head >= 0,
                                                                        expected_head, lane_id,
                                                                        hidden_int4, num_topk,
                                                                        combined_x + token_idx * hidden_int4,
                                                                        combined_topk_weights + token_idx * num_topk,
                                                                        bias_0 == nullptr ? nullptr : bias_0 + token_idx * hidden_int4,
                                                                        bias_1 == nullptr ? nullptr : bias_1 + token_idx * hidden_int4,
                                                                        num_max_rdma_chunked_recv_tokens, recv_fn, recv_tw_fn);
       }

        // Mark as retired
        __syncwarp();
        if (lane_id == 0)
            receiver_retired[warp_id] = true;
    } else {
        // kRDMAReceiverCoordinator - update remote heads
        sync_receiver_smem();

        int last_head = 0;
        int dst_rank = lane_id < kNumRanks ? lane_id : 0;
        EP_STATIC_ASSERT(kNumCombineSenderWarps <= 32, "Invalid number of PCIe sender warps");
        while (true) {
            // Retired
            if (__all_sync(0xffffffff, lane_id >= kNumCombineSenderWarps or receiver_retired[lane_id]))
                break;

            // Find minimum head for RDMA ranks
            int min_head = std::numeric_limits<int>::max();
            #pragma unroll
            for (int i = 0; i < kNumCombineSenderWarps; ++ i) if (not receiver_retired[i])
                min_head = min(min_head, receiver_channel_head[i][dst_rank]);
                    if (min_head != std::numeric_limits<int>::max() and min_head >= last_head + num_max_rdma_chunked_send_tokens and lane_id < kNumRanks) {
            nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rank), min_head - last_head,
                                            dst_rank, 
                                            channel_id + num_channels, dst_rank == rank);
            last_head = min_head;
        }

            // Nanosleep and let other warps work
            __nanosleep(NUM_WAIT_NANOSECONDS);
        }
    }
}

void combine_pcie(cudaDataType_t type,
                  void* combined_x, float* combined_topk_weights,
                  const void* recv_x, const float* recv_topk_weights,
                  const void* bias_0, const void* bias_1,
                  const int* combined_rdma_head,
                  const int* recv_rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
                  int num_recv_tokens, int num_combined_tokens, int hidden, int num_topk,
                  void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
                  int rank, int num_ranks, cudaStream_t stream, int num_channels) {
    constexpr int kNumCombineSenderWarps = 8;
    constexpr int kNumCombineReceiverWarps = 8;
    constexpr int kNumCombineCoordinatorWarps = 1;

    EP_HOST_ASSERT(type == CUDA_R_16BF);
    EP_HOST_ASSERT(hidden % (sizeof(int4) / sizeof(nv_bfloat16)) == 0);
   
#define COMBINE_PCIE_LAUNCH_CASE(num_ranks) { \
    auto combine_pcie_func = combine_pcie<num_ranks, nv_bfloat16, kNumCombineSenderWarps, kNumCombineReceiverWarps, kNumCombineCoordinatorWarps>; \
    LAUNCH_KERNEL(&cfg, combine_pcie_func, \
                  reinterpret_cast<int4*>(combined_x), combined_topk_weights, \
                  reinterpret_cast<const int4*>(recv_x), recv_topk_weights, \
                  reinterpret_cast<const int4*>(bias_0), reinterpret_cast<const int4*>(bias_1), \
                  combined_rdma_head, \
                  recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
                  num_recv_tokens, num_combined_tokens, hidden / (sizeof(int4) / sizeof(nv_bfloat16)), num_topk, \
                  rdma_buffer_ptr, num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, \
                  rank, num_ranks); } break

    SETUP_LAUNCH_CONFIG(num_channels * 2, (kNumCombineSenderWarps + kNumCombineCoordinatorWarps) * 32, stream);
    SWITCH_RANKS(COMBINE_PCIE_LAUNCH_CASE);
#undef COMBINE_PCIE_LAUNCH_CASE
}


} // namespace pcie

} // namespace deep_ep

// 伪代码，位于 __global__ void internode::dispatch(...) 中


//一些关键变量注释
    //num_topk:对应的选中的expert数量
    //kNumTopkRDMARanks: min(8,kNumRDMARanks)，最多8个rdma_rank被选中
    //num_topk_ranks: 真实的要发送的topk_ranks数量
    //num_topk_ranks<kNumTopkRDMARanks  num_topk_ranks<=num_topk  
// ==================================================================================================================================
//               1. Kernel输入与关键变量
// ==================================================================================================================================
// -- 全局内存指针 --
const int4* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights;

// 【【【 新增/确认的核心输入，来自notify_dispatch 】】】
// Rank级基础偏移: dst_rank从src_rank接收的数据块，在recv_x中的起始位置
const int* recv_rdma_rank_prefix_matrix;      // 逻辑形状: [dst_rank, src_rank]
// Channel级精细偏移: 在上述Rank级数据块内部，来自特定channel的子块的起始位置
const int* recv_rdma_channel_prefix_matrix; // 逻辑形状: [dst_rank, src_rank, channel]




// 用于构建Meta的预计算前缀和
const int* rdma_channel_prefix_matrix;      // 逻辑形状: [dst_rank, channel]

// 【【【 新增/确认的核心输入，来自notify_dispatch 】】】
// 每个token的topk_ranks_idx，用于计算目标rank
const int* topk_ranks_idx; // 逻辑形状: [token, kNumTopkRDMARanks]topk_idx计算获得




// 通信区
void* rdma_buffer_ptr; // 由 nvshmem_malloc 分配的巨大对称缓冲区

// 输出
int4* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights, SourceMeta* recv_src_meta;



// ==================================================================================================================================
//                2. Buffer布局与辅助类定义
// ==================================================================================================================================
// 划分 rdma_buffer_ptr，严格对齐原始代码的设计哲学

// a. 信令区: 每个(rank, channel)一个int[2]，用于发送meta (start/end)
auto rdma_channel_meta = SymBuffer<int>(rdma_buffer_ptr, 2, kNumRDMARanks, channel_id, num_channels);

// b. 流控反馈区: 每个(rank, channel)一个int，用于接收方反馈消费进度(head)
auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

// c. 数据到达通知区: 每个(rank, channel)一个int，用于发送方通知新数据(tail)
// 注意：在新meta设计下，tail严格来说是冗余的，但保留它用于双重确认或调试
auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

// d. 数据载荷区: 环形缓冲区
auto rdma_channel_data = SymBuffer<int8_t>(rdma_buffer_ptr, 
                                         num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token, 
                                         kNumRDMARanks, channel_id, num_channels);


// ==================================================================================================================================
//                3. 共享内存 (Shared Memory) 定义
// ==================================================================================================================================

// RDMA sender warp synchronization
// NOTES: `rdma_send_channel_tail` means the latest released tail
// NOTES: `rdma_send_channel_window` means the ongoing 32 transactions' status
__shared__ int rdma_send_channel_lock[kNumRDMARanks];
__shared__ int rdma_send_channel_tail[kNumRDMARanks];
__shared__ uint32_t rdma_send_channel_window[kNumRDMARanks];
auto sync_rdma_sender_smem = []() { asm volatile("bar.sync 0, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1) * 32)); };


// Forward warp synchronization
__shared__ volatile int forward_channel_head[kNumRDMARanks];
__shared__ volatile bool forward_channel_retired[kNumRDMARanks];

// ==================================================================================================================================
//                 4. Warp角色分配与执行
// ==================================================================================================================================
// ... (Warp角色分配逻辑，根据sm_id, warp_id等) ...

// line358-line386

switch (warp_role) {

// ==================================================================================================================================
//               WarpRole::kRDMASender (数据暂存者)
// ==================================================================================================================================
case WarpRole::kRDMASender:
{
    //Get task 
    int token_start_idx, token_end_idx;
    get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);


    //Get dst_rank_prefix_matrix
    //构建rdma_channel_meta 代表当前channel，src_rank往dst_rank发送的token数量
    for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
        auto dst_ptr = dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
        if(lane_id == 0) {
            dst_ptr[lane_id] = -(channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1]) - 1;
        }else if(lane_id == 1) {
            dst_ptr[lane_id] = -rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] - 1;
        }
        __syncwarp();

        //Issue RDMA for non-local ranks
        if(dst_rdma_rank != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * 2,
                                                  dst_rdma_rank,
                                                  channel_id, lane_id, 0);
        }
    }
    sync_rdma_sender_smem();

  

    int64_t token_idx;
    int cached_rdma_channel_head = 0;
    int global_rdma_tail_idx[kNumRDMARanks];//待优化 全局[warp][kNumRDMARanks]
    auto send_buffer; //self copy的数据保存位置
    // 逻辑：遍历属于自己的token，将其暂存到目标dst_rank的发送缓冲区
    for (token_idx = token_start_idx; token_idx < token_end_idx; ++ token_idx) {
            // Read RDMA rank existence
            uint64_t is_token_in_rank_uint64 = 0;    
            int target_rank = -1;
            if(lane_id < num_topk) {
                //！！！！！！！！！！这里在notify做一个新的map映射，将topk_idx映射到 topk_ranks_idx[token,num_topk]，就无须在sender中额外计算 
                target_rank = __ldg(reinterpret_cast<const int*>(topk_ranks_idx + token_idx * num_topk + lane_id));
                // target_rank = topk_idx[lane_id] % expertperrank;
                if (target_rank >= 0) {
                    is_token_in_rank_uint64 = 1;
                    global_rdma_tail_idx[target_rank]++;//所有warp里有效的thread都在同步更新全局idx，但只有目标warp会计算rdma_tail_idx
                }
                if(rdma_rank == target_rank) {
                    send_buffer = rdma_channel_data.recv_buffer(rdma_rank);
                }else{
                    send_buffer = rdma_channel_data.send_buffer(target_rank);
                }
            }
            __syncwarp();

            // Skip the token which does not belong to this warp
            if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
                continue;
            //<topk的thread在工作了
            auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx[target_rank] - 1;//获取当前 token 在其所属 warp 中计算出的逻辑尾索引，在dst_rank中的位置，-1表示不存在

            // Wait the remote buffer to be released
            auto start_time = clock64();
            while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
                cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(target_rank)));
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP RDMA sender coordinator timeout, channel: %d, IB: %d, dst IB: %d, tail: %d, remaining: %d\n",
                            channel_id, rdma_rank, target_rank, last_issued_tail, num_tokens_to_send);
                    trap();
                }
            }
            __syncwarp();

            // Store RDMA head for combine，待编辑
                //............

            // Broadcast tails
            SourceMeta src_meta;//only src_rank
            int num_topk_ranks = 0; //真实的要发送的topk_ranks数量，可能小于kNumTopkRDMARanks
            void* dst_send_buffers[num_topk];  //由于
            // int topk_ranks[kNumTopkRDMARanks]; 原代码中没用，先注释掉

            #pragma unroll
            for (int i = 0, slot_idx; i < num_topk; ++ i) if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {
                slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
                // topk_ranks[num_topk_ranks] = target_rank;
                // auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i);
                // auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);
                if (lane_id == num_topk_ranks)
                    src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);//每个lane保存当前要发送的token来自哪个src_rank
                dst_send_buffers[num_topk_ranks ++] = reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) + slot_idx * num_bytes_per_rdma_token;
            }
            EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRDMARanks);

            // Copy `x` into symmetric send buffer
            auto st_broadcast = [=](const int key, const int4& value) {
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++ j)
                    st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);
            };
            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, ld_nc_global, st_broadcast);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++ i)
                dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;        
            

            // Copy source metadata into symmetric send buffer
            if (lane_id < num_topk_ranks)
                st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++ i)
                dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;

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
                acquire_lock(rdma_send_channel_lock + target_rank);
                auto latest_tail = rdma_send_channel_tail[target_rank];
                auto offset = rdma_tail_idx - latest_tail;
                while (offset >= 32) {
                    release_lock(rdma_send_channel_lock + target_rank);
                    acquire_lock(rdma_send_channel_lock + target_rank);
                    latest_tail = rdma_send_channel_tail[target_rank];
                    offset = rdma_tail_idx - latest_tail;
                }

                // Release the transaction slot
                // Add the bit and move the ones if possible
                auto window = rdma_send_channel_window[target_rank] | (1u << offset);
                if (offset == 0) {
                    auto num_empty_slots = (~window) == 0 ? 32 : __ffs(~window) - 1;//计算窗口从开头开始有多少个连续的 1（即多少个连续完成的 token）。
                    st_release_cta(rdma_send_channel_tail + target_rank, latest_tail + num_empty_slots);
                    window >>= num_empty_slots;//清理窗口中已经完成的 token
                }
                rdma_send_channel_window[target_rank] = window;//保存更新后的窗口状态，并释放锁。

                // Release lock
                release_lock(rdma_send_channel_lock + target_rank);
            }
            __syncwarp();            


    }
    __syncwarp();
}






// ==================================================================================================================================
//            WarpRole::kSenderCoordinator (数据发送者)
// ==================================================================================================================================
case WarpRole::kSenderCoordinator:
{
    
    // NOTES: in case of splitting, the issued put at the end of the buffer
    EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);

    // Clean shared memory   
    //使用每个warp32个dst rank优先填满的策略
    dst_rank = (warp_id - kNumDispatchRDMASenderWarps) * 32 + lane_id;
    start_rank = (warp_id - kNumDispatchRDMASenderWarps) * 32;
    end_rank = min(start_rank + 32, kNumRDMARanks);
    //用于shuffle的偏移量
    rank_offset = end_rank - start_rank + 1;

    
    (dst_rank < kNumRDMARanks) ? (rdma_send_channel_lock[dst_rank] = 0) : 0;
    (dst_rank < kNumRDMARanks) ? (rdma_send_channel_tail[dst_rank] = 0) : 0;
    (dst_rank < kNumRDMARanks) ? (rdma_send_channel_window[dst_rank] = 0) : 0;  

    // Synchronize shared memory
    sync_rdma_sender_smem();

    int num_tokens_to_send = 0;
    if (dst_rank < kNumRDMARanks) {
        
        num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
        if (channel_id > 0)
            num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];
    }

    int last_issued_tail = 0;
    auto start_time = clock64();
    while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
            printf("DeepEP RDMA sender coordinator timeout, channel: %d, IB: %d, dst IB: %d, tail: %d, remaining: %d\n",
                    channel_id, rdma_rank, lane_id, last_issued_tail, num_tokens_to_send);
            trap();
        } 
    for(int i = start_rank,synced_num_tokens_to_send; i < end_rank; ++i) {
        //修改了shuffle的逻辑
        int dst_rdma_rank = (i - start_rank + channel_id + rdma_rank) % rank_offset + start_rank;
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
        if(dst_rdma_rank != rdma_rank) {
            auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
            EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
            const size_t num_bytes_per_msg = num_bytes_per_rdma_token * num_tokens_to_issue;
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + dst_slot_idx * num_bytes_per_rdma_token);
            const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + dst_slot_idx * num_bytes_per_rdma_token);
            nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                dst_rdma_rank, channel_id, lane_id, 0);
        }else{
            memory_fence();
        }
        
        __syncwarp();
        if(lane_id == dst_rdma_rank%32) {
            last_issued_tail += num_tokens_to_issue;
            num_tokens_to_send -= num_tokens_to_issue;
            nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank), num_tokens_to_issue,
                                            dst_rdma_rank, channel_id, dst_rdma_rank == rdma_rank);
        }
        __syncwarp();
    }
}


} // end switch


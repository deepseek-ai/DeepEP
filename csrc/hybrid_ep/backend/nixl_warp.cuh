// NIXL inter-node warp implementations (mirrored from hybrid-ep).
// Included only when USE_NIXL is defined.

#ifndef DEEPEP_NIXL_WARP_CUH
#define DEEPEP_NIXL_WARP_CUH

#include "nixl_types.h"

// NIXL inter-node dispatch warp function (1 warp per CUDA block).
// Transfers: tokens, probs (FORWARD_DISPATCH), scaling factors (FP8).
// All puts use no-flush; the final atomic signal flushes everything in one doorbell.
template<typename INTER_NODE_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH>
inline __device__ void N2N_warp_group_device_function(const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const bool *attn_to_rdma_map,
                                                      struct dispatch_gpu_nixl_ctx *nixl_ctx,
                                                      SMEM_TYPE* smem_buffer_ptr)
{
  static_assert(INTER_NODE_GROUP::size() == 32, "INTER_NODE_GROUP should be 1 warp.");

  const int NUM_OF_CHUNKS_PER_RANK = (num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
  constexpr size_t token_size = HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
  bool *smem_attn_to_rdma_map_ptr = smem_buffer_ptr->attn_to_rdma_map_buffer;

  constexpr size_t dst_mvh_stride = 1 + (int)FORWARD_DISPATCH + (int)std::is_same<TOKEN_DATA_TYPE, uint8_t>::value;

  for (int chunk_idx = blockIdx.x; chunk_idx < NUM_OF_CHUNKS_PER_RANK; chunk_idx += NUM_OF_BLOCKS) {
    const int chunk_base_token_idx = chunk_idx * NUM_OF_TOKENS_PER_CHUNK;
    int token_range = min(NUM_OF_TOKENS_PER_CHUNK, num_of_tokens_per_rank - chunk_base_token_idx);

    // Load routing map to shared memory.
    for (int m = INTER_NODE_GROUP::thread_rank(); m < token_range * (NUM_OF_NODES - 1); m += INTER_NODE_GROUP::size())
      smem_attn_to_rdma_map_ptr[m] = attn_to_rdma_map[chunk_base_token_idx * (NUM_OF_NODES - 1) + m];
    __syncwarp();

    for (int idx = 0; idx < NUM_OF_NODES - 1; ++idx) {
      const int remote_idx = (idx + node_rank) % (NUM_OF_NODES - 1);
      const int actual_remote_node_rank = remote_idx < node_rank ? remote_idx : (remote_idx + 1);
      const int my_node_rank_in_remote = (node_rank < actual_remote_node_rank) ? node_rank : (node_rank - 1);
      const size_t flag_offset = (my_node_rank_in_remote * NUM_OF_CHUNKS_PER_RANK + chunk_idx) * sizeof(uint64_t);
      int total_tokens = 0;

      for (int t = INTER_NODE_GROUP::thread_rank(); t < NUM_OF_TOKENS_PER_CHUNK; t += INTER_NODE_GROUP::size()) {
        const int token_idx = t + chunk_base_token_idx;
        const bool need_write = (t < token_range) && smem_attn_to_rdma_map_ptr[remote_idx + t * (NUM_OF_NODES - 1)];
        total_tokens += __popc(__ballot_sync(0xffffffff, need_write));

        if (need_write) {
          unsigned channel_id = blockIdx.x % nixl_ctx->num_channels;
          constexpr unsigned NODELAY_FLAG = 1;  // UCP_DEVICE_FLAG_NODELAY

          unsigned long buffer_sub_idx = 0;

          {
            size_t local_offset = token_idx * HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
            size_t remote_offset = token_idx * HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
            constexpr size_t token_size = HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);

            nixlMemViewElement src_desc{nixl_ctx->local_mvh, buffer_sub_idx, local_offset};
            nixlMemViewElement dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * dst_mvh_stride + buffer_sub_idx, remote_offset};

            unsigned put_flags = 0;

            nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
              src_desc, dst_desc, token_size, channel_id, put_flags);
            assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
          }

          if constexpr (FORWARD_DISPATCH) {
            buffer_sub_idx +=1;
            size_t local_offset = (token_idx * NUM_OF_NODES + actual_remote_node_rank) * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float);
            size_t remote_offset = token_idx * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float);
            constexpr size_t prob_size = (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float);

            nixlMemViewElement src_desc{nixl_ctx->local_mvh, buffer_sub_idx, local_offset};
            nixlMemViewElement dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * dst_mvh_stride + buffer_sub_idx, remote_offset};

            unsigned put_flags = 0;

            nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
              src_desc, dst_desc, prob_size, channel_id, put_flags);
            assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
          }

          if constexpr (std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
            buffer_sub_idx +=1;
            size_t local_offset = token_idx * (HIDDEN_DIM / 128) * sizeof(float);
            size_t remote_offset = token_idx * (HIDDEN_DIM / 128) * sizeof(float);
            constexpr size_t sf_size = (HIDDEN_DIM / 128) * sizeof(float);

            nixlMemViewElement src_desc{nixl_ctx->local_mvh, buffer_sub_idx, local_offset};
            nixlMemViewElement dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * dst_mvh_stride + buffer_sub_idx, remote_offset};

            unsigned put_flags = 0;

            nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
              src_desc, dst_desc, sf_size, channel_id, put_flags);
            assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
          }
        }
      }

      __syncwarp();
      if (total_tokens > 0 && INTER_NODE_GROUP::thread_rank() == 0) {
        const unsigned channel_id = blockIdx.x % nixl_ctx->num_channels;
        nixlMemViewElement sig{nixl_ctx->remote_signal_mvh, (size_t)remote_idx, flag_offset};
        assert(nixlAtomicAdd<nixl_gpu_level_t::THREAD>(1, sig, channel_id, /*NODELAY=*/1) >= NIXL_SUCCESS);
        atomicAdd((unsigned long long*)&nixl_ctx->local_flag_counters[remote_idx], 1ULL);
      }
    }
  }
}

// NIXL inter-node combine warp function (1 warp per CUDA block).
// Transfers: tokens, probs (BACKWARD_COMBINE).
// All puts use no-flush; the final atomic signal flushes everything in one doorbell.
template<typename INTER_NODE_RDMA_GROUP,
         typename SMEM_TYPE,
         int NUM_OF_STAGES_S2G,
         int HIDDEN_DIM,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_EXPERTS_PER_RANK,
         int NUM_OF_RANKS_PER_NODE,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool BACKWARD_COMBINE>
inline __device__ void inter_node_N2N_warp_group_device_function(
    const int node_rank,
    const int num_of_tokens_per_rank,
    const bool *rdma_to_attn_map,
    struct combine_gpu_nixl_ctx *nixl_ctx,
    SMEM_TYPE* smem_buffer_ptr)
{
  static_assert(INTER_NODE_RDMA_GROUP::size() == 32, "INTER_NODE_RDMA_GROUP should be 1 warp.");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % INTER_NODE_RDMA_GROUP::size() == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of 32.");

  constexpr size_t dst_mvh_stride = BACKWARD_COMBINE ? 2 : 1;

  int NUM_OF_CHUNKS_PER_RANK = (num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
  int TOTAL_NUM_OF_CHUNKS = (NUM_OF_NODES - 1) * NUM_OF_CHUNKS_PER_RANK;

  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  constexpr size_t token_size = HIDDEN_DIM * sizeof(uint16_t);

  uint32_t token_consumer_parity = 0;
  uint64_t (*mbarrier_ptr)[MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK] = nullptr;
  if constexpr(NUM_OF_NODES != 1)
    mbarrier_ptr = smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer;

  for (int i = blockIdx.x; i < TOTAL_NUM_OF_CHUNKS; i += NUM_OF_BLOCKS) {
    const int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
    const int chunk_id = i / (NUM_OF_NODES - 1);
    const int rdma_remote_node_id = node_id > node_rank ? node_id - 1 : node_id;
    const int remote_idx = rdma_remote_node_id;
    const int my_node_rank_in_remote = (node_rank < node_id) ? node_rank : (node_rank - 1);
    const int chunk_base_token_idx = node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    const int token_range = min(NUM_OF_TOKENS_PER_CHUNK, num_of_tokens_per_rank - chunk_id * NUM_OF_TOKENS_PER_CHUNK);

    while (!cuda::ptx::mbarrier_try_wait_parity(&mbarrier_ptr[rdma_remote_node_id][chunk_id], token_consumer_parity)) {}

    int total_tokens = 0;
    for (int t = INTER_NODE_RDMA_GROUP::thread_rank(); t < NUM_OF_TOKENS_PER_CHUNK; t += INTER_NODE_RDMA_GROUP::size()) {
      const int token_idx = t + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
      const int local_token_idx = rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + token_idx;
      const bool need_write = (t < token_range) && rdma_to_attn_map[t + chunk_base_token_idx];
      total_tokens += __popc(__ballot_sync(0xffffffff, need_write));

      if (need_write) {
        unsigned channel_id = blockIdx.x % nixl_ctx->num_channels;
        unsigned long buffer_sub_idx = 0;

        {
          size_t local_offset = local_token_idx * HIDDEN_DIM * sizeof(uint16_t);
          size_t remote_offset = token_idx * HIDDEN_DIM * sizeof(uint16_t);
          constexpr size_t token_size = HIDDEN_DIM * sizeof(uint16_t);

          nixlMemViewElement src_desc{nixl_ctx->local_mvh, 0, local_offset};
          nixlMemViewElement dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * dst_mvh_stride + 0, remote_offset};

          unsigned put_flags = 0;

          nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
            src_desc, dst_desc, token_size, channel_id, put_flags);
          assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
        }
        if constexpr(BACKWARD_COMBINE) {
          buffer_sub_idx++;
          size_t local_offset = local_token_idx * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float);
          size_t remote_offset = token_idx * (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float);
          constexpr size_t prob_size = (NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE) * sizeof(float);

          nixlMemViewElement src_desc{nixl_ctx->local_mvh, 1, local_offset};
          nixlMemViewElement dst_desc{nixl_ctx->remote_data_mvh, (size_t)remote_idx * dst_mvh_stride + 1, remote_offset};

          unsigned put_flags = 0;

          nixl_status_t status = nixlPut<nixl_gpu_level_t::THREAD>(
            src_desc, dst_desc, prob_size, channel_id, put_flags);
          assert(status == NIXL_SUCCESS || status == NIXL_IN_PROG);
        }
      }
    }

    __syncwarp();
    if (total_tokens > 0 && INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
      const size_t flag_offset = (my_node_rank_in_remote * NUM_OF_CHUNKS_PER_RANK + chunk_id) * sizeof(uint64_t);
      const unsigned channel_id = blockIdx.x % nixl_ctx->num_channels;
      nixlMemViewElement sig{nixl_ctx->remote_signal_mvh, (size_t)remote_idx, flag_offset};
      assert(nixlAtomicAdd<nixl_gpu_level_t::THREAD>(1, sig, channel_id, /*NODELAY=*/1) >= NIXL_SUCCESS);
      atomicAdd((unsigned long long*)&nixl_ctx->local_flag_counters[remote_idx], 1ULL);
    }
  }

  token_consumer_parity ^= 1;
}

#endif // DEEPEP_NIXL_WARP_CUH

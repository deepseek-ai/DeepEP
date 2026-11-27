#pragma once

#include <climits>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/ep/eplb.cuh>
#include <deep_ep/layout/ep/workspace.cuh>

namespace deep_ep::ep {

// Count routes per (block, expert) and record each route's local ordinal within its block.
template <int kNumExperts, int kNumSMs, int kNumThreads>
__device__ __forceinline__ void lb_count_ordinal(const topk_idx_t* topk_idx, const int num_routes, const int num_blocks,
                                                 const int sm_idx, const int thread_idx, int* block_expert_histogram,
                                                 int* expert_token_ordinal_in_block, int* num_warp_tokens) {
    constexpr int kNumWarps = kNumThreads / 32;
    const int warp_idx = ptx::get_warp_idx();
    const int lane_idx = ptx::get_lane_idx();
    for (int block_idx = sm_idx; block_idx < num_blocks; block_idx += kNumSMs) {
        for (int cell_idx = thread_idx; cell_idx < kNumWarps * kNumExperts; cell_idx += kNumThreads)
            num_warp_tokens[cell_idx] = 0;
        __syncthreads();

        const int route_idx = block_idx * kNumThreads + thread_idx;
        const auto value = route_idx < num_routes ? topk_idx[route_idx] : static_cast<topk_idx_t>(-1);
        EP_DEVICE_ASSERT(value < kNumExperts);
        const int route_expert_idx = value >= 0 ? static_cast<int>(value) : -1;
        const auto matching_lane_mask = ptx::match(route_expert_idx);
        int warp_expert_ordinal = 0;
        if (route_expert_idx >= 0) {
            warp_expert_ordinal = __popc(matching_lane_mask & ((1u << lane_idx) - 1));
            if (lane_idx == ptx::ffs(matching_lane_mask))
                num_warp_tokens[warp_idx * kNumExperts + route_expert_idx] = __popc(matching_lane_mask);
        }
        __syncthreads();

        // Turn the per-warp counts into this block's offsets, and store the block count in `[expert][block]` order
        for (int expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += kNumThreads) {
            int num_total_tokens = 0;
            #pragma unroll
            for (int source_warp_idx = 0; source_warp_idx < kNumWarps; ++ source_warp_idx) {
                const int num_tokens = num_warp_tokens[source_warp_idx * kNumExperts + expert_idx];
                num_warp_tokens[source_warp_idx * kNumExperts + expert_idx] = num_total_tokens;
                num_total_tokens += num_tokens;
            }
            block_expert_histogram[expert_idx * num_blocks + block_idx] = num_total_tokens;
        }
        __syncthreads();
        if (route_expert_idx >= 0)
            expert_token_ordinal_in_block[route_idx] =
                warp_expert_ordinal + num_warp_tokens[warp_idx * kNumExperts + route_expert_idx];
        __syncthreads();
    }
}

// Rewrite the block counts in place as block offsets, and publish each expert's total to every peer.
template <int kNumRDMARanks, int kNumNVLRanks, int kNumExperts, int kNumSMs, int kNumThreads, int64_t kNumTimeoutCycles>
__device__ __forceinline__ void lb_count_prefix_and_publish(const comm::NCCLGin& gin, const int num_blocks,
                                                            const int rank_idx, const int rdma_rank_idx,
                                                            const int nvl_rank_idx, const int sm_idx,
                                                            const int thread_idx, const uint64_t target_rdma_seq,
                                                            uint64_t* rdma_arrival_counter,
                                                            int* block_expert_histogram, int* num_tokens_per_rank_expert,
                                                            int* smem_num_expert_tokens) {
    constexpr int kNumWarps = kNumThreads / 32;
    const int warp_idx = ptx::get_warp_idx();
    const int lane_idx = ptx::get_lane_idx();
    int* local_num_tokens_per_expert = num_tokens_per_rank_expert + rank_idx * kNumExperts;
    bool issued_nvl_store = kNumNVLRanks > 1 and sm_idx * kNumWarps < kNumExperts;
    for (int expert_begin = sm_idx * kNumWarps; expert_begin < kNumExperts; expert_begin += kNumSMs * kNumWarps) {
        const int expert_idx = expert_begin + warp_idx;
        int num_total_tokens = 0;
        if (expert_idx < kNumExperts) {
            for (int block_begin = 0; block_begin < num_blocks; block_begin += 32) {
                const int block_idx = block_begin + lane_idx;
                const int num_tokens = block_idx < num_blocks ?
                    block_expert_histogram[expert_idx * num_blocks + block_idx] : 0;
                const int num_inclusive_tokens = ptx::warp_inclusive_sum(num_tokens, lane_idx);
                if (block_idx < num_blocks)
                    block_expert_histogram[expert_idx * num_blocks + block_idx] =
                        num_total_tokens + num_inclusive_tokens - num_tokens;
                num_total_tokens += ptx::exchange(num_inclusive_tokens, 31);
            }
            if (ptx::elect_one_sync())
                smem_num_expert_tokens[warp_idx] = num_total_tokens;
        }
        __syncthreads();

        // Transpose through shared memory so each warp writes 32 contiguous experts to one NVLink peer
        const int dst_expert_idx = expert_begin + lane_idx;
        if (dst_expert_idx < kNumExperts) {
            const int num_dst_tokens = smem_num_expert_tokens[lane_idx];
            if (warp_idx == 0)
                local_num_tokens_per_expert[dst_expert_idx] = num_dst_tokens;
            for (int peer_nvl_idx = warp_idx; peer_nvl_idx < kNumNVLRanks; peer_nvl_idx += kNumWarps) {
                if (peer_nvl_idx != nvl_rank_idx) {
                    auto* peer_num_tokens_per_expert = gin.get_sym_ptr<ncclTeamTagLsa>(local_num_tokens_per_expert, peer_nvl_idx);
                    peer_num_tokens_per_expert[dst_expert_idx] = num_dst_tokens;
                }
            }
        }
        __syncthreads();
    }

    if constexpr (kNumRDMARanks > 1) {
        cooperative_groups::this_grid().sync();
        if (sm_idx == 0) {
            for (int dst_rdma_idx = thread_idx; dst_rdma_idx < kNumRDMARanks; dst_rdma_idx += kNumThreads) {
                if (dst_rdma_idx != rdma_rank_idx) {
                    gin.put<ncclTeamTagRail>(
                        local_num_tokens_per_expert, local_num_tokens_per_expert,
                        kNumExperts * sizeof(int), dst_rdma_idx, 0,
                        ncclGin_StrongVASignalInc{
                            .signalWindow = gin.nccl_window,
                            .signalOffset = gin.get_sym_offset(rdma_arrival_counter + rdma_rank_idx),
                        });
                }
            }
            if (warp_idx == 0 and ptx::elect_one_sync())
                rdma_arrival_counter[rdma_rank_idx] = target_rdma_seq;
            for (int src_rdma_idx = thread_idx; src_rdma_idx < kNumRDMARanks; src_rdma_idx += kNumThreads) {
                if (src_rdma_idx != rdma_rank_idx) {
                    comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                        const auto arrived_seq = ptx::ld_acquire_sys<uint64_t>(rdma_arrival_counter + src_rdma_idx);
                        if (arrived_seq >= target_rdma_seq)
                            return true;
                        if (is_last_check) {
                            printf("DeepEP LB prepare RDMA timeout, rank: %d, rdma: %d<-%d, nvl: %d, arrived: %llu, target: %llu\n",
                                   rank_idx, rdma_rank_idx, src_rdma_idx, nvl_rank_idx,
                                   static_cast<unsigned long long>(arrived_seq), static_cast<unsigned long long>(target_rdma_seq));
                        }
                        return false;
                    });
                }
            }
            __syncthreads();
            if (warp_idx == 0)
                gin.flush<ncclCoopWarp>();
        }

        if constexpr (kNumNVLRanks > 1) {
            cooperative_groups::this_grid().sync();
            constexpr int kNumGlobalWarps = kNumSMs * kNumWarps;
            constexpr int kNumRemoteRows = kNumRDMARanks - 1;
            constexpr int kNumNVLPeers = kNumNVLRanks - 1;
            const int global_warp_idx = sm_idx * kNumWarps + warp_idx;
            for (int task_idx = global_warp_idx; task_idx < kNumRemoteRows * kNumNVLPeers; task_idx += kNumGlobalWarps) {
                const int remote_idx = task_idx / kNumNVLPeers;
                const int peer_idx = task_idx % kNumNVLPeers;
                const int src_rdma_idx = remote_idx + (remote_idx >= rdma_rank_idx);
                const int peer_nvl_idx = peer_idx + (peer_idx >= nvl_rank_idx);
                int* row_ptr = num_tokens_per_rank_expert + (src_rdma_idx * kNumNVLRanks + nvl_rank_idx) * kNumExperts;
                int* peer_row_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(row_ptr, peer_nvl_idx);
                for (int expert_idx = lane_idx; expert_idx < kNumExperts; expert_idx += 32)
                    peer_row_ptr[expert_idx] = row_ptr[expert_idx];
                issued_nvl_store = true;
            }
        }
    }

    if constexpr (kNumNVLRanks > 1) {
        if (issued_nvl_store)
            ptx::fence_acq_rel_sys();
    }
}

// Sum gathered counts across source ranks and record the prefix preceding this rank.
template <int kNumRanks, int kNumDomainExperts, int kNumExperts, int kNumThreads>
__device__ __forceinline__ void lb_aggregate(const int* num_tokens_per_rank_expert, const int rank_idx,
                                             const int thread_idx, int* num_tokens_per_expert, int* expert_token_offset,
                                             int* smem_num_token_sums) {
    constexpr int kNumGroups = 4;
    constexpr int kNumExpertsPerGroup = kNumThreads / kNumGroups;
    int* smem_num_total_tokens = smem_num_token_sums;
    int* smem_num_tokens_before_rank = smem_num_token_sums + kNumThreads;

    const int group_idx = thread_idx / kNumExpertsPerGroup;
    const int expert_idx_in_group = thread_idx % kNumExpertsPerGroup;
    for (int expert_begin_idx = 0; expert_begin_idx < kNumDomainExperts; expert_begin_idx += kNumExpertsPerGroup) {
        const int expert_idx = expert_begin_idx + expert_idx_in_group;
        int num_total_tokens = 0;
        int num_tokens_before_rank = 0;
        if (expert_idx < kNumDomainExperts) {
            for (int source_rank_idx = group_idx; source_rank_idx < kNumRanks; source_rank_idx += kNumGroups) {
                const int num_tokens = num_tokens_per_rank_expert[source_rank_idx * kNumExperts + expert_idx];
                num_total_tokens += num_tokens;
                if (source_rank_idx < rank_idx)
                    num_tokens_before_rank += num_tokens;
            }
        }
        smem_num_total_tokens[thread_idx] = num_total_tokens;
        smem_num_tokens_before_rank[thread_idx] = num_tokens_before_rank;
        __syncthreads();
        if (thread_idx < kNumExpertsPerGroup and expert_idx < kNumDomainExperts) {
            #pragma unroll
            for (int source_group_idx = 1; source_group_idx < kNumGroups; ++ source_group_idx) {
                num_total_tokens += smem_num_total_tokens[source_group_idx * kNumExpertsPerGroup + expert_idx_in_group];
                num_tokens_before_rank += smem_num_tokens_before_rank[source_group_idx * kNumExpertsPerGroup + expert_idx_in_group];
            }
            num_tokens_per_expert[expert_idx] = num_total_tokens;
            expert_token_offset[expert_idx] = num_tokens_before_rank;
        }
        __syncthreads();
    }
}

// Balance the aligned loads of one NVLink domain and compact each expert's active destination segments.
template <int kNumRanks, int kNumMainExpertsPerRank, int kNumRedundantExpertsPerRank, int kTokenAlignment, int kNumThreads>
__device__ __forceinline__ void lb_plan(const int* num_tokens_per_expert, const int physical_expert_offset,
                                        const int thread_idx, int* physical_expert_idx,
                                        int* psum_num_tokens_per_expert_rank, int* redundancy_mapping, int* smem_plan) {
    constexpr int kNumExperts = kNumRanks * kNumMainExpertsPerRank;
    constexpr int kNumPhysicalExpertsPerRank = kNumMainExpertsPerRank + kNumRedundantExpertsPerRank;
    constexpr int kNumWarps = kNumThreads / 32;
    constexpr int kNumRankSlots = math::constexpr_ceil_div(kNumRanks, 32);
    int* smem_num_main_loads = smem_plan;
    int* smem_num_redundant_loads = smem_num_main_loads + kNumExperts;
    int* smem_redundant_slot_idx = smem_num_redundant_loads + kNumExperts;
    int* smem_donor_rank_idx = smem_redundant_slot_idx + kNumExperts;
    int* smem_num_chain_loads = smem_donor_rank_idx + kNumRanks;
    const int warp_idx = ptx::get_warp_idx();
    const int lane_idx = ptx::get_lane_idx();

    // Pick the largest value across the warp and break ties by smallest key
    auto warp_max_with_key = [](int& value, int& key) {
        const int best_value = __reduce_max_sync(0xffffffffu, value);
        key = __reduce_min_sync(0xffffffffu, value == best_value ? key : INT_MAX);
        value = best_value;
    };

    // Home aligned loads and empty redundant slots (remote domains pass nullptr and only emit segment tables)
    for (int expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += kNumThreads) {
        smem_num_main_loads[expert_idx] = static_cast<int>(
            math::ceil_div<int64_t>(num_tokens_per_expert[expert_idx], kTokenAlignment));
        smem_num_redundant_loads[expert_idx] = 0;
    }
    if (redundancy_mapping != nullptr) {
        for (int cell_idx = thread_idx; cell_idx < kNumRanks * kNumRedundantExpertsPerRank; cell_idx += kNumThreads)
            redundancy_mapping[cell_idx] = -1;
    }
    __syncthreads();

    // Pair each deficit receiver with one surplus donor and record its requested chain loads
    if (warp_idx == 0) {
        int load_imbalance[kNumRankSlots];
        int local_donor_rank_idx[kNumRankSlots];
        int local_num_chain_loads[kNumRankSlots];
        int num_total_loads = 0;
        #pragma unroll
        for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
            const int rank_idx = slot_idx * 32 + lane_idx;
            int num_rank_loads = 0;
            if (rank_idx < kNumRanks) {
                for (int local_expert_idx = 0; local_expert_idx < kNumMainExpertsPerRank; ++ local_expert_idx)
                    num_rank_loads += smem_num_main_loads[rank_idx * kNumMainExpertsPerRank + local_expert_idx];
            }
            load_imbalance[slot_idx] = num_rank_loads;
            local_donor_rank_idx[slot_idx] = -1;
            local_num_chain_loads[slot_idx] = 0;
            num_total_loads += num_rank_loads;
        }
        num_total_loads = ptx::reduce_add(num_total_loads);
        const int num_avg_loads = num_total_loads / kNumRanks;
        const int num_remainder = num_total_loads % kNumRanks;
        #pragma unroll
        for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
            const int rank_idx = slot_idx * 32 + lane_idx;
            if (rank_idx < kNumRanks)
                load_imbalance[slot_idx] -= num_avg_loads + (rank_idx < num_remainder);
        }

        // One donor per receiver; a donor above the fair share keeps draining while it can cover the next deficit
        int sticky_donor_idx = -1;
        int num_surplus_loads = 0;
        #pragma unroll 1
        for (int step_idx = 0; step_idx < kNumRanks; ++ step_idx) {
            // Both picks read this step's snapshot, so their reductions overlap
            int num_deficit_loads = 0;
            int receiver_idx = kNumRanks;
            int num_best_surplus_loads = 0;
            int best_donor_idx = kNumRanks;
            #pragma unroll
            for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
                const int rank_idx = slot_idx * 32 + lane_idx;
                const int imbalance = load_imbalance[slot_idx];
                if (-imbalance > num_deficit_loads) {
                    num_deficit_loads = -imbalance;
                    receiver_idx = rank_idx;
                }
                if (imbalance > num_best_surplus_loads) {
                    num_best_surplus_loads = imbalance;
                    best_donor_idx = rank_idx;
                }
            }
            warp_max_with_key(num_deficit_loads, receiver_idx);
            warp_max_with_key(num_best_surplus_loads, best_donor_idx);
            const bool needs_donor = num_surplus_loads + num_surplus_loads / kNumMainExpertsPerRank < num_deficit_loads;
            if (num_deficit_loads <= 0 or (needs_donor and num_best_surplus_loads <= 0))
                break;
            if (needs_donor) {
                sticky_donor_idx = best_donor_idx;
                num_surplus_loads = num_best_surplus_loads > num_avg_loads ? num_best_surplus_loads : 0;
            }
            num_surplus_loads -= num_deficit_loads;
            #pragma unroll
            for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
                const int rank_idx = slot_idx * 32 + lane_idx;
                if (rank_idx == sticky_donor_idx)
                    load_imbalance[slot_idx] -= num_deficit_loads;
                if (rank_idx == receiver_idx) {
                    load_imbalance[slot_idx] += num_deficit_loads;
                    local_donor_rank_idx[slot_idx] = sticky_donor_idx;
                    local_num_chain_loads[slot_idx] = num_deficit_loads;
                }
            }
        }
        #pragma unroll
        for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
            const int rank_idx = slot_idx * 32 + lane_idx;
            if (rank_idx < kNumRanks) {
                smem_donor_rank_idx[rank_idx] = local_donor_rank_idx[slot_idx];
                smem_num_chain_loads[rank_idx] = local_num_chain_loads[slot_idx];
            }
        }
    }
    __syncthreads();

    // Each donor warp walks its own receivers, largest request first, and hands over whole experts
    if constexpr (kNumRedundantExpertsPerRank > 0) {
        for (int donor_idx = warp_idx; donor_idx < kNumRanks; donor_idx += kNumWarps) {
            while (true) {
                int num_remaining_loads = 0;
                int receiver_idx = kNumRanks;
                #pragma unroll
                for (int slot_idx = 0; slot_idx < kNumRankSlots; ++ slot_idx) {
                    const int rank_idx = slot_idx * 32 + lane_idx;
                    const int num_loads = rank_idx < kNumRanks and smem_donor_rank_idx[rank_idx] == donor_idx ?
                        smem_num_chain_loads[rank_idx] : 0;
                    if (num_loads > num_remaining_loads) {
                        num_remaining_loads = num_loads;
                        receiver_idx = rank_idx;
                    }
                }
                warp_max_with_key(num_remaining_loads, receiver_idx);
                if (num_remaining_loads <= 0)
                    break;
                if (ptx::elect_one_sync())
                    smem_num_chain_loads[receiver_idx] = 0;
                int redundant_expert_idx = 0;
                while (num_remaining_loads > 0 and redundant_expert_idx < kNumRedundantExpertsPerRank) {
                    int num_max_available_loads = 0;
                    int best_local_expert_idx = kNumMainExpertsPerRank;
                    for (int local_expert_idx = lane_idx; local_expert_idx < kNumMainExpertsPerRank; local_expert_idx += 32) {
                        const int num_available_loads = smem_num_main_loads[donor_idx * kNumMainExpertsPerRank + local_expert_idx];
                        if (num_available_loads > num_max_available_loads) {
                            num_max_available_loads = num_available_loads;
                            best_local_expert_idx = local_expert_idx;
                        }
                    }
                    warp_max_with_key(num_max_available_loads, best_local_expert_idx);
                    EP_DEVICE_ASSERT(num_max_available_loads > 0);
                    const int num_taken_loads = min(num_remaining_loads, num_max_available_loads);
                    const int expert_idx = donor_idx * kNumMainExpertsPerRank + best_local_expert_idx;
                    if (ptx::elect_one_sync()) {
                        smem_num_main_loads[expert_idx] = num_max_available_loads - num_taken_loads;
                        smem_num_redundant_loads[receiver_idx * kNumMainExpertsPerRank + best_local_expert_idx] = num_taken_loads;
                        smem_redundant_slot_idx[receiver_idx * kNumMainExpertsPerRank + best_local_expert_idx] = redundant_expert_idx;
                        if (redundancy_mapping != nullptr)
                            redundancy_mapping[receiver_idx * kNumRedundantExpertsPerRank + redundant_expert_idx] = expert_idx;
                    }
                    __syncwarp();
                    num_remaining_loads -= num_taken_loads;
                    ++ redundant_expert_idx;
                }
            }
        }
        __syncthreads();
    }

    // Compact each expert's active destination segments and cumulative token thresholds, one warp per expert
    for (int expert_idx = warp_idx; expert_idx < kNumExperts; expert_idx += kNumWarps) {
        const int owner_rank_idx = expert_idx / kNumMainExpertsPerRank;
        const int local_expert_idx = expert_idx % kNumMainExpertsPerRank;
        const int home_physical_expert_idx =
            physical_expert_offset + owner_rank_idx * kNumPhysicalExpertsPerRank + local_expert_idx;
        int tail_rank_idx = owner_rank_idx;
        int num_max_loads = 0;
        if (smem_num_main_loads[expert_idx] == 0) {
            for (int rank_idx = 0; rank_idx < kNumRanks; ++ rank_idx) {
                const int num_rank_loads = smem_donor_rank_idx[rank_idx] == owner_rank_idx ?
                    smem_num_redundant_loads[rank_idx * kNumMainExpertsPerRank + local_expert_idx] : 0;
                if (num_rank_loads > num_max_loads) {
                    num_max_loads = num_rank_loads;
                    tail_rank_idx = rank_idx;
                }
            }
        }
        const int num_tokens = num_tokens_per_expert[expert_idx];
        const int num_padding_tokens = num_tokens % kTokenAlignment == 0 ? 0 : kTokenAlignment - num_tokens % kTokenAlignment;
        int num_previous_rank_tokens = 0;
        int num_active_segments = 0;
        for (int rank_begin = 0; rank_begin < kNumRanks; rank_begin += 32) {
            const int rank_idx = rank_begin + lane_idx;
            const int num_copied_loads = rank_idx < kNumRanks and smem_donor_rank_idx[rank_idx] == owner_rank_idx ?
                smem_num_redundant_loads[rank_idx * kNumMainExpertsPerRank + local_expert_idx] : 0;
            const int dst_physical_expert_idx = num_copied_loads > 0 ?
                physical_expert_offset + rank_idx * kNumPhysicalExpertsPerRank + kNumMainExpertsPerRank +
                    smem_redundant_slot_idx[rank_idx * kNumMainExpertsPerRank + local_expert_idx] :
                home_physical_expert_idx;
            const int num_rank_loads = (rank_idx == owner_rank_idx ? smem_num_main_loads[expert_idx] : 0) + num_copied_loads;
            int num_rank_tokens = static_cast<int>(
                static_cast<int64_t>(num_rank_loads) * kTokenAlignment - (rank_idx == tail_rank_idx ? num_padding_tokens : 0));
            const uint32_t active_mask = ptx::gather(num_rank_tokens > 0);
            const int segment_idx = num_active_segments + __popc(active_mask & ((1u << lane_idx) - 1));
            num_rank_tokens = ptx::warp_inclusive_sum(num_rank_tokens, lane_idx);
            if (active_mask & (1u << lane_idx)) {
                physical_expert_idx[expert_idx * kNumRanks + segment_idx] = dst_physical_expert_idx;
                psum_num_tokens_per_expert_rank[expert_idx * kNumRanks + segment_idx] = num_previous_rank_tokens + num_rank_tokens;
            }
            num_previous_rank_tokens += ptx::exchange(num_rank_tokens, 31);
            num_active_segments += __popc(active_mask);
        }
    }
}

// Point every route at the physical expert that owns its stable index in that expert's sequence.
template <int kNumExperts, int kNumNVLRanks, int kNumSMs, int kNumThreads>
__device__ __forceinline__ void lb_remap(const topk_idx_t* topk_idx, const int* block_expert_histogram,
                                         const int* expert_token_ordinal_in_block, const int* local_num_tokens_per_expert,
                                         const int* expert_token_offset, const int* physical_expert_idx,
                                         const int* psum_num_tokens_per_expert_rank, const int num_routes,
                                         const int num_blocks, const int sm_idx, const int thread_idx,
                                         topk_idx_t* remapped_topk_idx, int* smem_remap) {
    int* smem_expert_token_offset = smem_remap;
    int* smem_single_physical_expert_idx = smem_expert_token_offset + kNumExperts;
    int* smem_start_segment_idx = smem_single_physical_expert_idx + kNumExperts;

    // Pre-advance each expert's segment pointer to this rank's starting offset
    for (int expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += kNumThreads) {
        const int num_local_tokens = local_num_tokens_per_expert[expert_idx];
        int first_ordinal = 0;
        int start_segment_idx = 0;
        int single_physical_expert_idx = 0;
        if (num_local_tokens > 0) {
            first_ordinal = expert_token_offset[expert_idx];
            const int last_ordinal = first_ordinal + num_local_tokens - 1;
            while (psum_num_tokens_per_expert_rank[expert_idx * kNumNVLRanks + start_segment_idx] <= first_ordinal)
                ++ start_segment_idx;
            single_physical_expert_idx = last_ordinal < psum_num_tokens_per_expert_rank[expert_idx * kNumNVLRanks + start_segment_idx] ?
                physical_expert_idx[expert_idx * kNumNVLRanks + start_segment_idx] : -1;
        }
        smem_expert_token_offset[expert_idx] = first_ordinal;
        smem_single_physical_expert_idx[expert_idx] = single_physical_expert_idx;
        smem_start_segment_idx[expert_idx] = start_segment_idx;
    }
    __syncthreads();

    for (int route_idx = sm_idx * kNumThreads + thread_idx; route_idx < num_routes; route_idx += kNumSMs * kNumThreads) {
        const auto value = topk_idx[route_idx];
        if (value < 0) {
            remapped_topk_idx[route_idx] = value;
            continue;
        }
        const int expert_idx = static_cast<int>(value);
        int dst_physical_expert_idx = smem_single_physical_expert_idx[expert_idx];
        if (dst_physical_expert_idx < 0) {
            const int ordinal = smem_expert_token_offset[expert_idx] +
                block_expert_histogram[expert_idx * num_blocks + route_idx / kNumThreads] +
                expert_token_ordinal_in_block[route_idx];
            int segment_idx = smem_start_segment_idx[expert_idx];
            while (psum_num_tokens_per_expert_rank[expert_idx * kNumNVLRanks + segment_idx] <= ordinal)
                ++ segment_idx;
            dst_physical_expert_idx = physical_expert_idx[expert_idx * kNumNVLRanks + segment_idx];
        }
        remapped_topk_idx[route_idx] = static_cast<topk_idx_t>(dst_physical_expert_idx);
    }
}

template <int kNumRDMARanks, int kNumNVLRanks, int kNumMainExpertsPerRank, int kNumRedundantExpertsPerRank,
          int kTokenAlignment, int kNumSMs, int kNumThreads, int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(kNumThreads, 1)
lb_get_redundancy_mapping(const topk_idx_t* topk_idx, const int num_routes, const int rank_idx, const int qp_idx,
                          const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                          int* redundancy_mapping, topk_idx_t* remapped_topk_idx, void* workspace) {
    constexpr int kNumRanks = kNumRDMARanks * kNumNVLRanks;
    constexpr int kNumDomainExperts = kNumNVLRanks * kNumMainExpertsPerRank;
    constexpr int kNumExperts = kNumRanks * kNumMainExpertsPerRank;
    constexpr int kNumPhysicalExpertsPerRank = kNumMainExpertsPerRank + kNumRedundantExpertsPerRank;
    extern __shared__ int smem_scratch[];
    const int sm_idx = static_cast<int>(blockIdx.x);
    const int thread_idx = static_cast<int>(threadIdx.x);
    const int rdma_rank_idx = rank_idx / kNumNVLRanks;
    const int nvl_rank_idx = rank_idx % kNumNVLRanks;
    const int num_blocks = math::ceil_div(num_routes, kNumThreads);
    EP_DEVICE_ASSERT(num_blocks * kNumExperts <= layout::LBPrepareSignals::kNumMaxPrepareRoutes);

    const auto gin = comm::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
    auto& signals = static_cast<layout::EPSignals*>(workspace)->lb_prepare_signals;
    const uint64_t rdma_seq = signals.rdma_arrival_counter[rdma_rank_idx];
    const int epoch = static_cast<int>(
        kNumRDMARanks > 1 ? rdma_seq : *signals.barrier_signals.get_nvl_barrier_counter_ptr());
    int* num_tokens_per_rank_expert = signals.num_tokens_per_rank_expert[epoch & 1];
    int* local_num_tokens_per_expert = num_tokens_per_rank_expert + rank_idx * kNumExperts;

    lb_count_ordinal<kNumExperts, kNumSMs, kNumThreads>(
        topk_idx, num_routes, num_blocks, sm_idx, thread_idx,
        signals.block_expert_histogram, signals.expert_token_ordinal_in_block, smem_scratch);
    cooperative_groups::this_grid().sync();

    lb_count_prefix_and_publish<kNumRDMARanks, kNumNVLRanks, kNumExperts, kNumSMs, kNumThreads, kNumTimeoutCycles>(
        gin, num_blocks, rank_idx, rdma_rank_idx, nvl_rank_idx, sm_idx, thread_idx,
        rdma_seq + 1, signals.rdma_arrival_counter,
        signals.block_expert_histogram, num_tokens_per_rank_expert, smem_scratch);
    comm::gpu_barrier<true, 1, kNumNVLRanks, kNumSMs, kNumThreads, 1, kNumTimeoutCycles, comm::kDeviceBarrierTag,
                      false, true, (kNumRDMARanks > 1 and kNumNVLRanks > 1)>(
        gin, signals.barrier_signals, 0, nvl_rank_idx, sm_idx, thread_idx, false, true);

    for (int domain_idx = sm_idx; domain_idx < kNumRDMARanks; domain_idx += kNumSMs) {
        if (domain_idx > sm_idx or (kNumRDMARanks == 1 and kNumNVLRanks > 1))
            __syncthreads();
        const int expert_offset = domain_idx * kNumDomainExperts;
        const int physical_expert_offset = domain_idx * kNumNVLRanks * kNumPhysicalExpertsPerRank;
        lb_aggregate<kNumRanks, kNumDomainExperts, kNumExperts, kNumThreads>(
            num_tokens_per_rank_expert + expert_offset, rank_idx, thread_idx,
            signals.num_tokens_per_expert + expert_offset, signals.expert_token_offset + expert_offset, smem_scratch);
        lb_plan<kNumNVLRanks, kNumMainExpertsPerRank, kNumRedundantExpertsPerRank, kTokenAlignment, kNumThreads>(
            signals.num_tokens_per_expert + expert_offset, physical_expert_offset, thread_idx,
            signals.physical_expert_idx + expert_offset * kNumNVLRanks,
            signals.psum_num_tokens_per_expert_rank + expert_offset * kNumNVLRanks,
            domain_idx == rdma_rank_idx ? redundancy_mapping : nullptr, smem_scratch);
    }
    cooperative_groups::this_grid().sync();

    lb_remap<kNumExperts, kNumNVLRanks, kNumSMs, kNumThreads>(
        topk_idx, signals.block_expert_histogram, signals.expert_token_ordinal_in_block, local_num_tokens_per_expert,
        signals.expert_token_offset, signals.physical_expert_idx, signals.psum_num_tokens_per_expert_rank,
        num_routes, num_blocks, sm_idx, thread_idx, remapped_topk_idx, smem_scratch);
}

}  // namespace deep_ep::ep

#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic::handle {

struct NCCLGin {
#define IS_TEAM_WORLD(code) if constexpr (std::is_same_v<team_t, ncclTeamTagWorld>) { code }
#define IS_TEAM_LSA(code) if constexpr (std::is_same_v<team_t, ncclTeamTagLsa>) { code }
#define IS_TEAM_RAIL(code) if constexpr (std::is_same_v<team_t, ncclTeamTagRail>) { code }
#define IS_TEAM_WORLD_RAIL(code) if constexpr (std::is_same_v<team_t, ncclTeamTagWorld> or std::is_same_v<team_t, ncclTeamTagRail>) { code }
#define IS_TEAM_WORLD_LSA(code) if constexpr (std::is_same_v<team_t, ncclTeamTagWorld> or std::is_same_v<team_t, ncclTeamTagLsa>) { code }
#define TEAM_WORLD_RAIL() ((std::is_same_v<team_t, ncclTeamTagWorld>) ? team_world : team_rail)

    const ncclDevComm_t& nccl_dev_comm;
    const ncclWindow_t& nccl_window;
    ncclGin gin;
    ncclTeam team_world, team_lsa, team_rail;
    uint64_t lsa_base_ptr;

    // TODO(NCCL): QP index should just be a hint or the users maintain the mapping?
    __device__ __forceinline__
    NCCLGin(const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
            const int& qp_idx = 0,
            const ncclGinResourceSharingMode& resource_sharing_mode = NCCL_GIN_RESOURCE_SHARING_GPU):
        nccl_dev_comm(nccl_dev_comm), nccl_window(nccl_window),
        gin(ncclGin(nccl_dev_comm, qp_idx, resource_sharing_mode)),
        team_world(ncclTeamWorld(nccl_dev_comm)), team_lsa(ncclTeamLsa(nccl_dev_comm)), team_rail(ncclTeamRail(nccl_dev_comm)) {
        // TODO: what if we only have 1 NVLink rank
        lsa_base_ptr = reinterpret_cast<uint64_t>(ncclGetLsaPointer(nccl_window, 0, team_lsa.rank));
    }

    template <typename team_t>
    __device__ __forceinline__ bool is_nvlink_accessible(const int& dst_rank_idx) const {
        IS_TEAM_LSA({
            return true;
        })

        IS_TEAM_WORLD({
            // TODO(NCCL): optimize this function's cycles
            // return ncclTeamRankIsMember(team_lsa, team_world, dst_rank_idx);
            return team_rail.rank * team_lsa.nRanks <= dst_rank_idx and
                   dst_rank_idx < (team_rail.rank + 1) * team_lsa.nRanks;
        })

        IS_TEAM_RAIL({
            // TODO(NCCL): some ranks may be connected by NVLink, e.g., "2 + 2 + 4"
            return team_rail.rank == dst_rank_idx;
        })
    }

    // ReSharper disable once CppNotAllPathsReturnValue
    template <typename dtype_t = void*>
    __device__ __forceinline__
    uint64_t get_sym_offset(dtype_t* ptr) const {
        return reinterpret_cast<uint64_t>(ptr) - lsa_base_ptr;
    }

    // ReSharper disable once CppNotAllPathsReturnValue
    template <typename team_t, typename dtype_t = void*>
    __device__ __forceinline__
    dtype_t* get_sym_ptr(dtype_t* ptr, const int& dst_rank_idx) const {
        IS_TEAM_RAIL({
            return team_rail.rank == dst_rank_idx ? ptr : nullptr;
        })

        IS_TEAM_WORLD_LSA({
            constexpr bool kIsTeamLSA = (std::is_same_v<team_t, ncclTeamTagLsa>);

            // Team world and not accessible by symmetric pointers
            if (not is_nvlink_accessible<team_t>(dst_rank_idx))
                return nullptr;

            // Translate into NVLink rank index
            const auto dst_nvl_rank_idx = kIsTeamLSA ?
                dst_rank_idx : (dst_rank_idx - team_rail.rank * team_lsa.nRanks);

            // Local rank bypass
            // TODO(NCCL): support this
            if (dst_nvl_rank_idx == team_lsa.rank)
                return ptr;

            // Get base ptr
            const auto dst_ptr = ncclGetLsaPointer(
                nccl_window, get_sym_offset(ptr), dst_nvl_rank_idx);
            return static_cast<dtype_t*>(dst_ptr);
        });
    }

    // NOTES: take care of this function when `team_t` is not LSA
    // Do not mix atomic add with gin signal into a single position
    template <typename team_t, typename dtype_t>
    __device__ __forceinline__
    void red_add_rel(dtype_t* sym_ptr, const dtype_t& value, const int& dst_rank_idx,
                     const int& extra_options = 0) const {
        const auto dst_ptr = get_sym_ptr<team_t>(sym_ptr, dst_rank_idx);
        // Use symmetric pointers as much as possible, RDMA otherwise
        if (dst_ptr != nullptr) {
            // NOTES: local rank (or even NVLink-connected) for tag rail can also bypass
            ptx::red_add_rel_sys(dst_ptr, value);
        } else {
            EP_DEVICE_ASSERT((not std::is_same_v<team_t, ncclTeamTagLsa>));
            EP_DEVICE_ASSERT((std::is_same_v<dtype_t, int64_t>) or (std::is_same_v<dtype_t, uint64_t>));
            // TODO(NCCL): support all dtypes
            gin.signal(TEAM_WORLD_RAIL(), dst_rank_idx,
                       ncclGin_VASignalAdd(nccl_window, reinterpret_cast<int64_t>(sym_ptr) - lsa_base_ptr, static_cast<uint64_t>(value)),
                       ncclCoopThread(),
                       ncclGin_None(),
                       cuda::thread_scope_thread,
                       cuda::thread_scope_device,
                       ncclGinOptFlagsDefault | extra_options);
        }
    }

    __device__ __forceinline__
    void wait(ncclGinRequest_t& request) const {
        gin.wait(request);
    }

    template <typename team_t, typename coop_t = ncclCoopThread, typename segment_t = ncclGin_SegmentDevice>
    __device__ __forceinline__
    void get(void* src_ptr, void* dst_ptr, const int& num_bytes, const int& src_rank_idx,
             const int& extra_options = 0) const {
        IS_TEAM_WORLD_RAIL({
            gin.get(
                TEAM_WORLD_RAIL(),
                src_rank_idx,
                nccl_window, reinterpret_cast<int64_t>(src_ptr) - lsa_base_ptr,
                nccl_window, reinterpret_cast<int64_t>(dst_ptr) - lsa_base_ptr,
                num_bytes,
                coop_t(),
                ncclGin_None(),
                ncclGinOptFlagsDefault | extra_options,
                segment_t()
            );
        });
    }

    template <typename coop_t = ncclCoopThread>
    __device__ __forceinline__
    void flush() const {
        gin.flush(coop_t());
    }

    template <typename team_t, typename coop_t = ncclCoopThread>
    __device__ __forceinline__
    void flush_async(const int& src_rank_idx, ncclGinRequest_t* request,
                     const int& extra_options = 0) const {
        IS_TEAM_WORLD_RAIL({
            gin.flushAsync(
                TEAM_WORLD_RAIL(),
                src_rank_idx,
                request,
                coop_t(),
                ncclGinOptFlagsDefault | extra_options
            );
        });
    }

    template <typename team_t, typename remote_action_t>
    __device__ __forceinline__
    void signal(const int& dst_rank_idx, const remote_action_t& remote_action) const {
        IS_TEAM_WORLD_RAIL({
            gin.signal(TEAM_WORLD_RAIL(), dst_rank_idx, remote_action);
        });
    }

    template <typename team_t,
              typename remote_action_t = ncclGin_None>
    __device__ __forceinline__
    void put(void* recv_sym_ptr, void* send_sym_ptr, const int& num_bytes, const int& dst_rank_idx,
             const int& extra_options = 0,
             const remote_action_t& remote_action = remote_action_t()) const {
        // NOTES: local or NVLink put will also go through NIC via this API
        IS_TEAM_WORLD_RAIL({
            gin.put(TEAM_WORLD_RAIL(),
                    dst_rank_idx,
                    // TODO: can we don't repeat the window?
                    // TODO: can we pass raw pointers?
                    nccl_window, reinterpret_cast<int64_t>(recv_sym_ptr) - lsa_base_ptr,
                    nccl_window, reinterpret_cast<int64_t>(send_sym_ptr) - lsa_base_ptr,
                    num_bytes,
                    remote_action,
                    ncclGin_None(),
                    ncclCoopThread(),
                    ncclGin_None(),
                    cuda::thread_scope_thread,
                    cuda::thread_scope_device,
                    ncclGinOptFlagsDefault | extra_options);
        });
    }

    template <typename team_t, typename dtype_t>
    __device__ __forceinline__
    void put_value(dtype_t* sym_ptr, const dtype_t& value, const int& dst_rank_idx,
                   const int& extra_options = 0) const {
        const auto dst_ptr = get_sym_ptr<team_t>(sym_ptr, dst_rank_idx);
        if (dst_ptr != nullptr) {
            ptx::st_relaxed_sys(dst_ptr, value);
        } else {
            EP_DEVICE_ASSERT((not std::is_same_v<team_t, ncclTeamTagLsa>));
            gin.putValue(TEAM_WORLD_RAIL(),
                         dst_rank_idx,
                         nccl_window, reinterpret_cast<int64_t>(sym_ptr) - lsa_base_ptr,
                         value,
                         ncclGin_None(),
                         ncclCoopThread(),
                         ncclGin_None(),
                         cuda::thread_scope_thread,
                         cuda::thread_scope_device,
                         ncclGinOptFlagsDefault | extra_options);
        }
    }

#undef IS_TEAM_WORLD
#undef IS_TEAM_LSA
#undef IS_TEAM_RAIL
#undef IS_TEAM_WORLD_RAIL
#undef IS_TEAM_WORLD_LSA
#undef TEAM_WORLD_RAIL
};

}  // namespace deep_ep::elastic::handle

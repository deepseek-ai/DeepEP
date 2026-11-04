#include <algorithm>
#include <utility>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <deep_ep/common/compiled.cuh>

#include "../../utils/system.hpp"
#include "kernels.cuh"
#include "runtime.hpp"

namespace deep_ep::magic {

using namespace kernels;

static constexpr unsigned kMaxCombineRowBytes = 32 << 10;

__global__ void magic_barrier_kernel(ncclDevComm_t dev_comm) {
    const auto team = ncclTeamWorld(dev_comm);
    ncclGin gin(dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_GPU);
    gin.flush(ncclCoopThread{});
    for (int peer = 0; peer < team.nRanks; ++peer)
        gin.signal(team, peer, ncclGin_SignalInc{static_cast<ncclGinSignal_t>(team.rank)});
    gin.flush(ncclCoopThread{});
    for (int peer = 0; peer < team.nRanks; ++peer)
        gin.waitSignal(ncclCoopThread{}, static_cast<ncclGinSignal_t>(peer), 1);
}

struct Runtime::Context {
    ncclComm_t host_comm = nullptr;
    int device = -1;
    MagicCommState comm_state{};
    MagicAllToAllState* all_to_all_state = nullptr;
    void* lsa_peer_workspaces[kMaxWorldSize]{};
    uint64_t* lsa_peer_completion_ptrs[kMaxWorldSize]{};
    uint8_t* combine_staging = nullptr;
    unsigned* send_counts = nullptr;
    unsigned* recv_counts = nullptr;
    unsigned* recv_peer_row_start = nullptr;
    unsigned* staging_row_indices = nullptr;
    unsigned* token_row_counts = nullptr;
    unsigned dispatch_num_tokens = 0, dispatch_recv_capacity = 0;
    bool dispatch_pending = false;
    unsigned rank, world_size, lsa_size, max_tokens_per_rank;
    unsigned num_lsa_send_ctas, num_gin_send_ctas, num_recv_ctas, lsa_bulk_rows, gin_bulk_rows;
    unsigned dispatch_lsa_bulk_rows, dispatch_gin_bulk_rows;
    size_t rank_slot_bytes;

    void destroy() {
        const c10::cuda::CUDAGuard device_guard(device);
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
        magic_barrier_kernel<<<1, 1>>>(comm_state.dev_comm);
        CUDA_RUNTIME_CHECK(cudaGetLastError());
        CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        NCCL_CHECK(ncclDevCommDestroy(host_comm, &comm_state.dev_comm));

        for (auto* region : {&comm_state.completion, &comm_state.recv_workspace, &comm_state.send_staging}) {
            NCCL_CHECK(ncclCommWindowDeregister(host_comm, region->window));
            NCCL_CHECK(ncclMemFree(region->base));
        }

        void* buffers[] = {token_row_counts, staging_row_indices, send_counts, combine_staging, all_to_all_state};
        for (void* buffer : buffers)
            CUDA_RUNTIME_CHECK(cudaFree(buffer));
    }

    void allocate_window(ncclComm_t host_comm, MagicMemoryRegion& region, size_t bytes, int flags) {
        NCCL_CHECK(ncclMemAlloc(&region.base, bytes));
        NCCL_CHECK(ncclCommWindowRegister(host_comm, region.base, bytes, &region.window, flags));
    }

    MagicSlotLayout slot_layout(unsigned row_bytes, bool is_dispatch = false) const {
        const MagicSlotLayout slot{rank_slot_bytes,
                                   row_bytes,
                                   is_dispatch ? dispatch_lsa_bulk_rows : lsa_bulk_rows,
                                   is_dispatch ? dispatch_gin_bulk_rows : gin_bulk_rows,
                                   rank / lsa_size * lsa_size,
                                   lsa_size};
        for (const unsigned peer : {slot.lsa_first_rank, (slot.lsa_first_rank + lsa_size) % world_size})
            EP_HOST_ASSERT(slot.row_offset(peer, max_tokens_per_rank) <= rank_slot_bytes);
        return slot;
    }

    template <int kNumRanks, typename OperationArgs>
    void launch_all_to_all(const MagicSlotLayout& slot,
                           unsigned* send_counts,
                           unsigned* recv_counts,
                           const OperationArgs& operation_args,
                           cudaStream_t stream) {
        MagicAllToAllArgs<kNumRanks, OperationArgs> args{};
        args.send_counts = send_counts;
        args.recv_counts = recv_counts;
        args.state = all_to_all_state;
        std::copy_n(lsa_peer_completion_ptrs, kNumRanks, args.lsa_peer_completion_ptrs);
        std::copy_n(lsa_peer_workspaces, kNumRanks, args.lsa_peer_workspaces);
        args.comm_state = comm_state;
        args.slot = slot;
        args.operation_args = operation_args;
        args.num_lsa_send_ctas = num_lsa_send_ctas;
        args.num_gin_send_ctas = num_gin_send_ctas;
        void* kernel_args[] = {&args};
        const auto kernel = comm_state.dev_comm.lsaSize == kNumRanks ? magic_all_to_all_kernel<kNumRanks, false, OperationArgs>
                                                                     : magic_all_to_all_kernel<kNumRanks, true, OperationArgs>;
        const size_t smem_bytes = num_gin_send_ctas != 0 ? kAllToAllWarps * slot.gin_warp_smem_bytes() : 0;
        CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
        const dim3 grid(num_lsa_send_ctas + num_gin_send_ctas + num_recv_ctas);
        CUDA_RUNTIME_CHECK(cudaLaunchCooperativeKernel(kernel, grid, dim3(kAllToAllThreads), kernel_args, smem_bytes, stream));
    }

    template <typename OperationArgs>
    void all_to_all(const MagicSlotLayout& slot,
                    unsigned* send_counts,
                    unsigned* recv_counts,
                    const OperationArgs& operation_args,
                    cudaStream_t stream) {
        switch (world_size) {
            case 2: return launch_all_to_all<2>(slot, send_counts, recv_counts, operation_args, stream);
            case 4: return launch_all_to_all<4>(slot, send_counts, recv_counts, operation_args, stream);
            case 8: return launch_all_to_all<8>(slot, send_counts, recv_counts, operation_args, stream);
            case 16: return launch_all_to_all<16>(slot, send_counts, recv_counts, operation_args, stream);
            case 32: return launch_all_to_all<32>(slot, send_counts, recv_counts, operation_args, stream);
            case 64: return launch_all_to_all<64>(slot, send_counts, recv_counts, operation_args, stream);
            default: EP_HOST_UNREACHABLE("world_size must be 2, 4, 8, 16, 32 or 64");
        }
    }
};

Runtime::Runtime(unsigned rank, unsigned world_size, int64_t nccl_comm_ptr, unsigned max_tokens_per_rank, size_t workspace_size)
    : context(std::make_unique<Context>()) {
    const size_t rank_slot_bytes = workspace_size / kNumStages / world_size;
    EP_HOST_ASSERT(rank_slot_bytes != 0);
    EP_HOST_ASSERT(rank_slot_bytes % kHeaderBytes == 0);
    auto& ctx = *context;
    const auto host_comm = reinterpret_cast<ncclComm_t>(nccl_comm_ptr);
    ctx.host_comm = host_comm;
    ctx.rank = rank;
    ctx.world_size = world_size;
    ctx.max_tokens_per_rank = max_tokens_per_rank;
    ctx.rank_slot_bytes = rank_slot_bytes;

    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginContextCount = 1;
    reqs.ginExclusiveContexts = true;
    reqs.ginSignalCount = world_size;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    NCCL_CHECK(ncclDevCommCreate(host_comm, &reqs, &ctx.comm_state.dev_comm));

    const size_t bytes = kNumStages * world_size * rank_slot_bytes;
    ctx.allocate_window(host_comm, ctx.comm_state.send_staging, bytes, NCCL_WIN_DEFAULT);
    ctx.allocate_window(host_comm, ctx.comm_state.recv_workspace, bytes, NCCL_WIN_DEFAULT);
    ctx.allocate_window(host_comm, ctx.comm_state.completion, kNumStages * world_size * sizeof(uint64_t), NCCL_WIN_STRICT_ORDERING);
    ctx.lsa_size = static_cast<unsigned>(ctx.comm_state.dev_comm.lsaSize);
    EP_HOST_ASSERT(world_size % ctx.lsa_size == 0 && ctx.comm_state.dev_comm.lsaRank == rank % ctx.lsa_size);
    EP_HOST_ASSERT(ctx.comm_state.dev_comm.ginContextCount == 1);
    const bool use_gin = ctx.lsa_size != world_size;
    ctx.num_lsa_send_ctas = get_env<int>("EP_MAGIC_NUM_LSA_SEND_CTAS", use_gin ? 8 : 16);
    ctx.num_gin_send_ctas = (world_size - ctx.lsa_size + kAllToAllWarps - 1) / kAllToAllWarps;
    ctx.num_recv_ctas = world_size;
    ctx.lsa_bulk_rows = get_env<int>("EP_MAGIC_LSA_BULK_ROWS", 2);
    EP_HOST_ASSERT(ctx.lsa_bulk_rows >= 1);
    ctx.gin_bulk_rows = get_env<int>("EP_MAGIC_GIN_BULK_ROWS", 2);
    EP_HOST_ASSERT(ctx.gin_bulk_rows >= 1 && ctx.gin_bulk_rows <= 32);
    // Dispatch and combine may use different dtypes (e.g. FP4 dispatch + INT8 combine),
    // so allow independent dispatch bulk sizes. For FP4 dispatch, 4/4 is recommended.
    ctx.dispatch_lsa_bulk_rows = get_env<int>("EP_MAGIC_DISPATCH_LSA_BULK_ROWS", ctx.lsa_bulk_rows);
    EP_HOST_ASSERT(ctx.dispatch_lsa_bulk_rows >= 1);
    ctx.dispatch_gin_bulk_rows = get_env<int>("EP_MAGIC_DISPATCH_GIN_BULK_ROWS", ctx.gin_bulk_rows);
    EP_HOST_ASSERT(ctx.dispatch_gin_bulk_rows >= 1 && ctx.dispatch_gin_bulk_rows <= 32);

    const unsigned lsa_first_rank = rank / ctx.lsa_size * ctx.lsa_size;
    for (unsigned peer = lsa_first_rank; peer < lsa_first_rank + ctx.lsa_size; ++peer) {
        NCCL_CHECK(ncclGetLsaDevicePointer(ctx.comm_state.recv_workspace.window, 0, peer % ctx.lsa_size, &ctx.lsa_peer_workspaces[peer]));
        void* completion_ptr = nullptr;
        NCCL_CHECK(ncclGetLsaDevicePointer(ctx.comm_state.completion.window, 0, peer % ctx.lsa_size, &completion_ptr));
        ctx.lsa_peer_completion_ptrs[peer] = static_cast<uint64_t*>(completion_ptr);
    }

    CUDA_RUNTIME_CHECK(cudaMalloc(&ctx.all_to_all_state, sizeof(MagicAllToAllState)));
    CUDA_RUNTIME_CHECK(cudaMalloc(&ctx.combine_staging, static_cast<size_t>(max_tokens_per_rank) * world_size * kMaxCombineRowBytes));
    CUDA_RUNTIME_CHECK(cudaMalloc(&ctx.send_counts, 3 * world_size * sizeof(unsigned)));
    ctx.recv_counts = ctx.send_counts + world_size;
    ctx.recv_peer_row_start = ctx.recv_counts + world_size;
    CUDA_RUNTIME_CHECK(cudaMalloc(&ctx.staging_row_indices, static_cast<size_t>(world_size) * max_tokens_per_rank * sizeof(unsigned)));
    CUDA_RUNTIME_CHECK(cudaMalloc(&ctx.token_row_counts, max_tokens_per_rank * sizeof(unsigned)));
    CUDA_RUNTIME_CHECK(cudaMemset(ctx.all_to_all_state, 0, sizeof(MagicAllToAllState)));
    CUDA_RUNTIME_CHECK(cudaMemset(ctx.send_counts, 0, world_size * sizeof(unsigned)));

    std::vector<uint64_t> initial_completions(kNumStages * world_size);
    for (unsigned stage = 0; stage < kNumStages; ++stage)
        for (unsigned peer = 0; peer < world_size; ++peer)
            initial_completions[stage * world_size + peer] = static_cast<uint64_t>(stage + 1) - kNumStages;
    CUDA_RUNTIME_CHECK(cudaMemcpy(
        ctx.comm_state.completion.base, initial_completions.data(), initial_completions.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    initialize_magic_workspace_kernel<<<dim3(64), dim3(kAllToAllThreads)>>>(ctx.comm_state.recv_workspace.base, bytes);
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    CUDA_RUNTIME_CHECK(cudaGetDevice(&ctx.device));
}

Runtime::~Runtime() = default;

void Runtime::destroy() {
    EP_HOST_ASSERT(context != nullptr);
    auto ctx = std::move(context);
    ctx->destroy();
}

void Runtime::dispatch(const DispatchArgs& args, cudaStream_t stream) {
    EP_HOST_ASSERT(context != nullptr);
    auto& ctx = *context;
    EP_HOST_ASSERT(!ctx.dispatch_pending);
    EP_HOST_ASSERT(args.token_bytes % sizeof(uint4) == 0 && args.scale_bytes % sizeof(uint4) == 0);
    EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(args.x) % sizeof(uint4) == 0);
    EP_HOST_ASSERT(args.scale_bytes == 0 || reinterpret_cast<uintptr_t>(args.x_scale) % sizeof(uint4) == 0);
    const MagicRowLayout row{args.token_bytes, args.scale_bytes, args.topk};
    const MagicSlotLayout slot = ctx.slot_layout(row.row_bytes(), true);

    const DispatchOperationArgs operation_args{static_cast<const uint8_t*>(args.x),
                                               static_cast<const uint8_t*>(args.x_scale),
                                               args.topk_ids,
                                               args.topk_weights,
                                               ctx.staging_row_indices,
                                               ctx.token_row_counts,
                                               args.num_tokens,
                                               ctx.max_tokens_per_rank,
                                               args.num_local_experts,
                                               static_cast<uint8_t*>(args.recv_x),
                                               static_cast<uint8_t*>(args.recv_x_scale),
                                               args.recv_topk_ids,
                                               args.recv_topk_weights,
                                               args.psum_recv_per_rank,
                                               ctx.recv_peer_row_start,
                                               row};
    ctx.all_to_all(slot, ctx.send_counts, ctx.recv_counts, operation_args, stream);

    ctx.dispatch_num_tokens = args.num_tokens;
    ctx.dispatch_recv_capacity = args.recv_capacity;
    ctx.dispatch_pending = true;
}

void Runtime::combine(const CombineArgs& args, cudaStream_t stream) {
    EP_HOST_ASSERT(context != nullptr);
    auto& ctx = *context;
    EP_HOST_ASSERT(ctx.dispatch_pending);
    EP_HOST_ASSERT(args.num_tokens == ctx.dispatch_num_tokens && args.recv_capacity == ctx.dispatch_recv_capacity);
    EP_HOST_ASSERT(args.row_bytes <= kMaxCombineRowBytes);
    EP_HOST_ASSERT(args.row_bytes % sizeof(uint4) == 0);
    EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(args.x) % sizeof(uint4) == 0);
    const unsigned quant_block_size = args.combine_num_sf_packs != 0 ? args.hidden_size / args.combine_num_sf_packs : 0;
    if (args.combine_num_sf_packs != 0) {
        EP_HOST_ASSERT(args.hidden_size % args.combine_num_sf_packs == 0);
        EP_HOST_ASSERT(quant_block_size % sizeof(uint4) == 0);
    }
    const MagicSlotLayout slot = ctx.slot_layout(args.row_bytes);

    const CombineOperationArgs operation_args{static_cast<const uint8_t*>(args.x),
                                              ctx.recv_peer_row_start,
                                              ctx.combine_staging,
                                              ctx.staging_row_indices,
                                              ctx.send_counts,
                                              args.row_bytes,
                                              ctx.max_tokens_per_rank};
    ctx.all_to_all(slot, ctx.recv_counts, nullptr, operation_args, stream);

    if (args.num_tokens) {
        const auto reduce_kernel = args.combine_num_sf_packs != 0 ? magic_combine_reduce_kernel<true> : magic_combine_reduce_kernel<false>;
        reduce_kernel<<<args.num_tokens, kReduceThreads, 0, stream>>>(ctx.world_size,
                                                                      args.row_bytes,
                                                                      args.hidden_size,
                                                                      quant_block_size,
                                                                      ctx.combine_staging,
                                                                      ctx.token_row_counts,
                                                                      static_cast<__nv_bfloat16*>(args.out));
        CUDA_RUNTIME_CHECK(cudaGetLastError());
    }
    ctx.dispatch_pending = false;
}

}

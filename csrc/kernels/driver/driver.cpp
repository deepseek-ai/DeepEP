#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <deep_ep/common/exception.cuh>

#include <deep_jit/backend/cuda/driver.hpp>
#include "../comm/api.hpp"
#include "api.hpp"

namespace deep_ep::driver {

void copy_engine_push(const CUstream& stream,
                      const std::vector<void*>& dsts,
                      const std::vector<void*>& srcs,
                      const std::vector<size_t>& sizes) {
    EP_HOST_ASSERT(dsts.size() == srcs.size() and srcs.size() == sizes.size());
    if (dsts.empty())
        return;

    cudaStreamCaptureStatus status;
    CUDA_RUNTIME_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status != cudaStreamCaptureStatusNone) {
        for (size_t i = 0; i < dsts.size(); ++ i)
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDeviceToDevice, stream));
        return;
    }

    cudaMemcpyAttributes attrs = {};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t attr_idx = 0;
    CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(
        dsts.data(), srcs.data(), sizes.data(), dsts.size(),
        &attrs, &attr_idx, 1, stream));
}

void copy_engine_wait(const CUstream& stream, const std::vector<uint64_t*>& signals, const uint64_t& target) {
    static constexpr size_t kNumMaxBatchMemOps = 255;
    std::vector<CUstreamBatchMemOpParams> ops(signals.size());
    for (size_t i = 0; i < signals.size(); ++ i) {
        ops[i] = {};
        ops[i].operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
        ops[i].waitValue.address = reinterpret_cast<CUdeviceptr>(signals[i]);
        ops[i].waitValue.value64 = target;
        ops[i].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
    }
    for (size_t offset = 0; offset < ops.size(); offset += kNumMaxBatchMemOps) {
        const auto num_ops = static_cast<unsigned int>(std::min(kNumMaxBatchMemOps, ops.size() - offset));
        CUDA_DRIVER_CHECK(::deep_jit::cuda::driver::lazy_cuStreamBatchMemOp(stream, num_ops, ops.data() + offset, 0));
    }
}

void copy_engine_push_synced(const comm::Context& context,
                             layout::CopyEngineSignals& signals,
                             const std::vector<void*>& dsts,
                             const std::vector<void*>& srcs,
                             const std::vector<size_t>& sizes,
                             const CUstream& stream) {
    EP_HOST_ASSERT(context.num_nvl_ranks <= kNumMaxNVLRanks and
                   "Number of NVLink ranks exceeds the maximum workspace size");

    std::vector<void*> prologue_write_signals, epilogue_write_signals;
    std::vector<uint64_t*> prologue_wait_signals, epilogue_wait_signals;
    prologue_write_signals.reserve(context.num_nvl_ranks - 1);
    epilogue_write_signals.reserve(context.num_nvl_ranks - 1);
    prologue_wait_signals.reserve(context.num_nvl_ranks - 1);
    epilogue_wait_signals.reserve(context.num_nvl_ranks - 1);
    for (int rank_offset = 1; rank_offset < context.num_nvl_ranks; ++ rank_offset) {
        const auto peer_rank_idx = (context.nvl_rank_idx + rank_offset) % context.num_nvl_ranks;
        auto* peer_signals = static_cast<layout::CopyEngineSignals*>(context.get_sym_ptr(&signals, peer_rank_idx));
        prologue_write_signals.push_back(peer_signals->prologue + context.nvl_rank_idx);
        epilogue_write_signals.push_back(peer_signals->epilogue + context.nvl_rank_idx);
        prologue_wait_signals.push_back(signals.prologue + peer_rank_idx);
        epilogue_wait_signals.push_back(signals.epilogue + peer_rank_idx);
    }

    const auto sync = [&](uint64_t* local_signals,
                          const std::vector<void*>& write_signals,
                          const std::vector<uint64_t*>& wait_signals) {
        if (write_signals.empty())
            return;
        // Push arrival signals via CE instead of memops to avoid remote peer compatibility issues
        std::vector<void*> signal_srcs(write_signals.size(), &signals.arrived);
        std::vector<size_t> signal_sizes(write_signals.size(), sizeof(uint64_t));
        copy_engine_push(stream, write_signals, signal_srcs, signal_sizes);
        copy_engine_wait(stream, wait_signals, layout::CopyEngineSignals::kArrivedSignal);
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(local_signals, 0, context.num_nvl_ranks * sizeof(uint64_t), stream));
    };

    sync(signals.prologue, prologue_write_signals, prologue_wait_signals);
    copy_engine_push(stream, dsts, srcs, sizes);
    sync(signals.epilogue, epilogue_write_signals, epilogue_wait_signals);
}

}  // namespace deep_ep::driver

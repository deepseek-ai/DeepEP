#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <deep_ep/common/exception.cuh>

#include "../../utils/lazy_driver.hpp"
#include "api.hpp"

namespace deep_ep::driver {

uint64_t increase_seq_idx(const CUstream& stream, uint64_t& seq_idx) {
    cudaStreamCaptureStatus status;
    CUDA_RUNTIME_CHECK(cudaStreamIsCapturing(stream, &status));
    EP_HOST_ASSERT(status == cudaStreamCaptureStatusNone and "Sequence indices do not support CUDA graph capture");
    return ++ seq_idx;
}

static CUstreamBatchMemOpParams create_mem_op(
    uint64_t* signal,
    const uint64_t& value,
    const CUstreamBatchMemOpType& type,
    const CUstreamWaitValue_flags& wait_flag = CU_STREAM_WAIT_VALUE_EQ) {
    CUstreamBatchMemOpParams params = {};
    if (type == CU_STREAM_MEM_OP_WRITE_VALUE_64) {
        params.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
        params.writeValue.address = reinterpret_cast<CUdeviceptr>(signal);
        params.writeValue.value64 = value;
        params.writeValue.flags = 0;
    } else {
        params.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
        params.waitValue.address = reinterpret_cast<CUdeviceptr>(signal);
        params.waitValue.value64 = value;
        params.waitValue.flags = wait_flag;
    }
    return params;
}

static void launch_batched_mem_ops(CUstream stream, std::vector<CUstreamBatchMemOpParams>& ops) {
    static constexpr int kNumMaxBatchMemOps = 255;
    for (std::size_t offset = 0; offset < ops.size(); offset += kNumMaxBatchMemOps) {
        const auto num_ops = static_cast<unsigned int>(
            std::min<std::size_t>(kNumMaxBatchMemOps, ops.size() - offset));
        CUDA_DRIVER_CHECK(lazy_cuStreamBatchMemOp(stream, num_ops, ops.data() + offset, 0));
    }
}

void copy_engine_push(const CUstream& stream,
                      const std::vector<void*>& dsts,
                      const std::vector<void*>& srcs,
                      const std::vector<size_t>& sizes) {
    EP_HOST_ASSERT(dsts.size() == srcs.size() and srcs.size() == sizes.size());
    if (dsts.empty())
        return;

    cudaMemcpyAttributes attrs = {};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.flags = cudaMemcpyFlagPreferOverlapWithCompute;
    size_t attr_idx = 0;
    CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(
        dsts.data(), srcs.data(), sizes.data(), dsts.size(),
        &attrs, &attr_idx, 1, stream));
}

void copy_engine_sync(const CUstream& stream, const uint64_t& seq_idx,
                      const std::vector<uint64_t*>& write_signals,
                      const std::vector<uint64_t*>& wait_signals) {
    // Write and wait signals
    std::vector<CUstreamBatchMemOpParams> ops(write_signals.size() + wait_signals.size());
    for (int i = 0; i < write_signals.size(); ++ i) {
        ops[i] = create_mem_op(write_signals[i], seq_idx, CU_STREAM_MEM_OP_WRITE_VALUE_64);
    }
    for (int i = 0; i < wait_signals.size(); ++ i) {
        ops[write_signals.size() + i] = create_mem_op(
            wait_signals[i], seq_idx, CU_STREAM_MEM_OP_WAIT_VALUE_64, CU_STREAM_WAIT_VALUE_GEQ);
    }
    launch_batched_mem_ops(stream, ops);
}

void copy_engine_wait(const CUstream& stream, const std::vector<uint64_t*>& signals, const uint64_t& target) {
    std::vector<CUstreamBatchMemOpParams> ops(signals.size());
    for (int i = 0; i < signals.size(); ++ i)
        ops[i] = create_mem_op(signals[i], target, CU_STREAM_MEM_OP_WAIT_VALUE_64, CU_STREAM_WAIT_VALUE_GEQ);
    launch_batched_mem_ops(stream, ops);
}

}  // namespace deep_ep::driver

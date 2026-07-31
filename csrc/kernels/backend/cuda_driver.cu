#include <cuda.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <deep_ep/common/exception.cuh>

#include "api.cuh"
#include "../../utils/lazy_driver.hpp"

namespace deep_ep::cuda_driver {

static CUstreamBatchMemOpParams create_mem_op(
    void *ptr, const int& value,
    const CUstreamBatchMemOpType& type,
    const CUstreamWaitValue_flags& wait_flag = CU_STREAM_WAIT_VALUE_EQ) {
    CUstreamBatchMemOpParams params;
    if (type == CU_STREAM_MEM_OP_WRITE_VALUE_32) {
        params.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
        params.writeValue.address = reinterpret_cast<CUdeviceptr>(ptr);
        params.writeValue.value = value;
        params.writeValue.flags = 0;
    } else {
        params.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
        params.waitValue.address = reinterpret_cast<CUdeviceptr>(ptr);
        params.waitValue.value = value;
        params.waitValue.flags = wait_flag;
    }
    return params;
}

void batched_write(CUstream stream, const std::vector<void*>& ptrs, const int& value) {
    std::vector<CUstreamBatchMemOpParams> ops(ptrs.size());
    for (int i = 0; i < ptrs.size(); ++ i)
        ops[i] = create_mem_op(ptrs[i], value, CU_STREAM_MEM_OP_WRITE_VALUE_32);
    CUDA_DRIVER_CHECK(lazy_cuStreamBatchMemOp_v2(stream, ops.size(), ops.data(), 0));
}

void batched_wait(CUstream stream, const std::vector<void*>& ptrs, const int& value) {
    std::vector<CUstreamBatchMemOpParams> ops(ptrs.size());
    for (int i = 0; i < ptrs.size(); ++ i)
        ops[i] = create_mem_op(ptrs[i], value, CU_STREAM_MEM_OP_WAIT_VALUE_32, CU_STREAM_WAIT_VALUE_GEQ);
    CUDA_DRIVER_CHECK(lazy_cuStreamBatchMemOp_v2(stream, ops.size(), ops.data(), 0));
}

void batched_write_and_wait(CUstream stream, const std::vector<void*>& write_ptrs, const std::vector<void*>& wait_ptrs, const int& value) {
    std::vector<CUstreamBatchMemOpParams> ops(write_ptrs.size() + wait_ptrs.size());
    for (int i = 0; i < write_ptrs.size(); ++ i)
       ops[i] = create_mem_op(write_ptrs[i], value, CU_STREAM_MEM_OP_WRITE_VALUE_32);
    for (int i = 0; i < wait_ptrs.size(); ++ i)
       ops[write_ptrs.size() + i] = create_mem_op(wait_ptrs[i], value, CU_STREAM_MEM_OP_WAIT_VALUE_32, CU_STREAM_WAIT_VALUE_GEQ);
    CUDA_DRIVER_CHECK(lazy_cuStreamBatchMemOp_v2(stream, ops.size(), ops.data(), 0));
}

}  // namespace deep_ep::cuda_driver

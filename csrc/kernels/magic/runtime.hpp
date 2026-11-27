#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace deep_ep::magic {

struct DispatchArgs {
    const void* x;
    const void* x_scale;
    const int32_t* topk_ids;
    const float* topk_weights;
    void* recv_x;
    void* recv_x_scale;
    int32_t* recv_topk_ids;
    float* recv_topk_weights;
    int32_t* psum_recv_per_rank;
    unsigned num_tokens;
    unsigned recv_capacity;
    unsigned topk;
    unsigned token_bytes;
    unsigned scale_bytes;
    unsigned num_local_experts;
};

struct CombineArgs {
    const void* x;
    void* out;
    unsigned num_tokens;
    unsigned recv_capacity;
    unsigned hidden_size;
    unsigned row_bytes;
    unsigned combine_num_sf_packs;
};

class Runtime {
    struct Context;
    std::unique_ptr<Context> context;

public:
    Runtime(unsigned rank, unsigned world_size, int64_t nccl_comm_ptr, unsigned max_tokens_per_rank, size_t workspace_size);
    ~Runtime();

    void destroy();
    void dispatch(const DispatchArgs& args, cudaStream_t stream);
    void combine(const CombineArgs& args, cudaStream_t stream);
};

}

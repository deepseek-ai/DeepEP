"""TileLang reference for the BF16 RDMA reduce-scatter send conversion."""

from pathlib import Path

import tilelang
from tilelang import language as T
import torch


# Use the same non-saturating instruction and operand order as ptx.cuh so
# infinities and values that round to infinity also match the RDMA kernel.
_CONVERSION_SOURCE = r'''
__device__ __forceinline__ unsigned int deep_ep_cvt_rs_bf16x2(
    float x, float y, unsigned int rbits) {
    unsigned int result;
    asm("cvt.rs.bf16x2.f32 %0, %2, %1, %3;"
        : "=r"(result) : "f"(x), "f"(y), "r"(rbits));
    return result;
}
'''


@tilelang.jit(target='cuda')
def get_stochastic_round_bf16_kernel(num_ranks: int, elems_per_rank: int):
    num_threads = 128
    quads_per_rank = elems_per_rank // 4

    @T.prim_func
    def stochastic_round_bf16_kernel(
        x: T.Tensor[(num_ranks * elems_per_rank,), T.float32],
        out: T.Tensor[(num_ranks * elems_per_rank,), T.float32],
        self_rank: T.int32,
    ):
        with T.Kernel(T.ceildiv(quads_per_rank, num_threads), num_ranks,
                      threads=num_threads) as (block, dst_rank):
            T.import_source(_CONVERSION_SOURCE)
            quad = block * num_threads + T.get_thread_binding()
            values = T.alloc_local((4,), T.float32)
            packed = T.alloc_local((2,), T.uint32)
            h = T.alloc_var(T.uint32)
            h1 = T.alloc_var(T.uint32)
            h2 = T.alloc_var(T.uint32)

            if quad < quads_per_rank:
                elem = dst_rank * elems_per_rank + quad * 4
                for k in T.vectorized(4):
                    values[k] = x[elem + k]
                if dst_rank == self_rank:
                    for k in T.vectorized(4):
                        out[elem + k] = values[k]
                else:
                    # Index from the beginning of this bucket, not this rank's slice.
                    h = (T.reinterpret(values[0], T.uint32) * T.uint32(0x5671D42B)
                         + T.reinterpret(values[1], T.uint32) * T.uint32(0x9995E499)
                         + T.reinterpret(values[2], T.uint32) * T.uint32(0xACE1B8A5)
                         + T.reinterpret(values[3], T.uint32) * T.uint32(0xE153538D)
                         + T.cast(dst_rank, T.uint32) * T.uint32(quads_per_rank)
                         + T.cast(quad, T.uint32))
                    h = h ^ (h >> 23)
                    h = h * T.uint32(0x7FEB352D)
                    h = h ^ (h >> 16)
                    h1 = h * T.uint32(0x846CA68B)
                    h1 = h1 ^ (h1 >> 11)
                    h2 = h * T.uint32(0xD35A2D97)
                    h2 = h2 ^ (h2 >> 11)

                    packed[0] = T.call_pure_extern(
                        T.uint32, 'deep_ep_cvt_rs_bf16x2', values[0], values[1], h1)
                    packed[1] = T.call_pure_extern(
                        T.uint32, 'deep_ep_cvt_rs_bf16x2', values[2], values[3], h2)
                    for pair in T.unroll(2):
                        for k in T.vectorized(2):
                            bf16 = T.reinterpret(T.cast(packed[pair] >> (k * 16), T.uint16), T.bfloat16)
                            out[elem + pair * 2 + k] = T.cast(bf16, T.float32)

    return stochastic_round_bf16_kernel


def stochastic_round_bf16(x: torch.Tensor, self_rank: int, num_ranks: int, *,
                          dump_source: str | Path | None = None) -> torch.Tensor:
    """Return FP32 values with remote destination slices rounded through BF16.

    ``x`` is one contiguous, 16-byte-aligned 1D FP32 CUDA bucket containing
    ``num_ranks`` equal destination slices. Each slice must contain a multiple
    of four elements. ``self_rank`` is the caller's rank within this group.
    Its slice is copied bit for bit, and ``x`` is left unchanged.

    Rounding matches ``rdma_reduce_scatter_bf16``: each float4's bit patterns
    and bucket-relative index determine two distinct rounding words.
    Requires SM100a or SM103a.
    ``dump_source`` optionally saves the generated CUDA code.
    """
    if x.dtype != torch.float32 or not x.is_cuda:
        raise ValueError('x must be a FP32 CUDA tensor')
    if x.ndim != 1 or not x.is_contiguous():
        raise ValueError('x must be a contiguous 1D tensor')
    if x.data_ptr() % 16:
        raise ValueError('x must be 16-byte aligned for float4 loads')
    if x.numel() % (num_ranks * 4):
        raise ValueError('x.numel() must be divisible by num_ranks * 4')
    elems_per_rank = x.numel() // num_ranks
    if not 0 <= self_rank < num_ranks:
        raise ValueError('self_rank must index a destination slice of x')
    out = torch.empty_like(x)
    if elems_per_rank == 0:
        return out
    with torch.cuda.device(x.device):
        kernel = get_stochastic_round_bf16_kernel(num_ranks, elems_per_rank)
        if dump_source is not None:
            path = Path(dump_source)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(kernel.get_kernel_source())
        kernel(x, out, self_rank)
    return out

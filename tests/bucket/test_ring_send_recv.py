import os

# Must pop before PyTorch imports
os.environ.pop('NCCL_MIN_NCHANNELS', None)
os.environ.pop('NCCL_MAX_NCHANNELS', None)
os.environ.pop('NCCL_NVLS_ENABLE', None)

import argparse
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.math import align
from deep_ep.utils.testing import bench, bench_kineto, parse_num_bytes

all_gather_single = getattr(dist, 'all_gather_single', dist.all_gather_into_tensor)


def nccl_ring_send_recv(src: torch.Tensor, dst: torch.Tensor, shift: int, cyclic: bool, group: dist.ProcessGroup) -> None:
    rank, size = group.rank(), group.size()
    dst_rank = (rank + shift) % size if cyclic else rank + shift
    src_rank = (rank - shift) % size if cyclic else rank - shift
    ops = []
    if 0 <= dst_rank < size:
        ops.append(dist.P2POp(dist.isend, src, dist.get_global_rank(group, dst_rank), group))
    if 0 <= src_rank < size:
        ops.append(dist.P2POp(dist.irecv, dst, dist.get_global_rank(group, src_rank), group))
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_rdma_ranks, num_nvl_ranks = deep_ep.comm.get_physical_domain_size(group)
    torch.manual_seed(rank_idx)

    # Test sizes including unaligned byte counts (ptr-aligned, size-unaligned)
    test_shapes_and_dtypes = [
        ((13,), torch.uint8),
        ((100,), torch.uint8),
        ((1, 7, 17), torch.bfloat16),
        ((1, 64, 128), torch.bfloat16),
        ((4097,), torch.float32),
        ((1, 256, 512), torch.bfloat16),
        ((1 << 21,), torch.float32),  # 8 MiB (multi-chunk on hybrid)
    ]

    # Allocate registered buffer for direct BucketBuffer tests and session tests
    allocator = deep_ep.BufferAllocator()
    reg_srcs = [allocator.allocate(shape, dtype) for shape, dtype in test_shapes_and_dtypes]
    reg_dsts = [allocator.allocate(shape, dtype) for shape, dtype in test_shapes_and_dtypes]
    perf_src = allocator.allocate((args.num_bytes // 2,), torch.bfloat16)
    perf_dst = allocator.allocate((args.num_bytes // 2,), torch.bfloat16)
    buffer = deep_ep.BucketBuffer(group, allocator, explicitly_destroy=True)

    dist_print(f'Config:\n'
               f' > Ranks: {num_rdma_ranks} x {num_nvl_ranks}\n'
               f' > #Bytes: {args.num_bytes}\n',
               once_in_node=True)

    configs = [
        (1, False),
        (-1, False),
        (1, True),
        (-1, True),
    ]
    if num_ranks > 2:
        configs.extend([(2, False), (-2, True)])

    # 1. Direct BucketBuffer.ring_send_recv (single and batched, aligned and unaligned sizes)
    for shift, cyclic in configs:
        has_src = cyclic or (0 <= rank_idx - shift < num_ranks)
        for i, (src, dst) in enumerate(zip(reg_srcs, reg_dsts, strict=True)):
            if src.dtype == torch.uint8:
                src.random_(0, 255)
                dst.fill_(123)
            else:
                src.normal_().add_(rank_idx)
                dst.fill_(-7.0)
            sentinel = dst.clone()

            ref = torch.zeros_like(dst)
            nccl_ring_send_recv(src, ref, shift, cyclic, group)

            out = buffer.ring_send_recv(src, dsts=dst, shift=shift, cyclic=cyclic, group=group).wait()
            assert out.data_ptr() == dst.data_ptr()
            if has_src:
                assert torch.equal(out, ref), f'Mismatch at shape={src.shape}, dtype={src.dtype}, shift={shift}, cyclic={cyclic}'
            else:
                assert torch.equal(out, sentinel), f'Boundary dst modified at shift={shift}, cyclic={cyclic}'

        # Batched list test
        batch_srcs = reg_srcs[:4]
        batch_dsts = reg_dsts[:4]
        refs, sentinels = [], []
        for src, dst in zip(batch_srcs, batch_dsts, strict=True):
            if src.dtype == torch.uint8:
                src.random_(0, 255)
                dst.fill_(77)
            else:
                src.normal_().add_(rank_idx)
                dst.fill_(-3.0)
            sentinels.append(dst.clone())
            ref = torch.zeros_like(dst)
            nccl_ring_send_recv(src, ref, shift, cyclic, group)
            refs.append(ref)

        outs = buffer.ring_send_recv(batch_srcs, dsts=batch_dsts, shift=shift, cyclic=cyclic, group=group).wait()
        for out, dst, ref, sentinel in zip(outs, batch_dsts, refs, sentinels, strict=True):
            assert out.data_ptr() == dst.data_ptr()
            if has_src:
                assert torch.equal(out, ref)
            else:
                assert torch.equal(out, sentinel)

    # 2. BucketSession tests (unregistered, non-contiguous slices, optional dsts, mixed with all_gather)
    session_bytes = align(64 << 20, deep_ep.get_num_allocation_alignment())
    session_buffer = deep_ep.BucketBuffer(group, session_bytes, explicitly_destroy=True)

    for shift, cyclic in [(1, False), (-1, False), (1, True)]:
        has_src = cyclic or (0 <= rank_idx - shift < num_ranks)
        with session_buffer.session():
            # Unregistered non-contiguous slice with unaligned byte count
            full_k = torch.randn(2, 128, 64, dtype=torch.bfloat16, device='cuda').add_(rank_idx)
            full_v = torch.randn(2, 128, 64, dtype=torch.bfloat16, device='cuda').add_(rank_idx * 2)
            k_slice = full_k[:, -17:]
            v_slice = full_v[:, -17:]

            k_ref = torch.zeros_like(k_slice.contiguous())
            v_ref = torch.zeros_like(v_slice.contiguous())
            nccl_ring_send_recv(k_slice.contiguous(), k_ref, shift, cyclic, group)
            nccl_ring_send_recv(v_slice.contiguous(), v_ref, shift, cyclic, group)

            # Without dsts (returns registered views)
            k_out, v_out = session_buffer.ring_send_recv(
                [k_slice, v_slice], shift=shift, cyclic=cyclic, group=group).wait()
            if has_src:
                assert torch.equal(k_out, k_ref)
                assert torch.equal(v_out, v_ref)

            # With external unregistered dsts
            ext_dst = torch.full_like(k_ref, -1.0)
            ext_sentinel = ext_dst.clone()
            out = session_buffer.ring_send_recv(
                k_slice, dsts=ext_dst, shift=shift, cyclic=cyclic, group=group).wait()
            assert out.data_ptr() == ext_dst.data_ptr()
            if has_src:
                assert torch.equal(out, k_ref)
            else:
                assert torch.equal(out, ext_sentinel)

            # Follow up with all_gather in the same session to verify num_allocated_bytes stayed in sync across ranks
            ag_src = torch.randn(1024, dtype=torch.float32, device='cuda').add_(rank_idx)
            ag_ref = torch.empty(num_ranks * 1024, dtype=torch.float32, device='cuda')
            all_gather_single(ag_ref, ag_src, group=group)
            ag_out = session_buffer.all_gather(ag_src, group=group).wait()
            assert torch.equal(ag_out, ag_ref)

    # 2b. BucketSession with omitted default kwargs (`shift=1`, `cyclic=False`)
    with session_buffer.session():
        def_src = torch.randn(64, 128, dtype=torch.bfloat16, device='cuda').add_(rank_idx)
        def_ref = torch.zeros_like(def_src)
        nccl_ring_send_recv(def_src, def_ref, 1, False, group)
        def_out = session_buffer.ring_send_recv(def_src).wait()
        if rank_idx > 0:
            assert torch.equal(def_out, def_ref)

        cyc_ref = torch.zeros_like(def_src)
        nccl_ring_send_recv(def_src, cyc_ref, 1, True, group)
        cyc_out = session_buffer.ring_send_recv(def_src, cyclic=True).wait()
        assert torch.equal(cyc_out, cyc_ref)

    # 3. CUDA Graph capture & replay (both direct BucketBuffer and BucketSession, open-chain & cyclic)
    capture_stream = torch.cuda.Stream()
    for shift, cyclic in [(1, False), (-1, False), (1, True)]:
        has_src = cyclic or (0 <= rank_idx - shift < num_ranks)
        # 3a. Direct BucketBuffer.ring_send_recv in CUDA Graph (multi-chunk 8 MiB + unaligned 100B)
        graph_srcs = [reg_srcs[-1], reg_srcs[1]]
        graph_dsts = [reg_dsts[-1], reg_dsts[1]]
        graph = torch.cuda.CUDAGraph()
        # Warmup before capture
        with torch.cuda.stream(capture_stream):
            buffer.ring_send_recv(graph_srcs, dsts=graph_dsts, shift=shift, cyclic=cyclic, group=group).wait()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture_stream):
            buffer.ring_send_recv(graph_srcs, dsts=graph_dsts, shift=shift, cyclic=cyclic, group=group).wait()
        for step in range(4):
            graph_srcs[0].normal_().add_(rank_idx + step)
            graph_srcs[1].random_(0, 255)
            graph_dsts[0].fill_(-5.0)
            graph_dsts[1].fill_(55)
            sentinel0, sentinel1 = graph_dsts[0].clone(), graph_dsts[1].clone()
            ref0 = torch.zeros_like(graph_dsts[0])
            ref1 = torch.zeros_like(graph_dsts[1])
            nccl_ring_send_recv(graph_srcs[0], ref0, shift, cyclic, group)
            nccl_ring_send_recv(graph_srcs[1], ref1, shift, cyclic, group)
            graph.replay()
            torch.cuda.synchronize()
            if has_src:
                assert torch.equal(graph_dsts[0], ref0), f'Graph mismatch (8M) at shift={shift}, cyclic={cyclic}, step={step}'
                assert torch.equal(graph_dsts[1], ref1), f'Graph mismatch (100B) at shift={shift}, cyclic={cyclic}, step={step}'
            else:
                assert torch.equal(graph_dsts[0], sentinel0)
                assert torch.equal(graph_dsts[1], sentinel1)

        # 3b. BucketSession ring_send_recv + all_gather in CUDA Graph (unregistered non-contiguous slice)
        ext_src_full = torch.randn(2, 64, 128, dtype=torch.bfloat16, device='cuda').add_(rank_idx)
        ext_src = ext_src_full[:, ::2, :]
        ext_dst = torch.zeros_like(ext_src.contiguous())
        ag_src = torch.randn(1024, dtype=torch.float32, device='cuda').add_(rank_idx)
        ag_dst = torch.zeros(num_ranks * 1024, dtype=torch.float32, device='cuda')
        session_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            with session_buffer.session():
                session_buffer.ring_send_recv(ext_src, dsts=ext_dst, shift=shift, cyclic=cyclic, group=group).wait()
                session_buffer.all_gather(ag_src, dsts=ag_dst, group=group).wait()
        torch.cuda.synchronize()
        with torch.cuda.graph(session_graph, stream=capture_stream):
            with session_buffer.session():
                session_buffer.ring_send_recv(ext_src, dsts=ext_dst, shift=shift, cyclic=cyclic, group=group).wait()
                session_buffer.all_gather(ag_src, dsts=ag_dst, group=group).wait()
        for step in range(4):
            ext_src_full.normal_().add_(rank_idx + step)
            ext_dst.fill_(-9.0)
            ext_sentinel = ext_dst.clone()
            ag_src.normal_().add_(rank_idx + step)
            ag_dst.zero_()
            ring_ref = torch.zeros_like(ext_dst)
            nccl_ring_send_recv(ext_src.contiguous(), ring_ref, shift, cyclic, group)
            ag_ref = torch.empty_like(ag_dst)
            all_gather_single(ag_ref, ag_src, group=group)
            session_graph.replay()
            torch.cuda.synchronize()
            if has_src:
                assert torch.equal(ext_dst, ring_ref), f'Session graph ring mismatch at shift={shift}, cyclic={cyclic}, step={step}'
            else:
                assert torch.equal(ext_dst, ext_sentinel)
            assert torch.equal(ag_dst, ag_ref), f'Session graph all_gather mismatch at step={step}'

    dist_print('Correctness check passed.', once_in_node=True)

    # 4. Performance benchmark (open-chain shift=1)
    perf_src.normal_().add_(rank_idx)
    nccl_dst = torch.empty_like(perf_dst)
    nccl_ring_send_recv(perf_src, nccl_dst, 1, False, group)
    nccl_t = bench_kineto(
        lambda: nccl_ring_send_recv(perf_src, nccl_dst, 1, False, group),
        kernel_names='ncclDevKernel_SendRecv', barrier_comm_profiling=True,
        barrier=buffer.barrier)

    t, _, _ = bench(lambda: buffer.ring_send_recv(perf_src, dsts=perf_dst, shift=1, cyclic=False, group=group).wait())
    dist_print('Performance:', once_in_node=True)
    dist_print(
        f' > Perf (shift=1, cyclic=False): '
        f'{t * 1e6:7.1f} us | '
        f'{perf_src.nbytes / t / 1e9:5.1f} GB/s | '
        f'{nccl_t / t:.2f}x NCCL',
        once_in_node=True)

    session_buffer.destroy()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test BucketBuffer ring-send-recv against NCCL')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-bytes', type=parse_num_bytes, default=16 << 20,
                        help='Tensor size per rank, with an optional binary suffix: 16M (default), 64M, 512K, or plain bytes')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

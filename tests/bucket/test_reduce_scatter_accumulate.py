import os
import sys

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
from deep_ep.utils.testing import bench_kineto, parse_num_bytes

reduce_scatter_single = getattr(dist, 'reduce_scatter_single', dist.reduce_scatter_tensor)


def import_baseline():
    stochastic_round_bf16 = None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        stochastic_round_bf16 = tilelang_ops.stochastic_round_bf16
    except Exception as ex:
        dist_print(f'Failed to load bf16 stochastic rounding helper: {ex}, test does not guarantee bitwise alignment', once_in_node=True)
        dist_print(once_in_node=True)
    return stochastic_round_bf16


def reduce_scatter_reference(source, rank_idx, group, num_rdma_ranks, num_nvl_ranks,
                             use_bf16, stochastic_round_bf16):
    reference_input = source
    if use_bf16 and stochastic_round_bf16 is not None:
        reference_input = stochastic_round_bf16(source, rank_idx, group.size())

    reference = torch.zeros(source.numel() // group.size(), dtype=torch.float32, device=source.device)
    if num_nvl_ranks > 1:
        reduce_scatter_single(reference, reference_input, group=group)
    else:
        reference_input_all_to_all = torch.empty_like(reference_input)
        dist.all_to_all_single(reference_input_all_to_all, reference_input, group=group)
        # Match rdma_reduce_scatter: local contribution followed by peers in reverse cyclic order.
        for i in range(num_rdma_ranks):
            peer = (rank_idx - i) % num_rdma_ranks
            reference.add_(reference_input_all_to_all[peer * reference.numel(): (peer + 1) * reference.numel()])
    return reference


def assert_accumulator_close(output, reference, num_nvl_ranks, use_bf16, stochastic_round_bf16):
    if num_nvl_ranks > 1:
        torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)
    elif not use_bf16 or stochastic_round_bf16 is not None:
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    else:
        torch.testing.assert_close(output, reference, rtol=1e-1, atol=1e-1)


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_rdma_ranks, num_nvl_ranks = deep_ep.comm.get_physical_domain_size(group)
    assert num_rdma_ranks * num_nvl_ranks == num_ranks

    # Allocate registered sources and one registered accumulator per source.
    allocator = deep_ep.BufferAllocator()
    src_0 = allocator.allocate((num_ranks * 4096,), torch.float32)
    src_1 = allocator.allocate((num_ranks * (3 * 1024 + 8),), torch.float32)
    accumulator_0 = allocator.allocate((4096,), torch.float32)
    accumulator_1 = allocator.allocate((3 * 1024 + 8,), torch.float32)
    perf_src = allocator.allocate((align(args.num_bytes // 4, num_ranks * 8),), torch.float32)
    buffer = deep_ep.BucketBuffer(group, allocator, explicitly_destroy=True)
    num_sms = buffer.get_theoretical_num_sms('reduce_scatter', group) if args.num_sms is None else args.num_sms
    srcs = [src_0, src_1]

    # Print configs
    dist_print(f'Config:\n'
               f' > Ranks: {num_rdma_ranks} x {num_nvl_ranks}\n'
               f' > #SM: {num_sms}\n'
               f' > Communication precision: {args.comm_precision}\n',
               once_in_node=True)

    use_bf16 = num_rdma_ranks > 1 and num_nvl_ranks == 1 and args.comm_precision == 'bf16'
    stochastic_round_bf16 = import_baseline() if use_bf16 else None

    # Preserve deterministic source contents for every accumulation round.
    source_templates = []
    for bucket_idx, src in enumerate(srcs):
        torch.manual_seed(rank_idx * 17 + bucket_idx)
        src.normal_()
        source_templates.append(src.clone())

    def run_case(case_idx: int, label: str, targets, launch_sources, use_session: bool):
        # These independent tensors are copies of the target state before any accumulation.
        reference_accumulators = []
        for bucket_idx, target in enumerate(targets):
            torch.manual_seed(rank_idx * 1009 + case_idx * 97 + bucket_idx)
            target.normal_()
            reference_accumulators.append(target.clone())

        for round_idx in range(3):
            for launch_source, source_template in zip(launch_sources, source_templates, strict=True):
                launch_source.copy_(source_template)

            # The RDMA kernel may reuse the local source shard as partial-sum storage.
            sources_before = [source.clone() for source in launch_sources]
            if use_session:
                with buffer.session():
                    outputs = buffer.reduce_scatter(
                        launch_sources, dsts=targets, num_sms=num_sms, group=group,
                        scale=args.scale, comm_precision=args.comm_precision).wait()
            else:
                outputs = buffer.reduce_scatter(
                    launch_sources, dsts=targets, num_sms=num_sms, group=group,
                    scale=args.scale, comm_precision=args.comm_precision).wait()

            for source_before, target, output, reference_accumulator in zip(
                    sources_before, targets, outputs, reference_accumulators, strict=True):
                reduction = reduce_scatter_reference(
                    source_before, rank_idx, group, num_rdma_ranks, num_nvl_ranks,
                    use_bf16, stochastic_round_bf16)

                # Match the kernel's separate FP32 multiply and add operations.
                reduction.mul_(args.scale)
                reference_accumulator.add_(reduction)

                assert output.data_ptr() == target.data_ptr(), \
                    f'{label}, round {round_idx}: output was not the supplied accumulator'
                assert output.shape == target.shape, \
                    f'{label}, round {round_idx}: output shape changed'
                assert_accumulator_close(
                    output, reference_accumulator, num_nvl_ranks, use_bf16, stochastic_round_bf16)

        dist_print(f'Accumulation correctness passed: {label}', once_in_node=True)

    # A registered target is legal as long as it does not overlap the source bucket.
    run_case(0, 'registered accumulator', [accumulator_0, accumulator_1], srcs, False)

    # External targets exercise the direct native accumulation path.
    external_accumulators = [
        torch.empty(src.numel() // num_ranks, dtype=torch.float32, device='cuda')
        for src in srcs
    ]
    run_case(1, 'external accumulator', external_accumulators, srcs, False)

    # The session path stages unregistered sources into temporary bucket storage.
    session_accumulators = [torch.empty_like(accumulator) for accumulator in external_accumulators]
    external_sources = [torch.empty_like(src) for src in srcs]
    run_case(2, 'session with external source and accumulator', session_accumulators, external_sources, True)

    # Perf NCCL
    torch.manual_seed(rank_idx)
    original = torch.randn_like(perf_src)
    nccl_input = original.clone()
    nccl_output = torch.empty(perf_src.numel() // num_ranks, dtype=torch.float32, device='cuda')
    nccl_duration = bench_kineto(
        lambda: reduce_scatter_single(nccl_output, nccl_input, group=group),
        kernel_names='ncclDevKernel_ReduceScatter', barrier_comm_profiling=True,
        barrier=buffer.barrier)

    # Perf ours
    dist_print('Performance:', once_in_node=True)
    perf_src.copy_(original)
    duration = bench_kineto(
        lambda: buffer.reduce_scatter(
            perf_src, num_sms=num_sms, group=group, scale=args.scale, comm_precision=args.comm_precision).wait(),
        kernel_names='reduce_scatter', barrier_comm_profiling=True,
        barrier=buffer.barrier)
    dist_print(
        f' > Perf ({num_sms=}): '
        f'{duration * 1e6:7.1f} us | '
        f'{perf_src.nbytes / duration / 1e9:5.1f} GB/s | '
        f'{nccl_duration / duration:.2f}x NCCL',
        once_in_node=True)

    # Perf accumulation
    perf_src.copy_(original)
    target = torch.zeros_like(nccl_output)
    accumulate_duration = bench_kineto(
        lambda: buffer.reduce_scatter(
            perf_src, dsts=target, num_sms=num_sms, group=group,
            scale=args.scale, comm_precision=args.comm_precision).wait(),
        kernel_names='reduce_scatter', barrier_comm_profiling=True,
        barrier=buffer.barrier)
    dist_print(
        f' > Accumulate ({num_sms=}): '
        f'{accumulate_duration * 1e6:7.1f} us | '
        f'{perf_src.nbytes / accumulate_duration / 1e9:5.1f} GB/s | '
        f'{duration / accumulate_duration:.2f}x RS',
        once_in_node=True)

    # Destroy
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test reduce-scatter accumulation')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-bytes', type=parse_num_bytes, default=1 << 30)
    parser.add_argument('--num-sms', type=int)
    parser.add_argument('--comm-precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

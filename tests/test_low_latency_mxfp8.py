import argparse

import torch
import torch.distributed as dist

import deep_ep
from utils import get_global_token_idxs, init_dist


def linearize_scales(scales: torch.Tensor) -> torch.Tensor:
    """Reorder the swizzled scales into `[num_local_experts, padded_capacity, hidden // 32]`."""
    return scales.permute(5, 2, 1, 0, 4, 3).reshape(scales.size(5), scales.size(2) * 128, -1)


def test_mxfp8_dispatch(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                       rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer):
    """Compare fused dispatch with BF16 dispatch followed by MXFP8 quantization.

    Requires SM100+ and FlashInfer's grouped MXFP8 quantizer. Covers unequal token counts,
    empty experts/ranks, scale tile boundaries, receive hooks, and graph replay.
    """
    from flashinfer import mxfp8_grouped_quantize

    num_local_experts = num_experts // num_ranks
    capacity = num_tokens * num_ranks
    padded_capacity = (capacity + 127) // 128 * 128

    def linear_scales(scales):
        return linearize_scales(scales)[:, :capacity]

    def dequantize(values, scales):
        scales = linear_scales(scales).contiguous().view(torch.float8_e8m0fnu).float()
        return (values.float().view(num_local_experts, capacity, hidden // 32, 32)
                * scales.unsqueeze(-1)).reshape(num_local_experts, capacity, hidden).bfloat16()

    for pattern in ('uneven', 'skewed', 'extreme', 'empty_rank', 'empty'):
        torch.manual_seed(42 + rank)
        count = max(0, num_tokens - rank)
        if pattern == 'empty' or (pattern == 'empty_rank' and rank == num_ranks - 1):
            count = 0
        x = torch.randn((count, hidden), dtype=torch.bfloat16, device='cuda')
        # Different magnitudes in adjacent 32-value blocks expose accidental block-128 quantization.
        magnitudes = torch.exp2((torch.arange(hidden // 32, device='cuda') % 15 - 7).float())
        x = (x.view(count, hidden // 32, 32) * magnitudes[None, :, None]).reshape(count, hidden).bfloat16()
        x[:, :32] = 0
        if pattern == 'extreme':
            x[:, 32:64] = 1e-37
            x[:, 64:96] = float('nan')
            x[:, 96:128] = float('inf')
            x[:, 128:160] = torch.finfo(torch.bfloat16).max
            x[:, 160:192] = 1e-35
            x[:, 192:224] = 1e-34
            x[:, 224:256] = 448.0 / 32768
            x[:, 256] = float('inf')
            x[:, 289] = float('-inf')
        scores = torch.rand((count, num_experts), device='cuda')
        topk_idx = scores.topk(num_topk, dim=-1).indices
        if pattern == 'skewed':
            topk_idx[:] = torch.arange(num_topk, device='cuda')
        if count and pattern == 'uneven':
            topk_idx[0, -1] = -1
        weights = torch.rand((count, num_topk), dtype=torch.float32, device='cuda') / num_topk
        expected_counts = torch.bincount(topk_idx[topk_idx >= 0], minlength=num_experts).to(torch.int32)
        dist.all_reduce(expected_counts, group=group)
        expected_counts = expected_counts[rank * num_local_experts:(rank + 1) * num_local_experts]

        # Check invalid options before entering a collective.
        for incompatible in ({'use_fp8': True}, {'use_nvfp4': True}, {'round_scale': True},
                             {'use_ue8m0': True}, {'use_ue8m0_for_sf': True},
                             {'x_global_scale': torch.ones((), device='cuda')}):
            options = dict(use_fp8=False, use_mxfp8=True)
            options.update(incompatible)
            try:
                buffer.low_latency_dispatch(x, topk_idx, num_tokens, num_experts, **options)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Accepted incompatible MXFP8 options: {incompatible}')

        for mode in ('sync', 'async', 'hook', 'graph'):
            ref_x, ref_counts, ref_handle, _, _ = buffer.low_latency_dispatch(
                x, topk_idx, num_tokens, num_experts, use_fp8=False)
            torch.testing.assert_close(ref_counts, expected_counts, atol=0, rtol=0)
            ref_q, ref_s = mxfp8_grouped_quantize(ref_x, ref_counts, backend='cuda')
            ref_q = ref_q.permute(2, 0, 1)
            ref_ids = get_global_token_idxs(ref_counts, ref_handle[0], ref_handle[1],
                                           num_local_experts, num_ranks, num_tokens)
            ref_output, _, _ = buffer.low_latency_combine(
                dequantize(ref_q, ref_s), topk_idx, weights, ref_handle)

            def run():
                received, counts, handle, event, hook = buffer.low_latency_dispatch(
                    x, topk_idx, num_tokens, num_experts, use_fp8=False, use_mxfp8=True,
                    async_finish=mode == 'async', return_recv_hook=mode in ('hook', 'graph'))
                if hook is not None:
                    hook()
                elif mode == 'async':
                    event.current_stream_wait()
                combined, _, _ = buffer.low_latency_combine(
                    dequantize(*received), topk_idx, weights, handle)
                return received, counts, handle, combined

            result = run()
            if mode == 'graph':
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = run()
                for replay in range(4):
                    if pattern != 'extreme':
                        x.neg_()
                    graph.replay()
                    expected = -ref_output if pattern != 'extreme' and replay % 2 == 0 else ref_output
                    torch.testing.assert_close(result[3], expected, atol=0, rtol=0,
                                               equal_nan=pattern == 'extreme')
            received, counts, handle, output = result
            torch.testing.assert_close(counts, ref_counts, atol=0, rtol=0)
            q, scales = received
            assert q.shape == (num_local_experts, capacity, hidden)
            assert q.dtype == torch.float8_e4m3fn and q.is_contiguous()
            assert scales.shape == (32, 4, padded_capacity // 128, 4, hidden // 128, num_local_experts)
            assert scales.dtype == torch.uint8
            ids = get_global_token_idxs(counts, handle[0], handle[1],
                                       num_local_experts, num_ranks, num_tokens)
            s, ref_s = linear_scales(scales), linear_scales(ref_s)
            for expert, n in enumerate(counts.tolist()):
                order = ids[expert, :n].argsort()
                ref_order = ref_ids[expert, :n].argsort()
                torch.testing.assert_close(ids[expert, :n][order], ref_ids[expert, :n][ref_order], atol=0, rtol=0)
                torch.testing.assert_close(q[expert, :n].view(torch.uint8)[order],
                                           ref_q[expert, :n].view(torch.uint8)[ref_order], atol=0, rtol=0)
                torch.testing.assert_close(s[expert, :n][order], ref_s[expert, :n][ref_order], atol=0, rtol=0)
            torch.testing.assert_close(output, ref_output, atol=0, rtol=0, equal_nan=pattern == 'extreme')
            if rank == 0:
                print(f'MXFP8 passed: {pattern=}, {mode=}, {hidden=}, {capacity=}', flush=True)
    dist.barrier(group=group)


def test_mxfp8_nan(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                   rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer):
    """A NaN encodes as 448 and leaves the scale and the other values of its block unchanged.

    Only an all-NaN block has a NaN scale. All tokens are identical, so the receive order does not matter.
    """
    capacity = num_tokens * num_ranks

    torch.manual_seed(0)
    row = torch.randn(hidden, dtype=torch.bfloat16, device='cuda')
    is_nan = torch.zeros(hidden, dtype=torch.bool, device='cuda')
    is_nan[1:32:2] = True
    is_nan[32:64:2] = True
    is_nan[64] = True
    is_nan[127] = True
    is_nan[128:160] = True
    all_nan_blocks = is_nan.view(-1, 32).all(dim=-1)

    x = row.masked_fill(is_nan, float('nan')).repeat(num_tokens, 1)
    x_without_nan = row.masked_fill(is_nan, 0).repeat(num_tokens, 1)
    scores = torch.rand((num_tokens, num_experts), device='cuda')
    topk_idx = scores.topk(num_topk, dim=-1).indices

    def dispatch(tokens):
        (q, scales), counts, _, _, _ = buffer.low_latency_dispatch(
            tokens, topk_idx, num_tokens, num_experts, use_fp8=False, use_mxfp8=True)
        return q.float(), linearize_scales(scales), counts

    ref_q, ref_s, ref_counts = dispatch(x_without_nan)
    q, s, counts = dispatch(x)
    torch.testing.assert_close(counts, ref_counts, atol=0, rtol=0)
    for expert, n in enumerate(counts.tolist()):
        assert (q[expert, :n][:, is_nan] == 448).all()
        torch.testing.assert_close(q[expert, :n][:, ~is_nan], ref_q[expert, :n][:, ~is_nan], atol=0, rtol=0)
        assert (s[expert, :n][:, all_nan_blocks] == 255).all()
        torch.testing.assert_close(s[expert, :n][:, ~all_nan_blocks], ref_s[expert, :n][:, ~all_nan_blocks],
                                   atol=0, rtol=0)
    if rank == 0:
        print(f'MXFP8 NaN passed: {hidden=}, {capacity=}', flush=True)
    dist.barrier(group=group)


def test_mxfp8_topk_weights(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                            rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer):
    """Dispatching the top-k weights along with the tokens leaves the values and scales unchanged."""
    num_local_experts = num_experts // num_ranks
    capacity = num_tokens * num_ranks

    torch.manual_seed(7 + rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    topk_idx = torch.rand((num_tokens, num_experts), device='cuda').topk(num_topk, dim=-1).indices
    # Every token carries its global index as the weight of all its experts.
    weights = (rank * num_tokens + torch.arange(num_tokens, device='cuda')).float()
    weights = weights.view(-1, 1).expand(-1, num_topk).contiguous()

    def dispatch(**kwargs):
        (q, scales), counts, *extra, handle, _, hook = buffer.low_latency_dispatch(
            x, topk_idx, num_tokens, num_experts, use_fp8=False, use_mxfp8=True, return_recv_hook=True, **kwargs)
        hook()
        ids = get_global_token_idxs(counts, handle[0], handle[1], num_local_experts, num_ranks, num_tokens)
        return q.view(torch.uint8), linearize_scales(scales), counts, ids, extra

    ref_q, ref_s, ref_counts, ref_ids, _ = dispatch()
    q, s, counts, ids, (recv_weights, _) = dispatch(topk_weights=weights)
    torch.testing.assert_close(counts, ref_counts, atol=0, rtol=0)
    for expert, n in enumerate(counts.tolist()):
        order = ids[expert, :n].argsort()
        ref_order = ref_ids[expert, :n].argsort()
        torch.testing.assert_close(ids[expert, :n][order], ref_ids[expert, :n][ref_order], atol=0, rtol=0)
        torch.testing.assert_close(q[expert, :n][order], ref_q[expert, :n][ref_order], atol=0, rtol=0)
        torch.testing.assert_close(s[expert, :n][order], ref_s[expert, :n][ref_order], atol=0, rtol=0)
        torch.testing.assert_close(recv_weights[expert, :n], ids[expert, :n].float(), atol=0, rtol=0)
    if rank == 0:
        print(f'MXFP8 top-k weights passed: {hidden=}, {capacity=}', flush=True)
    dist.barrier(group=group)


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink, explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl)
    test_mxfp8_dispatch(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer)
    test_mxfp8_nan(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer)
    # The dispatch sends the top-k weights only between peer-to-peer mapped ranks.
    if not args.disable_nvlink:
        test_mxfp8_topk_weights(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test MXFP8 low-latency dispatch/combine (SM100+; requires FlashInfer)')
    parser.add_argument('--num-processes', type=int, default=8,
                        help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128,
                        help='Maximum tokens per rank (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168,
                        help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8,
                        help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=288,
                        help='Number of experts (default: 288)')
    parser.add_argument('--allow-mnnvl', action='store_true',
                        help='Allow MNNVL for communication')
    parser.add_argument('--disable-nvlink', action='store_true',
                        help='Disable NVLink and use RDMA for dispatch/combine')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)

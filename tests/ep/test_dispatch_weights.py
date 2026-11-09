"""Run with the same launcher and topology settings as test_ep.py."""

import argparse

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden, num_topk = 16, 512, 4
    num_local_experts = 4
    num_experts = num_ranks * num_local_experts
    buffer = deep_ep.EPBuffer(
        group, num_max_tokens_per_rank=num_tokens, hidden=hidden, num_topk=num_topk,
        allow_hybrid_mode=bool(args.allow_hybrid_mode), explicitly_destroy=True,
        num_cpu_timeout_secs=30, num_gpu_timeout_secs=30)
    dist_print(f'Dispatch weights: logical topology {buffer.get_logical_domain_size()}', once_in_node=True)

    # Construct an independent oracle for every source token, including its routing weights.
    global_tokens = torch.arange(num_ranks * num_tokens, device='cuda')
    lanes = torch.arange(num_topk, device='cuda')
    all_idx = ((global_tokens[:, None] + lanes * 3) % num_experts).to(deep_ep.topk_idx_t)
    all_weights = global_tokens[:, None].float() + lanes.float() / 8
    all_x = global_tokens[:, None].expand(-1, hidden).to(torch.bfloat16).contiguous()
    source_slice = slice(rank * num_tokens, (rank + 1) * num_tokens)
    x, topk_idx, weights = all_x[source_slice], all_idx[source_slice], all_weights[source_slice]

    local_routes = all_idx // num_local_experts == rank
    expected_sources = global_tokens[local_routes.any(dim=1)]

    def check_result(result, weight_offset):
        recv_x, recv_idx, recv_weights, handle, _ = result
        torch.cuda.synchronize()
        source_ids, order = handle.recv_src_metadata[:, 0].sort()
        assert torch.equal(source_ids, expected_sources)
        routes = local_routes[source_ids]
        expected_x = all_x[source_ids]
        expected_weights = all_weights[source_ids]
        if recv_idx is not None:
            expected_idx = (all_idx[source_ids] % num_local_experts).masked_fill(~routes, -1)
            assert torch.equal(recv_idx[order], expected_idx)
            assert torch.equal(recv_x[order], expected_x)
            if weight_offset is not None:
                assert torch.equal(recv_weights[order][routes], (expected_weights + weight_offset)[routes])
        else:
            slots = handle.recv_src_metadata[order, 2:].long()[routes]
            assert slots.numel() == recv_x.shape[0]  # No padding with expert_alignment=1.
            assert torch.equal(recv_x[slots], expected_x[:, None, :].expand(-1, num_topk, -1)[routes])
            if weight_offset is not None:
                assert torch.equal(recv_weights[slots], (expected_weights + weight_offset)[routes])
        if weight_offset is None:
            assert recv_weights is None

    for do_expand in (False, True):
        # Explicit SM count avoids the independent cached-dispatch auto-SM issue.
        fresh_args = dict(x=x, topk_idx=topk_idx, num_experts=num_experts,
                          num_sms=4, do_expand=do_expand, expert_alignment=1)
        result = buffer.dispatch(**fresh_args, topk_weights=weights)
        check_result(result, 0)
        handle = result[3]
        for cached in (False, True):
            dispatch_args = dict(x=x, handle=handle, num_sms=4, do_expand=do_expand) if cached else fresh_args
            # Start with the wider case so an unpatched build fails without an out-of-bounds read.
            for width in (num_topk + 1, num_topk - 1):
                # Allocate contiguous inputs: a narrow column view would fail the existing stride check.
                invalid_weights = torch.zeros((num_tokens, width), dtype=torch.float32, device='cuda')
                # Every rank submits the same invalid shape before the next collective.
                try:
                    buffer.dispatch(**dispatch_args, topk_weights=invalid_weights)
                except RuntimeError as exc:
                    assert 'num_topk == num_topk_' in str(exc), str(exc)
                else:
                    raise AssertionError(f'Accepted weight width {width} for top-k {num_topk}')

            # Check recovery after rejection and ensure cached dispatch uses the newly supplied weights.
            check_result(buffer.dispatch(**dispatch_args, topk_weights=weights + 0.5), 0.5)
            check_result(buffer.dispatch(**dispatch_args, topk_weights=None), None)
            dist_print(f'PASS: expand={do_expand}, cached={cached}: narrow, wide, matching, None',
                       once_in_node=True)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test dispatch routing-weight width validation')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--allow-hybrid-mode', type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

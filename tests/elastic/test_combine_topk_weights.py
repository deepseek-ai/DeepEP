"""Check non-expanded combine with receiver-local top-k weight gradients.

Run with torchrun on at least two scale-out domains with two GPUs per domain:
    torchrun --nnodes=2 --nproc-per-node=2 --node-rank=NODE_RANK \
        --master-addr=MASTER_ADDR --master-port=29500 \
        tests/elastic/test_combine_topk_weights.py

Repeat with --allow-multiple-reduction=1 as a control.
"""

import argparse
import os

import torch
import torch.distributed as dist

import deep_ep


@torch.inference_mode()
def main(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    dist.init_process_group('nccl', device_id=device)
    buffer = None
    try:
        num_tokens, num_topk, experts_per_rank = 4, 4, 2
        num_experts = dist.get_world_size() * experts_per_rank
        buffer = deep_ep.ElasticBuffer(
            dist.group.WORLD,
            num_max_tokens_per_rank=num_tokens,
            hidden=args.hidden,
            num_topk=num_topk,
            deterministic=True,
            allow_hybrid_mode=True,
            allow_multiple_reduction=bool(args.allow_multiple_reduction),
            explicitly_destroy=True,
        )
        num_scaleout_ranks, num_scaleup_ranks = buffer.get_logical_domain_size()
        assert num_scaleout_ranks >= 2 and num_scaleup_ranks >= 2, \
            'This regression needs at least two scale-out domains and two GPUs per domain'

        cases = {
            'same_gpu': (0, 1),
            'same_scaleout_different_gpu': (0, experts_per_rank),
            'different_scaleout': (0, num_scaleup_ranks * experts_per_rank),
        }
        for name, experts in cases.items():
            # Invalid slots also exercise masked routes and master-slot selection.
            topk_idx = torch.full((num_tokens, num_topk), -1, dtype=deep_ep.topk_idx_t, device=device)
            topk_idx[:, 1] = experts[0]
            topk_idx[:, 3] = experts[1]
            topk_weights = torch.zeros((num_tokens, num_topk), dtype=torch.float, device=device)
            topk_weights[:, 1], topk_weights[:, 3] = 0.25, 0.75
            x = torch.zeros((num_tokens, args.hidden), dtype=torch.bfloat16, device=device)
            recv_x, recv_topk_idx, _, handle, event = buffer.dispatch(
                x,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=num_experts,
                expert_alignment=1,
                do_expand=False,
                async_with_compute_stream=True,
            )
            event.current_stream_wait()

            # A plain roundtrip can hide a wrong source row: dispatched weights
            # are duplicated. Backward gradients are nonzero only for local routes.
            expected_row = torch.tensor([0, 11, 0, 22], dtype=torch.float, device=device)
            local_routes = (recv_topk_idx >= 0) & (recv_topk_idx < experts_per_rank)
            score_grad = torch.where(local_routes, expected_row, 0.0).contiguous()
            grad_x, returned_score_grad, event = buffer.combine(
                torch.zeros_like(recv_x),
                handle=handle,
                topk_weights=score_grad,
                async_with_compute_stream=True,
            )
            event.current_stream_wait()

            expected = expected_row.expand(num_tokens, -1)
            nonzero_grad_x = torch.count_nonzero(grad_x).item()
            passed = torch.equal(returned_score_grad, expected) and nonzero_grad_x == 0
            if not passed:
                print(
                    f'{name}: rank={dist.get_rank()}, returned={returned_score_grad.tolist()}, '
                    f'expected={expected.tolist()}, nonzero_grad_x={nonzero_grad_x}',
                    flush=True)
            all_passed = torch.tensor(int(passed), device=device)
            dist.all_reduce(all_passed, op=dist.ReduceOp.MIN)
            assert all_passed.item() == 1, f'{name}: incorrect score gradients on one or more ranks'
            if dist.get_rank() == 0:
                print(f'{name}: PASS', flush=True)
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden', type=int, default=3072)
    parser.add_argument('--allow-multiple-reduction', type=int, choices=(0, 1), default=0)
    main(parser.parse_args())

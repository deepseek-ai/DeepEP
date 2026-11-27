import argparse

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.utils.envs import dist_print, init_dist


def token_view(buffer: deep_ep.EPBuffer) -> torch.Tensor:
    # Test-only access: lb_storage is a narrow of the entire [token, LB] storage.
    storage = buffer.runtime.lb_storage
    assert storage.storage_offset() == buffer.num_bytes
    assert storage.untyped_storage().nbytes() == buffer.num_bytes + buffer.num_lb_bytes
    return storage.as_strided((buffer.num_bytes,), (1,), storage_offset=0)


def byte_pattern(num_bytes: int, key: int) -> torch.Tensor:
    # A non-power-of-two period detects misplaced, allocation-aligned copies.
    return ((torch.arange(num_bytes, dtype=torch.int32, device='cuda') + key) % 251).to(torch.uint8)


def make_plan(num_bytes: int) -> tuple[deep_ep.BufferAllocator, torch.Tensor]:
    plan = deep_ep.BufferAllocator()
    tensor = plan.allocate((num_bytes // 4,), torch.float32)
    assert plan.num_bytes == num_bytes
    return plan, tensor


def check_layout(buffer: deep_ep.EPBuffer, num_bytes: int, num_lb_bytes: int,
                 planned_tensor=None) -> None:
    assert (buffer.num_bytes, buffer.num_lb_bytes) == (num_bytes, num_lb_bytes)
    storage = buffer.runtime.lb_storage
    assert storage.dtype == torch.uint8 and storage.is_cuda and storage.is_contiguous()
    assert storage.numel() == num_lb_bytes
    token_view(buffer)
    if planned_tensor is not None:
        assert planned_tensor.is_cuda and planned_tensor.dtype == torch.float32
        assert planned_tensor.nbytes == num_lb_bytes
        assert planned_tensor.data_ptr() == storage.data_ptr()


def make_tokens(rank_idx: int, num_tokens: int, hidden: int) -> torch.Tensor:
    values = torch.arange(num_tokens * hidden, dtype=torch.int32, device='cuda')
    return ((values.reshape(num_tokens, hidden) + 17 * rank_idx) % 127).to(torch.bfloat16)


def wait_result(result):
    if isinstance(result, tuple):
        *values, event = result
        event.wait()
        return values
    return result.wait()


def check_ep(buffer: deep_ep.EPBuffer, num_tokens: int, hidden: int, do_expand: bool,
             defer_epilogue: bool = False) -> None:
    # One expert per rank, all tokens go to the next rank. The reference does not
    # reuse DeepEP layout helpers and all values are exactly representable in BF16.
    x = make_tokens(buffer.rank_idx, num_tokens, hidden)
    topk_idx = torch.full((num_tokens, 1), (buffer.rank_idx + 1) % buffer.num_ranks,
                         dtype=deep_ep.topk_idx_t, device='cuda')
    result = buffer.dispatch(
        x, topk_idx=topk_idx, num_experts=buffer.num_ranks,
        num_max_tokens_per_rank=num_tokens, expert_alignment=1,
        num_sms=2, async_with_compute_stream=True, do_expand=do_expand,
        defer_epilogue=defer_epilogue)
    recv_x, _, _, handle = wait_result(result)
    expected = make_tokens((buffer.rank_idx - 1) % buffer.num_ranks, num_tokens, hidden)
    assert recv_x.shape == expected.shape
    assert torch.equal(recv_x[recv_x[:, 0].argsort()], expected[expected[:, 0].argsort()]), \
        'Dispatch after resize received the wrong tokens'

    result = buffer.combine(recv_x, handle, num_sms=2, async_with_compute_stream=True,
                            defer_epilogue=defer_epilogue)
    combined, _ = wait_result(result)
    assert torch.equal(combined, x), 'Dispatch/combine round trip failed after resize'


def check_lb(buffer: deep_ep.EPBuffer) -> None:
    if buffer.num_lb_bytes == 0:
        return
    num_local_ranks = buffer.num_nvlink_ranks
    local_rank = buffer.rank_idx % num_local_ranks
    owners = (torch.arange(num_local_ranks, dtype=torch.int32, device='cuda') + 1) % num_local_ranks
    mapping = owners.reshape(num_local_ranks, 1).contiguous()

    num_payload_bytes = 16 * 1024
    weights = byte_pattern(num_payload_bytes, local_rank * 19).reshape(1, num_payload_bytes)
    redundant = buffer.runtime.lb_storage[:num_payload_bytes].reshape(1, num_payload_bytes)
    redundant.zero_()
    pending = buffer.lb_prefetch_weights(redundant, weights, mapping, num_sms=2)

    # Intentionally do not wait for this communication before resizing. Resize
    # must finish it before copying/freeing the old allocation.
    buffer.resize(buffer.num_bytes, buffer.num_lb_bytes)
    redundant = buffer.runtime.lb_storage[:num_payload_bytes].reshape(1, num_payload_bytes)
    expected = byte_pattern(num_payload_bytes, ((local_rank + 1) % num_local_ranks) * 19)
    assert torch.equal(redundant[0], expected), 'Resize lost an in-flight LB prefetch'
    pending.wait()

    hidden = 1024
    redundant_grads = buffer.runtime.lb_storage[:hidden * 4].view(torch.float32).reshape(1, hidden)
    redundant_grads.fill_(local_rank + 1)
    expert_grads = torch.full((1, hidden), 10, dtype=torch.float32, device='cuda')
    buffer.lb_reduce_grads(redundant_grads, expert_grads, mapping, num_sms=2).wait()
    sender_rank = (local_rank - 1) % num_local_ranks
    assert torch.equal(expert_grads, torch.full_like(expert_grads, 10 + sender_rank + 1)), \
        'LB gradient reduction failed after resize'


def check_rejected_resize(buffer: deep_ep.EPBuffer, alignment: int) -> None:
    old_sizes = buffer.num_bytes, buffer.num_lb_bytes
    old_storage = buffer.runtime.lb_storage
    expected = old_storage.clone()
    used_plan, _ = make_plan(buffer.num_lb_bytes)
    used_plan.materialize(old_storage)
    cases = (
        ('zero token size', 0, old_sizes[1]),
        ('negative token size', -alignment, old_sizes[1]),
        ('unaligned token size', old_sizes[0] + 1, old_sizes[1]),
        ('negative LB size', old_sizes[0] + alignment, -1),
        ('unaligned LB size', old_sizes[0], old_sizes[1] + 1),
        ('already materialized plan', old_sizes[0] + alignment, used_plan),
    )
    for label, num_bytes, lb in cases:
        try:
            buffer.resize(num_bytes, lb)
        except (AssertionError, RuntimeError, ValueError, TypeError):
            pass
        else:
            raise AssertionError(f'{label}: resize should have rejected the request')
        assert (buffer.num_bytes, buffer.num_lb_bytes) == old_sizes, f'{label}: failed resize changed Python size fields'
        storage = buffer.runtime.lb_storage
        assert storage.data_ptr() == old_storage.data_ptr() and storage.numel() == old_sizes[1], \
            f'{label}: failed resize replaced LB storage'
        assert torch.equal(storage, expected), f'{label}: failed resize changed LB contents'


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks >= 2, 'Use at least two ranks to exercise peer windows'
    assert hasattr(_C.EPBuffer, 'resize'), 'Rebuild the DeepEP extension with resize support first'
    dist_print(f'Python package: {deep_ep.__file__}\nExtension: {_C.__file__}', once_in_node=True)

    alignment = deep_ep.get_num_allocation_alignment()
    num_tokens, hidden = 8, 256
    base_bytes = deep_ep.EPBuffer.get_buffer_size_hint(
        group, num_tokens, hidden, num_topk=1, allow_hybrid_mode=bool(args.allow_hybrid_mode))
    plan, planned_tensor = make_plan(alignment)
    buffer = deep_ep.EPBuffer(
        group, num_bytes=base_bytes, num_max_tokens_per_rank=num_tokens,
        lb_allocation_plan_or_num_bytes=plan,
        allow_hybrid_mode=bool(args.allow_hybrid_mode), explicitly_destroy=True)
    check_layout(buffer, base_bytes, alignment, planned_tensor)
    check_ep(buffer, num_tokens, hidden, False)
    del planned_tensor, plan

    cases = (
        (base_bytes + alignment, 2 * alignment, True),  # Grow both; move LB offset.
        (base_bytes + 2 * alignment, alignment, False),  # Grow token, shrink LB.
        (base_bytes, 2 * alignment, True),  # Shrink token, grow LB.
        (base_bytes + alignment, 0, False),  # Disable LB.
        (base_bytes, alignment, True),  # Re-enable LB.
        (base_bytes, alignment, False),  # Same sizes; repeat registration.
    )
    producer = torch.cuda.Stream()
    for round_idx in range(args.num_rounds):
        for step_idx, (num_bytes, num_lb_bytes, with_plan) in enumerate(cases):
            where = f'round={round_idx}, step={step_idx}, rank={rank_idx}'
            old_sizes = buffer.num_bytes, buffer.num_lb_bytes
            expected_token = byte_pattern(old_sizes[0], rank_idx * 23 + step_idx + 3 * round_idx)
            expected_lb = byte_pattern(old_sizes[1], rank_idx * 31 + step_idx + 7 * round_idx + 100)
            old_token, old_lb = token_view(buffer), buffer.runtime.lb_storage
            producer.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(producer):
                old_token.copy_(expected_token)
                old_lb.copy_(expected_lb)

            lb_arg, planned_tensor = make_plan(num_lb_bytes) if with_plan else (num_lb_bytes, None)
            # No producer synchronization here: exercise resize's entry sync.
            buffer.resize(num_bytes, lb_arg)
            del old_token, old_lb
            check_layout(buffer, num_bytes, num_lb_bytes, planned_tensor)
            num_preserved_bytes = min(old_sizes[0], num_bytes)
            assert torch.equal(token_view(buffer)[:num_preserved_bytes], expected_token[:num_preserved_bytes]), \
                f'{where}: token prefix changed'
            num_preserved_bytes = min(old_sizes[1], num_lb_bytes)
            assert torch.equal(buffer.runtime.lb_storage[:num_preserved_bytes], expected_lb[:num_preserved_bytes]), \
                f'{where}: LB prefix changed'
            if with_plan:
                assert lb_arg.materialized
                assert torch.equal(planned_tensor.view(torch.uint8)[:num_preserved_bytes], expected_lb[:num_preserved_bytes]), \
                    f'{where}: new allocation plan points to the wrong data'
            # check_lb performs another resize, so retire the plan's views now.
            del planned_tensor, lb_arg

            buffer.barrier(with_cpu_sync=True)
            check_ep(buffer, num_tokens, hidden, bool(step_idx % 2), defer_epilogue=step_idx % 3 == 0)
            check_lb(buffer)
            dist_print(f'Resize {round_idx}/{step_idx}: token={num_bytes}, LB={num_lb_bytes}: passed',
                       once_in_node=True)

    check_rejected_resize(buffer, alignment)
    check_ep(buffer, num_tokens, hidden, False)
    buffer.destroy()
    dist.destroy_process_group()
    if local_rank == 0:
        print('EP resize: passed (including rejection and destruction checks)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test EPBuffer resize, data migration, and communication')
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--num-rounds', type=int, default=2)
    parser.add_argument('--allow-hybrid-mode', type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    assert args.num_processes > 0 and args.num_rounds > 0
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)

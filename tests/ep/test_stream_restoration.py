import argparse
from itertools import product

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist


def assert_current_stream(caller: torch.cuda.Stream) -> None:
    actual = torch.cuda.current_stream()
    assert actual == caller, f'Caller stream was not restored: expected {caller}, got {actual}'


def expect_error(call, message: str, caller: torch.cuda.Stream) -> None:
    try:
        call()
    except RuntimeError as exc:
        assert message in str(exc), f'Unexpected exception: {exc}'
    else:
        raise AssertionError(f'Expected a host assertion containing {message!r}')
    # Check before leaving torch.cuda.stream(), whose __exit__ would hide a leaked stream.
    assert_current_stream(caller)


def finish(result, caller: torch.cuda.Stream, async_with_compute_stream: bool, defer_epilogue: bool):
    # Deferred calls must restore the caller before the epilogue hook is invoked.
    assert_current_stream(caller)
    if defer_epilogue:
        result = result.current_stream_wait()
    else:
        *result, event = result
        if async_with_compute_stream:
            event.current_stream_wait()
    assert_current_stream(caller)
    return result


@torch.inference_mode()
def test(local_rank: int, num_local_ranks: int, operation: str) -> None:
    _, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks > 1, 'The expert-count regression needs at least two ranks'
    num_tokens, hidden, num_experts = 32, 256, num_ranks * 2
    buffer = deep_ep.EPBuffer(
        group, num_max_tokens_per_rank=num_tokens, hidden=hidden, num_topk=1,
        explicitly_destroy=True, num_cpu_timeout_secs=30, num_gpu_timeout_secs=30)
    caller = torch.cuda.Stream()
    assert caller != torch.cuda.default_stream()
    assert caller != buffer.get_comm_stream()

    try:
        with torch.cuda.stream(caller):
            # Every rank uses the same shapes and routing, including every rejected call.
            x = (torch.arange(num_tokens * hidden, dtype=torch.int32, device='cuda') % 13) \
                .reshape(num_tokens, hidden).to(torch.bfloat16)
            topk_idx = (torch.arange(num_tokens, dtype=deep_ep.topk_idx_t, device='cuda') % num_experts) \
                .reshape(num_tokens, 1)
            # The hidden payload alone exceeds the communication buffer, without a large allocation.
            bad_hidden = (buffer.num_bytes // (num_tokens * x.element_size()) // 32 + 1) * 32

            for allocate, asynchronous, deferred, with_previous_event in product((False, True), repeat=4):
                if (deferred and not asynchronous) or (with_previous_event and not allocate):
                    continue

                def stream_args():
                    return dict(
                        allocate_on_comm_stream=allocate,
                        async_with_compute_stream=asynchronous,
                        defer_epilogue=deferred,
                        previous_event=buffer.capture() if with_previous_event else None)

                def dispatch(**overrides):
                    # Avoid Python's automatic SM estimator rejecting num_experts before C++ switches streams.
                    params = dict(x=x, topk_idx=topk_idx, num_experts=num_experts, num_sms=4, num_qps=1)
                    params.update(stream_args())
                    params.update(overrides)
                    return buffer.dispatch(**params)

                def valid_dispatch():
                    return finish(dispatch(), caller, asynchronous, deferred)

                def combine(recv_x, handle):
                    return buffer.combine(recv_x, handle, num_sms=4, num_qps=1, **stream_args())

                def valid_combine(recv_x, handle):
                    combined_x, _ = finish(combine(recv_x, handle), caller, asynchronous, deferred)
                    assert torch.equal(combined_x, x), 'Dispatch/combine round trip changed the input'

                if operation in ('all', 'dispatch'):
                    expect_error(lambda: dispatch(num_experts=num_experts + 1),
                                 'num_experts % context->num_ranks == 0', caller)
                    # Retry directly on the same stream and buffer, without manually restoring the stream.
                    recv_x, _, _, handle = valid_dispatch()
                    valid_combine(recv_x, handle)

                    bad_max_tokens = buffer.num_bytes // (hidden * x.element_size()) + 1
                    expect_error(lambda: dispatch(num_max_tokens_per_rank=bad_max_tokens),
                                 'get_dispatch_buffer_size', caller)
                    recv_x, _, _, handle = valid_dispatch()
                    valid_combine(recv_x, handle)

                if operation in ('all', 'combine'):
                    recv_x, _, _, handle = valid_dispatch()
                    # Keep genuine dispatch metadata and a valid BF16 input shape; only capacity is invalid.
                    oversized_x = torch.empty((recv_x.size(0), bad_hidden), dtype=x.dtype, device=x.device)
                    expect_error(lambda: combine(oversized_x, handle), 'get_combine_buffer_size', caller)
                    valid_combine(recv_x, handle)
                    recv_x, _, _, handle = valid_dispatch()
                    valid_combine(recv_x, handle)

                dist_print(f'Stream restoration passed: {operation=}, {allocate=}, {asynchronous=}, '
                           f'{deferred=}, {with_previous_event=}', once_in_node=True)
    finally:
        # Exiting the caller context also lets a failing baseline clean up on its original stream.
        torch.cuda.synchronize()
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test dispatch/combine CUDA stream restoration on success and exceptions')
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--operation', choices=('all', 'dispatch', 'combine'), default='all')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args.operation), nprocs=args.num_processes)

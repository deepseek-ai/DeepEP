import argparse
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist
from deep_ep.utils.refs import dispatch as reference_dispatch


class SavedReceive(torch.autograd.Function):
    """A consumer that keeps the received prefix for backward, like an expert input."""

    @staticmethod
    def forward(ctx, received, weight):
        ctx.save_for_backward(received)
        return (received.float() * weight).sum()

    @staticmethod
    def backward(ctx, grad_output):
        received, = ctx.saved_tensors
        return None, received.float().sum(dim=0) * grad_output


def make_inputs(rank: int, num_ranks: int, num_tokens: int, hidden: int, pattern: str):
    rows = num_tokens - rank % min(num_tokens, num_ranks)
    row = torch.arange(rows, device='cuda')
    col = torch.arange(hidden, device='cuda')
    x = ((row[:, None] + col[None, :] + rank * 3) % 17 - 8).bfloat16()
    destination = (row + rank) % num_ranks if pattern != 'skewed' else torch.zeros_like(row)
    routes = (destination[:, None] * 2 + torch.arange(2, device='cuda')[None, :]).to(deep_ep.topk_idx_t)
    weights = torch.ones((rows, 2), dtype=torch.float32, device='cuda')
    if pattern == 'masked':
        routes[row % 3 == 0, 1] = -1
        weights[row % 3 == 0, 1] = 0
    return x, routes, weights


def test_saved_prefix(buffer: deep_ep.EPBuffer, x: torch.Tensor, routes: torch.Tensor, weights: torch.Tensor,
                      expected: tuple, capacity: int, allocate_on_comm_stream: bool, delay_cycles: int):
    def dispatch(value: torch.Tensor):
        return buffer.dispatch(value, topk_idx=routes, topk_weights=weights, num_experts=buffer.num_ranks * 2,
                               num_max_tokens_per_rank=capacity, num_sms=8, do_cpu_sync=False,
                               async_with_compute_stream=True, allocate_on_comm_stream=allocate_on_comm_stream)

    # Only the first `count` rows of the capacity-sized outputs are valid
    recv_x, recv_topk_idx, recv_topk_weights, handle, event = dispatch(x)
    event.current_stream_wait()
    count = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    source = handle.recv_src_metadata[:count, 0]
    order = source.argsort()
    prefix = recv_x[:count]
    for actual, reference in zip((prefix[order], recv_topk_idx[:count][order], recv_topk_weights[:count][order], source[order]), expected[:4]):
        assert torch.equal(actual, reference)
    if buffer.deterministic:
        assert torch.equal(source, expected[3])
    # A prefix view keeps the whole capacity allocation alive
    assert prefix.untyped_storage().nbytes() == recv_x.numel() * recv_x.element_size()

    # Hand the prefix to a consumer stream and save it for backward; the event covers the weight initialization
    weight = torch.ones(x.shape[1], dtype=torch.float32, device='cuda', requires_grad=True)
    ready = torch.cuda.Event()
    ready.record()
    reader = torch.cuda.Stream()
    with torch.cuda.stream(reader):
        reader.wait_event(ready)
        prefix.record_stream(reader)
        loss = SavedReceive.apply(prefix, weight)
    # Only the saved prefix view keeps the receive storage alive from here on
    storage_ptr = prefix.untyped_storage().data_ptr()
    del recv_x, recv_topk_idx, recv_topk_weights, handle, event, prefix

    # Later dispatches through the same buffer must neither reuse nor overwrite the saved activation
    for multiplier in (2, 3, 4):
        later = dispatch(x * multiplier)
        later[-1].current_stream_wait()
        assert later[0].untyped_storage().data_ptr() != storage_ptr
        del later
    with torch.cuda.stream(reader):
        loss.backward()
    torch.cuda.synchronize()
    assert torch.equal(weight.grad, expected[0].float().sum(dim=0))
    del loss, weight

    # Exercise allocator reuse independently of the saved autograd activation
    test_stream_pressure(dispatch, buffer, x, expected, allocate_on_comm_stream, delay_cycles)


def test_stream_pressure(dispatch, buffer: deep_ep.EPBuffer, x: torch.Tensor, expected: tuple,
                         allocate_on_comm_stream: bool, delay_cycles: int):
    # A descheduled host can miss the delay window. Retry in lockstep, never silently skip it
    for attempt in range(3):
        received, indices, weights, handle, event = dispatch(x)
        event.current_stream_wait()
        count = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
        assert count == expected[0].shape[0]
        prefix = received[:count]
        order = handle.recv_src_metadata[:count, 0].argsort()
        shape = received.shape
        comm_stream = buffer.get_comm_stream()
        owner = comm_stream if allocate_on_comm_stream else torch.cuda.current_stream()
        # Warm allocation and create the copy target before the delay: cudaMalloc can synchronize it away
        with torch.cuda.stream(owner):
            warmup = [torch.empty(shape, dtype=x.dtype, device=x.device) for _ in range(4)]
        del warmup
        reader = torch.cuda.Stream()
        while reader.cuda_stream in (owner.cuda_stream, comm_stream.cuda_stream):
            reader = torch.cuda.Stream()
        with torch.cuda.stream(reader):
            copied = torch.empty_like(prefix)
        ready, read_started = torch.cuda.Event(), torch.cuda.Event()
        ready.record()
        with torch.cuda.stream(reader):
            reader.wait_event(ready)
            torch.cuda._sleep(delay_cycles * 4**attempt)
            read_started.record()
            prefix.record_stream(reader)
            copied.copy_(prefix)
        pending_before = not read_started.query()
        # Drop the event too: EP_AVOID_RECORD_STREAM=1 retains the native outputs there
        del received, indices, weights, handle, event, prefix
        with torch.cuda.stream(owner):
            pressure = [torch.empty(shape, dtype=x.dtype, device=x.device).fill_(-101) for _ in range(4)]
        owner.synchronize()
        pending_after = not read_started.query()
        torch.cuda.synchronize()
        local_correct = torch.equal(copied[order], expected[0])
        outcome = torch.tensor([local_correct, pending_before and pending_after], dtype=torch.int, device=x.device)
        dist.all_reduce(outcome, op=dist.ReduceOp.MIN)
        correct, effective = outcome.tolist()
        assert correct, f'CROSS_STREAM_RECEIVE_MISMATCH rank={dist.get_rank()} {local_correct=}'
        del pressure
        if effective:
            dist_print(f'PASS stream pressure {attempt=} {pending_before=} {pending_after=}', once_in_node=True)
            return
    raise RuntimeError('Could not establish an in-flight pressure window; increase --delay-cycles')


# noinspection PyUnusedLocal
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks, seed=42)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    capacity = args.num_tokens * 4
    for deterministic in (False, True):
        buffer = deep_ep.EPBuffer(group, num_max_tokens_per_rank=capacity, hidden=args.hidden, num_topk=2,
                                  deterministic=deterministic, allow_hybrid_mode=False, explicitly_destroy=True)
        for allocate_on_comm_stream in (False, True):
            for pattern in ('balanced', 'skewed', 'masked'):
                x, routes, weights = make_inputs(rank, num_ranks, args.num_tokens, args.hidden, pattern)
                expected = reference_dispatch(x, routes, weights, capacity, num_ranks * 2)
                test_saved_prefix(buffer, x, routes, weights, expected, capacity, allocate_on_comm_stream, args.delay_cycles)
                dist.barrier(group=group)
                dist_print(f'PASS {deterministic=} {allocate_on_comm_stream=} {pattern=}', once_in_node=True)
        buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test sync-free receive tensor lifetimes')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=64, help='Number of tokens per rank')
    parser.add_argument('--hidden', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--delay-cycles', type=int, default=20000000, help='Initial delay for the cross-stream reuse check')
    args = parser.parse_args()
    assert args.num_tokens > 0 and args.hidden > 0 and args.delay_cycles > 0
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)

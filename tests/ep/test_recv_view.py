import argparse
import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import dist_print, init_dist


def exact(actual: torch.Tensor, expected: torch.Tensor):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if actual.numel():
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def rejected(fn, error):
    try:
        fn()
    except error:
        return
    raise AssertionError(f'Expected {error.__name__}')


def test_recv_view(group: dist.ProcessGroup, deterministic: bool, asynchronous: bool, with_weights: bool, skew: bool):
    rank, world = group.rank(), group.size()
    torch.manual_seed(42 + rank)
    capacity, hidden, topk, local_experts = 128, 256, 2, 2
    tokens = 65 - rank
    full_x = torch.randn((capacity, hidden), device='cuda', dtype=torch.bfloat16)
    x = full_x[:tokens]
    full_indices = torch.full((capacity, topk), -1, device='cuda', dtype=deep_ep.topk_idx_t)
    indices = full_indices[:tokens]
    indices.copy_(torch.rand((tokens, local_experts * world), device='cuda').argsort(dim=1)[:, :topk])
    if skew:
        indices[:] = torch.arange(topk, device='cuda', dtype=deep_ep.topk_idx_t)
    indices[::7] = -1
    full_weights = torch.randn((capacity, topk), device='cuda', dtype=torch.float32) if with_weights else None
    weights = full_weights[:tokens] if with_weights else None

    def gather(tensor: torch.Tensor):
        gathered = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(gathered, tensor, group=group)
        return torch.cat(gathered)

    # Expected receives, computed from the gathered inputs
    all_x, all_indices = gather(full_x), gather(full_indices)
    local_routes = (all_indices >= rank * local_experts) & (all_indices < (rank + 1) * local_experts)
    expected_source = torch.arange(world * capacity, device='cuda')[local_routes.any(dim=1)]
    expected_lane = torch.where(local_routes[expected_source], torch.arange(topk, device='cuda'), -1).amax(dim=1)
    expected_peer_topk = expected_source // capacity * topk + expected_lane
    expected_x = all_x[expected_source]
    expected_ids = torch.where(local_routes, all_indices - rank * local_experts, -1)[expected_source]
    expected_weights = gather(full_weights)[expected_source] if with_weights else None

    def check(payload, recv_ids, recv_weights, handle):
        source = handle.recv_src_metadata[:, 0].long()
        order = slice(None) if deterministic else source.argsort()
        exact(source[order], expected_source)
        exact(handle.recv_src_metadata[:, 1].long()[order], expected_peer_topk)
        exact(payload[order], expected_x)
        exact(recv_ids[order], expected_ids)
        if with_weights:
            exact(recv_weights[order], expected_weights)
        else:
            assert recv_weights is None

    buffer = deep_ep.EPBuffer(group, num_max_tokens_per_rank=capacity, hidden=hidden, num_topk=topk,
                              deterministic=deterministic, explicitly_destroy=True)
    assert buffer.num_scaleout_ranks == 1 and buffer.num_rdma_ranks == 1, 'Borrowed receive storage requires one NVLink domain'
    kwargs = dict(topk_idx=indices, topk_weights=weights, num_experts=local_experts * world, num_sms=8,
                  async_with_compute_stream=asynchronous, allocate_on_comm_stream=asynchronous)

    def dispatch(borrow: bool, value: torch.Tensor = x):
        result = buffer.dispatch(value, borrow_recv=borrow, **kwargs)
        if asynchronous:
            if borrow:
                # The view is unusable until the dispatch event is waited
                for access in (result[0].wait, lambda: result[0].slab, lambda: result[0].row_indices, result[0].release):
                    rejected(access, RuntimeError)
            result[-1].current_stream_wait()
        return result[:4]

    check(*dispatch(False))
    view, recv_ids, recv_weights, handle = dispatch(True)
    payload = view.slab.index_select(0, view.row_indices)
    check(payload, recv_ids, recv_weights, handle)

    # An active view blocks other buffer operations until it is released
    rejected(lambda: buffer.dispatch(x, **kwargs), RuntimeError)
    rejected(lambda: buffer.combine(payload, handle), RuntimeError)
    rejected(buffer.destroy, RuntimeError)
    view.release()
    for access in (view.wait, lambda: view.slab, lambda: view.row_indices, view.release):
        rejected(access, RuntimeError)

    # The handle still supports ordinary cached replay and combine
    replay, _, _, _, event = buffer.dispatch(x, handle=handle, num_sms=8, async_with_compute_stream=True)
    event.current_stream_wait()
    exact(replay, payload)
    # Integer sums are exact even with an unspecified receive order
    combined, _, event = buffer.combine(torch.ones_like(replay), handle, async_with_compute_stream=True)
    event.current_stream_wait()
    expected_count = torch.stack([((indices >= 0) & (indices // local_experts == peer)).any(dim=1)
                                  for peer in range(world)]).sum(dim=0).to(x.dtype)
    exact(combined, expected_count[:, None].expand_as(x))

    # Wait on a third stream first; the release must protect both delayed readers from the next dispatch
    view, delayed_ids, delayed_weights, delayed_handle, event = \
        buffer.dispatch(x, borrow_recv=True, **(kwargs | dict(async_with_compute_stream=True)))
    waiter = torch.cuda.Stream()
    with torch.cuda.stream(waiter):
        torch.cuda._sleep(4_000_000)
        event.current_stream_wait()
    readers = [torch.cuda.Stream(), torch.cuda.Stream()]
    delayed = []
    for reader, delay in zip(readers, (4_000_000, 64_000_000)):
        with torch.cuda.stream(reader):
            torch.cuda._sleep(delay)
            view.wait()
            delayed.append(view.slab.index_select(0, view.row_indices))
    view.release(*readers)
    assert not readers[-1].query(), 'Delayed reader finished before buffer reuse was tested'
    dispatch(False, x + 1)
    for reader, received in zip(readers, delayed):
        reader.synchronize()
        check(received, delayed_ids, delayed_weights, delayed_handle)

    # Unsupported modes are rejected before launching
    for extra in (dict(do_expand=True), dict(defer_epilogue=True), dict(do_cpu_sync=False), dict(handle=handle)):
        rejected(lambda: buffer.dispatch(x, borrow_recv=True, **(kwargs | extra)), ValueError)
    rejected(lambda: buffer.dispatch(x.float(), borrow_recv=True, **kwargs), ValueError)
    fp8 = (x.to(torch.float8_e4m3fn), torch.ones((tokens, hidden // 128), device='cuda'))
    rejected(lambda: buffer.dispatch(fp8, borrow_recv=True, **kwargs), ValueError)
    dist.barrier(group=group)
    buffer.destroy()


def test_double_buffer(group: dist.ProcessGroup):
    rank, world = group.rank(), group.size()
    capacity, hidden, topk = 128, 256, 2
    tokens = world * (capacity // world // 2)
    row = torch.arange(tokens, device='cuda')
    indices = (2 * (row[:, None] % world) + torch.arange(topk, device='cuda')).to(deep_ep.topk_idx_t)
    x = (1 + rank * capacity + row[:, None]).expand(tokens, hidden).to(torch.bfloat16).contiguous()
    other_x = -x
    expert_output = torch.ones_like(x)
    slot_bytes = deep_ep.EPBuffer.get_buffer_size_hint(group, capacity, hidden, topk)
    lb_plan = deep_ep.BufferAllocator()
    lb_storage = lb_plan.allocate((128,), torch.uint8)
    buffer = deep_ep.EPBuffer(group, num_bytes=slot_bytes, num_max_tokens_per_rank=capacity,
                              num_buffers=2, lb_allocation_plan_or_num_bytes=lb_plan, explicitly_destroy=True)
    lb_storage.fill_(39)
    assert buffer.num_bytes == 2 * slot_bytes
    ready = buffer.capture()
    async_args = dict(previous_event=ready, allocate_on_comm_stream=True, async_with_compute_stream=True)
    dispatch_args = dict(topk_idx=indices, num_experts=2 * world, num_sms=8, borrow_recv=True, **async_args)

    reader = torch.cuda.Stream()
    for release_before_dispatch in (False, True):
        view0, _, _, handle0, event0 = buffer.dispatch(x, buffer_idx=0, **dispatch_args)
        event0.wait()
        ptr0 = view0.slab.data_ptr()
        with torch.cuda.stream(reader):
            view0.wait()
            if release_before_dispatch:
                # The first iteration warms both communication paths before the delayed read.
                torch.cuda._sleep(512_000_000)
            payload0 = view0.slab.index_select(0, view0.row_indices)
            reader_done = reader.record_event()
        if release_before_dispatch:
            view0.release(reader)

        view1, _, _, handle1, event1 = buffer.dispatch(other_x, buffer_idx=1, **dispatch_args)
        event1.wait()
        assert view1.slab.data_ptr() == ptr0 + slot_bytes
        assert lb_storage.data_ptr() == ptr0 + 2 * slot_bytes
        rejected(lambda: buffer.combine(expert_output, handle1, buffer_idx=1, **async_args), RuntimeError)
        rejected(buffer.destroy, RuntimeError)
        if release_before_dispatch:
            torch.cuda.current_stream().synchronize()
            assert not reader_done.query(), 'Dispatch on slot 1 waited for the slot 0 reader'
        else:
            rejected(lambda: buffer.dispatch(other_x, buffer_idx=0, **dispatch_args), RuntimeError)
            view0.release(reader)

        combined0, _, event = buffer.combine(expert_output, handle0, buffer_idx=0, **async_args)
        event.wait()
        torch.cuda.current_stream().wait_event(reader_done)
        expected0 = (1 + handle0.recv_src_metadata[:, :1]).expand_as(payload0).to(x.dtype)
        exact(payload0, expected0)
        payload1 = view1.slab.index_select(0, view1.row_indices)
        expected1 = -(1 + handle1.recv_src_metadata[:, :1]).expand_as(payload1).to(x.dtype)
        exact(payload1, expected1)
        view1.release()
        combined1, _, event = buffer.combine(expert_output, handle1, buffer_idx=1, **async_args)
        event.wait()
        exact(combined0, expert_output)
        exact(combined1, expert_output)

    exact(lb_storage, torch.full_like(lb_storage, 39))
    rejected(lambda: buffer.dispatch(x, buffer_idx=2, **dispatch_args), ValueError)
    buffer.destroy()


# noinspection PyUnusedLocal
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    for deterministic in (False, True):
        for asynchronous in (False, True):
            for with_weights in (False, True):
                for skew in (False, True):
                    test_recv_view(group, deterministic, asynchronous, with_weights, skew)
                    dist_print(f'PASS {deterministic=} {asynchronous=} {with_weights=} {skew=}', once_in_node=True)
    test_double_buffer(group)
    dist_print('PASS double-buffered receive views', once_in_node=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test borrowed receive storage')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)

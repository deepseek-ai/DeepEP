"""Run with torchrun --standalone --nproc-per-node=8 tests/ep/test_recv_view.py."""
import os

import torch
import torch.distributed as dist


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if actual.numel():
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def rejected(fn, error):
    try:
        fn()
    except error:
        return
    raise AssertionError(f'Expected {error.__name__}')


def test_case(deep_ep, deterministic, asynchronous, with_weights, skew):
    rank, world = dist.get_rank(), dist.get_world_size()
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

    def gather(tensor):
        gathered = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered)

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

    buffer = deep_ep.EPBuffer(dist.group.WORLD,
                              num_max_tokens_per_rank=capacity,
                              hidden=hidden,
                              num_topk=topk,
                              deterministic=deterministic,
                              explicitly_destroy=True)
    kwargs = dict(topk_idx=indices,
                  topk_weights=weights,
                  num_experts=local_experts * world,
                  num_sms=8,
                  async_with_compute_stream=asynchronous,
                  allocate_on_comm_stream=asynchronous)

    def dispatch(borrow, value=x):
        result = buffer.dispatch(value, borrow_recv=borrow, **kwargs)
        if asynchronous:
            result[-1].current_stream_wait()
        return result[:4]

    check(*dispatch(False))
    view, recv_ids, recv_weights, handle = dispatch(True)
    payload = view.slab.index_select(0, view.row_indices)
    check(payload, recv_ids, recv_weights, handle)
    rejected(lambda: buffer.dispatch(x, **kwargs), RuntimeError)
    rejected(lambda: buffer.combine(payload, handle), RuntimeError)
    rejected(buffer.destroy, RuntimeError)
    view.release()
    rejected(lambda: view.slab, RuntimeError)

    replay, _, _, _, event = buffer.dispatch(x, handle=handle, num_sms=8, async_with_compute_stream=True)
    event.current_stream_wait()
    exact(replay, payload)
    # Integer sums are exact even with an unspecified receive order.
    combined, _, event = buffer.combine(torch.ones_like(replay), handle, async_with_compute_stream=True)
    event.current_stream_wait()
    expected_count = torch.stack([((indices >= 0) & (indices // local_experts == peer)).any(dim=1)
                                  for peer in range(world)]).sum(dim=0).to(x.dtype)
    exact(combined, expected_count[:, None].expand_as(x))

    # First wait on a third stream; release must protect both delayed readers.
    view, delayed_ids, delayed_weights, delayed_handle, event = buffer.dispatch(x,
                                                                                borrow_recv=True,
                                                                                **(kwargs | dict(async_with_compute_stream=True)))
    ready = torch.cuda.Stream()
    with torch.cuda.stream(ready):
        torch.cuda._sleep(4_000_000)
        event.current_stream_wait()
    readers = [torch.cuda.Stream(), torch.cuda.Stream()]
    delayed = []
    for reader, delay in zip(readers, (4_000_000, 64_000_000)):
        with torch.cuda.stream(reader):
            torch.cuda._sleep(delay)
            delayed.append(view.slab.index_select(0, view.row_indices))
    view.release(*readers)
    assert not readers[-1].query(), 'Delayed reader finished before buffer reuse was tested'
    dispatch(False, x + 1)
    for reader, received in zip(readers, delayed):
        reader.synchronize()
        check(received, delayed_ids, delayed_weights, delayed_handle)

    for extra in [dict(do_expand=True), dict(defer_epilogue=True), dict(do_cpu_sync=False), dict(handle=handle)]:
        rejected(lambda extra=extra: buffer.dispatch(x, borrow_recv=True, **(kwargs | extra)), ValueError)
    rejected(lambda: buffer.dispatch(x.float(), borrow_recv=True, **kwargs), ValueError)
    dist.barrier()
    buffer.destroy()


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    import deep_ep

    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    for deterministic in [False, True]:
        for asynchronous in [False, True]:
            for with_weights in [False, True]:
                for skew in [False, True]:
                    test_case(deep_ep, deterministic, asynchronous, with_weights, skew)
                    if rank == 0:
                        print(f'PASS {deterministic=} {asynchronous=} {with_weights=} {skew=}', flush=True)
    deep_ep.destroy_all_managed_nccl_comm()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

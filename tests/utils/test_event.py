import torch

from deep_ep import EventHandle, EventOverlap


def test_multistream_hook():
    hook_stream, reader_stream = torch.cuda.Stream(), torch.cuda.Stream()
    payload = torch.zeros(1, device='cuda', dtype=torch.int32)
    control, received = torch.empty_like(payload), torch.empty_like(payload)
    control_done, hook_done = torch.cuda.Event(), torch.cuda.Event()

    # Warm up the kernels and event API before constructing the delayed dependency.
    control.copy_(payload)
    received.copy_(payload)
    torch.cuda._sleep(1)
    EventHandle().current_stream_wait()
    torch.cuda.synchronize()

    event = EventHandle()
    overlap = EventOverlap(event)
    calls = 0

    def hook():
        nonlocal calls
        calls += 1
        torch.cuda._sleep(200_000_000)
        payload.fill_(123)
        hook_done.record()
        return payload

    overlap.register_hook_after_wait(hook)
    with torch.cuda.stream(hook_stream):
        assert overlap.current_stream_wait() is payload
    epilogue_event = overlap._epilogue_event
    assert epilogue_event is not None

    with torch.cuda.stream(reader_stream):
        # The original event precedes the hook: this negative control must see zero.
        event.current_stream_wait()
        control.copy_(payload)
        control_done.record()
        # A second wait must additionally wait for the first stream's hook writes.
        assert overlap.wait() is None
        received.copy_(payload)

    control_done.synchronize()
    assert not hook_done.query(), 'The hook finished before the cross-stream dependency was exercised'
    assert control.item() == 0, 'The original event unexpectedly covered the hook'
    reader_stream.synchronize()
    assert received.item() == 123
    assert calls == 1 and overlap.hook_after_wait is None
    assert overlap._epilogue_event is epilogue_event

    with torch.cuda.stream(reader_stream):
        assert overlap.current_stream_wait(release_handle=True) is None
    assert overlap.event is None and overlap._epilogue_event is None
    assert calls == 1


def test_no_hook():
    overlap = EventOverlap(EventHandle())
    assert overlap.wait() is None
    assert overlap._epilogue_event is None
    with overlap(release_handle=True):
        pass
    assert overlap.event is None and overlap._epilogue_event is None


if __name__ == '__main__':
    test_multistream_hook()
    test_no_hook()
    print('EventOverlap tests passed')

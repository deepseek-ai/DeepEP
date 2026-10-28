import torch

from deep_ep.utils.event import EventHandle, EventOverlap


def run_scenarios() -> None:
    for first_wait in ('wait', 'current_stream_wait', 'context_manager'):
        output = torch.zeros(1024, device='cuda')
        observed = torch.empty_like(output)
        torch.cuda.synchronize()
        handle = EventOverlap(EventHandle())
        original_event = handle.event
        producer = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        calls = []

        def epilogue():
            calls.append(1)
            torch.cuda._sleep(100_000_000)
            output.fill_(17)
            return output

        handle.register_hook_after_wait(epilogue)
        with torch.cuda.stream(producer):
            if first_wait == 'context_manager':
                with handle:
                    pass
            else:
                result = getattr(handle, first_wait)()
                if result is not output:
                    raise RuntimeError('The first wait lost the epilogue result')
        with torch.cuda.stream(consumer):
            handle.current_stream_wait()
            observed.copy_(output)
        torch.cuda.synchronize()
        if not torch.equal(observed, torch.full_like(observed, 17)):
            raise RuntimeError(f'{first_wait}: a later consumer read before the epilogue completed')
        if calls != [1] or handle.event is not original_event:
            raise RuntimeError('Repeated waits changed the hook count or event ownership')

    handle = EventOverlap(EventHandle())
    calls = []

    def retryable_epilogue():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError('retry epilogue')
        return 23

    handle.register_hook_after_wait(retryable_epilogue)
    try:
        handle.wait()
    except ValueError as error:
        if str(error) != 'retry epilogue':
            raise
    else:
        raise RuntimeError('The epilogue failure was lost')
    if handle.current_stream_wait(release_handle=True) != 23 or handle.event is not None:
        raise RuntimeError('Retry or explicit handle release failed')
    if calls != [1, 1]:
        raise RuntimeError('The failed epilogue was not retried exactly once')

    if torch.cuda.device_count() > 1:
        with torch.cuda.device(0):
            handle = EventOverlap(EventHandle())
        original_event = handle.event
        handle.register_hook_after_wait(lambda: 31)
        with torch.cuda.device(1):
            if handle.wait() != 31:
                raise RuntimeError('The cross-device wait lost the hook result')
        with torch.cuda.device(0):
            handle.wait()
        if handle.event is not original_event:
            raise RuntimeError('Cross-device completion replaced the event handle')

    output = torch.zeros(1024, device='cuda')
    observed = torch.empty_like(output)
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    capture_stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        handle = EventOverlap(EventHandle())
        handle.register_hook_after_wait(lambda: output.fill_(29))
        with torch.cuda.stream(producer):
            handle.wait()
        with torch.cuda.stream(consumer):
            handle.wait()
            observed.copy_(output)
        capture_stream.wait_stream(consumer)
    for _ in range(3):
        output.zero_()
        observed.zero_()
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(observed, torch.full_like(observed, 29)):
            raise RuntimeError('Graph replay lost epilogue completion ordering')
    print('Event epilogue waits, failure retry, release and graph replay passed')


if __name__ == '__main__':
    run_scenarios()

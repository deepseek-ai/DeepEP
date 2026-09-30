__all__ = ['EventHandle']


def __getattr__(name):
    if name == 'EventHandle':
        # Keep the forward-compatible export without loading deep_ep._C while
        # the package-level NCCL compatibility checks are still running.
        from .event import EventHandle
        globals()[name] = EventHandle
        return EventHandle
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

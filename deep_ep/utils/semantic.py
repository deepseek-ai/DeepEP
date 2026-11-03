from typing import Any, Optional
import weakref
import functools

import torch.distributed as dist

def value_or(value: Optional[Any], default: Any) -> Any:
    return default if value is None else value


def weak_lru(maxsize: Optional[int] = 128, typed: bool = False):
    """
    LRU Cache decorator that keeps a weak reference to `self`
    Useful for caching methods in classes that may cause memory leaks if `functools.lru_cache` is used directly.
    From https://stackoverflow.com/a/68052994/16569836
    """
    def wrapper(func):

        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper


def check_group_consistent_settings(group: dist.ProcessGroup, **settings: Any) -> None:
    """
    Check that settings which must be identical on every rank of `group` really are,
    raising an actionable `ValueError` on every rank at call time.

    Arguments:
        group: the communication group (the call is collective on all of its ranks).
        settings: the named settings to validate, e.g., buffer geometry values.
    """
    # NOTES: callers run this inside an already collective `__init__`, so one extra
    # host-only collective is safe and cheap, and every rank raises instead of hanging
    gathered = [None] * group.size()
    dist.all_gather_object(gathered, settings, group=group)
    local_settings = gathered[group.rank()]
    for rank_idx, other_settings in enumerate(gathered):
        if other_settings != local_settings:
            diff = ', '.join(f'`{key}`: rank {group.rank()} = {local_settings[key]!r} vs rank {rank_idx} = {other_settings[key]!r}'
                             for key in local_settings if local_settings[key] != other_settings[key])
            raise ValueError(
                f'Settings must be identical on all ranks, but rank {rank_idx} disagrees ({diff}). Hint: per-rank '
                f'values such as `num_max_tokens_per_rank` must be derived from a group-wide capacity (e.g., the '
                f'max over all ranks) rather than from the local token count, otherwise the ranks allocate '
                f'mismatched buffers, corrupting the collectives and only failing much later as an opaque CUDA '
                f'device-side assertion')

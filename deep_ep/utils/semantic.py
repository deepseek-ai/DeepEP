from typing import Any, Optional
import weakref
import functools
import threading

def value_or(value: Optional[Any], default: Any) -> Any:
    return default if value is None else value


def weak_lru(maxsize: Optional[int] = 128, typed: bool = False):
    """
    LRU cache for methods, with a separate cache and size limit per receiver.
    Weak keys allow entries to be discarded when their receiver is collected.
    As with lru_cache, arguments and results are retained until eviction; a
    cached argument or result that refers to the receiver can keep it alive.
    The original weak-reference wrapper was based on https://stackoverflow.com/a/68052994/16569836.
    """
    def wrapper(func):
        caches = weakref.WeakKeyDictionary()
        lock = threading.RLock()

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            with lock:
                cached = caches.get(self)
                if cached is None:
                    receiver = weakref.ref(self)

                    @functools.lru_cache(maxsize, typed)
                    def cached(*args, **kwargs):
                        return func(receiver(), *args, **kwargs)

                    caches[self] = cached
            return cached(*args, **kwargs)

        return inner

    return wrapper

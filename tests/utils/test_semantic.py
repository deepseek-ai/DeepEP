import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relpath: str):
    """Load a host-side module by path, avoiding `deep_ep`'s GPU initialization."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


semantic = _load_module('deep_ep_utils_semantic', 'deep_ep/utils/semantic.py')


def test_value_or():
    assert semantic.value_or(None, 5) == 5
    assert semantic.value_or(0, 5) == 0
    assert semantic.value_or(False, True) is False


def test_weak_lru_caches_per_instance():
    calls = []

    class Obj:

        @semantic.weak_lru(maxsize=None)
        def compute(self, key):
            calls.append((id(self), key))
            return key * 2

    a, b = Obj(), Obj()
    assert a.compute(3) == 6
    assert a.compute(3) == 6
    assert len(calls) == 1, 'repeated calls on the same instance must hit the cache'
    assert b.compute(3) == 6
    assert len(calls) == 2, 'a different instance must get its own cache entry'


def test_weak_lru_releases_referents():
    class Obj:

        @semantic.weak_lru(maxsize=None)
        def compute(self, key):
            return key

    obj = Obj()
    assert obj.compute(1) == 1
    # A weak reference must not keep the instance alive.
    import gc
    import weakref
    ref = weakref.ref(obj)
    del obj
    gc.collect()
    assert ref() is None


if __name__ == '__main__':
    for _name, _fn in sorted(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn()
    print('All semantic tests passed')

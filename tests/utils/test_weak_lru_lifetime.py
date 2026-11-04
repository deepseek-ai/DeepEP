import gc
import importlib.util
import unittest
import weakref
from pathlib import Path

_path = Path(__file__).resolve().parents[2] / 'deep_ep' / 'utils' / 'semantic.py'
_spec = importlib.util.spec_from_file_location('deep_ep_semantic_test', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
weak_lru = _module.weak_lru


class Value:
    pass


class Memoized:
    def __init__(self):
        self.calls = []

    @weak_lru(maxsize=None)
    def calculate(self, key):
        self.calls.append(key)
        return Value()


class BoundedMemoized(Memoized):
    @weak_lru(maxsize=2, typed=True)
    def calculate(self, key):
        self.calls.append(key)
        return Value()


class WeakLRUTest(unittest.TestCase):
    def test_entries_are_released_after_receiver_is_collected(self):
        receiver = Memoized()
        argument = Value()
        result = receiver.calculate(argument)
        receiver_ref, argument_ref, result_ref = map(weakref.ref, (receiver, argument, result))
        del receiver, argument, result
        gc.collect()
        self.assertIsNone(receiver_ref())
        self.assertIsNone(argument_ref())
        self.assertIsNone(result_ref())

    def test_hits_and_eviction(self):
        receiver = BoundedMemoized()
        first = receiver.calculate(1)
        receiver.calculate(2)
        self.assertIs(first, receiver.calculate(1))
        receiver.calculate(3)
        receiver.calculate(2)
        self.assertEqual(receiver.calls, [1, 2, 3, 2])

    def test_receiver_lifetimes_are_independent(self):
        first, second = Memoized(), Memoized()
        first.calculate(1)
        cached = second.calculate(1)
        del first
        gc.collect()
        self.assertIs(cached, second.calculate(1))
        self.assertEqual(second.calls, [1])

    def test_typed_keys_remain_distinct(self):
        receiver = BoundedMemoized()
        self.assertIsNot(receiver.calculate(1), receiver.calculate(1.0))
        self.assertEqual(len(receiver.calls), 2)

    def test_size_limit_is_per_receiver(self):
        first, second = BoundedMemoized(), BoundedMemoized()
        cached = first.calculate(1)
        first.calculate(2)
        second.calculate(3)
        second.calculate(4)
        self.assertIs(cached, first.calculate(1))
        self.assertEqual(first.calls, [1, 2])

    def test_exceptions_are_not_cached(self):
        class FailsOnce:
            def __init__(self):
                self.calls = 0

            @weak_lru(maxsize=None)
            def calculate(self):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError('retry')
                return 7

        receiver = FailsOnce()
        with self.assertRaisesRegex(ValueError, 'retry'):
            receiver.calculate()
        self.assertEqual(receiver.calculate(), 7)
        self.assertEqual(receiver.calculate(), 7)
        self.assertEqual(receiver.calls, 2)


if __name__ == '__main__':
    unittest.main()

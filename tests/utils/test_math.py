import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relpath: str):
    """Load a host-side module by path.

    Importing `deep_ep` runs `check_nccl_so()` and `init_jit()`, which need the
    compiled extension and a NCCL install. These helpers are pure host logic, so
    they are loaded directly to keep the tests runnable without a GPU build.
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


math = _load_module('deep_ep_utils_math', 'deep_ep/utils/math.py')


def test_ceil_div_and_align():
    assert math.ceil_div(0, 128) == 0
    assert math.ceil_div(1, 128) == 1
    assert math.ceil_div(128, 128) == 1
    assert math.ceil_div(129, 128) == 2
    assert math.align(0, 128) == 0
    assert math.align(1, 128) == 128
    assert math.align(128, 128) == 128
    assert math.align(129, 128) == 256


def test_safe_div():
    assert math.safe_div(6, 3) == 2
    assert math.safe_div(0, 5) == 0
    assert math.safe_div(0, 0) == 0
    try:
        math.safe_div(1, 0)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError('A non-zero numerator over zero must re-raise')


def test_calc_diff():
    x = torch.randn(4, 8)
    assert math.calc_diff(x, x.clone()) == 0


def test_inplace_unique():
    x = torch.tensor([[0, 1, 0, 2], [3, 3, -1, -1]])
    math.inplace_unique(x, num_slots=4)
    expected = [{0, 1, 2}, {3}]
    for row, kept_expected in zip(x.tolist(), expected):
        kept = [value for value in row if value >= 0]
        assert len(kept) == len(set(kept)), 'kept slots must be unique'
        assert set(kept) == kept_expected


def test_hash_tensor_accepts_any_dtype_and_layout():
    values = {
        'bool': torch.tensor([True, False, True, False]),
        'int64': torch.arange(8),
        'float16': torch.arange(8).half(),
        'float32': torch.arange(8),
        'float64': torch.arange(8).double(),
    }
    for name, t in values.items():
        assert isinstance(math.hash_tensor(t), int)
        # Stable for equal content.
        assert math.hash_tensor(t) == math.hash_tensor(t.clone())
    # Non-contiguous and empty tensors must not raise.
    math.hash_tensor(torch.arange(16).reshape(4, 4).t())
    math.hash_tensor(torch.tensor([]))


def test_hash_tensors_nested_and_none():
    a, b = torch.arange(4), torch.zeros(4)
    expected = math.hash_tensor(a) ^ math.hash_tensor(b)
    assert math.hash_tensors(a, b, None) == expected
    assert math.hash_tensors(a, [b], None) == expected


def test_count_bytes():
    a = torch.zeros(3, dtype=torch.float32)
    b = torch.zeros(2, dtype=torch.int64)
    assert math.count_bytes(a, b) == 3 * 4 + 2 * 8
    assert math.count_bytes([a, b], None) == 3 * 4 + 2 * 8
    assert math.count_bytes(None) == 0


if __name__ == '__main__':
    for _name, _fn in sorted(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn()
    print('All math tests passed')

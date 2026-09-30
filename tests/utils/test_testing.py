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


testing = _load_module('deep_ep_utils_testing', 'deep_ep/utils/testing.py')


def test_parse_num_bytes_accepts_binary_suffixes():
    assert testing.parse_num_bytes('1') == 1
    assert testing.parse_num_bytes('1B') == 1
    assert testing.parse_num_bytes('1k') == 1 << 10
    assert testing.parse_num_bytes('1K') == 1 << 10
    assert testing.parse_num_bytes('64M') == 1 << 26
    assert testing.parse_num_bytes('2G') == 1 << 31
    assert testing.parse_num_bytes('1GiB') == 1 << 30
    assert testing.parse_num_bytes(' 4 K ') == 1 << 12
    assert testing.parse_num_bytes('1.5M') == int(1.5 * (1 << 20))


def test_parse_num_bytes_rejects_bad_input():
    for bad in ('', 'abc', '-1', '0', '1e3', '1.2.3', '1G1', 'nan'):
        try:
            testing.parse_num_bytes(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad!r} should have been rejected')


if __name__ == '__main__':
    for _name, _fn in sorted(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn()
    print('All testing-utils tests passed')

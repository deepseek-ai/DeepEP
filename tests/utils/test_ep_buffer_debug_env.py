"""Regression for #718: EP_BUFFER_DEBUG=0 must disable Python-side debug output.

Host-side only. Does not import `deep_ep` (that path needs the compiled extension
and NCCL). Validates the same `int(os.environ.get(...))` idiom used at the two
call sites in `deep_ep/buffers/ep.py`, and checks those call sites still wrap
the lookup in `int()`.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path


def _ep_buffer_debug_enabled() -> bool:
    # Same expression as deep_ep/buffers/ep.py after the fix for #718.
    return bool(int(os.environ.get('EP_BUFFER_DEBUG', 0)))


def test_ep_buffer_debug_truth_table() -> None:
    cases = [
        (None, False),  # unset → default 0
        ('0', False),
        ('1', True),
        ('2', True),
    ]
    original = os.environ.get('EP_BUFFER_DEBUG')
    try:
        for value, expected in cases:
            if value is None:
                os.environ.pop('EP_BUFFER_DEBUG', None)
            else:
                os.environ['EP_BUFFER_DEBUG'] = value
            assert _ep_buffer_debug_enabled() is expected, f'EP_BUFFER_DEBUG={value!r}'
    finally:
        if original is None:
            os.environ.pop('EP_BUFFER_DEBUG', None)
        else:
            os.environ['EP_BUFFER_DEBUG'] = original


def test_ep_py_call_sites_use_int() -> None:
    """Both EP_BUFFER_DEBUG guards in ep.py must convert with int() before testing."""
    ep_py = Path(__file__).resolve().parents[2] / 'deep_ep' / 'buffers' / 'ep.py'
    tree = ast.parse(ep_py.read_text())
    matches = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match: int(os.environ.get('EP_BUFFER_DEBUG', ...))
        if not isinstance(node.func, ast.Name) or node.func.id != 'int':
            continue
        if len(node.args) != 1:
            continue
        inner = node.args[0]
        if not isinstance(inner, ast.Call):
            continue
        if not (isinstance(inner.func, ast.Attribute) and inner.func.attr == 'get'):
            continue
        if not inner.args:
            continue
        key = inner.args[0]
        if isinstance(key, ast.Constant) and key.value == 'EP_BUFFER_DEBUG':
            matches += 1
    assert matches == 2, f'expected 2 int(os.environ.get("EP_BUFFER_DEBUG", ...)) sites, found {matches}'


if __name__ == '__main__':
    test_ep_buffer_debug_truth_table()
    test_ep_py_call_sites_use_int()
    print('test_ep_buffer_debug_env: ok')

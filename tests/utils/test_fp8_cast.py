import importlib.util
import sys
from pathlib import Path

import pytest
import torch


# Importing deep_ep initializes the native extension and NCCL. This utility
# only needs PyTorch, so load it directly for standalone regression testing.
_spec = importlib.util.spec_from_file_location(
    'deep_ep_fp8_cast_math', Path(__file__).resolve().parents[2] / 'deep_ep/utils/math.py')
math = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = math
_spec.loader.exec_module(math)


def _check_cast(cast, device, num_tokens, hidden, value):
    x = torch.full((num_tokens, hidden), value, dtype=torch.bfloat16, device=device)
    quantized, scales = cast(x)
    assert quantized.shape == x.shape
    assert quantized.dtype == torch.float8_e4m3fn
    assert quantized.device == x.device
    assert quantized.is_contiguous()
    assert scales.shape == (num_tokens, (hidden + 127) // 128)
    assert scales.dtype == torch.float32
    assert scales.device == x.device
    torch.testing.assert_close(
        quantized.float(), torch.full_like(x, 448.0 if value else 0.0).float(), rtol=0, atol=0)
    torch.testing.assert_close(
        scales, torch.full_like(scales, (1.0 if value else 1e-4) / 448.0), rtol=1e-6, atol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('num_tokens', [0, 1, 3])
@pytest.mark.parametrize('hidden', [1, 127, 128, 129, 256, 257])
@pytest.mark.parametrize('value', [0.0, 1.0])
def test_per_token_cast_to_fp8_eager(device, num_tokens, hidden, value):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    # Exercise the original function body without compiling CPU operations.
    _check_cast(math.per_token_cast_to_fp8.__wrapped__, device, num_tokens, hidden, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
@pytest.mark.parametrize('num_tokens', [0, 1, 3])
@pytest.mark.parametrize('hidden', [128, 129])
@pytest.mark.parametrize('value', [0.0, 1.0])
def test_per_token_cast_to_fp8_compiled(num_tokens, hidden, value):
    _check_cast(math.per_token_cast_to_fp8, 'cuda', num_tokens, hidden, value)

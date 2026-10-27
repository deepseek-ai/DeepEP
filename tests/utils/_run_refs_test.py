"""Standalone 2-rank worker for the optional-weights dispatch test.

Run with torchrun (one process per rank); each rank loads refs.py in isolation
(stubbing the compiled ``deep_ep._C`` / NCCL init), runs dispatch with and
without ``topk_weights``, and prints a PASS/FAIL line that the pytest wrapper
parses. Kept as a separate module so spawn does not need to pickle the worker.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REFS_PATH = _REPO_ROOT / "deep_ep" / "utils" / "refs.py"

_NUM_RANKS = 2
_NUM_EXPERTS = 8
_NUM_TOPK = 4
_HIDDEN = 16
_NUM_TOKENS = 12
_NUM_MAX_TOKENS_PER_RANK = 16


def _load_refs() -> types.ModuleType:
    for name in ("deep_ep", "deep_ep.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["deep_ep"].__path__ = [str(_REPO_ROOT / "deep_ep")]  # type: ignore[attr-defined]
    envs_stub = types.ModuleType("deep_ep.utils.envs")
    envs_stub.get_global_seed = lambda: 0
    sys.modules["deep_ep.utils.envs"] = envs_stub
    math_stub = types.ModuleType("deep_ep.utils.math")
    math_stub.ceil_div = lambda x, y: (x + y - 1) // y
    sys.modules["deep_ep.utils.math"] = math_stub
    spec = importlib.util.spec_from_file_location("deep_ep.utils.refs", _REFS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deep_ep.utils.refs"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    # Supports two launch modes:
    #  - torchrun: RANK/WORLD_SIZE env vars (uses c10d store, needs libuv)
    #  - manual:   argv rank world_size + DEEPEP_DIST_INIT_FILE env (file store, no libuv)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        rank = int(sys.argv[1])
        world_size = int(sys.argv[2])
        init_file = os.environ.get("DEEPEP_DIST_INIT_FILE")
        assert init_file is not None, "manual launch needs DEEPEP_DIST_INIT_FILE env"
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, init_method=f"file://{init_file}")
    assert world_size == _NUM_RANKS, f"this script expects {_NUM_RANKS} ranks, got {world_size}"

    refs = _load_refs()
    torch.manual_seed(1234 + rank)
    device = torch.device("cpu")
    x = torch.randn(_NUM_TOKENS, _HIDDEN, dtype=torch.float32, device=device)
    scores = torch.randn(_NUM_TOKENS, _NUM_EXPERTS, dtype=torch.float32, device=device)
    topk_weights, topk_idx = torch.topk(scores, _NUM_TOPK, dim=-1, largest=True, sorted=False)
    topk_weights = topk_weights.softmax(dim=-1).to(torch.float32)

    out_with = refs.dispatch(x, topk_idx, topk_weights, _NUM_MAX_TOKENS_PER_RANK, _NUM_EXPERTS)
    recv_x_w, recv_idx_w, recv_w_w, recv_src_w, n_per_rank_w = out_with

    out_none = refs.dispatch(x, topk_idx, None, _NUM_MAX_TOKENS_PER_RANK, _NUM_EXPERTS)
    recv_x_n, recv_idx_n, recv_w_n, recv_src_n, n_per_rank_n = out_none

    failures: list[str] = []
    if recv_w_n is not None:
        failures.append(f"recv_topk_weights should be None, got {type(recv_w_n)}")
    if not torch.equal(recv_x_w, recv_x_n):
        failures.append("recv_x differs between with/without weights")
    if not torch.equal(recv_idx_w, recv_idx_n):
        failures.append("recv_topk_idx differs between with/without weights")
    if not torch.equal(recv_src_w, recv_src_n):
        failures.append("recv_src_token_idx differs between with/without weights")
    if not torch.equal(n_per_rank_w, n_per_rank_n):
        failures.append("num_recv_tokens_per_rank differs between with/without weights")
    if recv_w_w is None or recv_w_w.shape != recv_idx_w.shape:
        failures.append(f"recv_topk_weights shape mismatch with weights: {None if recv_w_w is None else recv_w_w.shape}")

    # Tuple input (the "fp8" code path, use_fp8 = isinstance(x, tuple)) + weights=None.
    # Real FP8 dtypes need a GPU; here both members are float32 so gloo can exchange them,
    # exercising the SF send/recv branch under weights=None on CPU.
    x_pair = (x, torch.rand(_NUM_TOKENS, 1, dtype=torch.float32, device=device))
    out_pair = refs.dispatch(x_pair, topk_idx, None, _NUM_MAX_TOKENS_PER_RANK, _NUM_EXPERTS)
    (recv_x_p, recv_sf_p), recv_idx_p, recv_w_p, _, n_p = out_pair
    if recv_w_p is not None:
        failures.append("tuple recv_topk_weights should be None")
    if recv_sf_p is None:
        failures.append("tuple recv_sf should not be None")
    else:
        expected = int(n_per_rank_n.sum().item())
        if recv_x_p.shape[0] != expected or recv_sf_p.shape[0] != expected:
            failures.append("tuple recv tensors shape mismatch")

    dist.barrier()
    dist.destroy_process_group()

    if failures:
        for f in failures:
            print(f"[rank {rank}] FAIL {f}", flush=True)
        print(f"RESULT rank={rank} PASS=0", flush=True)
        return 1
    print(f"RESULT rank={rank} PASS=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pure-CPU tests for ``deep_ep.utils.refs.dispatch`` optional ``topk_weights``.

The reference ``dispatch`` mirrors the real EPBuffer dispatch contract
(``deep_ep/buffers/ep.py``), where ``topk_weights: Optional[torch.Tensor] = None``
is supported and ``recv_topk_weights`` is ``None`` when no weights are provided.
The reference used to assume ``topk_weights`` was always present; these tests
lock the now-aligned optional behaviour.

Runs on CPU with the ``gloo`` backend and **no** compiled ``deep_ep._C``, NCCL,
GPU, or NPU: the worker (``_run_refs_test.py``) loads ``refs`` via ``importlib``
with the package-level C/NCCL imports stubbed. Two processes are launched with a
file-based gloo init (avoids the libuv dependency of the c10d TCP store, which
the Windows CPU torch build lacks). Each rank prints a ``RESULT rank=N PASS=M``
line that this test parses.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_WORKER = _TESTS_DIR / "_run_refs_test.py"
_NUM_RANKS = 2


def _run_rank(rank: int, init_file: str, env: dict[str, str]) -> tuple[int, str]:
    proc_env = {**os.environ, **env, "DEEPEP_DIST_INIT_FILE": init_file, "USE_LIBUV": "0"}
    proc = subprocess.Popen(
        [sys.executable, str(_WORKER), str(rank), str(_NUM_RANKS)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=proc_env,
    )
    return proc


@pytest.mark.skipif(not _WORKER.is_file(), reason="worker script missing")
def test_dispatch_optional_weights() -> None:
    with tempfile.NamedTemporaryFile(suffix="_deepinfer_refs_init", delete=False) as f:
        init_file = f.name
    os.remove(init_file)  # gloo file init recreates it; start clean
    env = {"USE_LIBUV": "0"}
    # Launch all ranks concurrently: gloo collectives block until every rank joins,
    # so serial launch would deadlock on all_to_all_single.
    procs = [_run_rank(r, init_file, env) for r in range(_NUM_RANKS)]
    outputs: list[str] = []
    for p in procs:
        try:
            out, _ = p.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            out = "[TIMEOUT]\n" + out
        outputs.append(out)
    if os.path.exists(init_file):
        os.remove(init_file)

    passes: list[int] = []
    failures: list[str] = []
    for idx, out in enumerate(outputs):
        if "[TIMEOUT]" in out:
            failures.append(f"rank {idx} timed out: {out[-500:]}")
            continue
        found = False
        for line in out.splitlines():
            m = re.search(r"RESULT rank=(\d+) PASS=(\d)", line)
            if m:
                found = True
                passes.append(int(m.group(2)))
            if "FAIL" in line:
                failures.append(line.strip())
        if not found:
            failures.append(f"rank {idx} produced no RESULT line:\n{out[-800:]}")

    assert len(passes) == _NUM_RANKS, (
        f"expected {_NUM_RANKS} RESULT lines, got {len(passes)}\nfailures:\n" + "\n".join(failures)
    )
    assert all(p == 1 for p in passes), f"some ranks did not pass: {passes}\nfailures:\n" + "\n".join(failures)


if __name__ == "__main__":
    test_dispatch_optional_weights()
    print("OK")

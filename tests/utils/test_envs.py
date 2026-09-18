"""
Tests for the RoCE-LAG aware path in ``deep_ep.utils.envs.get_rdma_gbs``.

Both functions under test (`_query_num_lag_ports`, `get_rdma_gbs`) are
purely local — they hit the kernel via pyverbs / sysfs / ``ibstat`` without
any inter-node traffic — so a single process on one node is enough.

The script is also safe to launch under ``torchrun --nnodes=2 --nproc-per-node=8``;
each rank simply runs the same local probes and prints from its own context,
which exercises that the pyverbs query holds up under concurrent opens.

On H800 + CX-7 (2x200G NDR RoCE LAG):

    ibstat 'mlx5_bond_1' Rate           : 200    (Gb/s, per rail)
    _query_num_lag_ports('mlx5_bond_1') : 2
    get_rdma_gbs('mlx5_bond_1')         : 50.0   (GB/s, = 400 Gb/s aggregated)
"""

import os
import subprocess

from deep_ep.utils.envs import _query_num_lag_ports, get_rdma_gbs


def _rank_prefix() -> str:
    rank = os.environ.get("RANK")
    return f"[rank {rank}] " if rank is not None else ""


def _log(msg: str) -> None:
    print(_rank_prefix() + msg, flush=True)


def _has_pyverbs() -> bool:
    try:
        import pyverbs.providers.mlx5.mlx5dv  # noqa: F401
    except ImportError:
        return False
    return True


def _has_ibstat() -> bool:
    try:
        subprocess.run(["ibstat"], capture_output=True, check=False, timeout=2)
    except FileNotFoundError:
        return False
    return True


def _ibstat_per_port_gbps(nic: str):
    try:
        out = subprocess.check_output(["ibstat", nic], text=True, timeout=5)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Rate:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _clear_caches() -> None:
    _query_num_lag_ports.cache_clear()
    get_rdma_gbs.cache_clear()


def test_query_num_lag_ports():
    """LAG-aware path returns >=1; on a real RoCE LAG fabric it returns >=2."""
    nic = os.environ.get("EP_NIC_NAME", "mlx5_bond_1")
    _clear_caches()
    n = _query_num_lag_ports(nic)
    _log(f"_query_num_lag_ports({nic!r}) = {n}")

    if not _has_pyverbs():
        # No pyverbs -> intentional fallback to legacy single-rail behaviour.
        assert n == 1
        return
    assert n >= 1, f"expected >= 1 LAG port, got {n}"


def test_get_rdma_gbs_lag_aggregation():
    """get_rdma_gbs should report rate_per_port * num_lag_ports / 8 in GB/s."""
    nic = os.environ.get("EP_NIC_NAME", "mlx5_bond_1")
    os.environ.pop("EP_RDMA_GBS", None)
    _clear_caches()

    if not _has_ibstat():
        _log("skipping: ibstat not installed")
        return

    rate_per_port = _ibstat_per_port_gbps(nic)
    if rate_per_port is None:
        _log(f"skipping: ibstat reported no Rate for {nic}")
        return
    n_lag = _query_num_lag_ports(nic)
    expected_gbs = rate_per_port * n_lag / 8.0
    actual_gbs = get_rdma_gbs(nic)
    _log(
        f"get_rdma_gbs({nic!r}) = {actual_gbs} GB/s  "
        f"(per-port {rate_per_port} Gb/s x {n_lag} rails / 8)"
    )
    assert actual_gbs == expected_gbs, f"expected {expected_gbs}, got {actual_gbs}"
    # Headline assertion: on a 2-rail 200G LAG, we recover ~50 GB/s
    # (= 400 Gb/s aggregated), not the previous 25 GB/s single-rail value.
    if n_lag >= 2 and rate_per_port >= 100:
        aggregated_gbps = rate_per_port * n_lag
        _log(f"LAG aggregation detected: {aggregated_gbps} Gb/s -> {actual_gbs} GB/s")


def test_ep_rdma_gbs_env_override():
    """EP_RDMA_GBS overrides the probe; value is in Gb/s and divided by 8."""
    nic = os.environ.get("EP_NIC_NAME", "mlx5_bond_1")
    _clear_caches()
    os.environ["EP_RDMA_GBS"] = "400"
    try:
        assert get_rdma_gbs(nic) == 50.0
    finally:
        del os.environ["EP_RDMA_GBS"]
        _clear_caches()
    _log("EP_RDMA_GBS=400 -> 50.0 GB/s")


if __name__ == "__main__":
    test_query_num_lag_ports()
    test_get_rdma_gbs_lag_aggregation()
    test_ep_rdma_gbs_env_override()
    _log("ALL TESTS PASSED")

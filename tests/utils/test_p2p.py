import importlib.util
from pathlib import Path

import pytest


# Load this pure control-plane module without importing ``deep_ep.__init__``,
# which requires the CUDA extension to have been built first.
_module_path = Path(__file__).parents[2] / 'deep_ep' / 'utils' / 'p2p.py'
_module_spec = importlib.util.spec_from_file_location('deep_ep_p2p', _module_path)
assert _module_spec is not None and _module_spec.loader is not None
_p2p = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_p2p)

build_local_peer_access_results = _p2p.build_local_peer_access_results
find_unsupported_peer_pairs = _p2p.find_unsupported_peer_pairs
format_p2p_preflight_error = _p2p.format_p2p_preflight_error


def test_build_local_peer_access_results_queries_only_directed_intranode_peers():
    rank_devices = [('node-a', 0), ('node-a', 1), ('node-b', 0)]
    queried_pairs = []

    def can_access_peer(device, peer_device):
        queried_pairs.append((device, peer_device))
        return True

    assert build_local_peer_access_results(0, rank_devices, can_access_peer) == [(1, True)]
    assert queried_pairs == [(0, 1)]


def test_find_unsupported_peer_pairs_reports_full_directed_matrix():
    rank_devices = [('node-a', 0), ('node-a', 1), ('node-a', 2), ('node-a', 3)]
    peer_access_results = [
        [(1, True), (2, False), (3, False)],
        [(0, True), (2, False), (3, False)],
        [(0, False), (1, False), (3, True)],
        [(0, False), (1, False), (2, True)],
    ]

    unsupported_pairs, num_required_pairs = find_unsupported_peer_pairs(rank_devices, peer_access_results)

    assert num_required_pairs == 12
    assert unsupported_pairs == [
        (0, 0, 2, 2),
        (0, 0, 3, 3),
        (1, 1, 2, 2),
        (1, 1, 3, 3),
        (2, 2, 0, 0),
        (2, 2, 1, 1),
        (3, 3, 0, 0),
        (3, 3, 1, 1),
    ]


def test_issue_584_partial_eight_gpu_topology_reports_48_of_56_pairs():
    rank_devices = [('node-a', device) for device in range(8)]
    peer_access_results = []
    for src_device in range(8):
        peer_access_results.append([
            (dst_device, src_device // 2 == dst_device // 2)
            for dst_device in range(8)
            if src_device != dst_device
        ])

    unsupported_pairs, num_required_pairs = find_unsupported_peer_pairs(rank_devices, peer_access_results)

    assert len(unsupported_pairs) == 48
    assert num_required_pairs == 56
    assert unsupported_pairs[0] == (0, 0, 2, 2)
    assert unsupported_pairs[-1] == (7, 7, 5, 5)
    assert '(48/56 directed pairs)' in format_p2p_preflight_error(unsupported_pairs, num_required_pairs)


def test_find_unsupported_peer_pairs_skips_inter_node_pairs():
    rank_devices = [('node-a', 0), ('node-a', 1), ('node-b', 0), ('node-b', 1)]
    peer_access_results = [
        [(1, True)],
        [(0, True)],
        [(3, True)],
        [(2, True)],
    ]

    unsupported_pairs, num_required_pairs = find_unsupported_peer_pairs(rank_devices, peer_access_results)

    assert unsupported_pairs == []
    assert num_required_pairs == 4


def test_find_unsupported_peer_pairs_rejects_missing_required_result():
    with pytest.raises(ValueError, match='Missing P2P result for directed pair 0->1'):
        find_unsupported_peer_pairs([('node-a', 0), ('node-a', 1)], [[], [(0, True)]])


def test_format_p2p_preflight_error_is_aggregated_and_actionable():
    message = format_p2p_preflight_error([(0, 0, 2, 2), (2, 2, 0, 0)], 6)

    assert 'DeepEP P2P preflight failed' in message
    assert '(rank 0 GPU 0 -> rank 2 GPU 2)' in message
    assert '(rank 2 GPU 2 -> rank 0 GPU 0)' in message
    assert '(2/6 directed pairs)' in message
    assert 'different MoE all-to-all backend' in message

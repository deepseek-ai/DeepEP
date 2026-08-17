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
build_physical_peer_access_checker = _p2p.build_physical_peer_access_checker
find_unsupported_peer_pairs = _p2p.find_unsupported_peer_pairs
format_p2p_preflight_error = _p2p.format_p2p_preflight_error
validate_rank_devices = _p2p.validate_rank_devices


def test_build_local_peer_access_results_queries_only_directed_intranode_peers():
    rank_devices = [('node-a', 'GPU-0'), ('node-a', 'GPU-1'), ('node-b', 'GPU-2')]
    queried_pairs = []

    def can_access_peer(device, peer_device):
        queried_pairs.append((device, peer_device))
        return True

    assert build_local_peer_access_results(0, rank_devices, can_access_peer) == [(1, True)]
    assert queried_pairs == [('GPU-0', 'GPU-1')]


def test_physical_peer_checker_maps_reordered_visible_devices_to_local_ordinals():
    visible_queries = []
    hidden_queries = []
    checker = build_physical_peer_access_checker(['GPU-B', 'GPU-A'], lambda device, peer: visible_queries.append((device, peer)) or True,
                                                 lambda device, peer: hidden_queries.append((device, peer)) or False)

    assert checker('GPU-A', 'GPU-B')
    assert visible_queries == [(1, 0)]
    assert hidden_queries == []


def test_physical_peer_checker_uses_physical_fallback_for_single_gpu_visibility():
    visible_queries = []
    hidden_queries = []
    checker = build_physical_peer_access_checker(['GPU-A'], lambda device, peer: visible_queries.append((device, peer)) or False,
                                                 lambda device, peer: hidden_queries.append((device, peer)) or True)

    assert checker('GPU-A', 'GPU-B')
    assert visible_queries == []
    assert hidden_queries == [('GPU-A', 'GPU-B')]


def test_physical_peer_checker_rejects_duplicate_visible_device_ids():
    with pytest.raises(ValueError, match='Duplicate visible physical GPU identifier GPU-A'):
        build_physical_peer_access_checker(['GPU-A', 'GPU-A'], lambda _device, _peer: True, lambda _device, _peer: True)


def test_validate_rank_devices_rejects_duplicate_physical_gpu_assignment():
    with pytest.raises(ValueError, match='Ranks 0 and 1 are assigned to the same physical GPU GPU-A'):
        validate_rank_devices([('node-a', 'GPU-A'), ('node-a', 'GPU-A')])


def test_find_unsupported_peer_pairs_reports_full_directed_matrix():
    rank_devices = [('node-a', f'GPU-{device}') for device in range(4)]
    peer_access_results = [
        [(1, True), (2, False), (3, False)],
        [(0, True), (2, False), (3, False)],
        [(0, False), (1, False), (3, True)],
        [(0, False), (1, False), (2, True)],
    ]

    unsupported_pairs, num_required_pairs = find_unsupported_peer_pairs(rank_devices, peer_access_results)

    assert num_required_pairs == 12
    assert unsupported_pairs == [
        (0, 'GPU-0', 2, 'GPU-2'),
        (0, 'GPU-0', 3, 'GPU-3'),
        (1, 'GPU-1', 2, 'GPU-2'),
        (1, 'GPU-1', 3, 'GPU-3'),
        (2, 'GPU-2', 0, 'GPU-0'),
        (2, 'GPU-2', 1, 'GPU-1'),
        (3, 'GPU-3', 0, 'GPU-0'),
        (3, 'GPU-3', 1, 'GPU-1'),
    ]


def test_issue_584_partial_eight_gpu_topology_reports_48_of_56_pairs():
    rank_devices = [('node-a', f'GPU-{device}') for device in range(8)]
    peer_access_results = []
    for src_device in range(8):
        peer_access_results.append([(dst_device, src_device // 2 == dst_device // 2) for dst_device in range(8)
                                    if src_device != dst_device])

    unsupported_pairs, num_required_pairs = find_unsupported_peer_pairs(rank_devices, peer_access_results)

    assert len(unsupported_pairs) == 48
    assert num_required_pairs == 56
    assert unsupported_pairs[0] == (0, 'GPU-0', 2, 'GPU-2')
    assert unsupported_pairs[-1] == (7, 'GPU-7', 5, 'GPU-5')
    assert '(48/56 directed pairs)' in format_p2p_preflight_error(unsupported_pairs, num_required_pairs)


def test_find_unsupported_peer_pairs_skips_inter_node_pairs():
    rank_devices = [('node-a', 'GPU-A0'), ('node-a', 'GPU-A1'), ('node-b', 'GPU-B0'), ('node-b', 'GPU-B1')]
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
        find_unsupported_peer_pairs([('node-a', 'GPU-0'), ('node-a', 'GPU-1')], [[], [(0, True)]])


def test_format_p2p_preflight_error_is_aggregated_and_actionable():
    message = format_p2p_preflight_error([(0, 'GPU-0', 2, 'GPU-2'), (2, 'GPU-2', 0, 'GPU-0')], 6)

    assert 'DeepEP P2P preflight failed' in message
    assert '(rank 0 GPU GPU-0 -> rank 2 GPU GPU-2)' in message
    assert '(rank 2 GPU GPU-2 -> rank 0 GPU GPU-0)' in message
    assert '(2/6 directed pairs)' in message
    assert 'different MoE all-to-all backend' in message


def test_format_p2p_preflight_error_caps_large_pair_lists():
    unsupported_pairs = [(0, 'GPU-0', peer, f'GPU-{peer}') for peer in range(1, 6)]
    message = format_p2p_preflight_error(unsupported_pairs, 20, max_reported_pairs=2)

    assert '(rank 0 GPU GPU-0 -> rank 1 GPU GPU-1)' in message
    assert '(rank 0 GPU GPU-0 -> rank 2 GPU GPU-2)' in message
    assert '(rank 0 GPU GPU-0 -> rank 3 GPU GPU-3)' not in message
    assert '3 additional pairs omitted' in message
    assert '(5/20 directed pairs)' in message

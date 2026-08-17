from typing import Callable, List, Sequence, Tuple


RankDevice = Tuple[str, int]
PeerAccessResult = Tuple[int, bool]
UnsupportedPeerPair = Tuple[int, int, int, int]


def build_local_peer_access_results(rank: int,
                                    rank_devices: Sequence[RankDevice],
                                    can_access_peer: Callable[[int, int], bool]) -> List[PeerAccessResult]:
    """Query directed P2P access from one rank to every intranode peer."""
    if not 0 <= rank < len(rank_devices):
        raise ValueError(f'Invalid rank {rank} for {len(rank_devices)} devices')

    local_host, local_device = rank_devices[rank]
    access_results: List[PeerAccessResult] = []
    for peer_rank, (peer_host, peer_device) in enumerate(rank_devices):
        if peer_rank != rank and peer_host == local_host:
            access_results.append((peer_rank, bool(can_access_peer(local_device, peer_device))))
    return access_results


def find_unsupported_peer_pairs(rank_devices: Sequence[RankDevice],
                                peer_access_results: Sequence[Sequence[PeerAccessResult]]) -> Tuple[List[UnsupportedPeerPair], int]:
    """Return unsupported and total directed intranode P2P pairs."""
    num_ranks = len(rank_devices)
    if len(peer_access_results) != num_ranks:
        raise ValueError(f'Expected P2P results from {num_ranks} ranks, got {len(peer_access_results)}')

    unsupported_pairs: List[UnsupportedPeerPair] = []
    num_required_pairs = 0
    for src_rank, (src_host, src_device) in enumerate(rank_devices):
        rank_results = peer_access_results[src_rank]
        access_by_dst = {}
        for dst_rank, can_access in rank_results:
            if not 0 <= dst_rank < num_ranks:
                raise ValueError(f'Invalid destination rank {dst_rank} in P2P results from rank {src_rank}')
            if dst_rank in access_by_dst:
                raise ValueError(f'Duplicate P2P result for directed pair {src_rank}->{dst_rank}')
            dst_host, _ = rank_devices[dst_rank]
            if src_rank == dst_rank or src_host != dst_host:
                raise ValueError(f'Unexpected P2P result for non-peer pair {src_rank}->{dst_rank}')
            access_by_dst[dst_rank] = can_access

        for dst_rank, (dst_host, dst_device) in enumerate(rank_devices):
            if src_rank == dst_rank or src_host != dst_host:
                continue
            num_required_pairs += 1
            if dst_rank not in access_by_dst:
                raise ValueError(f'Missing P2P result for directed pair {src_rank}->{dst_rank}')
            if not access_by_dst[dst_rank]:
                unsupported_pairs.append((src_rank, src_device, dst_rank, dst_device))
    return unsupported_pairs, num_required_pairs


def format_p2p_preflight_error(unsupported_pairs: Sequence[UnsupportedPeerPair], num_required_pairs: int) -> str:
    """Build one deterministic, actionable error for all unsupported pairs."""
    pair_list = ', '.join(
        f'(rank {src_rank} GPU {src_device} -> rank {dst_rank} GPU {dst_device})'
        for src_rank, src_device, dst_rank, dst_device in unsupported_pairs)
    return (
        'DeepEP P2P preflight failed. '
        f'Unsupported directed pairs: {pair_list} '
        f'({len(unsupported_pairs)}/{num_required_pairs} directed pairs). '
        'DeepEP requires full CUDA peer access across all participating intranode devices. '
        'Try a different MoE all-to-all backend or a topology with full P2P support.'
    )

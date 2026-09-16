from typing import Callable, Dict, List, Sequence, Tuple

PhysicalDeviceId = str
RankDevice = Tuple[str, PhysicalDeviceId]
PeerAccessResult = Tuple[int, bool]
UnsupportedPeerPair = Tuple[int, PhysicalDeviceId, int, PhysicalDeviceId]


def validate_rank_devices(rank_devices: Sequence[RankDevice]) -> None:
    """Reject duplicate rank assignments to one physical GPU."""
    device_owners: Dict[RankDevice, int] = {}
    for rank, rank_device in enumerate(rank_devices):
        node_id, device_id = rank_device
        if not node_id or not device_id:
            raise ValueError(f'Rank {rank} has an empty physical node or GPU identifier')
        if rank_device in device_owners:
            owner = device_owners[rank_device]
            raise ValueError(f'Ranks {owner} and {rank} are assigned to the same physical GPU {device_id}')
        device_owners[rank_device] = rank


def build_physical_peer_access_checker(
        visible_device_ids: Sequence[PhysicalDeviceId], can_access_visible_peer: Callable[[int, int], bool],
        can_access_hidden_peer: Callable[[PhysicalDeviceId, PhysicalDeviceId],
                                         bool]) -> Callable[[PhysicalDeviceId, PhysicalDeviceId], bool]:
    """Resolve physical GPU IDs to source-process ordinals, with a physical-ID fallback."""
    device_to_ordinal: Dict[PhysicalDeviceId, int] = {}
    for ordinal, device_id in enumerate(visible_device_ids):
        if device_id in device_to_ordinal:
            raise ValueError(f'Duplicate visible physical GPU identifier {device_id}')
        device_to_ordinal[device_id] = ordinal

    def can_access_peer(device_id: PhysicalDeviceId, peer_device_id: PhysicalDeviceId) -> bool:
        device_ordinal = device_to_ordinal.get(device_id)
        peer_device_ordinal = device_to_ordinal.get(peer_device_id)
        if device_ordinal is not None and peer_device_ordinal is not None:
            return bool(can_access_visible_peer(device_ordinal, peer_device_ordinal))
        return bool(can_access_hidden_peer(device_id, peer_device_id))

    return can_access_peer


def build_local_peer_access_results(rank: int, rank_devices: Sequence[RankDevice],
                                    can_access_peer: Callable[[PhysicalDeviceId, PhysicalDeviceId], bool]) -> List[PeerAccessResult]:
    """Query directed P2P access from one rank to every intranode peer."""
    if not 0 <= rank < len(rank_devices):
        raise ValueError(f'Invalid rank {rank} for {len(rank_devices)} devices')
    validate_rank_devices(rank_devices)

    local_node, local_device = rank_devices[rank]
    access_results: List[PeerAccessResult] = []
    for peer_rank, (peer_node, peer_device) in enumerate(rank_devices):
        if peer_rank != rank and peer_node == local_node:
            access_results.append((peer_rank, bool(can_access_peer(local_device, peer_device))))
    return access_results


def find_unsupported_peer_pairs(rank_devices: Sequence[RankDevice],
                                peer_access_results: Sequence[Sequence[PeerAccessResult]]) -> Tuple[List[UnsupportedPeerPair], int]:
    """Return unsupported and total directed intranode P2P pairs."""
    validate_rank_devices(rank_devices)
    num_ranks = len(rank_devices)
    if len(peer_access_results) != num_ranks:
        raise ValueError(f'Expected P2P results from {num_ranks} ranks, got {len(peer_access_results)}')

    unsupported_pairs: List[UnsupportedPeerPair] = []
    num_required_pairs = 0
    for src_rank, (src_node, src_device) in enumerate(rank_devices):
        rank_results = peer_access_results[src_rank]
        access_by_dst = {}
        for dst_rank, can_access in rank_results:
            if not 0 <= dst_rank < num_ranks:
                raise ValueError(f'Invalid destination rank {dst_rank} in P2P results from rank {src_rank}')
            if dst_rank in access_by_dst:
                raise ValueError(f'Duplicate P2P result for directed pair {src_rank}->{dst_rank}')
            dst_node, _ = rank_devices[dst_rank]
            if src_rank == dst_rank or src_node != dst_node:
                raise ValueError(f'Unexpected P2P result for non-peer pair {src_rank}->{dst_rank}')
            access_by_dst[dst_rank] = can_access

        for dst_rank, (dst_node, dst_device) in enumerate(rank_devices):
            if src_rank == dst_rank or src_node != dst_node:
                continue
            num_required_pairs += 1
            if dst_rank not in access_by_dst:
                raise ValueError(f'Missing P2P result for directed pair {src_rank}->{dst_rank}')
            if not access_by_dst[dst_rank]:
                unsupported_pairs.append((src_rank, src_device, dst_rank, dst_device))
    return unsupported_pairs, num_required_pairs


def format_p2p_preflight_error(unsupported_pairs: Sequence[UnsupportedPeerPair],
                               num_required_pairs: int,
                               max_reported_pairs: int = 64) -> str:
    """Build one deterministic, actionable error for all unsupported pairs."""
    if max_reported_pairs < 0:
        raise ValueError('max_reported_pairs must be non-negative')
    reported_pairs = unsupported_pairs[:max_reported_pairs]
    pair_list = ', '.join(f'(rank {src_rank} GPU {src_device} -> rank {dst_rank} GPU {dst_device})'
                          for src_rank, src_device, dst_rank, dst_device in reported_pairs)
    num_omitted_pairs = len(unsupported_pairs) - len(reported_pairs)
    if num_omitted_pairs:
        pair_list += f', ... {num_omitted_pairs} additional pairs omitted'
    return ('DeepEP P2P preflight failed. '
            f'Unsupported directed pairs: {pair_list} '
            f'({len(unsupported_pairs)}/{num_required_pairs} directed pairs). '
            'DeepEP requires full CUDA peer access across all participating intranode devices. '
            'Try a different MoE all-to-all backend or a topology with full P2P support.')

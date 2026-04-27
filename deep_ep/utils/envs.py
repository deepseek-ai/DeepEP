import functools
import inspect
import os
import random
import re
import subprocess
import torch
import torch.distributed as dist
from typing import Tuple

# noinspection PyUnresolvedReferences
import deep_ep._C as _C

from .comm import get_nccl_comm_handle

_local_rank = None
_local_seed = 0
_global_seed = 0

# Default NIC name for RDMA operations, configurable via environment variable
_DEFAULT_NIC_NAME = os.getenv('EP_NIC_NAME', 'mlx5_0')


def init_seed(global_seed: int) -> None:
    """
    Initialize the random seed for reproducibility. The local seed is derived from the global seed plus rank.

    Arguments:
        global_seed: the global random seed.
    """
    global _local_seed, _global_seed
    _local_seed = global_seed + dist.get_rank()
    _global_seed = global_seed
    torch.manual_seed(_local_seed)
    random.seed(_local_seed)


def get_local_seed() -> int:
    """
    Get the local random seed.

    Returns:
        seed: the local random seed.
    """
    return _local_seed


def get_global_seed() -> int:
    """
    Get the global random seed.

    Returns:
        seed: the global random seed.
    """
    return _global_seed


def dist_print(s: str = '', once_in_node: bool = False) -> None:
    """
    Print a message from all ranks, or only from rank 0 of each node, followed by a barrier.

    Arguments:
        s: the message to print.
        once_in_node: if `True`, only the first local rank in each node prints.
    """
    global _local_rank
    assert _local_rank is not None
    if not once_in_node or _local_rank == 0:
        print(s, flush=True)
    dist.barrier()


def init_dist(local_rank: int, num_local_ranks: int, seed: int = 0) -> Tuple[int, int, dist.ProcessGroup]:
    """
    Initialize the distributed environment with NCCL backend.

    Arguments:
        local_rank: the local rank index.
        num_local_ranks: the number of local ranks.
        seed: the global random seed.

    Returns:
        rank: the global rank index.
        world_size: the total number of ranks.
        group: the communication group.
    """
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    # Set local rank
    global _local_rank
    _local_rank = local_rank

    sig = inspect.signature(dist.init_process_group)
    params = {
        'backend': 'nccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in sig.parameters:
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    init_seed(seed)
    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


def get_physical_domain_size(group: dist.ProcessGroup) -> Tuple[int, int]:
    """
    Get the physical domain sizes (RDMA ranks and NVLink ranks).

    Arguments:
        group: the communication group.

    Returns:
        num_rdma_ranks: the number of physical RDMA ranks.
        num_nvlink_ranks: the number of physical NVLink ranks.
    """
    return _C.get_physical_domain_size(get_nccl_comm_handle(group).get())


def get_logical_domain_size(group: dist.ProcessGroup, allow_hybrid_mode: bool = True) -> Tuple[int, int]:
    """
    Get the logical domain sizes (scaleout ranks and scaleup ranks).

    Arguments:
        group: the communication group.
        allow_hybrid_mode: whether to enable hybrid mode.

    Returns:
        num_scaleout_ranks: the number of logical scaleout ranks.
        num_scaleup_ranks: the number of logical scaleup ranks.
    """
    return _C.get_logical_domain_size(get_nccl_comm_handle(group).get(), allow_hybrid_mode)


def check_nvlink_connections(group: dist.ProcessGroup) -> None:
    """
    Check NVLink connection between every pair of GPUs.

    Arguments:
        group: the communication group.
    """
    # Check NVLink connection
    # NOTES: some A100 PCIE GPUs only have pairwise NVLink connection, so that we can only use EP2
    # TODO: check all cases, all local-node GPUs in the group should be connected via NVLink
    if 'PCIE' in torch.cuda.get_device_name():
        assert group.size() <= 2, 'PCIe GPUs only have pairwise NVLink connections'

        # noinspection PyUnresolvedReferences
        import pynvml
        pynvml.nvmlInit()

        # noinspection PyTypeChecker
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7').strip(',').split(',')
        physical_device_idx = int(devices[torch.cuda.current_device()])
        physical_device_indices = [0, ] * group.size()
        dist.all_gather_object(physical_device_indices, physical_device_idx, group)

        # Check whether they are all connected via NVLink
        # Reference: https://github.com/vllm-project/vllm/blob/b8e809a057765c574726a6077fd124db5077ce1f/vllm/platforms/cuda.py#L438
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_indices]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i >= j:
                    continue
                status = pynvml.nvmlDeviceGetP2PStatus(handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                assert status == pynvml.NVML_P2P_STATUS_OK, \
                    f'GPU {physical_device_indices[i]} and GPU {physical_device_indices[j]} are not connected via NVLink'

        # Close NVML
        pynvml.nvmlShutdown()


def check_torch_deterministic() -> None:
    """
    Ensure PyTorch deterministic algorithms and fill_uninitialized_memory are not both enabled.
    When both are on, `torch.empty()` calls an initialization kernel that may overlap with communication streams,
    causing errors.
    """
    assert not (torch.are_deterministic_algorithms_enabled() and torch.utils.deterministic.fill_uninitialized_memory)


@functools.lru_cache()
def get_nvlink_gbs(factor: float = 0.9) -> float:
    """
    Get the total NVLink bandwidth in GB/s, cached.

    Arguments:
        factor: the bandwidth efficiency factor.

    Returns:
        gbs: the total NVLink bandwidth in GB/s (0 if detection fails).
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['nvidia-smi', 'nvlink', '-s'],
                                capture_output=True, text=True, check=True)
        output = result.stdout
        pattern = r'GPU \d+:.*?(?=^GPU \d+:|^$)'
        match = re.search(pattern, output, re.MULTILINE | re.DOTALL)
        assert match

        gpu_block = match.group(0)
        link_pattern = r'Link \d+:\s*([\d\.]+) GB/s'
        link_matches = re.findall(link_pattern, gpu_block)
        assert link_matches
        return sum(float(bw) for bw in link_matches) * factor
    except Exception as e:
        print(f'Failed to get NVLink connection speed: {e}')
        return 0


@functools.lru_cache()
def check_fast_rdma_atomic_support(nic_name: str = _DEFAULT_NIC_NAME) -> bool:
    """
    Check whether the NIC supports fast RDMA atomic operations (MT4131 or newer).

    Arguments:
        nic_name: the NIC device name.

    Returns:
        supported: `True` if fast RDMA atomics are supported.
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)
        output = result.stdout
        pattern = rf"CA '{nic_name}'.*?CA type:\s*(\S+)"
        match = re.search(pattern, output, re.DOTALL)
        assert match
        return match.group(1) == 'MT4131'
    except Exception:
        return False


@functools.lru_cache()
def get_rdma_gbs(nic_name: str = _DEFAULT_NIC_NAME) -> float:
    """
    Get the RDMA bandwidth in GB/s, cached.

    Arguments:
        nic_name: the NIC device name.

    Returns:
        gbs: the RDMA bandwidth in GB/s (0 if detection fails).
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)
        output = result.stdout

        pattern = rf"CA '{nic_name}'.*?Port \d+:\s*.*?Rate:\s*(\d+)"
        match = re.search(pattern, output, re.DOTALL)
        assert match
        rate = int(match.group(1))
        return rate / 8
    except Exception as e:
        print(f'Failed to get RDMA connection speed: {e}')
        return 0

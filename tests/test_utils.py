import os
import pytest
from unittest import mock
import torch
import torch.distributed as dist

from deep_ep.utils import check_nvlink_connections


@pytest.fixture
def mock_pcie_gpu():
    """Mock PCIE GPU to trigger NVLink connection check."""
    with mock.patch("torch.cuda.get_device_name", return_value="NVIDIA A100-PCIE-40GB"):
        yield


def test_nvlink_check_duplicate_physical_gpu(mock_pcie_gpu):
    """Test NVLink check works when multiple ranks share same physical GPU.
    
    Fixes #582: AssertionError 'No NVLink connection between GPU X and GPU X'
    """
    group = mock.Mock(spec=dist.ProcessGroup)
    group.size.return_value = 2

    # Mock CUDA_VISIBLE_DEVICES with duplicate ID
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,0"}):
        with mock.patch("torch.cuda.current_device", return_value=0):
            # Mock all_gather returns two same physical GPU IDs
            def mock_all_gather(output_list, _, __):
                output_list[:] = [0, 0]
            with mock.patch("torch.distributed.all_gather_object", side_effect=mock_all_gather):
                with mock.patch("pynvml.nvmlInit"):
                    with mock.patch("pynvml.nvmlDeviceGetHandleByIndex"):
                        with mock.patch("pynvml.nvmlShutdown"):
                            # Should not raise assertion for same GPU
                            check_nvlink_connections(group)

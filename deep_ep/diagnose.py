import threading
import time
import uuid
import os
import torch
import logging
import torch.distributed as dist
import numpy as np
from typing import Optional, Tuple
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


class Diagnose:
    """
    Diagnose collects statistics on the average time spent waiting to receive each token, enabling effective detection and precise localization of slow anomalies in distributed computation.
    Supports identifying:
        - 1. Slowdown caused by the destination rank.
        - 2. Slowdown caused by the source rank.
        - 3. Slowdown caused by the communication path between a specific source and destination rank.
    Maintains a statistical matrix of average receive wait times: Matrix[src_rank, dst_rank], where each row represents a source rank and each column represents a destination rank.

    Example anomaly localization:
    1. Abnormal column 3: indicates destination rank 3 is slow.
    16   13   10  117   18   18   19   12
    10   19   11  118   16   16   16   13
    18   18   12  110   18   19   18   13
    13   18   16  112   12   11   18   18
    14   20   10  114   14   16   18   16
    20   20   15  114   19   13   15   18
    18   17   19  116   10   17   17   19
    15   17   20  118   13   13   15   14

    2. Abnormal row 6: indicates source rank 6 is slow.
    16   13   10   17   18   18   19   12
    10   19   11   18   16   16   16   13
    18   18   12   10   18   19   18   13
    13   18   16   12   12   11   18   18
    14   20   10   14   14   16   18   16
    20   20   15   14   19   13   15   18
    138  137  139  137  130  137  137  139
    15   17   20   18   13   13   15   14

    3. Abnormal entry (3, 4): indicates the path from src=3 to dst=4 is slow.
    16   13   10   17   18   18   19   12
    10   19   11   18   16   16   16   13
    18   18   12   10   18   19   18   15
    13   18   16   12   125  11   18   18
    14   20   10   14   14   16   18   16
    20   20   15   14   19   13   15   18
    18   17   19   17   10   17   17   19
    15   17   20   18   13   13   15   14

    Attributes:
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        interval: diagnose interval.
        enable_ll_diagnose: enable low latency mode diagnose.
        enable_normal_diagnose: enable normal mode diagnose.
        stop_diagnose: whether to stop diagnose.
        uuid: diagnose instance id.
        logger: diagnose logger.
        ll_dispatch_wait_recv_cost_per_token_stats: a average wait time for receiving each token.
                                                    shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
        ll_combine_wait_recv_cost_per_token_stats: a average wait time for receiving each token.
                                                   shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
        normal_dispatch_wait_recv_cost_per_token_stats: a average wait time for receiving each token.
                                                        shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
        normal_combine_wait_recv_cost_per_token_stats: a average wait time for receiving each token.
                                                       shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
        gather_tensor: save all ranks diagnose stats to rank0.
        cpu_group: the communication of `gloo` group.

    Environment variables:
        DEEPEP_DIAGNOSE_ENABLE: determine diagnose enable switch from environment variable. Default 1.
        DEEPEP_DIAGNOSE_INTERVAL: controls the diagnose cycle period in seconds. Default 10.
        DEEPEP_DIAGNOSE_LOG_PATH: set the output file path for diagnose logs. Default ".".
    """

    def __init__(
            self,
            group: dist.ProcessGroup,
            interval: int = 10,
            enable_ll_diagnose: bool = True,
            enable_normal_diagnose: bool = False) -> None:
        """
        Initialize the diagnose.

        Arguments:
            group: the communication group.
            interval: diagnose interval.
            enable_ll_diagnose: enable low latency mode diagnose.
            enable_normal_diagnose: enable normal mode diagnose.

        """

        # Check parameters
        assert group.size() != 0 and interval > 0, 'invalid parameter for diagnose'

        # Determine diagnose enable switch from environment variable
        enable_diagnose = os.getenv(
            "DEEPEP_DIAGNOSE_ENABLE", "1").lower() not in (
            "0", "false", "off")

        # Initialize the diagnose
        self.rank = group.rank()
        self.group_size = group.size()
        self.group = group
        # Controls the diagnose cycle period in seconds. Default: 10
        self.interval = int(os.getenv("DEEPEP_DIAGNOSE_INTERVAL", interval))
        self.enable_ll_diagnose = enable_ll_diagnose and enable_diagnose
        self.enable_normal_diagnose = enable_normal_diagnose and enable_diagnose
        self.stop_diagnose = False

        self.logger = Diagnose._setup_logger_internal(rank=self.rank)
        # TODO: Use pinned memory optimization
        if self.enable_ll_diagnose:
            self.ll_dispatch_wait_recv_cost_per_token_stats = torch.zeros(
                (self.group_size, ), dtype=torch.int64, device='cuda')
            self.ll_combine_wait_recv_cost_per_token_stats = torch.zeros(
                (self.group_size, ), dtype=torch.int64, device='cuda')
        if self.enable_normal_diagnose:
            self.normal_dispatch_wait_recv_cost_per_token_stats = torch.zeros(
                (self.group_size, ), dtype=torch.int64, device='cuda')
            self.normal_combine_wait_recv_cost_per_token_stats = torch.zeros(
                (self.group_size, ), dtype=torch.int64, device='cuda')
        if self.enable_ll_diagnose or self.enable_normal_diagnose:
            if self.rank == 0:
                ubytes = torch.tensor(
                    list(
                        uuid.uuid4().bytes),
                    dtype=torch.uint8,
                    device='cuda')
            else:
                ubytes = torch.empty(16, dtype=torch.uint8)
            dist.broadcast(ubytes, src=0, group=group)
            self.uuid = uuid.UUID(bytes=ubytes.cpu().numpy().tobytes())
            stats_list = [
                self.ll_dispatch_wait_recv_cost_per_token_stats,
                self.ll_combine_wait_recv_cost_per_token_stats]
            # Using gloo to avoid affecting GPU communication
            self.cpu_group = dist.new_group(ranks=list(
                range(self.group_size)), backend='gloo')
            self.gather_tensor = [
                torch.zeros_like(
                    torch.stack(
                        stats_list,
                        dim=0)).cpu() for _ in range(
                    self.group_size)] if self.rank == 0 else None

    def get_stats_ll_stats_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the average wait time for receiving each token under low-latency mode for statistical purposes,
        which is useful for detecting and precisely localizing slow anomalies.
        The shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.

        Returns:
            tuple[0]: ll_dispatch_wait_recv_cost_per_token_stats.
            tuple[1]: ll_combine_wait_recv_cost_per_token_stats.
        """
        return (
            self.ll_dispatch_wait_recv_cost_per_token_stats if self.enable_ll_diagnose else None,
            self.ll_combine_wait_recv_cost_per_token_stats if self.enable_ll_diagnose else None)

    def get_stats_normal_stats_tensor(
            self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the average wait time for receiving each token under normal mode for statistical purposes,
        which is useful for detecting and precisely localizing slow anomalies.
        The shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.

        Returns:
            tuple[0]: normal_dispatch_wait_recv_cost_per_token_stats.
            tuple[1]: normal_combine_wait_recv_cost_per_token_stats.
        """
        return (
            self.normal_dispatch_wait_recv_cost_per_token_stats if self.enable_normal_diagnose else None,
            self.normal_combine_wait_recv_cost_per_token_stats if self.enable_normal_diagnose else None)

    def get_all_stats_tensor(self):
        """
        Get the all ranks stats tensor.

        Returns:
            gather_tensor: the gather_tensor for the instance.
        """
        if not self.enable_ll_diagnose and not self.enable_normal_diagnose:
            return None
        return torch.stack(self.gather_tensor,
                           dim=0).numpy() if self.rank == 0 else None

    @staticmethod
    def _setup_logger_internal(
            log_prefix="diagnose",
            when="midnight",
            interval=1,
            backupCount=2,
            rank=None):
        logger = logging.getLogger(
            f'diagnose_logger{"" if rank is None else f"_rank{rank}"}')
        logger.setLevel(logging.INFO)
        log_name = f"{log_prefix}{'' if rank is None else f'_rank{rank}'}.log"
        # Set the output file path for diagnose logs. Default ".".
        log_dir = os.environ.get('DEEPEP_DIAGNOSE_LOG_PATH', '.')
        os.makedirs(log_dir, exist_ok=True)
        file = os.path.join(log_dir, log_name)
        handler = TimedRotatingFileHandler(
            file,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding='utf-8')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(handler)
        logger.propagate = False
        return logger

    @staticmethod
    def diagnose_matrix(
        mat, thres_col=3.0, thres_row=3.0, thres_point=5.0,
        suppress_points_in_strong_rowscols=True
    ):
        """
        mat: 2D numpy array, mat[i, j] = the waiting time of src i waiting for dst j to receive the token
        Returns abnormal columns/rows/points.
        suppress_points_in_strong_rowscols: whether to remove points located in already detected abnormal rows or columns
        """
        # 1. Check for abnormal columns
        col_means = mat.mean(axis=0)
        # z_col = (col_means - col_means.mean()) / (col_means.std() + 1e-8)
        z_col = col_means / (col_means.mean() + 1e-8)
        abnormal_cols = np.where(z_col > thres_col)[0].tolist()

        # 2. Check for abnormal rows
        row_means = mat.mean(axis=1)
        # z_row = (row_means - row_means.mean()) / (row_means.std() + 1e-8)
        z_row = row_means / (row_means.mean() + 1e-8)
        abnormal_rows = np.where(z_row > thres_row)[0].tolist()

        # 3. Check for abnormal single points
        # z_all = (mat - mat.mean()) / (mat.std() + 1e-8)
        z_all = mat / (mat.mean() + 1e-8)
        # Get all positions with z-score > threshold
        abnormal_points = [
            (i, j, mat[i, j], z_all[i, j])
            for i in range(mat.shape[0])
            for j in range(mat.shape[1])
            if z_all[i, j] > thres_point
        ]
        # Optionally remove points that are in already detected abnormal rows
        # or columns
        if suppress_points_in_strong_rowscols:
            abnormal_points = [
                (i, j, v, z) for (i, j, v, z) in abnormal_points
                if i not in abnormal_rows and j not in abnormal_cols
            ]
        # 4. Return for automatic processing
        return {
            'abnormal_cols': abnormal_cols,
            'abnormal_rows': abnormal_rows,
            'abnormal_points': abnormal_points
        }

    def _diagnose_internal(self):
        def gather_and_print_diagnose_stats(
                gather_tensor,
                stats_list,
                stat_names,
                group,
                rank,
                num_ranks,
                uuid,
                logger):
            stats_tensor = torch.stack(
                stats_list, dim=0).cpu()    # (N, num_ranks)
            dist.gather(
                stats_tensor,
                gather_list=gather_tensor,
                group=group,
                dst=0)
            if rank == 0:
                arr = torch.stack(gather_tensor, dim=0).numpy()
                for i, name in enumerate(stat_names):
                    res = Diagnose.diagnose_matrix(arr[:, i, :])
                    logger.info(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Diagnose] InstanceID: {uuid} EPSize: {num_ranks}, diagnose: {res}, {name} Wait Recv Cost Per Token Matrix[src_rank, dst_rank]:")
                    for row in arr[:, i, :]:
                        logger.info(
                            f"[{' '.join(f'{val:8d}' for val in row)}]")
        while True:
            if self.stop_diagnose:
                self.logger.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Diagnose] InstanceID: {self.uuid} EPSize: {self.group_size} Rank: {self.rank}, diagnose thread daemon exit!!!")
                logging.shutdown()
                break
            time.sleep(self.interval)
            try:
                if self.enable_ll_diagnose:
                    gather_and_print_diagnose_stats(
                        self.gather_tensor, [
                            self.ll_dispatch_wait_recv_cost_per_token_stats, self.ll_combine_wait_recv_cost_per_token_stats], [
                            "Dispatch", "Combine"], self.cpu_group, self.rank, self.group_size, self.uuid, self.logger)
                if self.enable_normal_diagnose:
                    gather_and_print_diagnose_stats(
                        self.gather_tensor, [
                            self.normal_dispatch_wait_recv_cost_per_token_stats, self.normal_combine_wait_recv_cost_per_token_stats], [
                            "Dispatch", "Combine"], self.cpu_group, self.rank, self.group_size, self.uuid, self.logger)
            except Exception as e:
                self.logger.info(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Diagnose] InstanceID: {self.uuid} EPSize: {self.group_size} Rank: {self.rank} deepep/dist error: {e}, diagnose thread exit.")
                logging.shutdown()
                break

    def start_daemon(self):
        """
        Start the diagnose.

        Returns:
            thread: diagnose thread object.

        """
        if not self.enable_ll_diagnose and not self.enable_normal_diagnose:
            return None

        t = threading.Thread(target=self._diagnose_internal, daemon=True)
        t.start()
        return t

    def stop_daemon(self):
        """
        Stop the diagnose.
        """
        self.stop_diagnose = True

import importlib.util
import os
import pathlib
import socket

import torch.distributed as dist
import torch.multiprocessing as mp

# NOTES: loaded from the file path so that this test also runs on a host-only
# checkout, without the compiled CUDA extension (importing `deep_ep` triggers JIT)
_spec = importlib.util.spec_from_file_location(
    'deep_ep_semantic', pathlib.Path(__file__).resolve().parents[2] / 'deep_ep' / 'utils' / 'semantic.py')
_semantic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_semantic)
check_group_consistent_settings = _semantic.check_group_consistent_settings


def _consistent(rank, world_size):
    # The BF16 MoE-argument path: `num_bytes is None` on every rank
    return dict(num_bytes=None, num_max_tokens_per_rank=4096, hidden=4096, num_topk=8,
                use_fp8_dispatch=False, allow_hybrid_mode=False, allow_multiple_reduction=True)


def _num_max_tokens_mismatch(rank, world_size):
    # The reproducer from issue #807: every rank derives the capacity from its local token count
    settings = _consistent(rank, world_size)
    settings['num_max_tokens_per_rank'] = 4096 + rank
    return settings


def _num_bytes_mismatch(rank, world_size):
    settings = _consistent(rank, world_size)
    settings['num_bytes'] = 1024 * 1024 * (1 + rank)
    return settings


def _mixed_construction(rank, world_size):
    # One rank passes `num_bytes` while the others pass the MoE arguments
    settings = _consistent(rank, world_size)
    settings['num_bytes'] = 1024 * 1024 if rank == 0 else None
    return settings


def _flag_mismatch(rank, world_size):
    settings = _consistent(rank, world_size)
    settings['allow_hybrid_mode'] = (rank == 0)
    return settings


def _worker(rank, world_size, port, settings_fn, expect_mismatch, expect_key):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    try:
        try:
            check_group_consistent_settings(dist.group.WORLD, **settings_fn(rank, world_size))
            raised, message = False, ''
        except ValueError as e:
            raised, message = True, str(e)
        assert raised == expect_mismatch, f'rank {rank}: raised = {raised}, expect_mismatch = {expect_mismatch}'
        if raised:
            assert expect_key in message, f'rank {rank}: error does not name `{expect_key}`: {message}'
    finally:
        dist.destroy_process_group()


def _run(world_size, settings_fn, expect_mismatch, expect_key=None):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(world_size, port, settings_fn, expect_mismatch, expect_key), nprocs=world_size, join=True)


def test_group_consistent_settings():
    _run(2, _consistent, expect_mismatch=False)
    _run(3, _consistent, expect_mismatch=False)


def test_group_inconsistent_settings():
    # Issue #807: a silent geometry mismatch that used to only surface much later
    _run(2, _num_max_tokens_mismatch, expect_mismatch=True, expect_key='num_max_tokens_per_rank')
    _run(3, _num_max_tokens_mismatch, expect_mismatch=True, expect_key='num_max_tokens_per_rank')
    _run(2, _num_bytes_mismatch, expect_mismatch=True, expect_key='num_bytes')
    _run(2, _mixed_construction, expect_mismatch=True, expect_key='num_bytes')
    _run(2, _flag_mismatch, expect_mismatch=True, expect_key='allow_hybrid_mode')


if __name__ == '__main__':
    test_group_consistent_settings()
    print('>>> test_group_consistent_settings passed')
    test_group_inconsistent_settings()
    print('>>> test_group_inconsistent_settings passed')

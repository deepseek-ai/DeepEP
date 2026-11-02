import importlib.util
import os
import sys
import types

# Host-only test: stubs out `torch`, the compiled extension and `deep_ep.comm`,
# then loads `deep_ep/utils/envs.py` directly, so no GPU or build is needed
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'deep_ep')
gathered_by_rank = []


def _all_gather_object(output, obj, group):
    output[:] = gathered_by_rank


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


torch = _stub('torch', distributed=_stub('torch.distributed', all_gather_object=_all_gather_object, ProcessGroup=object))
package = _stub('deep_ep', __path__=[ROOT])
package._C = _stub('deep_ep._C')
package.comm = _stub('deep_ep.comm')
spec = importlib.util.spec_from_file_location('deep_ep.utils.envs', os.path.join(ROOT, 'utils', 'envs.py'))
_stub('deep_ep.utils', __path__=[os.path.join(ROOT, 'utils')])
envs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(envs)


class _Group:
    def size(self):
        return len(gathered_by_rank)


def _check(per_rank_settings, local_rank=0):
    gathered_by_rank[:] = per_rank_settings
    envs.check_group_consistency(_Group(), **per_rank_settings[local_rank])


def test_consistency():
    same = {'num_max_tokens_per_rank': 4096, 'hidden': 4096, 'num_topk': 8}
    _check([same, dict(same)])

    # The padding-free mistake: capacity derived from the local token count
    skewed = [{**same, 'num_max_tokens_per_rank': 4096 + rank} for rank in range(2)]
    try:
        _check(skewed)
        raise RuntimeError('mismatch was not detected')
    except ValueError as e:
        assert 'num_max_tokens_per_rank' in str(e) and '4097' in str(e), e

    # Every rank must raise, not only the one that differs
    try:
        _check(skewed, local_rank=1)
        raise RuntimeError('mismatch was not detected')
    except ValueError as e:
        assert 'num_max_tokens_per_rank' in str(e), e

    # `num_bytes` is validated too, and `None` equals `None`
    _check([{'num_bytes': None}, {'num_bytes': None}])
    try:
        _check([{'num_bytes': 2 << 20}, {'num_bytes': 4 << 20}])
        raise RuntimeError('mismatch was not detected')
    except ValueError as e:
        assert 'num_bytes' in str(e), e


if __name__ == '__main__':
    test_consistency()
    print('All group consistency tests passed')

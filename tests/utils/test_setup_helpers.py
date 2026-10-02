import importlib.util
import unittest
from pathlib import Path


def load_setup_module():
    path = Path(__file__).parents[2] / 'setup.py'
    spec = importlib.util.spec_from_file_location('deep_ep_setup_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PersistentEnvironmentTests(unittest.TestCase):
    def test_values_round_trip_through_generated_python(self):
        setup = load_setup_module()
        expected = {
            'EP_JIT_CACHE_DIR': "/tmp/o'neil\\jit-cache",
            'EP_NCCL_ROOT_DIR': '/opt/NCCL dir',
        }

        source = setup.render_default_envs(expected)
        namespace = {}
        exec(compile(source, 'envs.py', 'exec'), namespace)

        self.assertEqual(namespace['persistent_envs'], expected)

    def test_ignores_unrelated_environment_values(self):
        setup = load_setup_module()
        source = setup.render_default_envs({'UNRELATED': 'value'})
        namespace = {}
        exec(compile(source, 'envs.py', 'exec'), namespace)

        self.assertEqual(namespace['persistent_envs'], {})


if __name__ == '__main__':
    unittest.main()

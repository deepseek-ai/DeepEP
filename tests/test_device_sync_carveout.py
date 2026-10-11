"""CPU regression checks for the host preference with mocked CUDA attributes.

Compile the actual helper from the header. In particular, exercise a second
device and a reset context; a process-wide once flag would fail these cases.
GPU integration/performance tests remain separate.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DeviceSyncCarveoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("g++ is required for the CPU host-helper test")
        source = (ROOT / "csrc/hybrid_ep/backend/hybrid_ep_backend.cuh").read_text()
        helper = source.split("// Optional host preference for sharing an SM", 1)[1]
        helper = helper.split("// This kernel will update expected_rdma_flag_value", 1)[0]
        helper = helper[helper.index("template<int NUM_OF_RANKS_PER_NODE>"):]
        mock = r'''
#include <cstdio>
#include <cstdlib>
#include <cassert>
struct cudaFuncAttributes { int preferredShmemCarveout; };
enum { cudaFuncAttributePreferredSharedMemoryCarveout = 7 };
int active_device = 0;
int preferences[2] = {-1, -1};
int reads = 0, writes = 0;
template<int N> void device_sync_kernel() {}
template<class F> int cudaFuncGetAttributes(cudaFuncAttributes* attr, F function) {
  assert(function == device_sync_kernel<64>);
  ++reads; attr->preferredShmemCarveout = preferences[active_device]; return 0;
}
template<class F> int cudaFuncSetAttribute(F function, int kind, int value) {
  assert(function == device_sync_kernel<64>);
  assert(kind == cudaFuncAttributePreferredSharedMemoryCarveout);
  ++writes; preferences[active_device] = value; return 0;
}
#define CUDA_CHECK(expression) assert((expression) == 0)
'''
        main = r'''
int main(int argc, char** argv) {
  const int expected = std::atoi(argv[1]);
  configure_device_sync_shared_memory_carveout<64>();
  configure_device_sync_shared_memory_carveout<64>();
  active_device = 1;
  configure_device_sync_shared_memory_carveout<64>();
  preferences[1] = -1; // Attribute state after a context reset.
  configure_device_sync_shared_memory_carveout<64>();
  if (expected == -2) { assert(reads == 0 && writes == 0); }
  else {
    assert(reads == 4 && preferences[0] == expected && preferences[1] == expected);
    assert(writes == (expected == -1 ? 0 : 3));
  }
}
'''
        cls.temporary = tempfile.TemporaryDirectory()
        folder = Path(cls.temporary.name)
        cpp = folder / "host_helper.cpp"
        cpp.write_text(mock + helper + main)
        cls.binary = folder / "host_helper"
        subprocess.run([compiler, "-std=c++17", "-Wall", "-Werror", "-Wno-unused-parameter",
                        str(cpp), "-o", str(cls.binary)], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temporary"):
            cls.temporary.cleanup()

    def execute(self, value, expected):
        env = os.environ.copy()
        env.pop("HYBRID_EP_DEVICE_SYNC_SMEM_CARVEOUT", None)
        if value is not None:
            env["HYBRID_EP_DEVICE_SYNC_SMEM_CARVEOUT"] = value
        # Suppress core files from deliberately invalid configuration cases.
        import resource
        def disable_core():
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        return subprocess.run([str(self.binary), str(expected)], env=env,
                              capture_output=True, text=True, preexec_fn=disable_core)

    def test_default_has_no_cuda_attribute_calls(self):
        self.assertEqual(self.execute(None, -2).returncode, 0)

    def test_preferences_apply_to_each_device_and_reset_context(self):
        for value in (-1, 0, 50, 100):
            with self.subTest(value=value):
                self.assertEqual(self.execute(str(value), value).returncode, 0)

    def test_invalid_configuration_fails(self):
        for value in ("", "oops", "-2", "101", "100x", "999999999999999999999999999"):
            with self.subTest(value=value):
                result = self.execute(value, -2)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("HYBRID_EP_DEVICE_SYNC_SMEM_CARVEOUT", result.stderr)


if __name__ == "__main__":
    unittest.main()

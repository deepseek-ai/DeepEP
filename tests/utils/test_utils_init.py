import importlib.util
import sys
import types
from pathlib import Path


UTILS_DIR = Path(__file__).parents[2] / "deep_ep" / "utils"


def load_utils_package(monkeypatch):
    package_name = "deep_ep_utils_lazy_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(UTILS_DIR.parent)]
    monkeypatch.setitem(sys.modules, package_name, package)

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.utils",
        UTILS_DIR / "__init__.py",
        submodule_search_locations=[str(UTILS_DIR)],
    )
    utils = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, utils)
    spec.loader.exec_module(utils)
    return utils


def test_importing_utils_does_not_import_compiled_extension(monkeypatch):
    monkeypatch.delitem(sys.modules, "deep_ep._C", raising=False)

    utils = load_utils_package(monkeypatch)

    assert "deep_ep._C" not in sys.modules
    assert utils.__all__ == ["EventHandle"]

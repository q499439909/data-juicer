import importlib
import sys
from types import ModuleType


def test_skip_extensions_requested(monkeypatch):
    modules = ["hatchling", "hatchling.builders", "hatchling.builders.hooks", "hatchling.builders.hooks.plugin"]
    for name in modules:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    interface = ModuleType("hatchling.builders.hooks.plugin.interface")
    interface.BuildHookInterface = object
    monkeypatch.setitem(sys.modules, "hatchling.builders.hooks.plugin.interface", interface)
    sys.modules.pop("hatch_build", None)
    skip_extensions_requested = importlib.import_module("hatch_build").skip_extensions_requested

    monkeypatch.delenv("DJ_SKIP_BUILD_EXTENSIONS", raising=False)
    assert skip_extensions_requested() is False

    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("DJ_SKIP_BUILD_EXTENSIONS", value)
        assert skip_extensions_requested() is True

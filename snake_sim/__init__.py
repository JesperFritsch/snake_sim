# snake_sim/__init__.py
import sys
import types
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec

try:
    import evdev  # noqa: F401
except ImportError:
    class _MissingEvdevModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []   # marks it as a package; suppresses Python's probing

        def __getattr__(self, name):
            # dunder attrs Python probes for during import machinery — don't raise on these
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            raise ImportError(
                f"evdev is not installed (attribute access: {self.__name__}.{name}). "
                "Manual input mode requires the 'game' extra: "
                "pip install 'snake_sim[game]'"
            )

    class _EvdevFinder(MetaPathFinder, Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname == "evdev" or fullname.startswith("evdev."):
                return ModuleSpec(fullname, self)
            return None

        def create_module(self, spec):
            return _MissingEvdevModule(spec.name)

        def exec_module(self, module):
            pass  # nothing to execute

    sys.meta_path.insert(0, _EvdevFinder())
"""Select frozen packages inside an isolated rollout or inference process."""

from __future__ import annotations

import importlib
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder
from pathlib import Path


class SnapshotFinder(MetaPathFinder):
    """Resolve the two recorded package trees before editable-install finders."""

    def __init__(self, source):
        source = Path(source).resolve()
        self.packages = {
            "bvr_sim": source / "bvr_sim_source/bvr_sim",
            "bvr_behavior_prediction": source / "bvr_behavior_prediction",
        }
        for directory in self.packages.values():
            if not (directory / "__init__.py").is_file():
                raise FileNotFoundError(f"Frozen package is missing: {directory}")

    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split(".")
        package = self.packages.get(parts[0])
        if package is None:
            return None  # Third-party dependencies keep their normal import resolution.
        parent = package.joinpath(*parts[1:-1]) if len(parts) > 1 else package.parent
        spec = PathFinder.find_spec(fullname, [str(parent)])
        if spec is None:
            # Never fill gaps in the frozen tree with an installed simulator/policy.
            raise ModuleNotFoundError(
                f"{fullname} is absent from frozen package {package}", name=fullname
            )
        return spec


def activate_snapshot(source):
    """Pin imports, including cached packages and editable installation hooks.

    Call only in isolated worker/owner processes, before loading any pilot. Merely
    prepending sys.path cannot override sys.modules or a higher-priority finder.
    The verified archive itself is never edited.
    """
    finder = SnapshotFinder(source)
    sys.meta_path[:] = [item for item in sys.meta_path if not isinstance(item, SnapshotFinder)]
    sys.meta_path.insert(0, finder)
    for name in list(sys.modules):
        if name.split(".", 1)[0] in finder.packages:
            del sys.modules[name]
    importlib.invalidate_caches()

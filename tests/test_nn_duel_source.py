"""Frozen rollout imports must win over installed and already imported packages."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


@pytest.mark.parametrize("preload", [False, True])
@pytest.mark.parametrize("editable_hook", [False, True])
@pytest.mark.parametrize("speed_override", [False, True, None])
def test_worker_uses_snapshot_despite_installed_packages(
    tmp_path, preload, editable_hook, speed_override
):
    work = tmp_path / "work"
    source = work / "tactical_snapshot/source"
    installed = tmp_path / "installed"
    for root, marker in ((source, "frozen"), (installed, "installed")):
        files = {
            "bvr_sim_source/bvr_sim/__init__.py": f"marker = {marker!r}\n",
            "bvr_sim_source/bvr_sim/bvr_env.py": (
                f"marker = {marker!r}\nBYPASS_RL_THROTTLE_CONTROL = True\n"
            ),
            "bvr_sim_source/bvr_sim/installed_only.py": "",
            "bvr_behavior_prediction/__init__.py": f"marker = {marker!r}\n",
            "bvr_behavior_prediction/rl/__init__.py": "",
            "bvr_behavior_prediction/rl/pilot.py": f"marker = {marker!r}\n",
            "bvr_behavior_prediction/rl/multi_pilot.py": "def load_trained_pilot(): pass\n",
        }
        for relative, contents in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
    (source / "bvr_sim_source/bvr_sim/installed_only.py").unlink()
    config = {"pilots": {}}
    if speed_override is not None:
        config["native_throttle_override"] = speed_override
    (work / "config.json").write_text(json.dumps(config), encoding="utf-8")
    # Run in a child: initialization deliberately replaces these package trees.
    code = """
import importlib.abc
import importlib.machinery
import json
import os
import sys
import types
from pathlib import Path

scripts, work, installed = map(Path, sys.argv[1:4])
preload, editable_hook = map(int, sys.argv[4:])
sys.path[:0] = [str(scripts), str(installed), str(installed / "bvr_sim_source")]

class InstalledFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split(".")
        if parts[0] not in ("bvr_sim", "bvr_behavior_prediction"):
            return None
        root = installed / "bvr_sim_source" if parts[0] == "bvr_sim" else installed
        return importlib.machinery.PathFinder.find_spec(
            fullname, [str(root.joinpath(*parts[:-1]))])

if editable_hook:
    sys.meta_path.insert(0, InstalledFinder())
if preload:
    import bvr_sim.bvr_env
    import bvr_behavior_prediction.rl.pilot
    assert bvr_sim.bvr_env.marker == "installed"
    assert bvr_behavior_prediction.rl.pilot.marker == "installed"

# Exercise the real initializer without needing a trained model or Torch threads.
sys.modules["torch"] = types.SimpleNamespace(
    set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None)
import generate_nn_duel_corpus as duel
duel.initialize_worker(work)
import bvr_sim.bvr_env as simulator
import bvr_behavior_prediction.rl.pilot as pilot
source = work / "tactical_snapshot/source"
assert simulator.marker == pilot.marker == "frozen"
config = json.loads((work / "config.json").read_text())
assert simulator.BYPASS_RL_THROTTLE_CONTROL is config.get("native_throttle_override", True)
assert Path(simulator.__file__).is_relative_to(source)
assert Path(pilot.__file__).is_relative_to(source)
assert Path(os.environ["BVR_SIM_RESOURCE_DIR"]) == source / "bvr_sim_source/bvr_sim/resources"
try:
    import bvr_sim.installed_only
except ModuleNotFoundError as error:
    assert error.name == "bvr_sim.installed_only"
else:
    raise AssertionError("Missing frozen modules must not fall back to the installation")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(SCRIPTS),
            str(work),
            str(installed),
            str(int(preload)),
            str(int(editable_hook)),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

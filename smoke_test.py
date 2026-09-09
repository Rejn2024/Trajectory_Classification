from pathlib import Path
import json
import commentjson

import bvr_sim
from bvr_sim import BVR3DEnv, BVR3DEnvCpp


PROJECT_ROOT = Path(__file__).resolve().parent
BVR_ROOT = PROJECT_ROOT / "bvr_sim_source"
LOG_ROOT = PROJECT_ROOT / "smoke_logs"

(LOG_ROOT / "python").mkdir(parents=True, exist_ok=True)
(LOG_ROOT / "cpp").mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("BVR SIM INSTALLATION")
print("=" * 70)

loaded_from = Path(bvr_sim.__file__).resolve()

print("bvr_sim loaded from:")
print(loaded_from)

if not loaded_from.is_relative_to(BVR_ROOT):
    raise RuntimeError(
        f"Wrong BVR Sim installation loaded.\n"
        f"Expected source under: {BVR_ROOT}\n"
        f"Actually loaded:       {loaded_from}"
    )

print("Editable installation location: OK")


print("\n" + "=" * 70)
print("PYTHON BACKEND")
print("=" * 70)

python_config_path = (
    BVR_ROOT / "scripts" / "tests" / "demo_config.json"
)

with python_config_path.open("r", encoding="utf-8") as f:
    python_config = json.load(f)

python_env = BVR3DEnv(
    python_config,
    logdir=str(LOG_ROOT / "python"),
)

obs, info = python_env.reset()

print("Environment successfully created")
print("Observation type:", type(obs))
print("Initial info:", info)

steps = 0

for steps in range(1, 101):
    obs, reward, dones, info = python_env.step({})

    if info.get("episode_done", False):
        break

print("Python backend completed steps:", steps)
print("Episode done:", info.get("episode_done", False))
print("PYTHON BACKEND: PASS")


print("\n" + "=" * 70)
print("C++ BACKEND")
print("=" * 70)

cpp_config_path = (
    BVR_ROOT / "scripts" / "tests" / "demo_config_cpp.jsonc"
)

with cpp_config_path.open("r", encoding="utf-8") as f:
    cpp_config = commentjson.load(f)

cpp_env = BVR3DEnvCpp(
    cpp_config,
    rl_index=[0],
    log_file_path=str(LOG_ROOT / "cpp" / "bvr_sim.log"),
)

obs, info = cpp_env.reset()

print("Environment successfully created")
print("Observation type:", type(obs))
print("Initial info:", info)

steps = 0

for steps in range(1, 101):
    action = cpp_env.action_space.sample().reshape(1, -1)

    obs, reward, dones, info = cpp_env.step(action)

    if info.get("episode_done", False):
        break

print("C++ backend completed steps:", steps)
print("Episode done:", info.get("episode_done", False))
print("C++ BACKEND: PASS")


print("\n" + "=" * 70)
print("ALL BVR SIM SMOKE TESTS PASSED")
print("=" * 70)
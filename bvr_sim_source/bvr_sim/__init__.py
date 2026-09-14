"""
BVR Sim - 3D Beyond Visual Range Air Combat Environment

Clean, modular 3D BVR environment with:
- Full 3D aircraft physics with altitude dynamics
- Realistic missile guidance (AIM-120C parameters)
- Pluggable observation space system
- ACMI Tacview rendering
- Modular reward components
"""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.4.8"

__all__ = ["BVR3DEnv", "BVR3DEnvCpp", "all_envs", "make_bvr3d_env"]

from .resource_paths import configure_runtime_environment

configure_runtime_environment()

if TYPE_CHECKING:
    from .bvr_env import BVR3DEnv, make_bvr3d_env
    from .bvr_env_cpp import BVR3DEnvCpp


def __getattr__(name: str):
    """Load simulator backends only when their public classes are requested.

    Lightweight subpackages such as :mod:`bvr_sim.agents` do not require the
    simulator's optional runtime or compiled extension.  Keeping those imports
    lazy lets tooling and applications use them independently.
    """
    if name == "all_envs":
        value = [__getattr__("BVR3DEnv"), __getattr__("BVR3DEnvCpp")]
        globals()[name] = value
        return value

    module_by_name = {
        "BVR3DEnv": ".bvr_env",
        "make_bvr3d_env": ".bvr_env",
        "BVR3DEnvCpp": ".bvr_env_cpp",
    }
    try:
        module_name = module_by_name[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


"""
Wrappers for RL frameworks
"""
# for MARLBenchmark (on-policy, off-policy):
# compatible fork: https://github.com/lizi-Margin/off-policy
# original: https://github.com/marlbenchmark
# from rl_envs.env_marlbenchmark import BVRSimEnvMarlBenchmark, MultiAgentEnvWrapper

# for HARL
# compatible fork: https://github.com/lizi-Margin/HARL4BVRSim
# original: https://github.com/PKU-MARL/HARL
# from rl_envs.env_harl import BVRSimEnv, HARLLogger

# for UHRL: https://github.com/lizi-Margin/UHRL
# from rl_envs.env_wrapper import BVR3DWrapper, make_env, ScenarioConfig

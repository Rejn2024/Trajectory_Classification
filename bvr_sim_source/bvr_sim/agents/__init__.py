"""Agent interfaces.

The tactical skill catalogue has no LLM dependencies, so keep the strategic
agent imports lazy.  This lets callers use ``SkillManager`` without installing
the optional API clients used by ``StrategicLLM``.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["Agent", "SkillManager", "SkillSelection", "StrategicLLM", "TacticalSkill"]

if TYPE_CHECKING:
    from .agent import Agent
    from .bvr_strategist import SkillSelection, StrategicLLM
    from .skill_manager import SkillManager, TacticalSkill


def __getattr__(name: str):
    """Load public agent classes only when a caller requests them."""
    module_by_name = {
        "Agent": ".agent",
        "SkillManager": ".skill_manager",
        "TacticalSkill": ".skill_manager",
        "StrategicLLM": ".bvr_strategist",
        "SkillSelection": ".bvr_strategist",
    }
    try:
        module_name = module_by_name[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

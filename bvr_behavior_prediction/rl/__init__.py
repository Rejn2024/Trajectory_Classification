"""Reinforcement-learning pilot for skill-level 1-v-1 control.

Heavy optional dependencies (PyTorch and MLflow) are imported only by the training
entry point, so scenario and reward configuration remains usable in lightweight
dataset installations.
"""

from .config import PilotTrainingConfig, RewardWeights
from .reward import CombatReward
from .scenarios import EngagementScenario, ScenarioSampler

__all__ = [
    "CombatReward",
    "EngagementScenario",
    "PilotTrainingConfig",
    "RewardWeights",
    "ScenarioSampler",
]

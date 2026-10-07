"""Reinforcement-learning pilot for skill-level 1-v-1 control.

Heavy optional dependencies (PyTorch and MLflow) are imported only by the training
entry point, so scenario and reward configuration remains usable in lightweight
dataset installations.
"""

from .config import PilotTrainingConfig, RewardWeights
from .dogfight_reward import (
    DogfightReward,
    DogfightWeights,
    HybridDogfightReward,
    dogfight_population_weights,
    dogfight_reward_definition,
    hybrid_dogfight_reward_definition,
)
from .multi_pilot import PilotSpec, load_trained_pilot, train_pilots
from .reward import CombatReward, RewardDefinition, combat_reward_definition
from .reward_population import combat_population_weights
from .scenarios import EngagementScenario, ScenarioSampler

__all__ = [
    "CombatReward",
    "DogfightReward",
    "DogfightWeights",
    "EngagementScenario",
    "HybridDogfightReward",
    "PilotSpec",
    "PilotTrainingConfig",
    "RewardDefinition",
    "RewardWeights",
    "ScenarioSampler",
    "combat_population_weights",
    "combat_reward_definition",
    "dogfight_population_weights",
    "dogfight_reward_definition",
    "hybrid_dogfight_reward_definition",
    "load_trained_pilot",
    "train_pilots",
]

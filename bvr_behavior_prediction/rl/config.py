"""Configuration for the short-horizon hybrid-action pilot."""

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RewardWeights:
    """Event rewards, ordered by the desired tactical priority."""

    crashed: float = -100.0
    shot_down: float = -150.0
    ground_clearance: float = 0.02
    target_lock_acquired: float = 2.0
    fired_with_lock: float = 8.0
    incoming_missile_avoided: float = 20.0
    opponent_destroyed: float = 250.0
    fired_without_lock: float = -4.0


@dataclass(frozen=True)
class PilotTrainingConfig:
    """All frequently tuned experiment controls in one serialisable object."""

    planning_horizon_s: float = 1.0
    episode_duration_s: float = 90.0
    scenarios_per_epoch: int = 3
    epochs: int = 100
    simulation_dt_s: float = 0.1
    seed: int = 7
    hidden_size: int = 256
    history_duration_s: float = 2.0
    sample_interval_s: float = 0.1
    transformer_heads: int = 4
    transformer_layers: int = 2
    learning_rate: float = 3e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 256
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    gradient_clip: float = 0.5
    checkpoint_interval: int = 10
    diagnostic_interval: int = 1
    output_dir: Path = Path("artifacts/rl_pilot")
    mlflow_experiment: str = "jsbsim-blue-1v1-pilot"
    mlflow_tracking_uri: str = "sqlite:///artifacts/mlflow.db"
    reward: RewardWeights = field(default_factory=RewardWeights)

    def __post_init__(self):
        if self.planning_horizon_s <= 0:
            raise ValueError("planning_horizon_s must be positive")
        if not 0 < self.episode_duration_s <= 120:
            raise ValueError("episode_duration_s must be in (0, 120]")
        if self.scenarios_per_epoch < 3:
            raise ValueError("scenarios_per_epoch must be at least 3")
        if self.epochs < 1 or self.simulation_dt_s <= 0:
            raise ValueError("epochs and simulation_dt_s must be positive")
        if self.history_duration_s <= 0 or self.sample_interval_s <= 0:
            raise ValueError("history duration and sample interval must be positive")
        if not (self.history_duration_s / self.sample_interval_s).is_integer():
            raise ValueError("history_duration_s must be divisible by sample_interval_s")
        if self.transformer_heads < 1 or self.transformer_layers < 1:
            raise ValueError("transformer heads and layers must be positive")
        if self.hidden_size % self.transformer_heads:
            raise ValueError("hidden_size must be divisible by transformer_heads")

    @property
    def decisions_per_episode(self) -> int:
        return int(self.episode_duration_s / self.planning_horizon_s)

    @property
    def history_steps(self) -> int:
        return int(round(self.history_duration_s / self.sample_interval_s))

    def as_dict(self) -> dict:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        return result

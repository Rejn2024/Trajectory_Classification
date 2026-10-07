"""Independent evade, pursue, and eliminate preferences for simulated duels."""

import math
from dataclasses import asdict, dataclass
from functools import partial

from .reward import RewardDefinition


@dataclass(frozen=True)
class DogfightWeights:
    """Relative preferences, normalized to a fixed total rather than scaled together."""

    evade: float = 1 / 3
    pursue: float = 1 / 3
    eliminate: float = 1 / 3

    def __post_init__(self):
        values = (self.evade, self.pursue, self.eliminate)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("dogfight weights must be finite and non-negative")
        total = sum(values)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("at least one dogfight weight must be positive")
        for name, value in zip(("evade", "pursue", "eliminate"), values):
            object.__setattr__(self, name, value / total)


def dogfight_population_weights(count: int, minimum_weight: float = 0.1):
    """Spread any size roster across the preference triangle, including its centre.

    A deterministic triangular grid and farthest-point selection cover trade-offs.
    The default floor keeps all three objectives relevant, even at the extremes.
    These are experiment preferences, not asserted behavioural classifications.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if not math.isfinite(minimum_weight) or not 0 <= minimum_weight < 1 / 3:
        raise ValueError("minimum_weight must be in [0, 1/3)")
    resolution = max(1, math.ceil((math.sqrt(8 * count + 1) - 3) / 2))
    candidates = [
        (i / resolution, j / resolution, (resolution - i - j) / resolution)
        for i in range(resolution + 1)
        for j in range(resolution + 1 - i)
    ]
    selected = [(1 / 3, 1 / 3, 1 / 3)]
    distances = [float("inf")] * len(candidates)
    while len(selected) < count:
        latest = selected[-1]
        distances = [
            min(previous, sum((a - b) ** 2 for a, b in zip(candidate, latest)))
            for candidate, previous in zip(candidates, distances)
        ]
        index = max(range(len(candidates)), key=distances.__getitem__)
        selected.append(candidates[index])
        distances[index] = -1.0
    scale = 1 - 3 * minimum_weight
    return [DogfightWeights(*(minimum_weight + scale * value for value in row)) for row in selected]


class DogfightReward:
    """Bounded tactical objectives plus the same loss penalty for every pilot.

    Evade: unique defeated incoming missiles (capped), minus time under threat.
    Pursue: new best progress towards a nearby target ahead of the flight path.
    Eliminate: one confirmed missile kill, excluding unrelated opponent crashes.

    No reward is paid for selecting a skill, firing, acquiring a lock, altitude,
    or simply staying airborne. The shared loss penalty discourages suicide.
    """

    def __init__(
        self, weights=None, episode_duration_s=120.0, evasion_cap=4,
        pursuit_range_m=30_000.0, loss_penalty=1.0, reward_scale=250.0,
    ):
        self.weights = weights or DogfightWeights()
        for name, value in (
            ("episode_duration_s", episode_duration_s), ("pursuit_range_m", pursuit_range_m),
            ("reward_scale", reward_scale),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(evasion_cap) is not int or evasion_cap < 1:
            raise ValueError("evasion_cap must be a positive integer")
        if not math.isfinite(loss_penalty) or loss_penalty < 0:
            raise ValueError("loss_penalty must be finite and non-negative")
        self.episode_duration_s = episode_duration_s
        self.evasion_cap = evasion_cap
        self.pursuit_range_m = pursuit_range_m
        self.loss_penalty = loss_penalty
        self.reward_scale = reward_scale
        self.reset()

    def reset(self):
        self._best_pursuit = None
        self._evaded = 0
        self._exposure = 0.0
        self._eliminated = False
        self._lost = False

    def _pursuit_quality(self, distance, alignment):
        return max(0.0, min(1.0, float(alignment))) * math.exp(
            -max(0.0, float(distance)) / self.pursuit_range_m
        )

    def __call__(self, info):
        required = (
            "blue_alive", "target_range_m", "target_alignment",
            "initial_target_range_m", "initial_target_alignment",
            "missiles_avoided_total", "threatened_time_s", "opponent_eliminated",
        )
        missing = [key for key in required if key not in info]
        if missing:
            raise ValueError(f"dogfight reward requires simulator signals: {', '.join(missing)}")
        alive = bool(info["blue_alive"])
        if self._best_pursuit is None:
            self._best_pursuit = self._pursuit_quality(
                info["initial_target_range_m"], info["initial_target_alignment"]
            )
        quality = self._pursuit_quality(info["target_range_m"], info["target_alignment"])
        # A retreat-and-return loop cannot repeatedly collect the same progress.
        pursue = max(0.0, quality - self._best_pursuit) if alive else 0.0
        self._best_pursuit = max(self._best_pursuit, quality)
        evaded = min(self.evasion_cap, int(info["missiles_avoided_total"]))
        exposure = min(1.0, float(info["threatened_time_s"]) / self.episode_duration_s)
        evade = max(0, evaded - self._evaded) / self.evasion_cap - max(0.0, exposure - self._exposure)
        self._evaded, self._exposure = max(self._evaded, evaded), max(self._exposure, exposure)
        eliminated = bool(info["opponent_eliminated"])
        eliminate = float(eliminated and not self._eliminated)
        loss = float(not alive and not self._lost)
        self._eliminated |= eliminated
        self._lost |= not alive
        components = {
            "evade": self.reward_scale * self.weights.evade * evade,
            "pursue": self.reward_scale * self.weights.pursue * pursue,
            "eliminate": self.reward_scale * self.weights.eliminate * eliminate,
            "loss": -self.reward_scale * self.loss_penalty * loss,
        }
        return sum(components.values()), components


class HybridDogfightReward(DogfightReward):
    """Style preferences with shared combat outcomes and bounded learning guidance.

    The first lock and first locked launch pay once. Missile support pays only
    while a live friendly missile still needs the aircraft's target lock, up to
    a fixed time budget. These are heuristic bonuses, not policy-invariant shaping.
    """

    def __init__(
        self, weights=None, elimination_bonus=100.0, lock_bonus=2.0,
        launch_bonus=8.0, support_bonus=5.0, support_budget_s=20.0, **settings,
    ):
        for name, value in (
            ("elimination_bonus", elimination_bonus), ("lock_bonus", lock_bonus),
            ("launch_bonus", launch_bonus), ("support_bonus", support_bonus),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, value)
        if not math.isfinite(support_budget_s) or support_budget_s <= 0:
            raise ValueError("support_budget_s must be finite and positive")
        self.support_budget_s = support_budget_s
        # At the default scale this matches the original shot-down cost (-150).
        settings.setdefault("loss_penalty", 0.6)
        super().__init__(weights=weights, **settings)

    def reset(self):
        super().reset()
        self._lock_paid = False
        self._launch_paid = False
        self._support_paid = 0.0

    def _pursuit_quality(self, distance, alignment):
        # Once within the nominal engagement range, additional closing alone
        # does not earn progress. Alignment can still improve the position.
        return super()._pursuit_quality(max(0.0, distance - self.pursuit_range_m), alignment)

    def __call__(self, info):
        if "missile_support_time_s" not in info:
            raise ValueError("hybrid reward requires simulator signal: missile_support_time_s")
        previously_eliminated = self._eliminated
        _, components = super().__call__(info)
        counts = info.get("reward_event_counts", {})
        acquired = counts.get("target_locked", int(bool(info.get("target_locked", False)))) > 0
        launched = counts.get("fired_with_lock", int(bool(info.get("fired_with_lock", False)))) > 0
        support = min(1.0, max(0.0, float(info["missile_support_time_s"])) / self.support_budget_s)
        components.update({
            "combat_elimination": self.elimination_bonus * (
                self._eliminated and not previously_eliminated
            ),
            "lock_acquired": self.lock_bonus * (acquired and not self._lock_paid),
            "locked_launch": self.launch_bonus * (launched and not self._launch_paid),
            "missile_support": self.support_bonus * max(0.0, support - self._support_paid),
        })
        self._lock_paid |= acquired
        self._launch_paid |= launched
        self._support_paid = max(self._support_paid, support)
        return sum(components.values()), components


def _style_parameters(reward):
    return {
        "weights": asdict(reward.weights),
        "episode_duration_s": reward.episode_duration_s,
        "evasion_cap": reward.evasion_cap,
        "pursuit_range_m": reward.pursuit_range_m,
        "loss_penalty": reward.loss_penalty,
        "reward_scale": reward.reward_scale,
    }


def dogfight_reward_definition(name="dogfight", weights=None, **settings):
    """Persist the original three-objective recipe without shared combat guidance."""
    reward = DogfightReward(weights=weights, **settings)
    parameters = _style_parameters(reward)
    return RewardDefinition(
        name, partial(DogfightReward, weights=reward.weights, **{
            key: value for key, value in parameters.items() if key != "weights"
        }), parameters=parameters, version="1",
    )


def hybrid_dogfight_reward_definition(name="hybrid_dogfight", weights=None, **settings):
    """Create the notebook 06 recipe; the original combat and style recipes remain available."""
    reward = HybridDogfightReward(weights=weights, **settings)
    parameters = {
        **_style_parameters(reward),
        **{key: getattr(reward, key) for key in (
            "elimination_bonus", "lock_bonus", "launch_bonus", "support_bonus", "support_budget_s",
        )},
    }
    return RewardDefinition(
        name, partial(HybridDogfightReward, weights=reward.weights, **{
            key: value for key, value in parameters.items() if key != "weights"
        }), parameters=parameters, version="1",
    )

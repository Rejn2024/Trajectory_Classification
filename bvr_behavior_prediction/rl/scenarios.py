"""Seeded, safe initial conditions for blue-versus-red engagements."""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EngagementScenario:
    range_m: float
    bearing_deg: float
    blue_altitude_m: float
    red_altitude_m: float
    blue_heading_deg: float
    red_heading_deg: float
    blue_speed_mps: float = 250.0
    red_speed_mps: float = 250.0
    red_policy: str = "constant_course"

    def as_dict(self) -> dict:
        return asdict(self)


class ScenarioSampler:
    """Generate diverse geometry without unsafe first-horizon flight paths."""

    def __init__(
        self,
        seed=7,
        min_altitude_m=2500.0,
        max_altitude_m=11000.0,
        ranges_m=(18000.0, 30000.0, 45000.0),
        bearings_deg=(-35.0, 0.0, 35.0),
        altitude_offsets_m=(-1500.0, 0.0, 1500.0),
    ):
        self.rng = np.random.default_rng(seed)
        self.min_altitude_m = min_altitude_m
        self.max_altitude_m = max_altitude_m
        self.ranges_m = tuple(ranges_m)
        self.bearings_deg = tuple(bearings_deg)
        self.altitude_offsets_m = tuple(altitude_offsets_m)

    def sample_batch(self, count: int) -> list[EngagementScenario]:
        if count < 3:
            raise ValueError("A training iteration requires at least three set-ups")
        result = []
        # Cycling before jitter guarantees range, bearing and height variation even
        # for the minimum batch, while remaining exactly reproducible by seed.
        permutation = self.rng.permutation(count)
        for i in range(count):
            j = int(permutation[i])
            distance = self.ranges_m[j % len(self.ranges_m)]
            bearing = self.bearings_deg[j % len(self.bearings_deg)]
            offset = self.altitude_offsets_m[j % len(self.altitude_offsets_m)]
            blue_altitude = float(self.rng.uniform(5000.0, 8000.0))
            red_altitude = float(
                np.clip(blue_altitude + offset, self.min_altitude_m, self.max_altitude_m)
            )
            # Both start level; red's constant course is always horizontal and thus
            # cannot meet the ground or an unreasonable altitude.
            blue_heading = (bearing + self.rng.uniform(-12.0, 12.0)) % 360.0
            red_heading = (bearing + 180.0) % 360.0
            scenario = EngagementScenario(
                distance, bearing, blue_altitude, red_altitude, blue_heading, red_heading
            )
            self._validate(scenario, planning_horizon_s=1.0)
            result.append(scenario)
        return result

    def _validate(self, scenario: EngagementScenario, planning_horizon_s: float) -> None:
        if not self.min_altitude_m <= scenario.blue_altitude_m <= self.max_altitude_m:
            raise ValueError("blue altitude is unsafe")
        if not self.min_altitude_m <= scenario.red_altitude_m <= self.max_altitude_m:
            raise ValueError("red altitude is unsafe")
        closing_bound = (scenario.blue_speed_mps + scenario.red_speed_mps) * planning_horizon_s
        if scenario.range_m <= closing_bound + 1000.0 or not math.isfinite(scenario.range_m):
            raise ValueError("aircraft could collide during the first planning horizon")

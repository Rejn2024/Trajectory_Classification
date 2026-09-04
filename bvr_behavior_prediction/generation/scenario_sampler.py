from dataclasses import dataclass
import random


@dataclass(frozen=True)
class InitialScenario:
    range_nm: float; altitude_difference_m: float; speed_difference_ms: float
    heading_difference_deg: float; lateral_offset_nm: float; scenario_bin: str


class ScenarioSampler:
    RANGES = {"controlled": ((22, 28), (-500, 500), (-20, 20), (160, 200)),
              "broad": ((10, 50), (-4000, 4000), (-100, 100), (20, 200)),
              "edge": ((5, 60), (-7000, 7000), (-180, 180), (70, 110))}
    def __init__(self, stage="controlled"):
        if stage not in self.RANGES: raise ValueError(f"Unknown sampling stage: {stage}")
        self.stage = stage
    def sample(self, rng: random.Random):
        distance, altitude, speed, heading = self.RANGES[self.stage]
        return InitialScenario(rng.uniform(*distance), rng.uniform(*altitude), rng.uniform(*speed),
                               rng.uniform(*heading), rng.uniform(-5, 5), self.stage)


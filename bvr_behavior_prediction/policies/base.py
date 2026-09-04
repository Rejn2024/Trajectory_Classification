from abc import ABC, abstractmethod
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class NativeAction:
    heading: int = 7
    altitude: int = 7
    speed: int = 4
    fire: int = 0

    def as_tuple(self): return self.heading, self.altitude, self.speed, self.fire

    def __post_init__(self):
        if not (0 <= self.heading < 15 and 0 <= self.altitude < 15 and
                0 <= self.speed < 9 and 0 <= self.fire < 2):
            raise ValueError("Invalid BVR Sim native action")


class TacticalPolicy(ABC):
    label = "MAINTAIN"
    def reset(self, rng: random.Random) -> None: self.rng = rng
    @abstractmethod
    def act(self, own_state: dict, opponent_state: dict, time_s: float) -> NativeAction: ...


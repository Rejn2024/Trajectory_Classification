from dataclasses import dataclass
import random


@dataclass(frozen=True)
class WindowIndex:
    episode_id: str
    end: int
    target: int


def build_window_indices(rows_by_episode: dict[str, list[dict]], history: int, horizon: int = 0):
    if history < 1 or horizon < 0:
        raise ValueError("history must be positive and horizon non-negative")
    return [WindowIndex(eid, end, end + horizon) for eid, rows in rows_by_episode.items()
            for end in range(history - 1, len(rows) - horizon)]


def episode_split(episode_ids, seed=42, fractions=(0.7, 0.15, 0.15)):
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must be three values summing to one")
    ids = list(dict.fromkeys(episode_ids)); random.Random(seed).shuffle(ids)
    a, b = int(len(ids) * fractions[0]), int(len(ids) * sum(fractions[:2]))
    return {"train": ids[:a], "validation": ids[a:b], "test": ids[b:]}


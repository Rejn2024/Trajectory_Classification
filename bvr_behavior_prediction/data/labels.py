from enum import IntEnum


class Lateral(IntEnum): STRAIGHT = 0; TURN_LEFT = 1; TURN_RIGHT = 2
class Vertical(IntEnum): LEVEL = 0; CLIMB = 1; DESCEND = 2
class Energy(IntEnum): STEADY_SPEED = 0; ACCELERATE = 1; DECELERATE = 2
class Tactical(IntEnum):
    MAINTAIN = 0; PURSUE = 1; BEAM = 2; CRANK_LEFT = 3; CRANK_RIGHT = 4; EXTEND = 5
    RECOMMIT = 6; DEFEND = 7; MISSILE_EVASION = 8; SUPPORT = 9


def kinematic_labels(turn_rate: float, climb_rate: float, acceleration: float,
                     turn_threshold: float = 0.01, climb_threshold: float = 2.0,
                     acceleration_threshold: float = 0.5):
    lateral = Lateral.TURN_LEFT if turn_rate < -turn_threshold else (
        Lateral.TURN_RIGHT if turn_rate > turn_threshold else Lateral.STRAIGHT)
    vertical = Vertical.CLIMB if climb_rate > climb_threshold else (
        Vertical.DESCEND if climb_rate < -climb_threshold else Vertical.LEVEL)
    energy = Energy.ACCELERATE if acceleration > acceleration_threshold else (
        Energy.DECELERATE if acceleration < -acceleration_threshold else Energy.STEADY_SPEED)
    return lateral, vertical, energy


def annotate_transitions(rows: list[dict], horizons=(2.0, 4.0, 8.0)) -> list[dict]:
    """Annotate skill boundaries without looking beyond the current episode."""
    for i, row in enumerate(rows):
        current = row["target_skill"]
        next_i = next((j for j in range(i + 1, len(rows)) if rows[j]["target_skill"] != current), None)
        previous_i = next((j for j in range(i - 1, -1, -1) if rows[j]["target_skill"] != current), None)
        row["time_since_skill_change"] = (row["time_s"] - rows[previous_i + 1]["time_s"]
                                            if previous_i is not None else row["time_s"] - rows[0]["time_s"])
        time_to = rows[next_i]["time_s"] - row["time_s"] if next_i is not None else float("inf")
        row["time_to_next_skill_change"] = time_to
        row["transition_flag"] = int(i > 0 and rows[i - 1]["target_skill"] != current)
        for horizon in horizons:
            row[f"transition_within_{horizon:g}s"] = int(0 < time_to <= horizon)
        row["target_next_skill"] = rows[next_i]["target_skill"] if next_i is not None else current
    return rows


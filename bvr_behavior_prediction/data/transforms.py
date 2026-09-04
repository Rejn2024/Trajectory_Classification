import math


def wrap_angle(angle: float) -> float:
    """Wrap radians to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def rotate_xy(x: float, y: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def to_observer_body(vector: tuple[float, float, float], observer_heading: float):
    x, y = rotate_xy(vector[0], vector[1], -observer_heading)
    return x, y, vector[2]


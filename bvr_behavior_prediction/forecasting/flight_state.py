"""Continuous future speed, course heading and altitude, with circular heading targets."""

import numpy as np
import pandas as pd

from .data import DT, HORIZONS, observed_kinematics
from .data import episode_examples as motion_examples

CONTRACT = "future-speed-course-heading-altitude-v1"
QUANTITIES = ["speed_mps", "heading_deg", "altitude_m"]
ORIGIN_COLUMNS = ["observed_speed_mps", "observed_heading_rad", "observed_altitude_m"]


def state_from_kinematics(position, velocity):
    """True course clockwise from north in the simulator's NWU frame, in radians."""
    position, velocity = np.asarray(position), np.asarray(velocity)
    return np.stack((np.linalg.norm(velocity, axis=-1),
                     np.arctan2(-velocity[..., 1], velocity[..., 0]), position[..., 2]), axis=-1)


def encode(state, origin):
    angle = state[..., 1] - origin[..., 1]
    return np.stack((state[..., 0] - origin[..., 0], np.sin(angle),
                     np.cos(angle) - 1, state[..., 2] - origin[..., 2]), axis=-1)


def decode(residual, origin):
    residual, origin = np.asarray(residual), np.asarray(origin)
    angle = np.arctan2(residual[..., 1], residual[..., 2] + 1)
    return np.stack((np.maximum(0, origin[..., 0] + residual[..., 0]),
                     (origin[..., 1] + angle) % (2 * np.pi),
                     origin[..., 2] + residual[..., 3]), axis=-1)


def episode_examples(observations, tracks, history_steps=51, stride_steps=10):
    features, ends, _, valid, _ = motion_examples(observations, tracks, history_steps, stride_steps)
    position, velocity, acceleration = observed_kinematics(observations)
    future = np.minimum(ends[:, None] + np.rint(HORIZONS / DT).astype(int), len(tracks) - 1)
    truth_position = tracks[[f"target_{c}" for c in ("x", "y", "z")]].to_numpy()
    truth_velocity = tracks[[f"target_{c}" for c in ("vx", "vy", "vz")]].to_numpy()
    # Course is undefined for stationary/vertical tracks. Do not invent headings.
    valid &= np.linalg.norm(truth_velocity[future, :2], axis=-1) > 1e-6
    valid &= (np.linalg.norm(velocity[ends, :2], axis=-1) > 1e-6)[:, None]
    keep = valid.any(axis=1)
    ends, future, valid = ends[keep], future[keep], valid[keep]
    origin = state_from_kinematics(position[ends], velocity[ends])[:, None]
    residual = encode(state_from_kinematics(truth_position[future], truth_velocity[future]), origin)
    ca_position = position[ends, None] + velocity[ends, None] * HORIZONS[None, :, None]
    ca_position += 0.5 * acceleration[ends, None] * HORIZONS[None, :, None] ** 2
    ca_velocity = velocity[ends, None] + acceleration[ends, None] * HORIZONS[None, :, None]
    ca = encode(state_from_kinematics(ca_position, ca_velocity), origin)
    residual[~valid] = 0
    return features, ends, residual.astype(np.float32), valid, ca.astype(np.float32)


def metrics(prediction, target, valid, samples, model):
    origin = samples[ORIGIN_COLUMNS].to_numpy()[:, None]
    predicted, truth = decode(prediction, origin), decode(target, origin)
    errors = np.abs(predicted - truth)
    angle = predicted[..., 1] - truth[..., 1]
    errors[..., 1] = np.degrees(np.abs(np.arctan2(np.sin(angle), np.cos(angle))))
    rows = []
    for h, seconds in enumerate(HORIZONS):
        for index, (quantity, units) in enumerate(zip(("speed", "heading", "altitude"), ("m/s", "deg", "m"))):
            frame = samples.loc[valid[:, h], ["episode_id", "pilot_id"]].copy()
            if frame.empty:
                raise ValueError("No validation targets for a forecast horizon")
            frame["error"] = errors[valid[:, h], h, index]
            for pilot, group in [("population", frame), *list(frame.groupby("pilot_id", sort=True))]:
                values = group.error.to_numpy()
                rows.append({"model": model, "pilot_id": pilot, "horizon_s": float(seconds),
                             "quantity": quantity, "units": units, "samples": len(group),
                             "episodes": group.episode_id.nunique(),
                             "episode_mean_error": float(group.groupby("episode_id").error.mean().mean()),
                             "mean_error": float(values.mean()), "rmse": float(np.sqrt(np.mean(values ** 2))),
                             "median_error": float(np.median(values)), "p90_error": float(np.quantile(values, .9))})
    return pd.DataFrame(rows)


class FlightStatePredictor:
    """Load a regression checkpoint; return future physical states, never skill IDs."""

    def __init__(self, checkpoint, device="cpu"):
        import torch

        from .models import MotionForecaster

        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if saved.get("target_contract") != CONTRACT or saved["model_config"].get("output_size") != 4:
            raise ValueError("Expected a flight-state regression checkpoint; retrain for the new target contract")
        self.model = MotionForecaster(**saved["model_config"]).to(self.device).eval()
        self.model.load_state_dict(saved["model_state"])
        self.history_steps = saved["model_config"]["history_steps"]
        self.mean = np.asarray(saved["scaling"]["feature_mean"], dtype=np.float32)
        self.std = np.asarray(saved["scaling"]["feature_std"], dtype=np.float32)
        self.scale = np.asarray(saved["scaling"]["residual_scale"], dtype=np.float32)
        if self.scale.shape != (3, 4) or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid flight-state target scaling")

    def predict(self, observations):
        import torch

        from .inference import history_features

        features, position, velocity = history_features(observations, self.history_steps)
        origin = state_from_kinematics(position, velocity)
        if np.linalg.norm(velocity[:2]) <= 1e-6:
            raise ValueError("Course heading requires a nonzero horizontal target velocity")
        inputs = torch.as_tensor((features - self.mean) / self.std, device=self.device)[None]
        with torch.inference_mode():
            residual = self.model(inputs)[0].float().cpu().numpy() * self.scale
        future = decode(residual, origin)
        return {"horizons_s": HORIZONS.copy(), "speed_mps": future[:, 0],
                "heading_deg": np.degrees(future[:, 1]) % 360, "altitude_m": future[:, 2]}

"""Load a selected checkpoint and forecast from raw tactical-side observations."""

import numpy as np
import torch

from .data import HORIZONS, observed_kinematics
from .models import MotionForecaster


def history_features(observations, history_steps=51):
    """Five extra samples supply the backward acceleration at the oldest token."""
    observations = np.asarray(observations, dtype=np.float32)
    required = history_steps + 5
    if observations.ndim != 2 or observations.shape[1] != 51 or len(observations) < required:
        raise ValueError(f"Supply at least {required} consecutive 10 Hz observation frames with 51 features")
    if not np.isfinite(observations).all():
        raise ValueError("Observation history contains nonfinite values")
    if np.any(observations[-required:, 15] <= 0):
        raise ValueError("A live target track is required throughout the observation history")
    observations = observations[-required:]
    position, velocity, acceleration = observed_kinematics(observations)
    features = np.concatenate((observations, acceleration / 100.0), axis=-1).astype(np.float32)
    return features[-history_steps:], position[-1], velocity[-1]


class MotionPredictor:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = MotionForecaster(**saved["model_config"]).to(self.device).eval()
        self.model.load_state_dict(saved["model_state"])
        self.history_steps = saved["model_config"]["history_steps"]
        self.required_frames = self.history_steps + 5
        self.mean = np.asarray(saved["scaling"]["feature_mean"], dtype=np.float32)
        self.std = np.asarray(saved["scaling"]["feature_std"], dtype=np.float32)
        self.scale = np.asarray(saved["scaling"]["residual_scale"], dtype=np.float32)

    @torch.inference_mode()
    def predict(self, observations):
        features, position, velocity = history_features(observations, self.history_steps)
        inputs = torch.as_tensor((features - self.mean) / self.std, device=self.device)[None]
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            scaled_residual = self.model(inputs)
        residual = scaled_residual[0].float().cpu().numpy() * self.scale
        predicted_position = position + HORIZONS[:, None] * velocity + residual[:, :3]
        predicted_velocity = velocity + residual[:, 3:]
        return {"horizons_s": HORIZONS.copy(), "position_nwu_m": predicted_position,
                "velocity_nwu_mps": predicted_velocity}

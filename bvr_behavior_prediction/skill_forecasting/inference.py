"""Observation-only deployment interface for a saved catalogue forecaster."""

import numpy as np
import torch

from bvr_behavior_prediction.forecasting.inference import history_features

from .models import SkillForecaster


class SkillPredictor:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = SkillForecaster(**saved["model_config"]).to(self.device).eval()
        self.model.load_state_dict(saved["model_state"])
        self.classes = np.asarray(saved["classes"])
        self.horizons = np.asarray(saved["horizons_s"])
        self.history_steps = saved["model_config"]["history_steps"]
        self.required_frames = self.history_steps + 5
        self.mean = np.asarray(saved["scaling"]["feature_mean"], dtype=np.float32)
        self.std = np.asarray(saved["scaling"]["feature_std"], dtype=np.float32)

    @torch.inference_mode()
    def predict(self, observations):
        features, _, _ = history_features(observations, self.history_steps)
        inputs = torch.as_tensor((features - self.mean) / self.std, device=self.device)[None]
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            logits = self.model(inputs)
        probabilities = logits[0].float().softmax(-1).cpu().numpy()
        ranking = np.argsort(-probabilities, axis=-1, kind="stable")
        return {"horizons_s": self.horizons.copy(), "classes": self.classes.copy(),
                "probabilities": probabilities, "top3_skills": self.classes[ranking[:, :3]]}

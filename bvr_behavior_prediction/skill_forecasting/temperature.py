"""One-parameter confidence calibration fitted on internal stop groups only."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def fit_temperature(probabilities, targets):
    log_probabilities = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-30, 1))

    def objective(temperature):
        losses = []
        for h in range(3):
            valid = targets[:, h] >= 0
            scores = log_probabilities[valid, h] / temperature
            selected = scores[np.arange(len(scores)), targets[valid, h]]
            losses.append(np.mean(logsumexp(scores, axis=1) - selected))
        return float(np.mean(losses))

    result = minimize_scalar(objective, bounds=(0.5, 4.0), method="bounded", options={"xatol": 0.001})
    if not result.success:
        raise RuntimeError("Temperature fitting failed")
    return float(result.x)

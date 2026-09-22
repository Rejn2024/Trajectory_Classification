"""Hybrid neural policy: discrete tactical skill plus exact continuous parameters."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from bvr_sim.agents.skill_manager import SkillManager


class HybridSkillPilot(nn.Module):
    """Select any SkillManager skill and parameterise it on every decision.

    The categorical head covers the complete runtime catalogue (currently 33
    skills). The Gaussian head supplies one normalized value per numeric parameter;
    values are transformed through each skill's public bounds rather than replaced
    by fixed primitive defaults.
    """

    def __init__(
        self,
        observation_size: int,
        hidden_size: int = 256,
        manager=None,
        history_steps: int = 20,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        super().__init__()
        self.manager = manager or SkillManager()
        self.skill_names = tuple(self.manager.list_skills())
        numeric = set()
        for name in self.skill_names:
            properties = self.manager.get_contract(name)["parameter_schema"]["properties"]
            numeric.update(key for key, schema in properties.items() if schema["type"] == "number")
        self.parameter_names = tuple(sorted(numeric))
        if transformer_heads < 1 or transformer_layers < 1:
            raise ValueError("transformer heads and layers must be positive")
        if hidden_size % transformer_heads:
            raise ValueError("hidden_size must be divisible by transformer_heads")
        self.history_steps = history_steps
        self.input_projection = nn.Linear(observation_size, hidden_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, history_steps, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=transformer_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, transformer_layers, enable_nested_tensor=False
        )
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.skill_head = nn.Linear(hidden_size, len(self.skill_names))
        self.parameter_mean = nn.Linear(hidden_size, len(self.parameter_names))
        self.parameter_log_std = nn.Parameter(torch.full((len(self.parameter_names),), -0.5))
        self.value_head = nn.Linear(hidden_size, 1)

    def distributions(self, observations):
        if observations.ndim == 2:
            observations = observations.unsqueeze(1)
        if observations.ndim != 3:
            raise ValueError("observations must have shape (batch, time, features)")
        if observations.shape[1] > self.history_steps:
            raise ValueError("observation history exceeds configured history_steps")
        tokens = self.input_projection(observations)
        tokens = tokens + self.position_embedding[:, -tokens.shape[1] :]
        # The newest token attends to the complete chronological energy/flight history.
        hidden = self.encoder_norm(self.encoder(tokens)[:, -1])
        skill = Categorical(logits=self.skill_head(hidden))
        mean = self.parameter_mean(hidden)
        params = Normal(mean, self.parameter_log_std.clamp(-5, 2).exp().expand_as(mean))
        return skill, params, self.value_head(hidden).squeeze(-1)

    def act(self, observation, deterministic=False):
        device = next(self.parameters()).device
        obs = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            skill_dist, param_dist, value = self.distributions(obs)
            skill_index = skill_dist.probs.argmax(-1) if deterministic else skill_dist.sample()
            raw = param_dist.mean if deterministic else param_dist.sample()
            log_prob = skill_dist.log_prob(skill_index) + param_dist.log_prob(raw).sum(-1)
        index = int(skill_index.item())
        params = self.decode_parameters(index, raw[0].cpu().numpy())
        return (
            self.skill_names[index],
            params,
            float(log_prob.item()),
            float(value.item()),
            raw[0].cpu().numpy(),
        )

    def decode_parameters(self, skill_index: int, raw_values) -> dict:
        contract = self.manager.get_contract(self.skill_names[skill_index])
        schemas = contract["parameter_schema"]["properties"]
        raw_by_name = dict(zip(self.parameter_names, np.asarray(raw_values)))
        result = {}
        for name, schema in schemas.items():
            if schema["type"] == "number":
                unit = (np.tanh(raw_by_name[name]) + 1.0) / 2.0
                low = schema.get("minimum", schema["default"] - 1.0)
                high = schema.get("maximum", schema["default"] + 1.0)
                result[name] = float(low + unit * (high - low))
            else:
                # Non-numeric identifiers/sides are tactical metadata, not a
                # kinematic precision control; retain their valid contract default.
                result[name] = schema["default"]
        return result

    def evaluate(self, observations, skill_indices, raw_parameters):
        skill, params, values = self.distributions(observations)
        log_prob = skill.log_prob(skill_indices) + params.log_prob(raw_parameters).sum(-1)
        entropy = skill.entropy() + params.entropy().sum(-1)
        return log_prob, entropy, values

"""Hybrid neural policy: discrete tactical skill plus exact continuous parameters."""

import numpy as np
import torch
from bvr_sim.agents.skill_manager import SkillManager
from torch import nn
from torch.distributions import Categorical, Normal


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
        selected_skill_parameters: bool = False,
    ):
        super().__init__()
        self.manager = manager or SkillManager()
        self.skill_names = tuple(self.manager.list_skills())
        self._skill_indices = {name: index for index, name in enumerate(self.skill_names)}
        self._skill_contracts = tuple(
            self.manager.get_contract(name)["parameter_schema"]["properties"]
            for name in self.skill_names
        )
        numeric = set()
        for properties in self._skill_contracts:
            numeric.update(key for key, schema in properties.items() if schema["type"] == "number")
        self.parameter_names = tuple(sorted(numeric))
        self.selected_skill_parameters = selected_skill_parameters
        # Derived from the existing contracts, with no new learned weights or
        # checkpoint keys. Old checkpoints and action decoding stay compatible.
        self.register_buffer(
            "_skill_parameter_mask",
            torch.tensor([
                [float(name in properties and properties[name]["type"] == "number")
                 for name in self.parameter_names]
                for properties in self._skill_contracts
            ], dtype=torch.float32),
            persistent=False,
        )
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
        self.encoder = nn.TransformerEncoder(layer, transformer_layers, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.skill_head = nn.Linear(hidden_size, len(self.skill_names))
        self.parameter_mean = nn.Linear(hidden_size, len(self.parameter_names))
        self.parameter_log_std = nn.Parameter(torch.full((len(self.parameter_names),), -0.5))
        self.value_head = nn.Linear(hidden_size, 1)

    def distributions(self, observations, *, validate_args=True):
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
        skill = Categorical(logits=self.skill_head(hidden), validate_args=validate_args)
        mean = self.parameter_mean(hidden)
        params = Normal(
            mean, self.parameter_log_std.clamp(-5, 2).exp().expand_as(mean),
            validate_args=validate_args,
        )
        return skill, params, self.value_head(hidden).squeeze(-1)

    def act(self, observation, deterministic=False):
        actions = self.act_batch(np.asarray(observation)[None], deterministic=deterministic)
        return actions[0]

    def act_batch(self, observations, deterministic=False):
        """Choose actions for multiple simulators in one accelerator forward pass."""
        device = next(self.parameters()).device
        obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        # Rollouts only retain plain numbers, so inference mode can also skip
        # autograd's version-counter bookkeeping.
        with torch.inference_mode():
            skill_dist, param_dist, value = self.distributions(obs)
            skill_index = skill_dist.probs.argmax(-1) if deterministic else skill_dist.sample()
            raw = param_dist.mean if deterministic else param_dist.sample()
            log_prob = self._action_log_prob(skill_dist, param_dist, skill_index, raw)
        indices = skill_index.cpu().tolist()
        raw_cpu = raw.cpu().numpy()
        log_prob_cpu = log_prob.cpu().tolist()
        value_cpu = value.cpu().tolist()
        return tuple(
            (
                self.skill_names[index],
                self.decode_parameters(index, raw_values),
                float(action_log_prob),
                float(action_value),
                raw_values,
            )
            for index, raw_values, action_log_prob, action_value in zip(
                indices, raw_cpu, log_prob_cpu, value_cpu
            )
        )

    def decode_parameters(self, skill_index: int, raw_values) -> dict:
        schemas = self._skill_contracts[skill_index]
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

    def evaluate(self, observations, skill_indices, raw_parameters, *, validate_args=True):
        skill, params, values = self.distributions(observations, validate_args=validate_args)
        log_prob = self._action_log_prob(skill, params, skill_indices, raw_parameters)
        parameter_entropy = params.entropy()
        if self.selected_skill_parameters:
            # Entropy of the raw hybrid action: H(skill) + E_skill[H(parameters)].
            # The expectation must retain gradients through skill probabilities;
            # using just the sampled skill would omit this part of the objective.
            per_skill_entropy = parameter_entropy @ self._skill_parameter_mask.T
            entropy = skill.entropy() + (skill.probs * per_skill_entropy).sum(-1)
        else:
            entropy = skill.entropy() + parameter_entropy.sum(-1)
        return log_prob, entropy, values

    def _action_log_prob(self, skill, params, skill_indices, raw_parameters):
        parameter_log_prob = params.log_prob(raw_parameters)
        if self.selected_skill_parameters:
            parameter_log_prob = parameter_log_prob * self._skill_parameter_mask[skill_indices]
        return skill.log_prob(skill_indices) + parameter_log_prob.sum(-1)

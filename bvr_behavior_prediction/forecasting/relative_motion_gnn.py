"""HGT-v2 with causal temporal encoders for notebook-33 relative-motion data."""

import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from bvr_behavior_prediction.enemy_missile_models import (
    HGT_THREAT_EDGE_FEATURES,
    CausalTemporalConvEncoder,
    _HGTBlock,
    masked_summary,
)

from .relative_motion import (
    DATA_CONTRACT,
    HISTORY_STEPS,
    HORIZONS,
    SPLITS,
    TARGET_FIELDS,
    RelativeMotionData,
    read_json,
    sha256,
    write_json,
)

MODEL_CONTRACT = "relative-motion-hgt-v2-tcn-v1"
ARCHITECTURE = "hgt_v2_tcn_relative_motion"
TRANSFORMER_MODEL_CONTRACT = "relative-motion-hgt-v2-horizon-transformer-v1"
TRANSFORMER_ARCHITECTURE = "hgt_v2_horizon_transformer_relative_motion"
MODEL_CONTRACTS = {
    ARCHITECTURE: MODEL_CONTRACT,
    TRANSFORMER_ARCHITECTURE: TRANSFORMER_MODEL_CONTRACT,
}


def default_training_config(architecture=ARCHITECTURE):
    if architecture not in MODEL_CONTRACTS:
        raise ValueError(f"Unknown relative-motion architecture: {architecture}")
    config = {
        "architecture": architecture,
        "seed": 42,
        "width": 96,
        "graph_width": 64,
        "graph_layers": 2,
        "graph_heads": 4,
        "dropout": 0.25,
        "threat_cpa_horizon_s": 30.0,
        "threat_time_scale_s": 5.0,
        "threat_prior_strength": 1.0,
        "missile_distance_floor_m": 1000.0,
        "missile_distance_power": 1.0,
        "epochs": 100,
        "batch_size": 1024,
        "evaluation_batch_size": 1024,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "huber_beta": 1.0,
        "l1_lambda": 1e-6,
        "lr_schedule": "warmup_cosine",
        "warmup_fraction": 0.05,
        "warmup_start_ratio": 0.1,
        "min_lr_ratio": 0.05,
        "early_stopping": True,
        "early_stopping_min_epochs": 20,
        "early_stopping_patience": 10,
        "precision": "bf16",
        "cpu_threads": 4,
    }
    if architecture == ARCHITECTURE:
        config.update(tcn_kernel_size=3, tcn_dilations=[1, 2, 4, 8])
    else:
        config.update(
            transformer_layers=2,
            transformer_heads=4,
            transformer_ff_width=384,
            batch_size=256,
            evaluation_batch_size=256,
        )
    return config


class _RelativeMotionHGT(nn.Module):
    """Shared per-frame HGT-v2 graph encoder for the 91 policy features."""

    def __init__(
        self,
        *,
        width=96,
        graph_width=64,
        graph_layers=2,
        graph_heads=4,
        dropout=0.25,
        threat_cpa_horizon_s=30.0,
        threat_time_scale_s=5.0,
        threat_prior_strength=1.0,
        missile_distance_floor_m=1000.0,
        missile_distance_power=1.0,
        mean=None,
        std=None,
    ):
        super().__init__()
        for name, value in (
            ("width", width),
            ("graph_width", graph_width),
            ("graph_layers", graph_layers),
            ("graph_heads", graph_heads),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if graph_width % graph_heads:
            raise ValueError("graph_width must be divisible by graph_heads")
        for name, value in (
            ("threat_cpa_horizon_s", threat_cpa_horizon_s),
            ("threat_time_scale_s", threat_time_scale_s),
            ("threat_prior_strength", threat_prior_strength),
            ("missile_distance_floor_m", missile_distance_floor_m),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(missile_distance_power, bool)
            or not isinstance(missile_distance_power, (int, float))
            or not math.isfinite(missile_distance_power)
            or missile_distance_power < 0
        ):
            raise ValueError("missile_distance_power must be finite and nonnegative")
        mean = torch.zeros(91) if mean is None else torch.as_tensor(mean)
        std = torch.ones(91) if std is None else torch.as_tensor(std)
        if (
            mean.shape != (91,)
            or std.shape != (91,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError("Supply finite mean and positive std for the 91 policy features")
        self.register_buffer("feature_mean", mean.detach().clone().float())
        self.register_buffer("feature_std", std.detach().clone().float())
        self.graph_width = graph_width
        self.threat_cpa_horizon_s = float(threat_cpa_horizon_s)
        self.threat_time_scale_s = float(threat_time_scale_s)
        self.missile_distance_floor_m = float(missile_distance_floor_m)
        self.missile_distance_power = float(missile_distance_power)

        # All 91 fields enter one of these entity encoders. The observer receives
        # own state, incoming flag and own energies; the opponent receives its
        # relative state/bearing and energies. Missile types keep their native schemas.
        self.observer = nn.Sequential(nn.Linear(12, graph_width), nn.GELU())
        self.opponent = nn.Sequential(nn.Linear(11, graph_width), nn.GELU())
        self.friendly_missile = nn.Sequential(
            nn.Linear(7, graph_width), nn.GELU(), nn.Linear(graph_width, graph_width), nn.GELU()
        )
        self.hostile_missile = nn.Sequential(
            nn.Linear(10, graph_width), nn.GELU(), nn.Linear(graph_width, graph_width), nn.GELU()
        )
        self.role = nn.Embedding(4, graph_width)
        self.register_buffer("roles", torch.tensor([0, 1, 2, 2, 2, 2, 3, 3, 3, 3]))

        sources, destinations, relations = [0, 1], [1, 0], [0, 0]
        for missile in range(2, 10):
            owner = 0 if missile < 6 else 1
            for aircraft in range(2):
                owning = aircraft == owner
                sources.extend((aircraft, missile))
                destinations.extend((missile, aircraft))
                relations.extend((1 if owning else 3, 2 if owning else 4))
        self.register_buffer("sources", torch.tensor(sources))
        self.register_buffer("destinations", torch.tensor(destinations))
        self.register_buffer("relations", torch.tensor(relations))
        self.messages = nn.ModuleList(
            [
                _HGTBlock(
                    graph_width,
                    graph_heads,
                    dropout,
                    edge_features=len(HGT_THREAT_EDGE_FEATURES),
                    relation_family_gates=True,
                )
                for _ in range(graph_layers)
            ]
        )
        initial = threat_prior_strength + math.log(-math.expm1(-threat_prior_strength))
        self.threat_prior_raw_strength = nn.Parameter(torch.full((graph_heads,), initial))
        self.threat_prior = nn.Sequential(
            nn.Linear(len(HGT_THREAT_EDGE_FEATURES), 32),
            nn.GELU(),
            nn.Linear(32, graph_heads),
        )
        nn.init.zeros_(self.threat_prior[-1].weight)
        nn.init.zeros_(self.threat_prior[-1].bias)
        self.register_buffer("target_edge_indices", torch.where(self.relations == 4)[0])
        self.register_buffer(
            "edge_missile_indices",
            (torch.maximum(self.sources, self.destinations) - 2).clamp_min(0),
        )

        self.projection = nn.Linear(6 * graph_width + 2, width)

    def _decode_rows(self, frames):
        if frames.ndim != 2 or frames.shape[1] != 91:
            raise ValueError("Expected normalized (frames, 91) policy features")
        normalized = frames.float()
        raw = normalized * self.feature_std + self.feature_mean
        friendly_indices = torch.arange(19, 47, device=frames.device)
        friendly = raw[:, friendly_indices].reshape(-1, 4, 7)
        sentinel = (-self.feature_mean / self.feature_std)[friendly_indices]
        tolerance = 8 * torch.finfo(normalized.dtype).eps * (1 + sentinel.abs())
        friendly_present = (
            (normalized[:, friendly_indices] - sentinel).abs() > tolerance
        ).reshape(-1, 4, 7).any(-1)
        hostile = raw[:, 47:87].reshape(-1, 4, 10)
        hostile_present = hostile[:, :, 0] > 0.5
        friendly = friendly * friendly_present.unsqueeze(-1)
        hostile = hostile * hostile_present.unsqueeze(-1)

        observer_features = torch.cat((raw[:, :9], raw[:, 18:19], raw[:, 87:89]), -1)
        opponent_features = torch.cat((raw[:, 9:18], raw[:, 89:91]), -1)
        aircraft = torch.stack(
            (self.observer(observer_features), self.opponent(opponent_features)), 1
        )
        missiles = torch.cat(
            (self.friendly_missile(friendly), self.hostile_missile(hostile)), 1
        )
        nodes = torch.cat((aircraft, missiles), 1)
        present = torch.cat(
            (torch.ones_like(friendly_present[:, :2]), friendly_present, hostile_present), -1
        )
        nodes = (nodes + self.role(self.roles).to(nodes.dtype)) * present.unsqueeze(-1)
        return raw, friendly, hostile, nodes, present

    def _world_states(self, raw, friendly, hostile):
        position_scale = raw.new_tensor([50000, 50000, 10000])
        own_position = torch.zeros_like(raw[:, None, 9:12])
        opponent_position = raw[:, None, 9:12] * position_scale
        missile_positions = torch.cat(
            (friendly[:, :, :3], hostile[:, :, 1:4]), 1
        ) * position_scale
        positions = torch.cat((own_position, opponent_position, missile_positions), 1)

        own_velocity = raw[:, None, 3:6] * 600
        opponent_velocity = own_velocity + raw[:, None, 12:15] * 600
        missile_relative_velocity = torch.cat(
            (friendly[:, :, 3:6], hostile[:, :, 4:7]), 1
        ) * 1200
        missile_velocities = own_velocity + missile_relative_velocity
        velocities = torch.cat((own_velocity, opponent_velocity, missile_velocities), 1)
        return positions, velocities

    def _edge_geometry(self, raw, friendly, hostile):
        position_scale = raw.new_tensor([50000, 50000, 10000])
        positions, velocities = self._world_states(raw, friendly, hostile)
        relative_position = positions[:, self.sources] - positions[:, self.destinations]
        relative_velocity = velocities[:, self.sources] - velocities[:, self.destinations]
        distance = torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True)
        closing = -(
            relative_position * relative_velocity
        ).sum(-1, keepdim=True) / distance.clamp_min(1)
        base = torch.cat(
            (
                relative_position / position_scale,
                relative_velocity / 1200,
                distance / 50000,
                closing / 1200,
            ),
            -1,
        )

        speed_square = relative_velocity.square().sum(-1)
        approach_time = (
            -(relative_position * relative_velocity).sum(-1) / speed_square.clamp_min(1)
        ).clamp(min=0, max=self.threat_cpa_horizon_s)
        approach_distance = torch.linalg.vector_norm(
            relative_position + approach_time.unsqueeze(-1) * relative_velocity, dim=-1
        )
        source_velocity = velocities[:, self.sources]
        bearing_alignment = -(
            relative_position * source_velocity
        ).sum(-1) / (
            distance.squeeze(-1) * torch.linalg.vector_norm(source_velocity, dim=-1)
        ).clamp_min(1)
        bearing_alignment = bearing_alignment.clamp(-1, 1)
        relative_speed = torch.linalg.vector_norm(relative_velocity, dim=-1)

        observer_altitude = raw[:, None, 6] * 10000
        altitude = observer_altitude + positions[:, :, 2]
        kinetic = 0.5 * velocities.square().sum(-1)
        potential = 9.80665 * altitude
        energy = torch.stack((kinetic, potential, kinetic + potential), -1) / 1e6
        threat = torch.stack(
            (
                approach_time / self.threat_cpa_horizon_s,
                approach_distance / 50000,
                bearing_alignment,
                relative_speed / 1200,
            ),
            -1,
        )
        return torch.cat(
            (base, threat, energy[:, self.sources], energy[:, self.destinations]), -1
        )

    def _edge_log_prior(self, geometry, present):
        distance = geometry[:, self.target_edge_indices, 6].float() * 50000
        occupied = present[:, 2:].reshape(-1, 2, 4)
        distance = distance.reshape(-1, 2, 4)
        nearest = distance.masked_fill(~occupied, float("inf")).amin(-1, keepdim=True)
        nearest = torch.where(occupied.any(-1, keepdim=True), nearest, 0)
        log_weight = self.missile_distance_power * (
            (nearest + self.missile_distance_floor_m).log()
            - (distance + self.missile_distance_floor_m).log()
        )
        features = geometry[:, self.target_edge_indices].float()
        cpa_distance = features[:, :, 9].reshape(-1, 2, 4) * 50000
        cpa_time = features[:, :, 8].reshape(-1, 2, 4) * self.threat_cpa_horizon_s
        alignment = features[:, :, 10].reshape(-1, 2, 4)
        physical_score = (
            log_weight
            - torch.log1p(cpa_distance / self.missile_distance_floor_m)
            - torch.log1p(cpa_time / self.threat_time_scale_s)
            + alignment
        )
        residual = self.threat_prior(features).float().reshape(*physical_score.shape, -1)
        score = (
            physical_score.unsqueeze(-1)
            * nn.functional.softplus(self.threat_prior_raw_strength.float())
            + residual
        )
        mask = occupied.unsqueeze(-1)
        minimum = torch.finfo(score.dtype).min
        maximum = score.masked_fill(~mask, minimum).amax(2, keepdim=True)
        maximum = torch.where(mask.any(2, keepdim=True), maximum, 0)
        score = (score - maximum).masked_fill(~mask, 0).flatten(1, 2)
        return score[:, self.edge_missile_indices] * (
            self.relations >= 3
        )[None, :, None]

    def encode_frame_rows(self, frames):
        raw, friendly, hostile, nodes, present = self._decode_rows(frames)
        geometry = self._edge_geometry(raw, friendly, hostile)
        prior = self._edge_log_prior(geometry, present)
        for block in self.messages:
            nodes = block(
                nodes,
                present,
                geometry,
                self.sources,
                self.destinations,
                self.relations,
                self.roles,
                prior,
            )
        readout = torch.cat(
            (
                nodes[:, 0],
                nodes[:, 1],
                masked_summary(nodes[:, 2:6], present[:, 2:6]),
                masked_summary(nodes[:, 6:], present[:, 6:]),
            ),
            -1,
        )
        return self.projection(readout)

    def encode_frames(self, windows):
        if windows.ndim != 3 or tuple(windows.shape[1:]) != (HISTORY_STEPS, 91):
            raise ValueError(f"Expected (batch, {HISTORY_STEPS}, 91) histories")
        return self.encode_frame_rows(windows.reshape(-1, 91)).reshape(
            windows.shape[0], HISTORY_STEPS, -1
        )


class RelativeMotionHGTTCN(_RelativeMotionHGT):
    """Notebook-28 HGT-v2/TCN adapted to the exact 91-feature pilot frames."""

    def __init__(
        self,
        *,
        width=96,
        tcn_kernel_size=3,
        tcn_dilations=(1, 2, 4, 8),
        dropout=0.25,
        **graph_options,
    ):
        super().__init__(width=width, dropout=dropout, **graph_options)
        self.temporal = CausalTemporalConvEncoder(
            width, tcn_kernel_size, tcn_dilations, dropout
        )
        if self.temporal.receptive_field < HISTORY_STEPS:
            raise ValueError("TCN receptive field must cover all 20 history frames")
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, len(HORIZONS) * len(TARGET_FIELDS)),
        )

    def encode_history(self, windows):
        return self.norm(self.temporal(self.encode_frames(windows))[:, -1])

    def forward(self, windows):
        return self.head(self.encode_history(windows)).reshape(
            -1, len(HORIZONS), len(TARGET_FIELDS)
        )


class _TemporalTransformerSection(nn.Module):
    def __init__(self, width, layers, heads, ff_width, dropout):
        super().__init__()
        self.position = nn.Parameter(torch.empty(HISTORY_STEPS, width))
        nn.init.normal_(self.position, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=ff_width,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, len(TARGET_FIELDS)),
        )

    def encode(self, frames, causal_mask):
        tokens = frames + self.position.to(frames.dtype)
        for layer in self.layers:
            tokens = layer(tokens, src_mask=causal_mask)
        return tokens

    def forward(self, frames, causal_mask):
        return self.head(self.encode(frames, causal_mask)[:, -1])


class RelativeMotionHGTTransformer(_RelativeMotionHGT):
    """Shared spatial graphs and one independent temporal section per horizon."""

    def __init__(
        self,
        *,
        width=96,
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff_width=384,
        dropout=0.25,
        **graph_options,
    ):
        for name, value in (
            ("transformer_layers", transformer_layers),
            ("transformer_heads", transformer_heads),
            ("transformer_ff_width", transformer_ff_width),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        super().__init__(width=width, dropout=dropout, **graph_options)
        if width % transformer_heads:
            raise ValueError("width must be divisible by transformer_heads")
        self.sections = nn.ModuleList([
            _TemporalTransformerSection(
                width, transformer_layers, transformer_heads, transformer_ff_width, dropout
            )
            for _ in HORIZONS
        ])
        self.register_buffer(
            "causal_mask",
            torch.ones(HISTORY_STEPS, HISTORY_STEPS, dtype=torch.bool).triu(diagonal=1),
            persistent=False,
        )

    def forward(self, windows):
        frames = self.encode_frames(windows)
        return torch.stack([section(frames, self.causal_mask) for section in self.sections], dim=1)


def model_configuration(config):
    architecture = config.get("architecture")
    if architecture not in MODEL_CONTRACTS:
        raise ValueError(f"Unknown relative-motion architecture: {architecture}")
    keys = (
        "width",
        "graph_width",
        "graph_layers",
        "graph_heads",
        "dropout",
        "threat_cpa_horizon_s",
        "threat_time_scale_s",
        "threat_prior_strength",
        "missile_distance_floor_m",
        "missile_distance_power",
    )
    if architecture == ARCHITECTURE:
        keys += ("tcn_kernel_size", "tcn_dilations")
    else:
        keys += ("transformer_layers", "transformer_heads", "transformer_ff_width")
    return {key: config[key] for key in keys}


def make_model(config, data):
    options = model_configuration(config)
    model_type = (
        RelativeMotionHGTTCN if config["architecture"] == ARCHITECTURE
        else RelativeMotionHGTTransformer
    )
    return model_type(**options, mean=data.mean, std=data.std)


def masked_huber_loss(prediction, target, valid, beta=1.0):
    if prediction.shape != target.shape or prediction.shape[-2:] != (3, 6):
        raise ValueError("Expected matching (batch, 3, 6) predictions and targets")
    if valid.shape != prediction.shape[:2] or not valid.any():
        raise ValueError("Expected at least one valid target horizon")
    loss = nn.functional.smooth_l1_loss(prediction, target, beta=beta, reduction="none").mean(-1)
    return (loss * valid).sum() / valid.sum()


def l1_penalty(model):
    values = [parameter.abs().sum() for parameter in model.parameters() if parameter.ndim > 1]
    return torch.stack(values).sum() if values else next(model.parameters()).new_zeros(())


def effective_precision(device, requested):
    if requested not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    if requested == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def _autocast(device, precision):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16",
    )


def _batch(data, ids, device):
    ids = np.asarray(ids, dtype=np.int64)
    histories = torch.from_numpy(data.history(ids).astype(np.float32)).to(device)
    targets = torch.from_numpy(
        (data.residuals[ids] / data.target_scale).astype(np.float32)
    ).to(device)
    valid = torch.from_numpy(data.valid[ids]).to(device)
    return histories, targets, valid


def _metric_accumulators():
    return [
        {
            "count": 0,
            "normalized_squared": 0.0,
            "position": 0.0,
            "position_squared": 0.0,
            "velocity": 0.0,
            "velocity_squared": 0.0,
            "baseline_position": 0.0,
            "baseline_position_squared": 0.0,
            "baseline_velocity": 0.0,
            "baseline_velocity_squared": 0.0,
            "component_squared": np.zeros(6, dtype=np.float64),
        }
        for _ in HORIZONS
    ]


def evaluate(model, data, split, device, batch_size, huber_beta=1.0, collect=False):
    model.eval()
    ids = data.indices[split]
    if not len(ids):
        raise ValueError(f"No samples in {split}")
    accumulators = _metric_accumulators()
    loss_sum, valid_count = 0.0, 0
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            x, target_scaled, valid = _batch(data, batch_ids, device)
            with _autocast(device, "fp32"):
                prediction_scaled = model(x).float()
            loss = nn.functional.smooth_l1_loss(
                prediction_scaled, target_scaled, beta=huber_beta, reduction="none"
            ).mean(-1)
            loss_sum += float((loss * valid).sum().cpu())
            valid_count += int(valid.sum().cpu())
            prediction_scaled = prediction_scaled.cpu().numpy()
            predicted = (
                data.baseline[batch_ids]
                + prediction_scaled * data.target_scale[None]
            )
            truth = data.targets[batch_ids]
            baseline = data.baseline[batch_ids]
            mask = data.valid[batch_ids]
            if collect:
                predictions.append(predicted.astype(np.float32))
            normalized_error = prediction_scaled - (
                data.residuals[batch_ids] / data.target_scale[None]
            )
            physical_error = predicted - truth
            baseline_error = baseline - truth
            for horizon in range(len(HORIZONS)):
                selected = mask[:, horizon]
                if not selected.any():
                    continue
                errors = physical_error[selected, horizon]
                reference = baseline_error[selected, horizon]
                position = np.linalg.norm(errors[:, :3], axis=1)
                velocity = np.linalg.norm(errors[:, 3:], axis=1)
                baseline_position = np.linalg.norm(reference[:, :3], axis=1)
                baseline_velocity = np.linalg.norm(reference[:, 3:], axis=1)
                accumulator = accumulators[horizon]
                accumulator["count"] += int(selected.sum())
                accumulator["normalized_squared"] += float(
                    np.square(normalized_error[selected, horizon]).sum()
                )
                accumulator["position"] += float(position.sum())
                accumulator["position_squared"] += float(np.square(position).sum())
                accumulator["velocity"] += float(velocity.sum())
                accumulator["velocity_squared"] += float(np.square(velocity).sum())
                accumulator["baseline_position"] += float(baseline_position.sum())
                accumulator["baseline_position_squared"] += float(
                    np.square(baseline_position).sum()
                )
                accumulator["baseline_velocity"] += float(baseline_velocity.sum())
                accumulator["baseline_velocity_squared"] += float(
                    np.square(baseline_velocity).sum()
                )
                accumulator["component_squared"] += np.square(errors).sum(axis=0)
    if not valid_count:
        raise ValueError(f"No valid targets in {split}")
    rows = []
    for horizon, seconds in enumerate(HORIZONS):
        values = accumulators[horizon]
        count = values["count"]
        if not count:
            raise ValueError(f"No valid {split} targets at +{seconds:g}s")
        component_rmse = np.sqrt(values["component_squared"] / count)
        rows.append(
            {
                "split": split,
                "horizon_s": float(seconds),
                "samples": count,
                "normalized_rmse": math.sqrt(values["normalized_squared"] / (count * 6)),
                "position_mae_m": values["position"] / count,
                "position_rmse_m": math.sqrt(values["position_squared"] / count),
                "velocity_mae_mps": values["velocity"] / count,
                "velocity_rmse_mps": math.sqrt(values["velocity_squared"] / count),
                "baseline_position_mae_m": values["baseline_position"] / count,
                "baseline_position_rmse_m": math.sqrt(
                    values["baseline_position_squared"] / count
                ),
                "baseline_velocity_mae_mps": values["baseline_velocity"] / count,
                "baseline_velocity_rmse_mps": math.sqrt(
                    values["baseline_velocity_squared"] / count
                ),
                **{
                    f"{field}_rmse": float(component_rmse[index])
                    for index, field in enumerate(TARGET_FIELDS)
                },
            }
        )
    return {
        "loss": loss_sum / valid_count,
        "metrics": pd.DataFrame(rows),
        "predictions": np.concatenate(predictions) if collect else None,
        "ids": ids,
    }


def _warmup_cosine(update, total, warmup, start_ratio, minimum_ratio):
    if warmup and update < warmup:
        return start_ratio + (1 - start_ratio) * update / max(1, warmup - 1)
    progress = (update - warmup) / max(1, total - warmup - 1)
    return minimum_ratio + 0.5 * (1 - minimum_ratio) * (1 + math.cos(math.pi * progress))


def _atomic_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    history,
    best_loss,
    best_epoch,
    stale_epochs,
    optimizer_steps,
    rng,
    config,
    data,
    run,
):
    return {
        "model_contract": run["model_contract"],
        "model_state": model.state_dict(),
        "model_config": model_configuration(config),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "history": history,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "optimizer_steps": optimizer_steps,
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
        "training": config,
        "data_manifest_sha256": run["data_manifest_sha256"],
        "feature_names": data.manifest["features"],
        "target_contract": data.manifest["target_contract"],
        "feature_mean": data.mean,
        "feature_std": data.std,
        "target_scale": data.target_scale,
    }


def train_model(data, output, config=None, device=None, resume=False, callback=None):
    if not isinstance(data, RelativeMotionData) or data.manifest["data_contract"] != DATA_CONTRACT:
        raise ValueError("Expected notebook-33 relative-motion data")
    overrides = config or {}
    architecture = overrides.get("architecture", ARCHITECTURE)
    config = {**default_training_config(architecture), **overrides}
    output = Path(output)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if config["seed"] < 0 or config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("Seed, epochs, and batch size are invalid")
    if config["lr_schedule"] != "warmup_cosine":
        raise ValueError("Only warmup_cosine scheduling is supported")
    if config["cpu_threads"] < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(int(config["cpu_threads"]))
    torch.manual_seed(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))
    precision = effective_precision(device, config["precision"])
    model = make_model(config, data).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    steps_per_epoch = math.ceil(len(data.indices["train"]) / config["batch_size"])
    total_updates = config["epochs"] * steps_per_epoch
    warmup_updates = min(
        total_updates - 1, max(1, round(total_updates * config["warmup_fraction"]))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda update: _warmup_cosine(
            update,
            total_updates,
            warmup_updates,
            config["warmup_start_ratio"],
            config["min_lr_ratio"],
        ),
    )
    manifest_hash = sha256(data.directory / "manifest.json")
    source_path = Path(__file__)
    architecture_path = source_path.parents[1] / "enemy_missile_models.py"
    run = {
        "model_contract": MODEL_CONTRACTS[architecture],
        "architecture": architecture,
        "model": model_configuration(config),
        "training": config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "data_manifest_sha256": manifest_hash,
        "data_contract": data.manifest["data_contract"],
        "feature_names": data.manifest["features"],
        "input_contract": data.manifest["input_contract"],
        "target_contract": data.manifest["target_contract"],
        "target_fields": data.manifest["target_fields"],
        "horizons_s": data.manifest["horizons_s"],
        "history_steps": data.manifest["history_steps"],
        "source_sha256": sha256(source_path),
        "hgt_source_sha256": sha256(architecture_path),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "device": str(device),
        "effective_precision": precision,
        "selection": "Lowest equal-horizon mean test normalized RMSE",
        "validation_policy": "Validation evaluated only after checkpoint selection",
    }
    rng = np.random.default_rng(config["seed"])
    history, best_loss, best_epoch, stale_epochs = [], float("inf"), 0, 0
    optimizer_steps, first_epoch = 0, 1
    if resume:
        saved_run = read_json(output / "run.json")
        for key in ("model_contract", "model", "training", "data_manifest_sha256", "source_sha256", "hgt_source_sha256"):
            if saved_run.get(key) != run.get(key):
                raise ValueError("Resume requires identical model, data, configuration, and source")
        saved = torch.load(output / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        history = saved["history"]
        best_loss, best_epoch = saved["best_loss"], saved["best_epoch"]
        stale_epochs, optimizer_steps = saved["stale_epochs"], saved["optimizer_steps"]
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        first_epoch = saved["epoch"] + 1
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Run directory exists; resume it or choose a fresh directory")
        (output / "checkpoints").mkdir(parents=True)
        (output / "epochs").mkdir()
        write_json(output / "run.json", run)
        shutil.copy2(source_path, output / "training_source.py")
        shutil.copy2(architecture_path, output / "hgt_source.py")

    minimum_epoch = int(config["early_stopping_min_epochs"])
    patience = int(config["early_stopping_patience"])
    for epoch in range(first_epoch, config["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        order = rng.permutation(data.indices["train"])
        batch_losses = []
        for start in range(0, len(order), config["batch_size"]):
            x, target, valid = _batch(data, order[start:start + config["batch_size"]], device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                prediction = model(x)
                data_loss = masked_huber_loss(
                    prediction.float(), target, valid, config["huber_beta"]
                )
                penalty = config["l1_lambda"] * l1_penalty(model)
                loss = data_loss + penalty
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1
            batch_losses.append(float(data_loss.detach().cpu()))
        train_result = evaluate(
            model,
            data,
            "train",
            device,
            config["evaluation_batch_size"],
            config["huber_beta"],
        )
        test_result = evaluate(
            model,
            data,
            "test",
            device,
            config["evaluation_batch_size"],
            config["huber_beta"],
        )
        epoch_metrics = pd.concat(
            (train_result["metrics"], test_result["metrics"]), ignore_index=True
        )
        selection_loss = float(test_result["metrics"].normalized_rmse.mean())
        improved = selection_loss < best_loss
        if improved:
            best_loss, best_epoch, stale_epochs = selection_loss, epoch, 0
        elif epoch >= minimum_epoch:
            stale_epochs += 1
        stopped = bool(
            config["early_stopping"] and epoch >= minimum_epoch and stale_epochs >= patience
        )
        row = {
            "epoch": epoch,
            "train_huber": train_result["loss"],
            "test_huber": test_result["loss"],
            "test_normalized_rmse": selection_loss,
            "train_batch_huber": float(np.mean(batch_losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "improved": improved,
            "best_epoch": best_epoch,
            "best_test_normalized_rmse": best_loss,
            "stale_epochs": stale_epochs,
            "early_stopped": stopped,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        epoch_metrics.to_csv(output / "epochs" / f"epoch_{epoch:03d}_metrics.csv", index=False)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            history,
            best_loss,
            best_epoch,
            stale_epochs,
            optimizer_steps,
            rng,
            config,
            data,
            run,
        )
        if improved:
            checkpoint = output / "checkpoints" / f"epoch_{epoch:03d}.pt"
            _atomic_checkpoint(checkpoint, payload)
            temporary = output / "best.tmp"
            shutil.copy2(checkpoint, temporary)
            temporary.replace(output / "best.pt")
        _atomic_checkpoint(output / "last.pt", payload)
        write_json(
            output / "training_status.json",
            {
                "status": "early_stopped" if stopped else "running",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_test_normalized_rmse": best_loss,
            },
        )
        if callback:
            callback(row, epoch_metrics)
        if stopped:
            break
    final_status = "early_stopped" if history[-1]["early_stopped"] else "completed"
    write_json(
        output / "training_status.json",
        {
            "status": final_status,
            "epoch": history[-1]["epoch"],
            "best_epoch": best_epoch,
            "best_test_normalized_rmse": best_loss,
        },
    )
    return pd.DataFrame(history)


def load_selected_model(data, run, device="cpu"):
    run = Path(run)
    device = torch.device(device)
    metadata = read_json(run / "run.json")
    architecture = metadata.get("architecture")
    contract = MODEL_CONTRACTS.get(architecture)
    if contract is None or metadata.get("model_contract") != contract:
        raise ValueError("Unsupported relative-motion HGT-v2 run")
    if metadata["data_manifest_sha256"] != sha256(data.directory / "manifest.json"):
        raise ValueError("Run and prepared dataset differ")
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
    if checkpoint.get("model_contract") != contract:
        raise ValueError("Checkpoint contract differs")
    model = make_model({**checkpoint["model_config"], "architecture": architecture}, data)
    model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), checkpoint


def final_evaluation(data, run, device="cpu", batch_size=1024, overwrite=False):
    run = Path(run)
    required = [run / "final_metrics.csv"] + [run / f"predictions_{split}.npz" for split in SPLITS]
    if not overwrite and all(path.exists() for path in required):
        return pd.read_csv(run / "final_metrics.csv")
    model, checkpoint = load_selected_model(data, run, device)
    rows = []
    for split in SPLITS:
        result = evaluate(
            model,
            data,
            split,
            torch.device(device),
            batch_size,
            checkpoint["training"]["huber_beta"],
            collect=True,
        )
        rows.append(result["metrics"])
        ids = result["ids"]
        np.savez_compressed(
            run / f"predictions_{split}.npz",
            sample_ids=ids,
            prediction=result["predictions"],
            target=data.targets[ids],
            baseline=data.baseline[ids],
            valid=data.valid[ids],
        )
    metrics = pd.concat(rows, ignore_index=True)
    metrics.to_csv(run / "final_metrics.csv", index=False)
    write_json(
        run / "selection.json",
        {
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_test_normalized_rmse": float(checkpoint["best_loss"]),
            "validation_accessed_after_selection": True,
        },
    )
    return metrics

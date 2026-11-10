"""Slot-independent frame encoders for notebook 17's observer-centred histories.

Only frame-local missile sets are encoded: compact slots are never treated as
persistent tracks. Both models use notebook 18's temporal Transformer/readout.
The original training-only normalisation is stored in checkpoint buffers so
geometry and absence are recovered before using slot-independent encoders.
"""

import math

import torch
from torch import nn


HISTORY_STEPS = 51
HGT_THREAT_EDGE_FEATURES = (
    "relative_position_n", "relative_position_w", "relative_position_u",
    "relative_velocity_n", "relative_velocity_w", "relative_velocity_u", "distance", "closing_speed",
    "closest_approach_time", "closest_approach_distance", "bearing_alignment", "relative_speed",
    "source_kinetic_energy", "source_potential_energy", "source_total_energy",
    "destination_kinetic_energy", "destination_potential_energy", "destination_total_energy",
)


def masked_summary(values, present):
    """Mean, maximum and fraction present; empty sets have an all-zero summary."""
    mask = present.unsqueeze(-1)
    count = mask.sum(-2)
    mean = (values * mask).sum(-2) / count.clamp_min(1)
    maximum = values.masked_fill(~mask, torch.finfo(values.dtype).min).amax(-2)
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum, count.to(values.dtype) / values.shape[-2]), -1)


class _TypedMessageBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.relation = nn.Embedding(5, 8)
        self.message = nn.Sequential(nn.Linear(2 * width + 16, width), nn.GELU(),
                                     nn.Linear(width, width))
        self.update = nn.Sequential(nn.Linear(3 * width + 1, width), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, nodes, present, geometry, sources, destinations, relations):
        # Residual states and autocast Linear outputs need not share a dtype.
        # Keep reductions, their destinations and the max-mask sentinel in FP32;
        # using an FP32 minimum to fill BF16 messages overflows before reduction.
        nodes = nodes.float()
        source = nodes[:, sources]
        target = nodes[:, destinations]
        edge_type = self.relation(relations).to(nodes.dtype).expand(nodes.shape[0], -1, -1)
        features = torch.cat((source, target, geometry.to(nodes.dtype), edge_type), -1)
        messages = self.message(features).float()
        mask = (present[:, sources] & present[:, destinations]).unsqueeze(-1)
        destinations = destinations[None, :, None].expand_as(messages)
        summed = torch.zeros_like(nodes).scatter_add(1, destinations, messages * mask)
        count = nodes.new_zeros((*nodes.shape[:2], 1)).scatter_add(
            1, destinations[..., :1], mask.to(nodes.dtype))
        minimum = torch.finfo(messages.dtype).min
        maximum = torch.full_like(nodes, minimum).scatter_reduce(
            1, destinations, messages.masked_fill(~mask, minimum),
            reduce="amax", include_self=True)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        aggregate = torch.cat((nodes, summed / count.clamp_min(1), maximum, count / 9), -1)
        result = self.norm(nodes + self.update(aggregate).float())
        return result * present.unsqueeze(-1)


class _HGTBlock(nn.Module):
    """HGT with four node types, relation transforms and geometry-conditioned edges."""

    TYPE_SLICES = ((0, 1), (1, 2), (2, 6), (6, 10))

    def __init__(self, width, heads, dropout, edge_features=8, relation_family_gates=False):
        super().__init__()
        if not isinstance(heads, int) or isinstance(heads, bool) or heads < 1 or width % heads:
            raise ValueError("graph_heads must be positive and divide graph_width")
        self.heads, self.head_width = heads, width // heads
        self.relation_family_gates = relation_family_gates
        self.query = nn.ModuleList([nn.Linear(width, width) for _ in range(4)])
        self.key = nn.ModuleList([nn.Linear(width, width) for _ in range(4)])
        self.value = nn.ModuleList([nn.Linear(width, width) for _ in range(4)])
        self.output = nn.ModuleList([nn.Linear(width, width) for _ in range(4)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.skip = nn.Parameter(torch.zeros(4))
        self.relation_attention = nn.Parameter(torch.empty(5, heads, self.head_width, self.head_width))
        self.relation_message = nn.Parameter(torch.empty(5, heads, self.head_width, self.head_width))
        # HGT's learned priority for each (source type, relation, destination type).
        self.relation_priority = nn.Parameter(torch.ones(4 * 5 * 4, heads))
        self.edge_attention = nn.Linear(edge_features, heads, bias=False)
        self.edge_message = nn.Linear(edge_features, width, bias=False)
        if relation_family_gates:
            self.family_embedding = nn.Embedding(3, width)
            self.family_gate = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(),
                                             nn.Linear(width, heads))
        self.dropout = nn.Dropout(dropout)
        for parameter in (self.relation_attention, self.relation_message):
            for relation in range(5):
                for head in range(heads):
                    nn.init.xavier_uniform_(parameter[relation, head])

    def _project(self, nodes, projections):
        return torch.cat([layer(nodes[:, start:end]) for layer, (start, end)
                          in zip(projections, self.TYPE_SLICES)], 1)

    def forward(self, nodes, present, geometry, sources, destinations, relations,
                roles, edge_log_prior, return_attention=False, return_family_gates=False):
        nodes = nodes.float()
        shape = (*nodes.shape[:2], self.heads, self.head_width)
        query = self._project(nodes, self.query).reshape(shape).float()[:, destinations]
        key = self._project(nodes, self.key).reshape(shape).float()[:, sources]
        value = self._project(nodes, self.value).reshape(shape).float()[:, sources]
        edge_scores = self.edge_attention(geometry).float()
        edge_values = self.edge_message(geometry).float().reshape(
            *geometry.shape[:2], self.heads, self.head_width)
        meta = (roles[sources] * 5 + relations) * 4 + roles[destinations]
        mask = (present[:, sources] & present[:, destinations]).unsqueeze(-1)
        # Attention, softmax and reductions remain FP32 during BF16 training.
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            key = torch.einsum("behd,ehdf->behf", key, self.relation_attention[relations].float())
            value = torch.einsum("behd,ehdf->behf", value, self.relation_message[relations].float())
            scores = (query * key).sum(-1) / math.sqrt(self.head_width)
            scores = scores * self.relation_priority[meta].float() + edge_scores
            scores = scores + (edge_log_prior.float().unsqueeze(-1)
                               if edge_log_prior.ndim == 2 else edge_log_prior.float())
            # Ownership directions share one family; targeting directions another.
            family = torch.where(relations == 0, 0, torch.where(relations <= 2, 1, 2))
            groups = 3 if self.relation_family_gates else 1
            destination_group = destinations * groups + (family if groups == 3 else 0)
            index = destination_group[None, :, None].expand_as(scores)
            grouped_shape = (nodes.shape[0], nodes.shape[1] * groups, self.heads)
            minimum = torch.finfo(scores.dtype).min
            masked_scores = scores.masked_fill(~mask, minimum)
            maxima = nodes.new_full(grouped_shape, minimum).scatter_reduce(
                1, index, masked_scores, reduce="amax", include_self=True)
            numerator = (masked_scores - maxima.gather(1, index)).exp() * mask
            denominator = nodes.new_zeros(grouped_shape).scatter_add(1, index, numerator)
            attention = numerator / denominator.gather(1, index).clamp_min(1e-12)
            messages = (value + edge_values) * attention.unsqueeze(-1)
            aggregated = nodes.new_zeros((*grouped_shape, self.head_width)).scatter_add(
                1, index.unsqueeze(-1).expand_as(messages), messages)
            family_weights = None
            if self.relation_family_gates:
                summaries = aggregated.reshape(*nodes.shape[:2], 3, self.heads, self.head_width)
                gate_inputs = torch.cat((nodes[:, :, None].expand(-1, -1, 3, -1),
                                         summaries.flatten(-2),
                                         self.family_embedding.weight[None, None].expand(
                                             *nodes.shape[:2], -1, -1)), -1)
                gate_scores = self.family_gate(gate_inputs).float()
                active = denominator.reshape(*nodes.shape[:2], 3, self.heads) > 0
                gate_scores = gate_scores.masked_fill(~active, minimum)
                gate_values = (gate_scores - gate_scores.amax(2, keepdim=True)).exp() * active
                family_weights = gate_values / gate_values.sum(2, keepdim=True).clamp_min(1e-12)
                aggregated = (summaries * family_weights.unsqueeze(-1)).sum(2).flatten(-2)
                # Return effective edge weights, including the learned family gate.
                attention = attention * family_weights.flatten(1, 2).gather(1, index)
            else:
                aggregated = aggregated.flatten(-2)
        updated = self._project(torch.nn.functional.gelu(aggregated), self.output).float()
        updated = self.dropout(updated)
        result = []
        for type_id, (start, end) in enumerate(self.TYPE_SLICES):
            gate = self.skip[type_id].sigmoid()
            result.append(self.norm[type_id](gate * updated[:, start:end]
                                            + (1 - gate) * nodes[:, start:end]))
        result = torch.cat(result, 1) * present.unsqueeze(-1)
        if return_family_gates:
            return result, attention, family_weights
        return (result, attention) if return_attention else result


class _MissileTransformer(nn.Module):
    def __init__(self, num_classes=33, width=96, heads=6, layers=2, dropout=.25,
                 input_features=115, mean=None, std=None, graph_width=64, graph_layers=2,
                 observer_acceleration=False, target_relative_missiles=False,
                 target_relative_missile_velocities=False):
        super().__init__()
        if not isinstance(observer_acceleration, bool):
            raise ValueError("observer_acceleration must be boolean")
        self.observer_acceleration = observer_acceleration
        if (not isinstance(target_relative_missiles, bool)
                or (target_relative_missiles and not observer_acceleration)):
            raise ValueError("target_relative_missiles must be boolean and requires observer_acceleration")
        self.target_relative_missiles = target_relative_missiles
        if (not isinstance(target_relative_missile_velocities, bool)
                or (target_relative_missile_velocities and not target_relative_missiles)):
            raise ValueError("target_relative_missile_velocities must be boolean and requires target_relative_missiles")
        # Default False preserves geometry when restoring earlier position-only checkpoints.
        self.target_relative_missile_velocities = target_relative_missile_velocities
        if input_features != 82 + num_classes:
            raise ValueError("Expected notebook-17 features: 82 numeric plus one own-skill per class")
        if width < 1 or heads < 1 or width % heads or layers < 1:
            raise ValueError("Positive width/layers and a width divisible by heads are required")
        if graph_width < 1 or graph_layers < 1:
            raise ValueError("graph_width and graph_layers must be positive")
        mean = torch.zeros(input_features) if mean is None else torch.as_tensor(mean)
        std = torch.ones(input_features) if std is None else torch.as_tensor(std)
        if (mean.shape != (input_features,) or std.shape != (input_features,)
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or (std <= 0).any()):
            raise ValueError("Supply finite feature mean and positive std in notebook-17 order")
        self.register_buffer("feature_mean", mean.detach().clone().float())
        self.register_buffer("feature_std", std.detach().clone().float())
        self.num_classes, self.input_features = num_classes, input_features
        self.graph_width = graph_width
        # All non-missile channels are retained by these two aircraft encoders.
        self.observer = nn.Sequential(nn.Linear(12 + num_classes, graph_width), nn.GELU())
        self.opponent = nn.Sequential(nn.Linear(14, graph_width), nn.GELU())
        self.missile = nn.Sequential(nn.Linear(7, graph_width), nn.GELU(),
                                     nn.Linear(graph_width, graph_width), nn.GELU())
        self.role = nn.Embedding(4, graph_width)
        self.projection = nn.Linear(6 * graph_width + 2, width)
        self.position = nn.Parameter(torch.randn(1, HISTORY_STEPS, width) * .02)
        block = nn.TransformerEncoderLayer(width, heads, width * 4, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(width, 3 * num_classes))
        self.register_buffer("roles", torch.tensor([0, 1, 2, 2, 2, 2, 3, 3, 3, 3]))

    def _decode(self, windows):
        if windows.ndim != 3 or tuple(windows.shape[1:]) != (HISTORY_STEPS, self.input_features):
            raise ValueError(f"Expected (batch, 51, {self.input_features}) feature histories")
        x = windows.reshape(-1, self.input_features).float()
        raw = x * self.feature_std + self.feature_mean
        offset = 3 if self.observer_acceleration else 0
        slot_indices = torch.cat((torch.arange(19 - offset, 47 - offset, device=x.device),
                                  torch.arange(54, 82, device=x.device)))
        slots = raw[:, slot_indices].reshape(-1, 8, 7)
        # Comparing in standardised coordinates handles slot-specific statistics
        # without assuming that standardised absence is zero. Tolerance covers
        # floating-point roundoff at a raw-zero sentinel, not missing-track logic.
        sentinel = (-self.feature_mean / self.feature_std)[slot_indices]
        tolerance = 8 * torch.finfo(x.dtype).eps * (1 + sentinel.abs())
        present = ((x[:, slot_indices] - sentinel).abs() > tolerance).reshape(-1, 8, 7).any(-1)
        slots = slots * present.unsqueeze(-1)
        observer_parts = [raw[:, :9 - offset], raw[:, 18 - offset:19 - offset],
                          raw[:, 47 - offset:49 - offset]]
        if self.observer_acceleration:
            observer_parts.append(raw[:, 51:54])
        observer = torch.cat((*observer_parts, raw[:, 82:]), -1)
        # Opponent acceleration follows the 51 original channels (48 in notebook 22).
        opponent = torch.cat((raw[:, 9 - offset:18 - offset], raw[:, 49 - offset:54 - offset]), -1)
        aircraft = torch.stack((self.observer(observer), self.opponent(opponent)), 1)
        missiles = self.missile(slots)
        nodes = torch.cat((aircraft, missiles), 1)
        nodes = nodes + self.role(self.roles).to(nodes.dtype)
        node_present = torch.cat((torch.ones_like(present[:, :2]), present), -1)
        nodes = nodes * node_present.unsqueeze(-1)
        return raw, slots, nodes, node_present

    def _spatial(self, raw, slots, nodes, present):
        return nodes

    def encode_frames(self, windows):
        """Encode frames independently; missile slot histories are never joined."""
        raw, slots, nodes, present = self._decode(windows)
        nodes = self._spatial(raw, slots, nodes, present)
        readout = torch.cat((nodes[:, 0], nodes[:, 1],
                             masked_summary(nodes[:, 2:6], present[:, 2:6]),
                             masked_summary(nodes[:, 6:], present[:, 6:])), -1)
        return self.projection(readout).reshape(windows.shape[0], HISTORY_STEPS, -1)

    def forward(self, windows):
        encoded = self.encoder(self.encode_frames(windows) + self.position)
        return self.head(self.norm(encoded[:, -1])).reshape(-1, 3, self.num_classes)


class MissileSetTransformer(_MissileTransformer):
    """Shared missile encoder and separate friendly/enemy pooling; no messages."""


class TypedGraphTransformer(_MissileTransformer):
    """Two aircraft and eight optional missiles, with typed directed messages.

    Edges connect the aircraft pair and each missile to both aircraft in both
    directions. Ownership and potential-opponent relationships are distinct;
    no unrecorded seeker lock or target identity is assumed.
    """

    def __init__(self, *args, graph_width=64, graph_layers=2, **kwargs):
        super().__init__(*args, graph_width=graph_width, graph_layers=graph_layers, **kwargs)
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
        dropout = kwargs.get("dropout", .25)
        self.messages = nn.ModuleList([_TypedMessageBlock(graph_width, dropout)
                                       for _ in range(graph_layers)])

    def _world_states(self, raw, slots):
        scale = raw.new_tensor([50000, 50000, 10000])
        offset = 3 if self.observer_acceleration else 0
        relative_position = raw[:, None, 9 - offset:12 - offset]
        missile_positions = slots[:, :, :3] * scale
        if self.target_relative_missiles:
            # Node features keep the target-relative friendly geometry. Edges need
            # every position in one common observer-centred world coordinate frame.
            missile_positions = torch.cat((missile_positions[:, :4] + relative_position * scale,
                                           missile_positions[:, 4:]), 1)
        positions = torch.cat((torch.zeros_like(relative_position),
                               relative_position * scale, missile_positions), 1)
        own_velocity = raw[:, None, 3 - offset:6 - offset] * 600
        relative_velocity = raw[:, None, 12 - offset:15 - offset] * 600
        missile_velocities = own_velocity + slots[:, :, 3:6] * 1200
        if self.target_relative_missile_velocities:
            # Friendly inputs subtract opponent velocity; edges use common world velocities.
            missile_velocities = torch.cat((missile_velocities[:, :4] + relative_velocity,
                                            missile_velocities[:, 4:]), 1)
        velocities = torch.cat((own_velocity, own_velocity + relative_velocity,
                                missile_velocities), 1)
        return positions, velocities

    def _edge_geometry(self, raw, slots):
        scale = raw.new_tensor([50000, 50000, 10000])
        positions, velocities = self._world_states(raw, slots)
        delta_position = positions[:, self.sources] - positions[:, self.destinations]
        delta_velocity = velocities[:, self.sources] - velocities[:, self.destinations]
        distance = torch.linalg.vector_norm(delta_position, dim=-1, keepdim=True)
        closing = -(delta_position * delta_velocity).sum(-1, keepdim=True) / distance.clamp_min(1)
        geometry = torch.cat((delta_position / scale, delta_velocity / 1200,
                              distance / 50000, closing / 1200), -1)
        return geometry

    def _spatial(self, raw, slots, nodes, present):
        geometry = self._edge_geometry(raw, slots)
        for block in self.messages:
            nodes = block(nodes, present, geometry, self.sources, self.destinations, self.relations)
        return nodes


class HGTTransformer(TypedGraphTransformer):
    """HGT v1 preserves saved models; v2 adds gated families and learned threat features."""

    def __init__(self, *args, graph_heads=4, missile_distance_floor_m=1000.0,
                 missile_distance_power=1.0, graph_width=64, graph_layers=2, hgt_version=1,
                 threat_cpa_horizon_s=30.0, threat_time_scale_s=5.0,
                 threat_prior_strength=1.0, **kwargs):
        if (not math.isfinite(missile_distance_floor_m) or missile_distance_floor_m <= 0
                or not math.isfinite(missile_distance_power) or missile_distance_power < 0):
            raise ValueError("Missile distance floor must be positive and power nonnegative, both finite")
        if isinstance(hgt_version, bool) or hgt_version not in (1, 2):
            raise ValueError("hgt_version must be 1 or 2")
        for name, value in (("threat_cpa_horizon_s", threat_cpa_horizon_s),
                            ("threat_time_scale_s", threat_time_scale_s),
                            ("threat_prior_strength", threat_prior_strength)):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        super().__init__(*args, graph_width=graph_width, graph_layers=graph_layers, **kwargs)
        self.hgt_version = hgt_version
        self.threat_cpa_horizon_s = float(threat_cpa_horizon_s)
        self.threat_time_scale_s = float(threat_time_scale_s)
        self.missile_distance_floor_m = float(missile_distance_floor_m)
        self.missile_distance_power = float(missile_distance_power)
        self.messages = nn.ModuleList([_HGTBlock(
            graph_width, graph_heads, kwargs.get("dropout", .25),
            edge_features=len(HGT_THREAT_EDGE_FEATURES) if hgt_version == 2 else 8,
            relation_family_gates=hgt_version == 2)
                                       for _ in range(graph_layers)])
        if hgt_version == 2:
            # A nonnegative strength per attention head, plus a learnable residual
            # from all physical edge features. Start from the physical threat prior.
            initial = threat_prior_strength + math.log(-math.expm1(-threat_prior_strength))
            self.threat_prior_raw_strength = nn.Parameter(torch.full((graph_heads,), initial))
            self.threat_prior = nn.Sequential(nn.Linear(len(HGT_THREAT_EDGE_FEATURES), 32),
                                               nn.GELU(), nn.Linear(32, graph_heads))
            nn.init.zeros_(self.threat_prior[-1].weight)
            nn.init.zeros_(self.threat_prior[-1].bias)
        self.register_buffer("target_edge_indices", torch.where(self.relations == 4)[0])
        self.register_buffer("edge_missile_indices",
                             (torch.maximum(self.sources, self.destinations) - 2).clamp_min(0))

    def _edge_geometry(self, raw, slots):
        geometry = super()._edge_geometry(raw, slots)
        if self.hgt_version == 1:
            return geometry
        positions, velocities = self._world_states(raw, slots)
        r = positions[:, self.sources] - positions[:, self.destinations]
        v = velocities[:, self.sources] - velocities[:, self.destinations]
        approach_time = (-(r * v).sum(-1) / v.square().sum(-1).clamp_min(1)).clamp(
            min=0, max=self.threat_cpa_horizon_s)
        approach_distance = torch.linalg.vector_norm(r + approach_time.unsqueeze(-1) * v, dim=-1)
        distance = torch.linalg.vector_norm(r, dim=-1)
        source_velocity = velocities[:, self.sources]
        bearing_alignment = (-(r * source_velocity).sum(-1) / (
            distance * torch.linalg.vector_norm(source_velocity, dim=-1)).clamp_min(1)).clamp(-1, 1)
        relative_speed = torch.linalg.vector_norm(v, dim=-1)
        offset = 3 if self.observer_acceleration else 0
        # obs_06 retains altitude even when the absolute-position channels are removed.
        altitude = raw[:, None, 6 - offset] * 10000 + positions[:, :, 2]
        kinetic = .5 * velocities.square().sum(-1)
        potential = 9.80665 * altitude
        # Masses are not recorded: these are specific energies (J/kg), scaled by 1e6.
        energy = torch.stack((kinetic, potential, kinetic + potential), -1) / 1e6
        threat = torch.stack((approach_time / self.threat_cpa_horizon_s,
                              approach_distance / 50000, bearing_alignment, relative_speed / 1200), -1)
        return torch.cat((geometry, threat, energy[:, self.sources], energy[:, self.destinations]), -1)

    def _edge_log_prior(self, geometry, present):
        # There is one missile->target edge for every missile slot, in slot order.
        distance = geometry[:, self.target_edge_indices, 6].float() * 50000
        occupied = present[:, 2:].reshape(-1, 2, 4)
        distance = distance.reshape(-1, 2, 4)
        nearest = distance.masked_fill(~occupied, float("inf")).amin(-1, keepdim=True)
        nearest = torch.where(occupied.any(-1, keepdim=True), nearest, 0)
        log_weight = self.missile_distance_power * (
            (nearest + self.missile_distance_floor_m).log()
            - (distance + self.missile_distance_floor_m).log())
        if self.hgt_version == 2:
            target_features = geometry[:, self.target_edge_indices].float()
            # CPA uses current relative velocity, bounded to the configured horizon.
            cpa_distance = target_features[:, :, 9].reshape(-1, 2, 4) * 50000
            cpa_time = target_features[:, :, 8].reshape(-1, 2, 4) * self.threat_cpa_horizon_s
            alignment = target_features[:, :, 10].reshape(-1, 2, 4)
            physical_score = (log_weight - torch.log1p(cpa_distance / self.missile_distance_floor_m)
                              - torch.log1p(cpa_time / self.threat_time_scale_s) + alignment)
            residual = self.threat_prior(target_features).float().reshape(*physical_score.shape, -1)
            score = (physical_score.unsqueeze(-1)
                     * nn.functional.softplus(self.threat_prior_raw_strength.float()) + residual)
            mask = occupied.unsqueeze(-1)
            maximum = score.masked_fill(~mask, torch.finfo(score.dtype).min).amax(2, keepdim=True)
            maximum = torch.where(mask.any(2, keepdim=True), maximum, 0)
            score = (score - maximum).masked_fill(~mask, 0).flatten(1, 2)
            return score[:, self.edge_missile_indices] * (self.relations >= 3)[None, :, None]
        log_weight = log_weight.masked_fill(~occupied, 0).flatten(1)
        # Owner edges and aircraft-pair edges have unit prior. Target edges use
        # the same prior in both directions; absence is handled by attention masks.
        return log_weight[:, self.edge_missile_indices] * (self.relations >= 3)

    def _spatial(self, raw, slots, nodes, present):
        geometry = self._edge_geometry(raw, slots)
        log_prior = self._edge_log_prior(geometry, present)
        for block in self.messages:
            nodes = block(nodes, present, geometry, self.sources, self.destinations,
                          self.relations, self.roles, log_prior)
        return nodes

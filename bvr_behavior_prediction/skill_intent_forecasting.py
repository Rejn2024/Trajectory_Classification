"""Frozen skill-intent supervision layered over notebook 24's verified dataset."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .nn_duel_skill_forecasting import TARGET_RELATIVE_MISSILE_DATA_CONTRACT, TargetRelativeMissileNNDuelData
from .paired_skill_forecasting import HORIZONS, SPLITS, read_json, sha256, write_json


INTENT_DATA_CONTRACT = "nn-duel-target-relative-missile-skill-intent-horizon-v2"
INTENT_CLASSES = ["offensive", "defensive", "context_dependent"]
INTENT_GROUPS = {
    "offensive": ["pursue_target", "lead_pursuit", "recommit", "lock_target", "commit", "launch",
                  "support_missile", "secondary_shot", "short_range_attack"],
    "defensive": ["preemptive_crank", "notch_left", "notch_right", "beam_missile_left",
                  "beam_missile_right", "drag_missile", "dive_defense", "last_ditch_break",
                  "defensive_reversal", "turn_cold", "abort_support"],
    "context_dependent": ["maintain_heading", "turn_to_heading", "climb_to_altitude", "descend_to_altitude",
                          "accelerate_to_speed", "decelerate_to_speed", "search", "lag_pursuit", "extend",
                          "beam_target_left", "beam_target_right", "crank_target_left", "crank_target_right"],
}
# Preserve the historical taxonomy for archived 33-label datasets. The simplified
# primitives describe physical controls and do not imply a tactical intent.
SIMPLIFIED_INTENT_GROUPS = {
    "offensive": ["launch"],
    "defensive": [],
    "context_dependent": ["change_trajectory"],
}
LEGACY_SIMPLIFIED_INTENT_GROUPS = {
    "context_dependent": ["turn_to_heading", "set_altitude", "set_speed"],
}
SKILL_TO_INTENT = {skill: group for groups in (INTENT_GROUPS, LEGACY_SIMPLIFIED_INTENT_GROUPS, SIMPLIFIED_INTENT_GROUPS)
                   for group, skills in groups.items() for skill in skills}
SIMPLIFIED_SKILLS = {skill for skills in SIMPLIFIED_INTENT_GROUPS.values() for skill in skills}
_NEW_SKILLS = (SIMPLIFIED_SKILLS | {skill for skills in LEGACY_SIMPLIFIED_INTENT_GROUPS.values() for skill in skills}) - {
    skill for skills in INTENT_GROUPS.values() for skill in skills
}
MIRROR_PAIRS = {family: [f"{family}_left", f"{family}_right"]
                for family in ("beam_missile", "beam_target", "crank_target", "notch")}


def hierarchy_policy(classes):
    """Freeze a complete partition of leaf skills into families and optional directions."""
    intent_policy(classes)  # Unknown catalogue entries must never silently form new families.
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Hierarchy requires distinct, nonempty skill classes")
    pairs = {}
    for family, leaves in MIRROR_PAIRS.items():
        included = set(leaves) & set(classes)
        if included and included != set(leaves):
            raise ValueError(f"Incomplete mirror pair: {family}; include both left and right")
        if included:
            pairs[family] = leaves
    leaf_family = {skill: family for family, leaves in pairs.items() for skill in leaves}
    families = sorted({leaf_family.get(skill, skill) for skill in classes})
    paired = sorted(pairs)
    version = "skill-family-direction-v2" if set(classes) & _NEW_SKILLS else "skill-family-direction-v1"
    return {"version": version, "classes": list(classes), "families": families,
            "paired_families": paired, "directions": ["left", "right"], "mirror_pairs": pairs,
            "skill_to_family": [families.index(leaf_family.get(skill, skill)) for skill in classes],
            "skill_to_pair": [paired.index(leaf_family[skill]) if skill in leaf_family else -1 for skill in classes],
            "skill_to_direction": [pairs[leaf_family[skill]].index(skill) if skill in leaf_family else -1
                                   for skill in classes],
            "family_to_pair": [paired.index(family) if family in paired else -1 for family in families],
            "factorization": "P(skill)=P(family)*P(direction|family); singleton P(skill)=P(family)",
            "missing": "family=-1 for missing skill; direction=-1 for missing or non-directional skill"}


def intent_policy(classes):
    unknown = set(classes) - set(SKILL_TO_INTENT)
    if unknown:
        raise ValueError(f"Unknown skills in frozen intent taxonomy: {sorted(unknown)}")
    return {
        "version": "skill-intent-v2" if set(classes) & _NEW_SKILLS else "skill-intent-v1",
        "classes": list(INTENT_CLASSES),
        "skill_to_intent": [INTENT_CLASSES.index(SKILL_TO_INTENT[skill]) for skill in classes],
        "mapping": {skill: SKILL_TO_INTENT[skill] for skill in classes},
        "meaning": "catalogue skill grouping; context_dependent does not infer latent tactical intent",
        "missing": "preserve each original skill target's -1 mask at each horizon",
    }


def map_intent_targets(targets, mapping):
    targets, mapping = np.asarray(targets), np.asarray(mapping)
    if (targets.ndim != 2 or targets.shape[1] != len(HORIZONS)
            or not np.issubdtype(targets.dtype, np.integer)
            or np.any(targets < -1) or np.any(targets >= len(mapping))):
        raise ValueError("Invalid skill targets for intent mapping")
    return np.where(targets >= 0, mapping[np.maximum(targets, 0)], -1).astype(np.int64)


def prepare_intent_dataset(base_directory, output):
    """Write only new supervision/metadata; the immutable base arrays remain in place."""
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty intent output directory or load the existing dataset")
    base = TargetRelativeMissileNNDuelData(Path(base_directory).expanduser().resolve())
    policy = intent_policy(base.classes)
    targets = map_intent_targets(base.targets, policy["skill_to_intent"])
    hierarchy = hierarchy_policy(base.classes)
    family_targets = map_intent_targets(base.targets, hierarchy["skill_to_family"])
    direction_targets = map_intent_targets(base.targets, hierarchy["skill_to_direction"])
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "intent_targets.npy", targets)
    write_json(output / "intent_policy.json", policy)
    write_json(output / "skill_hierarchy.json", hierarchy)
    np.save(output / "family_targets.npy", family_targets)
    np.save(output / "direction_targets.npy", direction_targets)
    pd.DataFrame({"skill": base.classes,
                  "family": [hierarchy["families"][f] for f in hierarchy["skill_to_family"]],
                  "family_index": hierarchy["skill_to_family"],
                  "pair_index": hierarchy["skill_to_pair"], "direction_index": hierarchy["skill_to_direction"],
                  "direction": [hierarchy["directions"][d] if d >= 0 else "none"
                                for d in hierarchy["skill_to_direction"]]}).to_csv(
                                    output / "skill_family_direction_mapping.csv", index=False)
    pd.DataFrame({"skill": base.classes,
                  "intent": [policy["mapping"][skill] for skill in base.classes],
                  "intent_index": policy["skill_to_intent"]}).to_csv(output / "skill_intent_mapping.csv", index=False)
    rows = []
    for split in SPLITS:
        ids = base.indices[split]
        for h, horizon in enumerate(HORIZONS):
            for c, name in enumerate(INTENT_CLASSES):
                selected = ids[targets[ids, h] == c]
                rows.append({"split": split, "horizon_s": horizon, "intent": name,
                             "windows": len(selected),
                             "episodes": int(base.samples.iloc[selected].episode_id.nunique()),
                             "scenario_groups": int(base.samples.iloc[selected].scenario_group.nunique())})
    pd.DataFrame(rows).to_csv(output / "intent_support_by_split.csv", index=False)
    family_rows, direction_rows = [], []
    for split in SPLITS:
        ids = base.indices[split]
        for h, horizon in enumerate(HORIZONS):
            for f, family in enumerate(hierarchy["families"]):
                selected = family_targets[ids, h] == f
                family_rows.append({"split": split, "horizon_s": horizon, "family": family,
                                    "windows": int(selected.sum())})
                if family in hierarchy["paired_families"]:
                    for d, direction in enumerate(hierarchy["directions"]):
                        direction_rows.append({"split": split, "horizon_s": horizon, "family": family,
                                               "direction": direction, "windows": int(
                                                   (selected & (direction_targets[ids, h] == d)).sum())})
    pd.DataFrame(family_rows).to_csv(output / "family_support_by_split.csv", index=False)
    pd.DataFrame(direction_rows, columns=["split", "horizon_s", "family", "direction", "windows"]).to_csv(
        output / "direction_support_by_split.csv", index=False)
    # Small existing audits are copied for standalone inspection and tracking;
    # features, anchors, skill targets and sample metadata are read from the pinned base.
    for name in ("scaling.json", "scenario_splits.json", "split_assignments.csv", "skill_support_before_split.csv",
                 "skill_support_by_split.csv", "window_sampling_audit.csv", "nonoverlap_audit.json",
                 "enemy_missile_audit.csv"):
        if (base.directory / name).is_file():
            shutil.copy2(base.directory / name, output / name)
    manifest = {**base.manifest, "data_contract": INTENT_DATA_CONTRACT,
                "base_data_contract": TARGET_RELATIVE_MISSILE_DATA_CONTRACT,
                "parent_dataset": {"directory": str(base.directory),
                                   "manifest_sha256": sha256(base.directory / "manifest.json")},
                "intent_supervision": policy, "intent_preparation_sha256": sha256(__file__),
                "skill_hierarchy": hierarchy,
                "intent_target_contract": "frozen intent group of exact future opponent skill; independently missing=-1"}
    manifest["files"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "manifest.json", manifest)
    return manifest


class IntentNNDuelData(TargetRelativeMissileNNDuelData):
    """Hash-pinned intent overlay with the same batching interface and base feature map."""

    data_contract = INTENT_DATA_CONTRACT

    def __init__(self, directory):
        directory = Path(directory).expanduser().resolve()
        manifest = read_json(directory / "manifest.json")
        if (manifest.get("data_contract") != INTENT_DATA_CONTRACT
                or manifest.get("base_data_contract") != TARGET_RELATIVE_MISSILE_DATA_CONTRACT):
            raise ValueError("Expected notebook-26 v2 skill-intent/family/direction dataset; rerun notebook 26")
        for name, digest in manifest["files"].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"Prepared intent data changed: {name}")
        parent = manifest["parent_dataset"]
        base_directory = Path(parent["directory"])
        if sha256(base_directory / "manifest.json") != parent["manifest_sha256"]:
            raise ValueError("Parent prepared dataset changed since intent preparation")
        base = TargetRelativeMissileNNDuelData(base_directory)
        for key in ("classes", "features", "counts", "split_seed", "source_manifests"):
            if manifest.get(key) != base.manifest.get(key):
                raise ValueError(f"Intent dataset differs from its parent: {key}")
        policy = intent_policy(base.classes)
        if manifest.get("intent_supervision") != policy or read_json(directory / "intent_policy.json") != policy:
            raise ValueError("Intent taxonomy or skill-to-intent ordering changed")
        intent_targets = np.load(directory / "intent_targets.npy")
        expected = map_intent_targets(base.targets, policy["skill_to_intent"])
        if not np.issubdtype(intent_targets.dtype, np.integer) or not np.array_equal(intent_targets, expected):
            raise ValueError("Intent labels or missing-target masks differ from original skill labels")
        hierarchy = hierarchy_policy(base.classes)
        if manifest.get("skill_hierarchy") != hierarchy or read_json(directory / "skill_hierarchy.json") != hierarchy:
            raise ValueError("Skill family/direction mapping or ordering changed")
        family_targets = np.load(directory / "family_targets.npy")
        direction_targets = np.load(directory / "direction_targets.npy")
        for name, labels, mapping in (("Family", family_targets, hierarchy["skill_to_family"]),
                                       ("Direction", direction_targets, hierarchy["skill_to_direction"])):
            if (not np.issubdtype(labels.dtype, np.integer)
                    or not np.array_equal(labels, map_intent_targets(base.targets, mapping))):
                raise ValueError(f"{name} labels or masks differ from original skill labels")
        self.__dict__.update(base.__dict__)
        self.directory, self.manifest = directory, manifest
        self.frame_directory = base_directory  # Spawn workers open these pinned immutable arrays.
        self.intent_targets = intent_targets
        self.intent_classes = list(INTENT_CLASSES)
        self.family_targets, self.direction_targets = family_targets, direction_targets
        self.family_classes, self.direction_classes = hierarchy["families"], hierarchy["directions"]


def aggregate_intent_probabilities(probabilities, mapping):
    probabilities = np.asarray(probabilities)
    result = np.zeros((*probabilities.shape[:-1], len(INTENT_CLASSES)), dtype=probabilities.dtype)
    for c in range(len(INTENT_CLASSES)):
        result[..., c] = probabilities[..., np.asarray(mapping) == c].sum(-1)
    return result


def save_intent_diagnostics(run, data):
    """Compare the auxiliary head with grouped skill probabilities, on the same examples."""
    from .enemy_missile_diagnostics import _probability_metrics, _pyplot, _figure
    from .paired_skill_forecasting import metric_tables

    run = Path(run)
    metadata = read_json(run / "run.json")
    if metadata["data_manifest_sha256"] != sha256(data.directory / "manifest.json"):
        raise ValueError("Intent diagnostic data differs from training")
    if metadata.get("intent_supervision") != data.manifest["intent_supervision"]:
        raise ValueError("Intent diagnostic taxonomy differs from training")
    directory = run / "intent_diagnostics"
    directory.mkdir(exist_ok=True)
    summaries, per_class, reliabilities = [], [], []
    mapping = data.manifest["intent_supervision"]["skill_to_intent"]
    for split in SPLITS:
        with np.load(run / f"final_{split}.npz") as stored:
            ids = stored["sample_indices"]
            grouped = aggregate_intent_probabilities(stored["probabilities"], mapping)
        with np.load(run / f"final_intent_{split}.npz") as stored:
            if (not np.array_equal(ids, stored["sample_indices"])
                    or not np.array_equal(ids, data.indices[split])
                    or not np.array_equal(stored["targets"], data.intent_targets[ids])):
                raise ValueError("Skill and intent final predictions are not aligned")
            probabilities, targets, losses = stored["probabilities"], stored["targets"], stored["losses"]
        if (probabilities.shape != (len(ids), len(HORIZONS), len(INTENT_CLASSES))
                or losses.shape != targets.shape or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or not np.allclose(probabilities.sum(-1), 1, atol=1e-5)
                or not np.isfinite(losses).all() or np.any(losses < 0)):
            raise ValueError("Invalid saved intent probabilities or logit losses")
        for name, p in (("intent_head", probabilities), ("grouped_skill_head", grouped)):
            summary, scores, reliability, confusion = _probability_metrics(targets, p, data.intent_classes, split)
            if name == "intent_head":
                exact, exact_scores, _ = metric_tables(targets, p, data.intent_classes, split, losses)
                summary["nll"] = exact.nll
                scores["class_nll"] = exact_scores.class_nll
            summaries.append(summary.assign(model=name))
            per_class.append(scores.assign(model=name))
            reliabilities.append(reliability.assign(model=name))
            plt = _pyplot()
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for h, ax, matrix in zip(HORIZONS, axes, confusion):
                ax.imshow(matrix, cmap="Blues")
                ax.set_xticks(range(3), data.intent_classes, rotation=30, ha="right", fontsize=8)
                ax.set_yticks(range(3), data.intent_classes, fontsize=8)
                ax.set(title=f"{split}, +{h}s", xlabel="Predicted", ylabel="True")
                for (i, j), value in np.ndenumerate(matrix):
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
            _figure(fig, directory / f"confusion_{name}_{split}.png")
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(directory / "summary.csv", index=False)
    pd.concat(per_class, ignore_index=True).to_csv(directory / "per_class.csv", index=False)
    pd.concat(reliabilities, ignore_index=True).to_csv(directory / "reliability.csv", index=False)
    write_json(directory / "manifest.json", {
        "format": "skill-intent-diagnostics-v1", "intent_supervision": data.manifest["intent_supervision"],
        "data_manifest_sha256": metadata["data_manifest_sha256"],
        "validation_used_for_selection": False,
        "artifacts": {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != "manifest.json"}})
    return summary

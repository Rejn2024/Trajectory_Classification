"""A negative primary result must stay visible beside an optional positive blend."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_transformer_experiments import (
    comparison_statement,
    output_roles,
    quality_notes,
    render,
    sha,
    write_json,
)


def portable_fixture(destination, secondary=True):
    """Only invented numbers; no experiment results, observation data or weights."""
    data = destination / "data"
    data.mkdir(parents=True)
    primary_name, other_name = "history60_temporal_transformer", "history60_patch_transformer"
    primary = {"kind": "primary_transformer", "recipe": "pure_pair", "new_weight": 1.0,
               "components": [primary_name, other_name], "active_members": 6, "active_parameters": 18_000_000,
               "quality_guards_passed_both_folds": False}
    mixed = {"kind": "secondary_blend", "recipe": "all_new", "new_weight": 0.5,
             "components": [primary_name, other_name, "history60_hybrid_transformer"],
             "active_members": 24, "active_parameters": 60_000_000,
             "quality_guards_passed_both_folds": True} if secondary else None
    protocol = {"baseline_members": 15, "baseline_parameters": 31_730_031,
                "guards": {"maximum_nll_increase": .02, "maximum_macro_f1_decrease": .01,
                           "maximum_horizon_accuracy_decrease": .003}}
    write_json(data / "protocol.json", protocol)
    write_json(data / "internal_baseline_metrics.json", [
        {"fold": 0, "members": 15, "mean_accuracy": .612345, "nll": 1.25, "mean_macro_f1": .39},
        {"fold": 1, "members": 15, "mean_accuracy": .632345, "nll": 1.20, "mean_macro_f1": .41}])
    (data / "frozen_protocol.json").write_bytes(b'{"fixture": true}\r\n')
    selection = {"primary_transformer": primary, "secondary_blend": mixed,
                 "protocol_sha256": sha(data / "frozen_protocol.json")}
    write_json(data / "frozen_selection.json", selection)
    roles = output_roles(selection)
    # The report never needs to unpack the ZIP or access local training state.
    (destination / "source_snapshot.zip").write_bytes(b"PK fixture placeholder")
    write_json(data / "provenance.json", {
        "frozen_protocol_sha256": selection["protocol_sha256"],
        "source_archive": {"sha256": sha(destination / "source_snapshot.zip")}})
    recipes = {}
    for role in roles[1:]:
        gain = -0.0123 if role == "primary_transformer" else 0.0045
        recipes[role] = {"mean_accuracy_gain": gain,
                         "mean_accuracy_interval": {"lower_95": gain - 0.01, "upper_95": gain + 0.01}}
    classes = ["abort_support", "accelerate_to_speed", "beam_missile_left", "beam_missile_right",
               "beam_target_left", "beam_target_right", "climb_to_altitude", "commit", "crank_target_left",
               "crank_target_right", "decelerate_to_speed", "defensive_reversal", "descend_to_altitude",
               "dive_defense", "drag_missile", "extend", "lag_pursuit", "last_ditch_break", "launch",
               "lead_pursuit", "lock_target", "maintain_heading", "notch_left", "notch_right",
               "preemptive_crank", "pursue_target", "recommit", "search", "secondary_shot",
               "short_range_attack", "support_missile", "turn_cold", "turn_to_heading"]
    metrics, pilots, skills = [], [], []
    for role in roles:
        for horizon in (1, 2, 5):
            metrics.append({"model": role, "pilot_id": "population", "horizon_s": horizon,
                            "top1_accuracy": .66 + recipes.get(role, {}).get("mean_accuracy_gain", 0),
                            "top3_accuracy": .87, "macro_f1_catalogue": .42,
                            "nll": 1.21 if role == "primary_transformer" else 1.2,
                            "brier": .42 if role == "primary_transformer" else .43,
                            "ece_10_bins": .03 if role == "primary_transformer" else .02})
            for index in range(10):
                gain = (index - 4) / 100
                if role != "baseline_v4":
                    pilots.append({"model": role, "pilot_id": f"pilot_{index + 1:02}", "horizon_s": horizon,
                                   "count": 2_900 + index, "baseline_top1": .65, "candidate_top1": .65 + gain,
                                   "gain": gain, "gain_lower": gain - .02, "gain_upper": gain + .02})
            for index, skill in enumerate(classes):
                skills.append({"model": role, "pilot_id": "population", "horizon_s": horizon,
                               "skill": skill, "support": 0 if index > 29 else 1_000 + index, "f1": .5})
    frame = pd.DataFrame(metrics)
    frame.to_csv(data / "metrics.csv", index=False)
    pd.DataFrame(pilots).to_csv(data / "pilot_comparison.csv", index=False)
    pd.DataFrame(skills).to_csv(data / "pilot_skill_metrics.csv", index=False)
    screen = [{"recipe": f"history{history}_{family}_transformer", "kind": "architectural_hybrid"
               if family == "hybrid" else "pure_transformer", "mean_accuracy": .63, "nll": 1.2,
               "mean_macro_f1": .4} for family in ("pooled", "temporal", "patch", "hybrid") for history in (20, 60)]
    pd.DataFrame(screen).to_csv(data / "transformer_screen.csv", index=False)
    confirmation = [{"kind": "primary_transformer", "recipe": name, "fold": fold,
                     "mean_accuracy": .64, "gain": -.01, "eligible": False}
                    for name in (primary_name, other_name, "pure_pair") for fold in (0, 1)]
    pd.DataFrame(confirmation).to_csv(data / "transformer_confirmation.csv", index=False)
    inventory = [{"candidate": primary_name, "phase": phase, "reused_baseline": False,
                  "elapsed_seconds": 100., "epochs_completed": 20}
                 for phase in ("internal", "internal", "refit")]
    pd.DataFrame(inventory).to_csv(data / "run_inventory.csv", index=False)
    histories = [{"candidate": primary_name, "phase": "internal", "seed": 42, "fold": fold,
                  "reused_baseline": False, "epoch": epoch, "stop_mean_accuracy": .5 + epoch / 100}
                 for fold in (0, 1) for epoch in range(1, 11)]
    pd.DataFrame(histories).to_csv(data / "training_histories.csv", index=False)
    evidence = {"selection": selection, "roles": roles, "evaluation": {"recipes": recipes}, "classes": classes,
                "new_completed_runs": 3, "reused_internal_runs": 30, "new_training_seconds": 300,
                "quality_notes": {role: quality_notes(frame, role) for role in roles[1:]}}
    write_json(data / "evidence.json", evidence)
    write_json(data / "verification.json", {"preserved_artifacts_checked": 123})
    write_json(data / "checksums.json", {path.name: sha(path) for path in data.iterdir()})
    return evidence


def test_primary_question_survives_a_better_secondary_score(tmp_path):
    evidence = portable_fixture(tmp_path)
    assert output_roles(evidence["selection"]) == ["baseline_v4", "primary_transformer", "secondary_blend"]
    assert "-1.23 pp" in comparison_statement("Transformer only", evidence["evaluation"]["recipes"]["primary_transformer"])
    invalid = dict(evidence["selection"], primary_transformer=evidence["selection"]["secondary_blend"])
    with pytest.raises(ValueError, match="all-Transformer primary"):
        output_roles(invalid)


def test_quality_regressions_cannot_hide_behind_improved_brier(tmp_path):
    evidence = portable_fixture(tmp_path)
    note = evidence["quality_notes"]["primary_transformer"]
    assert "NLL worsened at all horizons" in note
    assert "Brier improved at all horizons" in note
    assert "ECE worsened at all horizons" in note
    frame = pd.read_csv(tmp_path / "data/metrics.csv")
    frame.loc[frame.horizon_s.eq(5), "horizon_s"] = 2
    with pytest.raises(ValueError, match="three fixed horizons"):
        quality_notes(frame, "primary_transformer")


@pytest.mark.parametrize("secondary", [False, True])
def test_portable_negative_result_renders_all_pages_without_experiment_data(tmp_path, secondary):
    portable_fixture(tmp_path, secondary)
    result = render(tmp_path)
    assert result["pages"] == 6
    assert result["primary_transformer_reported"]
    assert result["secondary_blend_reported"] is secondary
    pdf = Path(result["pdf"]).read_bytes()
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 20_000
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "-1.23 pp" in readme
    assert "NLL worsened" in readme
    assert "61.2345%" in readme
    assert "NLL <= v4 +0.02" in readme
    assert "mean F1/33 >= v4 -0.01" in readme
    assert "each horizon accuracy >= v4 -0.003 (-0.30 pp)" in readme
    assert "mean accuracy gain >0 in both partitions" in readme
    assert "preliminary screen compares individual candidate models" in readme
    assert "multi-seed ensemble" in readme
    notes = (tmp_path / "speaker_notes.md").read_text(encoding="utf-8")
    assert "61.2345%" in notes and "NLL <= v4 +0.02" in notes
    assert ".venv\\Scripts\\python.exe" in readme
    render(tmp_path)
    assert Path(result["pdf"]).read_bytes() == pdf


def test_changed_frozen_bytes_block_a_portable_rebuild(tmp_path):
    portable_fixture(tmp_path)
    protocol = tmp_path / "data/frozen_protocol.json"
    protocol.write_bytes(protocol.read_bytes().replace(b"\r\n", b"\n"))
    with pytest.raises(ValueError, match="Portable evidence changed"):
        render(tmp_path)

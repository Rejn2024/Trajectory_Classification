"""Target-relative missile position/velocity preserves protocol and physical edges."""

import numpy as np
import pandas as pd
import pytest
import torch

from bvr_behavior_prediction import nn_duel_skill_forecasting as duel
from bvr_behavior_prediction import paired_skill_forecasting as paired
from bvr_behavior_prediction.enemy_missile_models import TypedGraphTransformer
from tests.test_enemy_missile_models import example
from tests.test_nn_duel_skill_forecasting import CLASSES, write_source


def test_target_relative_preparation_changes_only_friendly_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(duel, "catalogue", lambda root: CLASSES)
    source = write_source(tmp_path, frames=181)
    analysis = duel.analyze_corpus(tmp_path, source)
    for name, enabled in (("old", False), ("new", True)):
        duel.prepare_dataset(tmp_path, tmp_path / name, source, analysis=analysis,
                             nonoverlapping_train=True, enemy_missiles=True,
                             observer_acceleration=True, target_relative_missiles=enabled)
    old = duel.ObserverAccelerationEnemyMissileNNDuelData(tmp_path / "old")
    new = duel.TargetRelativeMissileNNDuelData(tmp_path / "new")
    assert new.manifest["features"] == old.manifest["features"]
    assert new.manifest["missile_position_reference"] == duel.TARGET_RELATIVE_MISSILE_POLICY
    assert new.manifest["data_contract"] == duel.TARGET_RELATIVE_MISSILE_DATA_CONTRACT
    changed = list(range(16, 44))
    unchanged = np.setdiff1d(np.arange(new.features.shape[1]), changed)
    np.testing.assert_array_equal(new.features[:, unchanged], old.features[:, unchanged])
    np.testing.assert_array_equal(new.mean[unchanged], old.mean[unchanged])
    np.testing.assert_array_equal(new.std[unchanged], old.std[unchanged])
    old_slots = old.features[:, 16:44].reshape(-1, 4, 7)
    new_slots = new.features[:, 16:44].reshape(-1, 4, 7)
    expected_positions = old_slots[:, :, :3] - old.features[:, None, 6:9]
    np.testing.assert_allclose(new_slots[:, :, :3], expected_positions, atol=1e-7)
    np.testing.assert_allclose(new_slots[:, :, 3:6],
                               old_slots[:, :, 3:6] - .5 * old.features[:, None, 9:12], atol=1e-7)
    np.testing.assert_allclose(new_slots[:, :, 6],
                               np.linalg.norm(expected_positions * [50000, 50000, 10000], axis=-1) / 50000)
    np.testing.assert_array_equal(new.targets, old.targets)
    np.testing.assert_array_equal(new.anchors, old.anchors)
    pd.testing.assert_frame_equal(new.samples, old.samples)
    for name in ("split_assignments.csv", "skill_support_by_split.csv", "enemy_missile_audit.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(new.directory / name), pd.read_csv(old.directory / name))
    retained = np.concatenate([np.arange(end - 50, end + 1) for end in new.anchors[new.indices["train"]]])
    values = new.features[retained].astype(float)
    np.testing.assert_allclose(new.mean[:82], values.mean(0)[:82], atol=1e-7)
    np.testing.assert_allclose(new.std[:82], np.maximum(values.std(0)[:82], .01), atol=1e-7)
    for loader, path in ((duel.TargetRelativeMissileNNDuelData, old.directory),
                         (duel.ObserverAccelerationEnemyMissileNNDuelData, new.directory)):
        with pytest.raises(ValueError, match="Expected own-skill-conditioned"):
            loader(path)
    for target_positions in (False, True):
        with pytest.raises(ValueError, match="matching notebook"):
            paired.train_model(new, tmp_path / "wrong-reference", {
                "architecture": "typed_graph", "observer_acceleration": True,
                "target_relative_missiles": target_positions,
            }, device="cpu")
    # The same input width must not allow the earlier position-only v1 cache.
    earlier = {**new.manifest, "data_contract": duel.TARGET_RELATIVE_MISSILE_DATA_CONTRACT[:-1] + "1"}
    duel.write_json(new.directory / "manifest.json", earlier)
    with pytest.raises(ValueError, match="Expected own-skill-conditioned"):
        duel.TargetRelativeMissileNNDuelData(new.directory)
    new.manifest["missile_position_reference"]["friendly_position"] = "observer"
    duel.write_json(new.directory / "manifest.json", new.manifest)
    with pytest.raises(ValueError, match="position reference policy"):
        duel.TargetRelativeMissileNNDuelData(new.directory)


def test_transformation_is_frame_local_and_does_not_mutate_source():
    raw = np.full((100, 51), .2, dtype=np.float32)
    original = raw.copy()
    before = duel.target_relative_friendly_missiles(raw)
    np.testing.assert_array_equal(raw, original)
    raw[61:, 9:12] = 100
    raw[61:, 12:15] = -200
    raw[61:, 19:47] = 500
    np.testing.assert_array_equal(duel.target_relative_friendly_missiles(raw)[:61], before[:61])


@pytest.mark.parametrize("value,acceleration", [("yes", True), (True, False)])
def test_invalid_reference_settings_fail_before_writing(tmp_path, value, acceleration):
    with pytest.raises(ValueError, match="target_relative_missiles"):
        duel.prepare_dataset(tmp_path, tmp_path / "unused", nonoverlapping_train=True,
                             enemy_missiles=True, observer_acceleration=acceleration,
                             target_relative_missiles=value)
    assert not (tmp_path / "unused").exists()
    with pytest.raises(ValueError, match="target_relative_missiles"):
        TypedGraphTransformer(target_relative_missiles=value, observer_acceleration=acceleration)


@pytest.mark.parametrize("target_velocities", [False, True])
def test_graph_edges_recover_common_geometry_and_preserve_masks_and_gradients(target_velocities):
    torch.set_num_threads(1)
    raw, mean, std = example()
    order = [*range(3, 54), 0, 1, 2, *range(54, 115)]
    raw, mean, std = raw[:, :, order], mean[order], std[order]
    settings = dict(width=12, heads=6, layers=1, dropout=0, graph_width=8, graph_layers=1,
                    observer_acceleration=True, mean=mean, std=std)
    old = TypedGraphTransformer(**settings).eval()
    new = TypedGraphTransformer(**settings, target_relative_missiles=True,
                                target_relative_missile_velocities=target_velocities).eval()
    changed = raw.clone()
    slots = changed[:, :, 16:44].reshape(2, 51, 4, 7)
    present = slots.ne(0).any(-1)
    slots[..., :3] -= changed[:, :, None, 6:9]
    if target_velocities:
        slots[..., 3:6] -= .5 * changed[:, :, None, 9:12]
    slots[..., 6] = torch.linalg.vector_norm(slots[..., :3] * torch.tensor([50000, 50000, 10000]), dim=-1) / 50000
    slots[~present] = 0
    changed[:, :, 16:44] = slots.reshape(2, 51, 28)
    geometries = []
    handles = [model.messages[0].register_forward_pre_hook(
        lambda module, args: geometries.append((args[2].detach(), args[1].detach())))
        for model in (old, new)]
    try:
        old((raw - mean) / std)
        normalized = ((changed - mean) / std).requires_grad_()
        output = new(normalized)
        geometry_old, mask_old = geometries[0]
        geometry_new, mask_new = geometries[1]
        torch.testing.assert_close(mask_old, mask_new)
        active_edges = mask_new[:, new.sources] & mask_new[:, new.destinations]
        torch.testing.assert_close(geometry_old[active_edges], geometry_new[active_edges], atol=3e-6, rtol=3e-6)
        output.square().mean().backward()
        assert torch.isfinite(normalized.grad).all()
        assert normalized.grad[:, :, 16:19].abs().sum() > 0
        assert normalized.grad[:, :, 19:22].abs().sum() > 0
        assert not normalized.grad[:, :, 23:30].any()  # Empty friendly slot.
        permuted = changed.clone()
        permuted[:, :, 16:44] = slots[:, :, [3, 2, 0, 1]].reshape(2, 51, 28)
        with torch.no_grad():
            torch.testing.assert_close(output, new((permuted - mean) / std), atol=2e-6, rtol=2e-6)
    finally:
        for handle in handles:
            handle.remove()

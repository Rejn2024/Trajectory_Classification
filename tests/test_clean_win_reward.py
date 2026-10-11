"""Terminal bonus semantics and identical scoring across collection paths."""

import json
from dataclasses import replace

import numpy as np
import pytest

from bvr_behavior_prediction.rl.config import PilotTrainingConfig, RewardWeights
from bvr_behavior_prediction.rl.reward import (
    CleanWinCombatReward, CombatReward, clean_win_reward_definition, combat_reward_definition,
)


def info(*, ended=False, alive=True, eliminated=False, destroyed=False):
    return {"episode_ended": ended, "blue_alive": alive, "blue_altitude_m": 6000,
            "opponent_eliminated": eliminated, "opponent_destroyed": destroyed,
            "shot_down": not alive}


@pytest.mark.parametrize("alive,eliminated,destroyed,bonus", [
    (True, True, True, 250), (True, False, False, 0),
    (True, False, True, 0), (False, True, True, 0),
    (False, False, True, 0), (False, False, False, 0),
])
def test_bonus_matches_confirmed_clean_win_and_preserves_event_components(alive, eliminated, destroyed, bonus):
    weights = replace(RewardWeights(), shot_down=-371, opponent_destroyed=250)
    original, changed = CombatReward(weights), CleanWinCombatReward(weights)
    final = info(ended=True, alive=alive, eliminated=eliminated, destroyed=destroyed)
    old_total, old_components = original(final)
    new_total, new_components = changed(final)
    assert new_total == old_total + bonus
    assert new_components == {**old_components, "clean_win_bonus": bonus}
    assert changed(final)[1]["clean_win_bonus"] == 0  # Never pay a terminal event twice.
    changed.reset()
    assert changed(final)[1]["clean_win_bonus"] == bonus


@pytest.mark.parametrize("survives", [False, True])
def test_waits_until_episode_end_including_a_later_death(survives):
    reward = CleanWinCombatReward()
    for _ in range(5):
        assert reward(info(eliminated=True, destroyed=True))[1]["clean_win_bonus"] == 0
    _, components = reward(info(ended=True, alive=survives, eliminated=True, destroyed=True))
    assert components["clean_win_bonus"] == (250 if survives else 0)


def test_missing_or_inconsistent_terminal_information_is_not_silently_rewarded():
    with pytest.raises(ValueError, match="episode_ended"):
        CleanWinCombatReward()({"blue_alive": True, "opponent_eliminated": True})
    with pytest.raises(ValueError, match="opponent_eliminated"):
        CleanWinCombatReward()({"episode_ended": True, "blue_alive": True})
    with pytest.raises(ValueError, match="requires opponent destruction"):
        CleanWinCombatReward()(info(ended=True, eliminated=True, destroyed=False))


@pytest.mark.parametrize("bonus", [0, -1, float("nan"), float("inf"), True])
def test_invalid_bonus_fails_before_training(bonus):
    with pytest.raises(ValueError, match="finite and positive"):
        clean_win_reward_definition(clean_win_bonus=bonus)


def test_recipe_version_and_independent_episode_state():
    recipe = clean_win_reward_definition("pilot_005", clean_win_bonus=250)
    metadata = json.loads(json.dumps(recipe.as_dict()))
    assert metadata["version"] == "3"
    assert metadata["parameters"]["clean_win_bonus"] == 250
    assert combat_reward_definition().version == "2"
    first, second = recipe.factory(), recipe.factory()
    final = info(ended=True, eliminated=True, destroyed=True)
    assert first(final)[1]["clean_win_bonus"] == second(final)[1]["clean_win_bonus"] == 250


class TerminalEnvironment:
    """Opponent is eliminated first; episode ends one decision later."""
    def __init__(self, truncated=False, survives=True):
        self.truncated, self.survives = truncated, survives

    def reset(self, seed):
        self.steps = 0
        return np.zeros((1, 1), np.float32)

    def step(self, *_):
        self.steps += 1
        ended = self.steps == 2
        # Supply a deliberately wrong flag: collector must use its actual flags.
        result = info(ended=not ended, alive=self.survives if ended else True,
                      eliminated=True, destroyed=True)
        return np.zeros((1, 1), np.float32), ended and not self.truncated, ended and self.truncated, result

    def close(self):
        pass


@pytest.mark.parametrize("collection", ["single", "thread", "process"])
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("survives", [False, True])
def test_training_and_validation_bonus_benchmark_exclusion_and_scaled_gae(tmp_path, collection, truncated, survives):
    torch = pytest.importorskip("torch")
    from bvr_behavior_prediction.rl.trainer import PPOTrainer

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    weights = replace(RewardWeights(), ground_clearance=0)
    config = PilotTrainingConfig(episode_duration_s=2, hidden_size=8, transformer_heads=2,
                                 transformer_layers=1, history_duration_s=1, sample_interval_s=1,
                                 simulator_workers=2, simulator_executor="process" if collection == "process" else "thread",
                                 training_reward_scale=.01, output_dir=tmp_path)
    trainer = PPOTrainer(lambda *_: TerminalEnvironment(truncated, survives), 1, config, "cpu",
                         clean_win_reward_definition(weights=weights))
    try:
        if collection == "single":
            episodes = [trainer._episode(None, 7)]
        else:
            episodes = trainer._training_episodes()
        for rollout, total, final in episodes:
            bonus = 250 if survives else 0
            assert rollout.dones == [False, True]
            assert final["episode_ended"] is True
            assert final["reward_component_totals"]["clean_win_bonus"] == bonus
            assert total == (500 if survives else 100)
            advantages, targets = trainer._advantages(rollout)
            assert targets[-1] == pytest.approx(rollout.rewards[-1] * .01, abs=1e-6)
        validation = trainer.evaluate_episodes()
        assert all(e[2]["reward_component_totals"]["clean_win_bonus"] == (250 if survives else 0)
                   for e in validation)
        benchmark = trainer.evaluate_episodes(reward_definition=combat_reward_definition(weights=weights))
        assert all("clean_win_bonus" not in e[2]["reward_component_totals"] for e in benchmark)
        assert all(e[1] == (250 if survives else 100) for e in benchmark)
    finally:
        trainer.close()
        torch.set_num_threads(previous_threads)

"""Persistent hierarchical agent built around :mod:`skill_manager`."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from .bvr_strategist import StrategicLLM
from .skill_manager import SkillManager


class AgentConfig:
    model = "qwen3-8b"
    planning_interval = 15.0


class Agent:
    """Use a slow strategist to select persistent, fast tactical skills."""

    def __init__(self, strategist=None, skill_manager=None):
        self.skill_manager = skill_manager or SkillManager()
        self.strategist = strategist or StrategicLLM()
        self.step_count = 0
        self.llm_calls = 0
        self.current_skill = None
        self.active_skill = None
        self.active_selection = None
        self.skill_events = []

    def step(self, obs: Mapping[str, Any], info=None):
        if not isinstance(obs, Mapping):
            raise TypeError("Agent.step expects a structured observation mapping")
        self.step_count += 1
        selection = asyncio.run(self.strategist.select_skill(obs))
        if selection is None:
            raise RuntimeError("strategist did not provide a skill selection")
        if selection.source == "llm":
            self.llm_calls += 1

        selection_key = (selection.skill_name, _freeze_mapping(selection.skill_params))
        now = obs.get("time_s", obs.get("time", 0.0))
        if self.active_skill is None or selection_key != self.active_selection:
            if self.active_skill is not None:
                self.active_skill.interrupt("superseded_by_%s" % selection.skill_name)
                self.skill_events.append({
                    "event": "interrupted", "skill": self.active_skill.skill_name,
                    "reason": self.active_skill.interruption_reason, "time_s": now,
                })
            self.active_skill = self.skill_manager.create_skill(
                selection.skill_name, selection.skill_params
            )
            self.active_selection = selection_key
            self.skill_events.append({
                "event": "started", "skill": self.active_skill.skill_name,
                "source": selection.source, "reasoning": selection.reasoning, "time_s": now,
            })

        action, completed = self.active_skill.execute(obs)
        self.current_skill = self.active_skill.skill_name
        if completed:
            self.skill_events.append({
                "event": "completed", "skill": self.active_skill.skill_name, "time_s": now,
            })
            self.active_skill = None
            self.active_selection = None
        return action

    def reset(self, episode_config=None):
        if self.active_skill is not None:
            self.active_skill.interrupt("episode_reset")
        self.step_count = 0
        self.llm_calls = 0
        self.current_skill = None
        self.active_skill = None
        self.active_selection = None
        self.skill_events = []


def _freeze_mapping(value):
    if isinstance(value, Mapping):
        return tuple(sorted((key, _freeze_mapping(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_mapping(item) for item in value)
    return value

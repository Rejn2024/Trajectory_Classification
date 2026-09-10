"""Slow strategic skill selection with deterministic emergency overrides."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .api.api_router import get_api_class
from .skill_manager import SkillManager

HUMAN_PROMPT = "MISSION: Air Superiority. Engage enemies beyond 20nm. Maintain altitude advantage."


@dataclass(frozen=True)
class SkillSelection:
    skill_name: str
    skill_params: Mapping[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.8
    source: str = "rule"


class StrategicLLM:
    """Select infrequent tactical skills; it never directly flies the aircraft."""

    def __init__(self, api=None):
        from .agent import AgentConfig

        self.model = AgentConfig.model
        self.planning_interval = AgentConfig.planning_interval
        self.last_plan_time = 0.0
        self.api = api if api is not None else get_api_class(self.model)()
        self.skill_manager = SkillManager()
        self.last_selection = None
        self.last_error = None

    def should_replan(self, obs):
        current_time = float(obs.get("time_s", obs.get("time", 0.0)))
        return self.last_selection is None or current_time - self.last_plan_time >= self.planning_interval

    async def select_skill(self, obs):
        if not self.should_replan(obs):
            return self.last_selection

        emergency_skill = self._check_emergencies(obs)
        if emergency_skill:
            self.last_selection = emergency_skill
        else:
            try:
                response = await self.api.chat_completion(self._create_planning_prompt(obs, HUMAN_PROMPT))
                skill_name, skill_params = self._parse_llm_response(response)
                self.last_selection = SkillSelection(skill_name, skill_params, source="llm")
                self.last_error = None
            except (RuntimeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self.last_error = "%s: %s" % (type(exc).__name__, exc)
                self.last_selection = SkillSelection(
                    "crank_maneuver",
                    {"direction": "left", "offset_angle": 30},
                    reasoning="planner_fallback: %s" % self.last_error,
                    confidence=0.0,
                    source="fallback",
                )

        self.last_plan_time = float(obs.get("time_s", obs.get("time", 0.0)))
        return self.last_selection

    def _check_emergencies(self, obs):
        threats = str(obs.get("threat_assessment", "")).lower()
        if obs.get("missile_warning") or "missile" in threats:
            return SkillSelection(
                "missile_evasion", {"break_direction": "right"},
                reasoning="missile warning", confidence=1.0, source="emergency_rule",
            )

        fuel = obs.get("self_status", {}).get("resources", {}).get("fuel", "100%")
        try:
            fuel_percent = float(str(fuel).replace("%", ""))
        except ValueError:
            fuel_percent = 100.0
        if fuel_percent < 15.0:
            return SkillSelection(
                "disengage", {}, reasoning="low fuel", confidence=1.0, source="emergency_rule"
            )
        return None

    def _create_planning_prompt(self, obs, mission):
        return """
%s

Current situation:
%s
%s
%s

Available skills and parameter schemas: %s
Select a skill and parameters in JSON format: {"skill": "name", "params": {"param": "value"}}
""" % (
            mission,
            obs.get("situation_summary", ""),
            obs.get("self_status", ""),
            obs.get("threat_assessment", ""),
            json.dumps(self.skill_manager.schemas(), sort_keys=True),
        )

    def _parse_llm_response(self, response):
        parsed = json.loads(response)
        if not isinstance(parsed, dict):
            raise ValueError("planner response must be a JSON object")
        skill_name = parsed.get("skill")
        skill_params = parsed.get("params", {})
        if not isinstance(skill_params, dict):
            raise ValueError("planner params must be a JSON object")
        self.skill_manager.create_skill(skill_name, skill_params)  # validate selection
        return skill_name, skill_params

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))
from .api.api_router import get_api_class

from .skill_manager import SkillManager

HUMAN_PROMPT = "MISSION: Air Superiority. Engage enemies beyond 20nm. Maintain altitude advantage."

class StrategicLLM:
    """LLM-powered strategic planning"""

    def __init__(self):
        from .agent import AgentConfig
        self.model = AgentConfig.model
        self.planning_interval = AgentConfig.planning_interval
        self.last_plan_time = 0
        self.api = get_api_class(self.model)()
        self.skill_manager = SkillManager()
        self.last_selection = None

    def should_replan(self, obs):
        """Check if planning is needed"""
        current_time = obs.get("time", 0)
        return current_time - self.last_plan_time >= self.planning_interval

    async def select_skill(self, obs):
        """Select skill with parameters"""
        if self.last_selection is not None and not self.should_replan(obs):
            return self.last_selection

        # LLM planning
        prompt = self._create_planning_prompt(obs, HUMAN_PROMPT)

        try:
            response = await self.api.chat_completion(prompt)
            skill_name, skill_params = self._parse_llm_response(response, obs)
            self.last_selection = SkillSelection(skill_name, skill_params)
        except Exception:
            self.last_selection = self._default_selection(obs)

        self.last_plan_time = obs.get("time", 0)
        return self.last_selection

    @staticmethod
    def _default_selection(obs):
        """Request zero heading change when a plan cannot be decoded."""
        del obs
        return SkillSelection("turn_to_heading", {"delta_heading_deg": 0.0})

    def _create_planning_prompt(self, obs, mission):
        """Create planning prompt for LLM"""
        return f"""
{mission}

Current situation:
{obs.get('situation_summary', '')}
{obs.get('self_status', '')}
{obs.get('threat_assessment', '')}

Available skills: {list(self.skill_manager.skills.keys())}
Select a skill and parameters in JSON format: {{"skill": "name", "params": {{"param": "value"}}}}
"""

    def _parse_llm_response(self, response, obs=None):
        """Parse LLM response"""
        try:
            import json
            data = json.loads(response)
            name, params = data["skill"], data.get("params", {})
            self.skill_manager.create_skill(name, params)
            return name, params
        except (KeyError, TypeError, ValueError):
            selection = self._default_selection(obs or {})
            return selection.skill_name, selection.skill_params


class SkillSelection:
    """Skill selection with parameters"""
    def __init__(self, skill_name, skill_params, reasoning="", confidence=0.8):
        self.skill_name = skill_name
        self.skill_params = skill_params
        self.reasoning = reasoning
        self.confidence = confidence

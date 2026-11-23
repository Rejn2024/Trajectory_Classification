import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from .bvr_strategist import StrategicLLM
from .skill_manager import SkillManager

class AgentConfig:
    model = "qwen3-8b"
    planning_interval = 15.0


class Agent:
    """BVR LLM Red Agent"""

    def __init__(self):
        self.planning_interval = AgentConfig.planning_interval

        self.skill_manager = SkillManager()
        self.strategist = StrategicLLM()

        self.step_count = 0
        self.llm_calls = 0
        self.current_skill = None
        self._active_skill = None
        self._active_parameters = None

    def step(self, obs: dict, info=None):
        """Agent step"""
        self.step_count += 1
        if not hasattr(obs, "get"):
            raise TypeError("Agent.step expects a structured observation mapping")

        # Select skill
        import asyncio
        skill_selection = asyncio.run(self.strategist.select_skill(obs))
        if skill_selection:
            self.llm_calls += 1
            parameters = dict(skill_selection.skill_params or {})
            if (self._active_skill is None
                    or self._active_skill.skill_name != skill_selection.skill_name
                    or self._active_parameters != parameters):
                self._active_skill = self.skill_manager.create_skill(
                    skill_selection.skill_name, parameters
                )
                self._active_parameters = parameters
            action, completed = self._active_skill.execute(obs)
            self.current_skill = skill_selection.skill_name
            if completed:
                self._active_skill = None
                self._active_parameters = None
        else:
            raise ValueError("No skill selected")
            

        # Convert to environment action format
        return action

    def reset(self, episode_config=None):
        """Reset agent"""
        self.step_count = 0
        self.current_skill = None
        self._active_skill = None
        self._active_parameters = None
        self.strategist.last_selection = None
        self.strategist.last_plan_time = 0

import logging
from typing import Any

from agent_framework.agents.base import BaseAgent
from agent_framework.llm.config import LLMConfig


logger = logging.getLogger(__name__)


class DefaultAgent(BaseAgent):
    """
    The default agent for orchestrating and managing the overall job.
    """

    default_llm_config = LLMConfig(prompt_modules=["coordination/root_agent"])

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        logger.info("DefaultAgent initialized.")

    max_iterations = 200

    async def run_job(self, job_config: dict[str, Any]) -> dict[str, Any]:
        """
        Executes the job based on the task description found in state.
        Callers must provide a full task description.
        """
        task_description = self.state.task or ""

        # Append user instructions/context if not already present in task
        user_instructions = job_config.get("user_instructions", "")
        context = job_config.get("context")

        if context and context not in task_description:
            task_description += f"\n\nContext:\n{context}"

        if user_instructions and user_instructions not in task_description:
            task_description += f"\n\nUser Instructions: {user_instructions}"

        return await self.start_lifecycle(primary_task=task_description)

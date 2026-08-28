import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from agent_framework.agents.base import BaseAgent
from agent_framework.llm.types import LLMResponse


class TestAgentWaitingResumption(unittest.TestCase):
    @patch("agent_framework.agents.base.db")
    def test_resume_updates_redis_status(self, mock_db):
        agent_config = {
            "state": {"agent_id": "test_agent_resume", "status": "running"},
            "llm_config": {"model_name": "test-model"},
        }

        mock_db.pop_all_messages_for_agent.return_value = [
            {"type": "message", "from": "user", "content": "hello"}
        ]
        mock_db.get_agent_status.return_value = "waiting"

        agent = BaseAgent(agent_config)

        agent.context.set_waiting()

        asyncio.run(agent._check_messages())

        self.assertFalse(agent.context.waiting_for_input)

        status_update_calls = [
            args[1]
            for args in [
                call.args for call in mock_db.update_agent_status.call_args_list
            ]
            if args[0] == "test_agent_resume"
        ]

        self.assertIn(
            "running",
            status_update_calls,
            "Agent did not update Redis status to 'running' upon resumption",
        )

    @patch("agent_framework.agents.base.db")
    def test_reasoning_only_response_retries_instead_of_parking(self, mock_db):
        """No content and no parseable tool call must retry, even with reasoning_content present."""
        agent_config = {
            "state": {"agent_id": "test_agent_reasoning_only", "status": "running"},
            "llm_config": {"model_name": "test-model"},
        }
        mock_db.get_agent_status.return_value = "running"

        agent = BaseAgent(agent_config)
        agent.llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="",
                tool_invocations=None,
                reasoning_content=(
                    "Let me start with the host discovery ping sweep."
                    "<function=get_findings>\n\n"
                ),
            )
        )

        is_done = asyncio.run(agent._execute_cycle())

        self.assertFalse(is_done)
        self.assertFalse(
            agent.context.waiting_for_input,
            "A reasoning-only turn with no tool call parked the agent in "
            "set_waiting() instead of retrying it",
        )
        self.assertEqual(agent.context.consecutive_empty_responses, 1)

    @patch("agent_framework.agents.base.db")
    def test_plain_text_response_retries_instead_of_parking(self, mock_db):
        """Real content but no tool call must also retry, not silently wait."""
        agent_config = {
            "state": {"agent_id": "test_agent_plain_text", "status": "running"},
            "llm_config": {"model_name": "test-model"},
        }
        mock_db.get_agent_status.return_value = "running"

        agent = BaseAgent(agent_config)
        agent.llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="I've found the live hosts, let me think about next steps.",
                tool_invocations=None,
                reasoning_content=None,
            )
        )

        is_done = asyncio.run(agent._execute_cycle())

        self.assertFalse(is_done)
        self.assertFalse(
            agent.context.waiting_for_input,
            "A plain-text turn with no tool call parked the agent instead of retrying it",
        )
        self.assertEqual(agent.context.consecutive_empty_responses, 1)

    @patch("agent_framework.agents.base.db")
    def test_wait_state_expires_and_resumes(self, mock_db):
        agent_config = {
            "state": {"agent_id": "test_agent_wait_timeout", "status": "running"},
            "llm_config": {"model_name": "test-model"},
        }
        mock_db.get_agent_status.return_value = "waiting"
        agent = BaseAgent(agent_config)
        agent.context.set_waiting(timeout=0)

        asyncio.run(agent._wait_cycle())

        self.assertFalse(agent.context.waiting_for_input)
        self.assertEqual(agent.context.status, "running")
        mock_db.update_agent_status.assert_any_call(
            "test_agent_wait_timeout", "running"
        )


if __name__ == "__main__":
    unittest.main()

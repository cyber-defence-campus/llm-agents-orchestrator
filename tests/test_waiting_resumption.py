import asyncio
import unittest
from unittest.mock import MagicMock, patch
from agent_framework.agents.base import BaseAgent


class TestAgentWaitingResumption(unittest.TestCase):
    """
    Tests that the agent correctly handles waiting mode and resumption.
    """

    @patch("agent_framework.agents.base.db")
    def test_resume_updates_redis_status(self, mock_db):
        """
        Verify that when an agent resumes from waiting, it explicitly updates
        its status in Redis back to 'running'.
        """
        # Setup
        agent_config = {
            "state": {"agent_id": "test_agent_resume", "status": "running"},
            "llm_config": {"model_name": "test-model"},
        }

        # Configure mocks
        # Simulate one message in the queue to trigger processing
        mock_db.pop_all_messages_for_agent.return_value = [
            {"type": "message", "from": "user", "content": "hello"}
        ]
        # Redis initially thinks it's waiting (simulating what enter_wait_mode tool does)
        mock_db.get_agent_status.return_value = "waiting"

        # Initialize Agent
        agent = BaseAgent(agent_config)

        # Put agent in waiting mode locally
        agent.context.set_waiting()

        # Act: Check messages (trigger resume)
        # using asyncio.run because _check_messages is async
        asyncio.run(agent._check_messages())

        # Assertions
        # 1. Agent context should be active again
        self.assertFalse(agent.context.waiting_for_input)

        # 2. Redis status MUST be updated to 'running'
        # We check the calls to update_agent_status
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


if __name__ == "__main__":
    unittest.main()

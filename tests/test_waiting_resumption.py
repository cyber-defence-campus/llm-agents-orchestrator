import asyncio
import unittest
from unittest.mock import MagicMock, patch
from agent_framework.agents.base import BaseAgent


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


if __name__ == "__main__":
    unittest.main()

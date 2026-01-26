
import asyncio
import unittest
from unittest.mock import MagicMock, patch
from agent_framework.agents.base import BaseAgent
from agent_framework.agents.state import AgentContext

class ReproduceStatusBug(unittest.TestCase):
    @patch('agent_framework.agents.base.db')
    def test_resume_does_not_update_status(self, mock_db):
        # Setup
        agent_config = {
             "state": {
                 "agent_id": "test_agent",
                 "status": "running"
             },
            "llm_config": {"model_name": "test-model"}
        }
        
        # Configure mocks
        mock_db.pop_all_messages_for_agent.return_value = [
            {"type": "message", "from": "user", "content": "hello"}
        ]
        mock_db.get_agent_status.return_value = "waiting" # Redis thinks it's waiting
        
        # Initialize Agent
        agent = BaseAgent(agent_config)
        
        # Simulate Agent entering waiting mode matches state in Redis
        agent.context.set_waiting()
        # In the bug scenario, Redis status is "waiting" (set by tool), 
        # distinct from context.status (which might still be running or waiting depending on implementation)
        # But crucially, we want to see if _check_messages updates it back to "running".
        
        # Act: Check messages (trigger resume)
        asyncio.run(agent._check_messages())
        
        # Assertions
        # 1. Agent should be active again
        self.assertFalse(agent.context.waiting_for_input)
        
        # 2. BUG REPRODUCTION: 
        # We expect that update_agent_status was NOT called with "running".
        # If the bug exists, this assertion should PASS (verifying the bug).
        # Once fixed, we expect update_agent_status to be called with "running".
        
        calls = [args[0] for args in mock_db.update_agent_status.call_args_list] if mock_db.update_agent_status.called else []
        # Checks if ('test_agent', 'running') was passed
        status_update_calls = [
            args[1] for args in  [call.args for call in mock_db.update_agent_status.call_args_list]
            if args[0] == "test_agent"
        ]
        
        print(f"Status updates: {status_update_calls}")
        
        if "running" not in status_update_calls:
            print("Status was NOT updated to 'running' in Redis! Bug reproduced.")
        else:
            print("Status WAS updated to 'running'. Bug not present?")
            
        # For the test to pass as a "reproduction", we assert the current buggy behavior
        # After fix, we will flip this assertion.
        self.assertNotIn("running", status_update_calls)

if __name__ == "__main__":
    unittest.main()

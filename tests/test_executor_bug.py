import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agent_framework.tools.executor import execute_tool_with_validation

@pytest.mark.asyncio
async def test_execute_tool_with_name_argument_collision():
    # Mock validate_tool_availability
    with patch('agent_framework.tools.executor.validate_tool_availability', return_value=(True, "")):
        # Mock execute_tool (which is likely imported or defined globally)
        # We need to patch where it is used in executor.py
        with patch('agent_framework.tools.executor.execute_tool', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = "Success"
            
            # Call with a tool name AND a kwargs containing 'tool_name'
            # This previously caused TypeError: execute_tool() got multiple values for argument 'tool_name'
            await execute_tool_with_validation(
                "add_action", 
                agent_state=None, 
                tool_name="run_shell_command", 
                description="test"
            )
            
            # Verify call
            mock_exec.assert_called_once()
            args, kwargs = mock_exec.call_args
            
            # First arg should be 'add_action'
            assert args[0] == "add_action"
            # kwargs should contain tool_name='run_shell_command'
            assert kwargs['tool_name'] == "run_shell_command"

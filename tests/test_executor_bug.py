import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agent_framework.tools.executor import execute_tool_with_validation


@pytest.mark.asyncio
async def test_execute_tool_with_name_argument_collision():
    with patch(
        "agent_framework.tools.executor.validate_tool_availability",
        return_value=(True, ""),
    ):
        with patch(
            "agent_framework.tools.executor.execute_tool", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = "Success"

            # This previously caused TypeError: execute_tool() got multiple values for argument 'tool_name'
            await execute_tool_with_validation(
                "add_action",
                agent_state=None,
                tool_name="run_shell_command",
                description="test",
            )

            mock_exec.assert_called_once()
            args, kwargs = mock_exec.call_args

            assert args[0] == "add_action"
            assert kwargs["tool_name"] == "run_shell_command"

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agent_framework.tools.executor import execute_tool, execute_tool_with_validation


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


@pytest.mark.asyncio
async def test_dispatch_preserves_tool_name_argument():
    with patch(
        "agent_framework.tools.executor._dispatch_tool", new_callable=AsyncMock
    ) as dispatch:
        dispatch.return_value = {"success": True}

        result = await execute_tool(
            "add_action",
            None,
            tool_name="run_shell_command",
            description="test",
        )

    assert result == {"success": True}
    dispatch.assert_awaited_once_with(
        "add_action",
        None,
        tool_name="run_shell_command",
        description="test",
    )


@pytest.mark.asyncio
async def test_missing_required_argument_is_structured_error():
    def required_tool(summary: str):
        return summary

    with patch(
        "agent_framework.tools.executor.get_tool_by_name",
        return_value=required_tool,
    ), patch(
        "agent_framework.tools.executor.needs_agent_state", return_value=False
    ):
        from agent_framework.tools.executor import _execute_tool_locally

        result = await _execute_tool_locally("required_tool", None)

    assert result["type"] == "ArgumentError"
    assert "summary" in result["error"]

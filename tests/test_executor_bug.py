import asyncio
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


@pytest.mark.asyncio
async def test_a_tool_argument_named_tool_name_reaches_the_tool():
    """`add_action` takes a `tool_name` argument, and the dispatch chain used to
    collide with it: every forwarding hop named its own first parameter
    `tool_name`, so passing one raised "got multiple values for argument
    'tool_name'" before the tool was reached. `execute_tool` had been fixed;
    the four hops below it had not."""
    import inspect

    from agent_framework.tools import executor

    for name in (
        "_dispatch_tool",
        "_try_local_orchestration",
        "_try_direct_sandbox_execution",
        "_delegate_tool_to_core_api",
        "_execute_tool_locally",
    ):
        parameters = inspect.signature(getattr(executor, name)).parameters
        assert parameters["tool_name"].kind is inspect.Parameter.POSITIONAL_ONLY, (
            f"{name} takes tool_name positionally-or-by-keyword, so a tool "
            f"argument of that name collides with it instead of being forwarded"
        )


@pytest.mark.asyncio
async def test_tool_invocations_in_one_turn_do_not_overlap():
    """A beacon's wire carries Ready -> Do -> Result on one connection, so two
    capability calls from the same turn must not be in flight together. They
    were gathered concurrently, and the second came back "beacon did not
    answer"; the agent then reinstalled a carrier that had never died."""
    import agent_framework.tools.executor as executor

    live = 0
    overlapped = False

    async def fake_execute(tool_inv, agent_state=None):
        nonlocal live, overlapped
        live += 1
        if live > 1:
            overlapped = True
        await asyncio.sleep(0)          # yield, so a gather would interleave
        live -= 1
        return {"ok": True}

    state = MagicMock()
    state.iteration = 1
    with patch.object(executor, "execute_tool_invocation", fake_execute):
        await executor.process_tool_invocations(
            [{"toolName": "read_file", "args": {}},
             {"toolName": "list_directory", "args": {}},
             {"toolName": "http_request", "args": {}}],
            [], state)

    assert not overlapped, "two capability calls were in flight at once"

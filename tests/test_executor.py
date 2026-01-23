import pytest
from unittest.mock import patch, MagicMock, AsyncMock, ANY
from agent_framework.tools.executor import (
    execute_tool,
    execute_tool_with_validation,
    process_tool_invocations,
    should_execute_in_sandbox,
)


# Mock AgentState
@pytest.fixture
def mock_agent_state():
    state = MagicMock()
    state.agent_id = "test-agent"
    state.sandbox_info = {"job_id": "test-job"}
    state.iteration = 1
    return state


@pytest.fixture
def mock_sandbox_client():
    with patch("agent_framework.services.sandbox_client.sandbox_client") as mock:
        mock.is_available = True
        mock.base_url = "http://mock-sandbox"
        mock.execute_tool = AsyncMock()
        yield mock


class TestExecutorExecution:
    @pytest.mark.asyncio
    async def test_execute_local_tool_success(self, mock_agent_state):
        # Mock a local tool
        mock_tool = MagicMock(return_value="tool output")
        mock_tool.__name__ = "local_tool"

        with patch(
            "agent_framework.tools.executor.get_tool_by_name", return_value=mock_tool
        ), patch(
            "agent_framework.tools.executor.should_execute_in_sandbox",
            return_value=False,
        ), patch("agent_framework.tools.executor.is_sandbox_runtime", False):
            result = await execute_tool("local_tool", mock_agent_state, arg1="val1")

            assert result == "tool output"
            mock_tool.assert_called_once()
            # Verify args were passed (checking simplified call)
            assert mock_tool.call_args[1]["arg1"] == "val1"

    @pytest.mark.asyncio
    async def test_execute_sandboxed_tool_direct(
        self, mock_agent_state, mock_sandbox_client
    ):
        # Setup environment for sandbox
        with patch.dict(
            "os.environ", {"AGENT_SANDBOX_URL": "http://mock-sandbox"}
        ), patch(
            "agent_framework.tools.executor.should_execute_in_sandbox",
            return_value=True,
        ), patch("agent_framework.tools.executor.is_sandbox_runtime", False):
            mock_sandbox_client.execute_tool.return_value = {"result": "sandbox output"}

            result = await execute_tool("sandbox_tool", mock_agent_state, arg1="val1")

            assert result == {"result": "sandbox output"}
            mock_sandbox_client.execute_tool.assert_awaited_once()

            # Verify arguments passed to sandbox client
            call_kwargs = mock_sandbox_client.execute_tool.call_args[1]
            assert call_kwargs["tool_name"] == "sandbox_tool"
            assert call_kwargs["session_id"] == "test-job"
            assert call_kwargs["kwargs"]["arg1"] == "val1"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, mock_agent_state):
        with patch(
            "agent_framework.tools.executor.get_tool_by_name", return_value=None
        ), patch(
            "agent_framework.tools.executor.should_execute_in_sandbox",
            return_value=False,
        ):
            result = await execute_tool("missing_tool", mock_agent_state)

            assert isinstance(result, dict)
            assert result["type"] == "NotFoundError"

    @pytest.mark.asyncio
    async def test_validation_before_execution(self, mock_agent_state):
        with patch(
            "agent_framework.tools.executor.get_tool_names", return_value=["valid_tool"]
        ), patch(
            "agent_framework.tools.executor.execute_tool", new_callable=AsyncMock
        ) as mock_exec:
            # Test valid tool
            await execute_tool_with_validation("valid_tool", mock_agent_state)
            mock_exec.assert_called_once()

            # Test invalid tool
            mock_exec.reset_mock()
            result = await execute_tool_with_validation(
                "invalid_tool", mock_agent_state
            )
            assert result["type"] == "ValidationError"
            mock_exec.assert_not_called()


class TestToolProcessing:
    @pytest.mark.asyncio
    async def test_process_tool_invocations_parallel(self, mock_agent_state):
        # Mock execution to be fast
        with patch(
            "agent_framework.tools.executor.execute_tool_invocation",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_exec.side_effect = ["Result A", "Result B"]

            invocations = [
                {"toolName": "toolA", "args": {}},
                {"toolName": "toolB", "args": {}},
            ]
            history = []

            await process_tool_invocations(invocations, history, mock_agent_state)

            assert mock_exec.call_count == 2
            # Check history population
            assert len(history) >= 2  # Calls and Results might be appended

            # Check we have tool_calls and tool_results
            roles = [msg["role"] for msg in history]
            assert "tool_call" in roles
            assert "tool_result" in roles

            # Verify user observation message was added (XML)
            assert history[-1]["role"] == "user"
            assert "<tool_result>" in history[-1]["content"]

    @pytest.mark.asyncio
    async def test_process_tool_error_handling(self, mock_agent_state):
        with patch(
            "agent_framework.tools.executor.execute_tool_invocation",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_exec.return_value = {
                "error": "Something went wrong",
                "type": "ExecutionError",
            }

            invocations = [{"toolName": "fail_tool", "args": {}}]
            history = []

            await process_tool_invocations(invocations, history, mock_agent_state)

            # Verify result in history
            result_entry = next(
                entry for entry in history if entry["role"] == "tool_result"
            )
            assert result_entry["content"]["isError"] is True
            assert result_entry["content"]["error"] == {
                "error": "Something went wrong",
                "type": "ExecutionError",
            }

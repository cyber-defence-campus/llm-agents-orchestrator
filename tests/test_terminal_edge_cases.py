import asyncio
import pytest
from agent_framework.tools.terminal.shell_session import ShellExecutor
from agent_framework.tools.terminal.terminal_manager import TerminalToolManager


class TestBusySessionHandling:
    @pytest.fixture
    async def executor(self):
        exc = ShellExecutor("busy-test")
        await asyncio.sleep(0.3)
        yield exc
        exc.terminate()

    @pytest.mark.asyncio
    async def test_busy_session_error_when_command_running(self, executor):
        """
        CRITICAL TEST: Verify that when a command is running,
        new commands (without require_input) get a clear error.

        This is the exact scenario from the user report where agents
        got stuck in loops because they couldn't understand/handle the busy state.
        """
        result1 = await executor.run("sleep 10", timeout=0.5)

        assert result1["status"] == "running"
        assert executor.busy is True

        result2 = await executor.run("echo 'test'", timeout=1.0)

        assert result2["status"] == "error"
        assert "busy" in result2.get("error", "").lower()
        assert "terminal_id" in result2

    @pytest.mark.asyncio
    async def test_interrupt_with_ctrl_c(self, executor):
        result1 = await executor.run("sleep 30", timeout=0.5)
        assert result1["status"] == "running"
        assert executor.busy is True

        interrupt_result = await executor.run("^C", timeout=2.0)

        assert interrupt_result["status"] == "completed"
        assert interrupt_result["exit_code"] == 130  # SIGINT exit code
        assert executor.busy is False

        result3 = await executor.run("echo 'back to normal'", timeout=5.0)
        assert result3["status"] == "completed"
        assert "back to normal" in result3["content"]

    @pytest.mark.asyncio
    async def test_require_input_bypasses_busy_check(self, executor):
        result1 = await executor.run("cat", timeout=0.5)
        assert result1["status"] == "running"

        input_result = await executor.run("hello world", timeout=1.0, is_input=True)

        assert input_result.get("status") != "error"
        assert "busy" not in input_result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_busy_state_clears_on_completion(self, executor):
        result = await executor.run("echo 'quick'", timeout=5.0)

        assert result["status"] == "completed"
        assert executor.busy is False

        result2 = await executor.run("echo 'also quick'", timeout=5.0)
        assert result2["status"] == "completed"


class TestParallelSessions:
    @pytest.fixture
    def manager(self):
        # Create a new instance (bypass singleton for testing)
        mgr = TerminalToolManager.__new__(TerminalToolManager)
        mgr._executors = {}
        mgr._manager_lock = __import__("threading").Lock()
        mgr._default_id = "default"
        mgr._initialized = True
        yield mgr
        mgr.shutdown_all()

    @pytest.mark.asyncio
    async def test_different_session_ids_are_isolated(self, manager):
        result1 = await manager.execute_command(
            "sleep 10", terminal_id="session1", timeout=0.5
        )
        assert result1["status"] == "running"

        result2 = await manager.execute_command(
            "echo 'session2 works'", terminal_id="session2", timeout=5.0
        )
        assert result2["status"] == "completed"
        assert "session2 works" in result2["content"]

        result3 = await manager.execute_command(
            "echo 'session1 test'", terminal_id="session1", timeout=1.0
        )
        assert result3["status"] == "error"
        assert "busy" in result3.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_can_switch_between_sessions(self, manager):
        sessions = ["main", "scan", "exploit"]

        for session in sessions:
            result = await manager.execute_command(
                f"echo 'hello from {session}'", terminal_id=session, timeout=5.0
            )
            assert result["status"] == "completed"
            assert f"hello from {session}" in result["content"]

        assert len(manager._executors) == 3

        for session in sessions:
            result = await manager.execute_command(
                "pwd", terminal_id=session, timeout=5.0
            )
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_default_session_per_agent(self, manager):
        result = await manager.execute_command("echo 'default'", timeout=5.0)

        assert result["status"] == "completed"
        assert result["terminal_id"] == "default"
        assert "default" in manager._executors

    @pytest.mark.asyncio
    async def test_session_isolation_prevents_cross_contamination(self, manager):
        await manager.execute_command(
            "export MY_VAR='session1_value'", terminal_id="session1", timeout=5.0
        )

        result1 = await manager.execute_command(
            "echo $MY_VAR", terminal_id="session1", timeout=5.0
        )
        assert "session1_value" in result1["content"]

        result2 = await manager.execute_command(
            'echo "VAR=$MY_VAR"', terminal_id="session2", timeout=5.0
        )
        assert "session1_value" not in result2["content"]


class TestConcurrentStress:
    @pytest.fixture
    def manager(self):
        # Create a new instance (bypass singleton for testing)
        mgr = TerminalToolManager.__new__(TerminalToolManager)
        mgr._executors = {}
        mgr._manager_lock = __import__("threading").Lock()
        mgr._default_id = "default"
        mgr._initialized = True
        yield mgr
        mgr.shutdown_all()

    @pytest.mark.asyncio
    async def test_many_parallel_sessions(self, manager):
        num_sessions = 5

        async def run_in_session(session_id: str):
            result = await manager.execute_command(
                f"echo 'result from {session_id}' && sleep 0.1",
                terminal_id=session_id,
                timeout=5.0,
            )
            return session_id, result

        tasks = [run_in_session(f"stress-{i}") for i in range(num_sessions)]
        results = await asyncio.gather(*tasks)

        for session_id, result in results:
            assert result["status"] == "completed", f"Session {session_id} failed"
            assert f"result from {session_id}" in result["content"]

        assert len(manager._executors) == num_sessions

    @pytest.mark.asyncio
    async def test_rapid_command_succession(self, manager):
        num_commands = 10

        for i in range(num_commands):
            result = await manager.execute_command(
                f"echo 'command {i}'", terminal_id="rapid", timeout=5.0
            )
            assert result["status"] == "completed"
            assert f"command {i}" in result["content"]

    @pytest.mark.asyncio
    async def test_session_recovery_after_interrupt(self, manager):
        session = "recovery-test"

        await manager.execute_command("sleep 30", terminal_id=session, timeout=0.5)

        interrupt_result = await manager.execute_command(
            "^C", terminal_id=session, timeout=2.0
        )
        assert interrupt_result["status"] == "completed"

        for i in range(5):
            result = await manager.execute_command(
                f"echo 'recovery {i}'", terminal_id=session, timeout=5.0
            )
            assert result["status"] == "completed"
            assert f"recovery {i}" in result["content"]


class TestRealWorldScenarios:
    @pytest.fixture
    def manager(self):
        # Create a new instance (bypass singleton for testing)
        mgr = TerminalToolManager.__new__(TerminalToolManager)
        mgr._executors = {}
        mgr._manager_lock = __import__("threading").Lock()
        mgr._default_id = "default"
        mgr._initialized = True
        yield mgr
        mgr.shutdown_all()

    @pytest.mark.asyncio
    async def test_scan_in_background_work_in_foreground(self, manager):
        scan_result = await manager.execute_command(
            "sleep 5",
            terminal_id="scan",
            timeout=0.5,
        )
        assert scan_result["status"] == "running"

        work_results = []
        for cmd in [
            "echo 'checking target'",
            "echo 'analyzing data'",
            "echo 'preparing exploit'",
        ]:
            result = await manager.execute_command(cmd, terminal_id="main", timeout=5.0)
            work_results.append(result)

        for result in work_results:
            assert result["status"] == "completed"

        check_scan = await manager.execute_command(
            "echo 'test'", terminal_id="scan", timeout=1.0
        )
        assert check_scan["status"] == "error"
        assert "busy" in check_scan.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_agent_properly_uses_ctrl_c_recovery(self, manager):
        await manager.execute_command("sleep 60", terminal_id="stuck", timeout=0.5)

        retry = await manager.execute_command(
            "echo 'retry'", terminal_id="stuck", timeout=1.0
        )
        assert retry["status"] == "error"

        interrupt = await manager.execute_command(
            "^C", terminal_id="stuck", timeout=2.0
        )
        assert interrupt["status"] == "completed"

        recovery = await manager.execute_command(
            "echo 'recovered!'", terminal_id="stuck", timeout=5.0
        )
        assert recovery["status"] == "completed"
        assert "recovered!" in recovery["content"]

    @pytest.mark.asyncio
    async def test_process_with_password_prompt(self, manager):
        await manager.execute_command(
            "read -p 'Enter: ' var", terminal_id="interactive", timeout=0.5
        )

        input_result = await manager.execute_command(
            "myinput", terminal_id="interactive", is_input=True, timeout=2.0
        )

        assert (
            input_result.get("status") != "error"
            or "busy" not in input_result.get("error", "").lower()
        )


class TestWaitingFunctionality:
    @pytest.fixture
    async def executor(self):
        exc = ShellExecutor("wait-test")
        await asyncio.sleep(0.3)
        yield exc
        exc.terminate()

    @pytest.mark.asyncio
    async def test_wait_for_completion_success(self, executor):
        import time

        result1 = await executor.run("sleep 2; echo 'done'", timeout=0.1)
        assert result1["status"] == "running"
        assert executor.busy is True

        start_wait = time.time()
        result2 = await executor.run("# wait for it", timeout=5.0)
        duration = time.time() - start_wait

        assert duration >= 1.0
        assert result2["status"] == "completed"
        assert "done" in result2["content"]
        assert executor.busy is False

    @pytest.mark.asyncio
    async def test_wait_times_out(self, executor):
        import time

        await executor.run("sleep 10", timeout=0.1)

        start_wait = time.time()
        result2 = await executor.run("", timeout=1.0)
        duration = time.time() - start_wait

        assert duration < 2.0
        assert result2["status"] == "running"
        assert executor.busy is True

    @pytest.mark.asyncio
    async def test_busy_error_for_real_commands(self, executor):
        await executor.run("sleep 5", timeout=0.1)

        result = await executor.run("echo 'no wait'", timeout=1.0)

        assert result["status"] == "error"
        assert "busy" in result.get("error", "").lower()


class TestAgentExpectedBehavior:
    """
    Tests that verify the exact behaviors documented in the system prompt.

    These tests serve as documentation and regression tests for the terminal
    tool guidance given to agents. If any of these fail, the prompt documentation
    is out of sync with the actual behavior.
    """

    @pytest.fixture
    def manager(self):
        # Create a new instance (bypass singleton for testing)
        mgr = TerminalToolManager.__new__(TerminalToolManager)
        mgr._executors = {}
        mgr._manager_lock = __import__("threading").Lock()
        mgr._default_id = "default"
        mgr._initialized = True
        yield mgr
        mgr.shutdown_all()

    @pytest.fixture
    async def executor(self):
        exc = ShellExecutor("agent-behavior-test")
        await asyncio.sleep(0.3)
        yield exc
        exc.terminate()

    @pytest.mark.asyncio
    async def test_status_completed_means_command_finished(self, manager):
        """
        PROMPT DOCUMENTATION: 'status: "completed"' means the command finished.

        The agent should trust this field to know when a session is available.
        """
        result = await manager.execute_command("echo 'hello'", timeout=5.0)

        assert result["status"] == "completed"
        assert result["exit_code"] is not None
        result2 = await manager.execute_command("echo 'world'", timeout=5.0)
        assert result2["status"] == "completed"

    @pytest.mark.asyncio
    async def test_status_running_means_session_busy(self, manager):
        """
        PROMPT DOCUMENTATION: 'status: "running"' means the session is BUSY.

        The agent MUST NOT send a new command to a busy session.
        """
        result = await manager.execute_command(
            "sleep 10", terminal_id="busy-test", timeout=0.5
        )

        assert result["status"] == "running"
        assert result["exit_code"] is None

        result2 = await manager.execute_command(
            "echo 'this should fail'", terminal_id="busy-test", timeout=1.0
        )
        assert result2["status"] == "error"
        assert "busy" in result2.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_status_running_tells_agent_to_use_different_session(self, manager):
        """
        PROMPT DOCUMENTATION: If status is 'running', use a different session_id.

        This simulates the correct agent behavior.
        """
        result_a = await manager.execute_command(
            "sleep 10", terminal_id="session_a", timeout=0.5
        )
        assert result_a["status"] == "running"

        result_b = await manager.execute_command(
            "echo 'session B is free'", terminal_id="session_b", timeout=5.0
        )
        assert result_b["status"] == "completed"
        assert "session B is free" in result_b["content"]

    @pytest.mark.asyncio
    async def test_waiting_with_empty_string(self, executor):
        """
        PROMPT DOCUMENTATION: Send command="" to wait for a busy session.
        """
        await executor.run("sleep 1; echo 'finished'", timeout=0.1)
        assert executor.busy is True

        result = await executor.run("", timeout=5.0)

        assert result["status"] == "completed"
        assert "finished" in result["content"]
        assert executor.busy is False

    @pytest.mark.asyncio
    async def test_waiting_with_comment(self, executor):
        """
        PROMPT DOCUMENTATION: Send command="# waiting" to wait for a busy session.
        """
        await executor.run("sleep 1; echo 'done waiting'", timeout=0.1)
        assert executor.busy is True

        result = await executor.run("# waiting", timeout=5.0)

        assert result["status"] == "completed"
        assert "done waiting" in result["content"]

    @pytest.mark.asyncio
    async def test_waiting_timeout_returns_running(self, executor):
        """
        PROMPT DOCUMENTATION: If wait times out, status is still 'running'.

        This helps the agent understand it should wait longer or interrupt.
        """
        await executor.run("sleep 30", timeout=0.1)

        result = await executor.run("", timeout=0.5)

        assert result["status"] == "running"
        assert executor.busy is True

    @pytest.mark.asyncio
    async def test_ctrl_c_recovers_stuck_session(self, manager):
        """
        PROMPT DOCUMENTATION: Send command="^C" to interrupt a stuck session.
        """
        await manager.execute_command("sleep 60", terminal_id="stuck", timeout=0.5)

        interrupt_result = await manager.execute_command(
            "^C", terminal_id="stuck", timeout=2.0
        )

        assert interrupt_result["status"] == "completed"
        assert interrupt_result["exit_code"] == 130  # SIGINT

        result = await manager.execute_command(
            "echo 'recovered'", terminal_id="stuck", timeout=5.0
        )
        assert result["status"] == "completed"
        assert "recovered" in result["content"]

    @pytest.mark.asyncio
    async def test_chained_command_timeout_shows_running(self, manager):
        """
        PROMPT DOCUMENTATION: Avoid '&&' chaining with long commands.

        This test shows WHY: if the first command is long, the chain
        appears as 'running' even though partial output may exist.
        """
        result = await manager.execute_command(
            "sleep 5 && echo 'chain complete'", terminal_id="chain", timeout=0.5
        )

        assert result["status"] == "running"
        assert "chain complete" not in result.get("content", "")

    @pytest.mark.asyncio
    async def test_quick_chained_commands_work(self, manager):
        result = await manager.execute_command(
            "echo 'one' && echo 'two' && echo 'three'", timeout=5.0
        )

        assert result["status"] == "completed"
        assert "one" in result["content"]
        assert "two" in result["content"]
        assert "three" in result["content"]

    @pytest.mark.asyncio
    async def test_require_input_sends_to_busy_session(self, manager):
        """
        PROMPT DOCUMENTATION: Use require_input=true to respond to prompts.
        """
        await manager.execute_command(
            'read var; echo "Got: $var"', terminal_id="input-test", timeout=0.5
        )

        result = await manager.execute_command(
            "my_input_value", terminal_id="input-test", is_input=True, timeout=2.0
        )

        assert (
            result.get("status") != "error"
            or "busy" not in result.get("error", "").lower()
        )

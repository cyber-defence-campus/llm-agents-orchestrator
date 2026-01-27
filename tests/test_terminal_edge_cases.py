"""
Comprehensive integration tests for terminal session management edge cases.

These tests specifically cover the scenarios that cause issues in production:
1. Busy session detection and proper error handling
2. Parallel sessions with different session_ids
3. ^C interrupt mechanism on busy sessions
4. TerminalToolManager multi-session handling
5. Stress testing with concurrent sessions
"""

import asyncio
import pytest
from agent_framework.tools.terminal.shell_session import ShellExecutor
from agent_framework.tools.terminal.terminal_manager import TerminalToolManager


class TestBusySessionHandling:
    """Tests for proper handling of busy terminal sessions."""

    @pytest.fixture
    async def executor(self):
        """Create a ShellExecutor for testing."""
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
        # Start a long-running command (will timeout)
        result1 = await executor.run("sleep 10", timeout=0.5)
        
        # Session should be busy (running status)
        assert result1["status"] == "running"
        assert executor.busy is True
        
        # Try to run another command without require_input
        result2 = await executor.run("echo 'test'", timeout=1.0)
        
        # Should get a clear error about busy session
        assert result2["status"] == "error"
        assert "busy" in result2.get("error", "").lower()
        assert "terminal_id" in result2  # Should tell which terminal is busy

    @pytest.mark.asyncio
    async def test_interrupt_with_ctrl_c(self, executor):
        """
        CRITICAL TEST: Verify that sending ^C interrupts a running command.
        
        This is the mechanism agents should use when stuck.
        """
        # Start a long-running command
        result1 = await executor.run("sleep 30", timeout=0.5)
        assert result1["status"] == "running"
        assert executor.busy is True
        
        # Send ^C to interrupt
        interrupt_result = await executor.run("^C", timeout=2.0)
        
        # Should complete with interrupt signal
        assert interrupt_result["status"] == "completed"
        assert interrupt_result["exit_code"] == 130  # SIGINT exit code
        assert executor.busy is False  # Session should be free now
        
        # Now we should be able to run new commands
        result3 = await executor.run("echo 'back to normal'", timeout=5.0)
        assert result3["status"] == "completed"
        assert "back to normal" in result3["content"]

    @pytest.mark.asyncio
    async def test_require_input_bypasses_busy_check(self, executor):
        """
        Verify that require_input=True allows sending input to a running process.
        """
        # Start cat (waits for input)
        result1 = await executor.run("cat", timeout=0.5)
        assert result1["status"] == "running"
        
        # Send input with require_input=True
        input_result = await executor.run("hello world", timeout=1.0, is_input=True)
        
        # Should succeed (not get busy error)
        assert input_result.get("status") != "error"
        assert "busy" not in input_result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_busy_state_clears_on_completion(self, executor):
        """Verify that busy state properly clears when a command completes."""
        # Run a quick command
        result = await executor.run("echo 'quick'", timeout=5.0)
        
        assert result["status"] == "completed"
        assert executor.busy is False
        
        # Should be able to run more commands immediately
        result2 = await executor.run("echo 'also quick'", timeout=5.0)
        assert result2["status"] == "completed"


class TestParallelSessions:
    """
    Tests for running multiple terminal sessions in parallel.
    
    This covers the scenario where agents should use different session_ids
    to avoid blocking each other.
    """

    @pytest.fixture
    def manager(self):
        """Create a fresh TerminalToolManager for testing."""
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
        """
        CRITICAL TEST: Different session_ids should create completely 
        isolated terminals that don't block each other.
        """
        # Start a long command in session1
        result1 = await manager.execute_command(
            "sleep 10", 
            terminal_id="session1", 
            timeout=0.5
        )
        assert result1["status"] == "running"
        
        # session2 should work immediately
        result2 = await manager.execute_command(
            "echo 'session2 works'", 
            terminal_id="session2", 
            timeout=5.0
        )
        assert result2["status"] == "completed"
        assert "session2 works" in result2["content"]
        
        # Session1 should still be busy
        result3 = await manager.execute_command(
            "echo 'session1 test'", 
            terminal_id="session1", 
            timeout=1.0
        )
        assert result3["status"] == "error"
        assert "busy" in result3.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_can_switch_between_sessions(self, manager):
        """Test rapidly switching between different sessions."""
        sessions = ["main", "scan", "exploit"]
        
        # Create all sessions with a quick command
        for session in sessions:
            result = await manager.execute_command(
                f"echo 'hello from {session}'",
                terminal_id=session,
                timeout=5.0
            )
            assert result["status"] == "completed"
            assert f"hello from {session}" in result["content"]
        
        # Should have 3 separate executors
        assert len(manager._executors) == 3
        
        # Each session should have its own state
        for session in sessions:
            result = await manager.execute_command(
                "pwd",
                terminal_id=session,
                timeout=5.0
            )
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_default_session_per_agent(self, manager):
        """Test that omitting session_id uses the default."""
        # Command without session_id
        result = await manager.execute_command("echo 'default'", timeout=5.0)
        
        assert result["status"] == "completed"
        assert result["terminal_id"] == "default"
        assert "default" in manager._executors

    @pytest.mark.asyncio
    async def test_session_isolation_prevents_cross_contamination(self, manager):
        """Verify that sessions don't see each other's environment variables."""
        # Set a variable in session1
        await manager.execute_command(
            "export MY_VAR='session1_value'",
            terminal_id="session1",
            timeout=5.0
        )
        
        # Check it's set in session1
        result1 = await manager.execute_command(
            "echo $MY_VAR",
            terminal_id="session1",
            timeout=5.0
        )
        assert "session1_value" in result1["content"]
        
        # session2 should NOT have this variable
        result2 = await manager.execute_command(
            "echo \"VAR=$MY_VAR\"",
            terminal_id="session2",
            timeout=5.0
        )
        # The variable should be empty or not set
        assert "session1_value" not in result2["content"]


class TestConcurrentStress:
    """Stress tests for concurrent session usage."""

    @pytest.fixture
    def manager(self):
        """Create a fresh TerminalToolManager for testing."""
        mgr = TerminalToolManager.__new__(TerminalToolManager)
        mgr._executors = {}
        mgr._manager_lock = __import__("threading").Lock()
        mgr._default_id = "default"
        mgr._initialized = True
        yield mgr
        mgr.shutdown_all()

    @pytest.mark.asyncio
    async def test_many_parallel_sessions(self, manager):
        """
        Stress test: Create and use many parallel sessions.
        This simulates an agent spawning multiple scan sessions.
        """
        num_sessions = 5
        
        # Create tasks to run commands in parallel
        async def run_in_session(session_id: str):
            result = await manager.execute_command(
                f"echo 'result from {session_id}' && sleep 0.1",
                terminal_id=session_id,
                timeout=5.0
            )
            return session_id, result
        
        tasks = [run_in_session(f"stress-{i}") for i in range(num_sessions)]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        for session_id, result in results:
            assert result["status"] == "completed", f"Session {session_id} failed"
            assert f"result from {session_id}" in result["content"]
        
        # Should have created all sessions
        assert len(manager._executors) == num_sessions

    @pytest.mark.asyncio
    async def test_rapid_command_succession(self, manager):
        """Test running many commands in rapid succession in same session."""
        num_commands = 10
        
        for i in range(num_commands):
            result = await manager.execute_command(
                f"echo 'command {i}'",
                terminal_id="rapid",
                timeout=5.0
            )
            assert result["status"] == "completed"
            assert f"command {i}" in result["content"]

    @pytest.mark.asyncio
    async def test_session_recovery_after_interrupt(self, manager):
        """Test that sessions properly recover after ^C interrupt."""
        session = "recovery-test"
        
        # Start a long command
        await manager.execute_command("sleep 30", terminal_id=session, timeout=0.5)
        
        # Interrupt it
        interrupt_result = await manager.execute_command("^C", terminal_id=session, timeout=2.0)
        assert interrupt_result["status"] == "completed"
        
        # Run several commands to verify session is healthy
        for i in range(5):
            result = await manager.execute_command(
                f"echo 'recovery {i}'",
                terminal_id=session,
                timeout=5.0
            )
            assert result["status"] == "completed"
            assert f"recovery {i}" in result["content"]


class TestRealWorldScenarios:
    """Tests that simulate actual agent usage patterns."""

    @pytest.fixture
    def manager(self):
        """Create a fresh TerminalToolManager for testing."""
        mgr = TerminalToolManager.__new__(TerminalToolManager)
        mgr._executors = {}
        mgr._manager_lock = __import__("threading").Lock()
        mgr._default_id = "default"
        mgr._initialized = True
        yield mgr
        mgr.shutdown_all()

    @pytest.mark.asyncio
    async def test_scan_in_background_work_in_foreground(self, manager):
        """
        Simulate: Agent starts a scan in dedicated session,
        then continues working in default session.
        """
        # Start "scan" (simulated with sleep) in dedicated session
        scan_result = await manager.execute_command(
            "sleep 5",  # Simulate long-running scan
            terminal_id="scan",
            timeout=0.5
        )
        assert scan_result["status"] == "running"
        
        # Work in default session while scan runs
        work_results = []
        for cmd in ["echo 'checking target'", "echo 'analyzing data'", "echo 'preparing exploit'"]:
            result = await manager.execute_command(cmd, terminal_id="main", timeout=5.0)
            work_results.append(result)
        
        # All work commands should succeed
        for result in work_results:
            assert result["status"] == "completed"
        
        # Scan session should still be busy
        check_scan = await manager.execute_command("echo 'test'", terminal_id="scan", timeout=1.0)
        assert check_scan["status"] == "error"
        assert "busy" in check_scan.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_agent_properly_uses_ctrl_c_recovery(self, manager):
        """
        Simulate: Agent realizes session is stuck and uses ^C to recover.
        """
        # Agent starts a command that hangs
        await manager.execute_command("sleep 60", terminal_id="stuck", timeout=0.5)
        
        # Agent tries another command, gets busy error
        retry = await manager.execute_command("echo 'retry'", terminal_id="stuck", timeout=1.0)
        assert retry["status"] == "error"
        
        # Agent sends ^C to recover
        interrupt = await manager.execute_command("^C", terminal_id="stuck", timeout=2.0)
        assert interrupt["status"] == "completed"
        
        # Now agent can continue working
        recovery = await manager.execute_command("echo 'recovered!'", terminal_id="stuck", timeout=5.0)
        assert recovery["status"] == "completed"
        assert "recovered!" in recovery["content"]

    @pytest.mark.asyncio
    async def test_process_with_password_prompt(self, manager):
        """
        Simulate: Interactive process that requires input.
        """
        # Start a process that reads input
        await manager.execute_command("read -p 'Enter: ' var", terminal_id="interactive", timeout=0.5)
        
        # Send input using require_input
        input_result = await manager.execute_command(
            "myinput",
            terminal_id="interactive",
            is_input=True,
            timeout=2.0
        )
        
        # Should not get a busy error
        assert input_result.get("status") != "error" or "busy" not in input_result.get("error", "").lower()

class TestWaitingFunctionality:
    """Tests for the new 'waiting' capability in ShellExecutor."""

    @pytest.fixture
    async def executor(self):
        """Create a ShellExecutor for testing."""
        exc = ShellExecutor("wait-test")
        await asyncio.sleep(0.3)
        yield exc
        exc.terminate()

    @pytest.mark.asyncio
    async def test_wait_for_completion_success(self, executor):
        """
        Test waiting for a running command using a comment/empty command.
        """
        import time
        
        # Start a command that takes ~2 seconds
        # Using session_id="wait-test" implicit in executor
        # We start it with a short timeout so it returns 'running'
        result1 = await executor.run("sleep 2; echo 'done'", timeout=0.1)
        assert result1["status"] == "running"
        assert executor.busy is True
        
        # Now "wait" for it by sending a comment
        start_wait = time.time()
        result2 = await executor.run("# wait for it", timeout=5.0)
        duration = time.time() - start_wait
        
        # Should have waited effectively
        assert duration >= 1.0 
        assert result2["status"] == "completed"
        assert "done" in result2["content"]
        assert executor.busy is False

    @pytest.mark.asyncio
    async def test_wait_times_out(self, executor):
        """
        Test that the wait logic respects the new timeout.
        """
        import time
        
        # Start a very long command
        await executor.run("sleep 10", timeout=0.1)
        
        # Wait for only 1 second
        start_wait = time.time()
        result2 = await executor.run("", timeout=1.0) # Empty string also triggers wait
        duration = time.time() - start_wait
        
        # Should timeout but update content
        assert duration < 2.0
        assert result2["status"] == "running"
        assert executor.busy is True

    @pytest.mark.asyncio
    async def test_busy_error_for_real_commands(self, executor):
        """
        Verify we still get 'busy' error if we try to run a REAL command
        while busy.
        """
        await executor.run("sleep 5", timeout=0.1)
        
        # Try to run 'echo' logic
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
        """Create a fresh TerminalToolManager for testing."""
        mgr = TerminalToolManager.__new__(TerminalToolManager)
        mgr._executors = {}
        mgr._manager_lock = __import__("threading").Lock()
        mgr._default_id = "default"
        mgr._initialized = True
        yield mgr
        mgr.shutdown_all()

    @pytest.fixture
    async def executor(self):
        """Create a ShellExecutor for testing."""
        exc = ShellExecutor("agent-behavior-test")
        await asyncio.sleep(0.3)
        yield exc
        exc.terminate()

    # =========================================================================
    # STATUS FIELD SEMANTICS
    # These tests verify that the status field is reliable for agent decisions.
    # =========================================================================

    @pytest.mark.asyncio
    async def test_status_completed_means_command_finished(self, manager):
        """
        PROMPT DOCUMENTATION: 'status: "completed"' means the command finished.
        
        The agent should trust this field to know when a session is available.
        """
        result = await manager.execute_command("echo 'hello'", timeout=5.0)
        
        assert result["status"] == "completed"
        assert result["exit_code"] is not None
        # Session should be immediately available for another command
        result2 = await manager.execute_command("echo 'world'", timeout=5.0)
        assert result2["status"] == "completed"

    @pytest.mark.asyncio
    async def test_status_running_means_session_busy(self, manager):
        """
        PROMPT DOCUMENTATION: 'status: "running"' means the session is BUSY.
        
        The agent MUST NOT send a new command to a busy session.
        """
        result = await manager.execute_command(
            "sleep 10", 
            terminal_id="busy-test", 
            timeout=0.5
        )
        
        assert result["status"] == "running"
        assert result["exit_code"] is None  # Not finished
        
        # This is the key assertion: trying to send another command should fail
        result2 = await manager.execute_command(
            "echo 'this should fail'",
            terminal_id="busy-test",
            timeout=1.0
        )
        assert result2["status"] == "error"
        assert "busy" in result2.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_status_running_tells_agent_to_use_different_session(self, manager):
        """
        PROMPT DOCUMENTATION: If status is 'running', use a different session_id.
        
        This simulates the correct agent behavior.
        """
        # Start long command in session A
        result_a = await manager.execute_command(
            "sleep 10",
            terminal_id="session_a",
            timeout=0.5
        )
        assert result_a["status"] == "running"
        
        # Agent sees "running" and correctly uses session B instead
        result_b = await manager.execute_command(
            "echo 'session B is free'",
            terminal_id="session_b",
            timeout=5.0
        )
        assert result_b["status"] == "completed"
        assert "session B is free" in result_b["content"]

    # =========================================================================
    # WAITING PATTERNS
    # These tests verify the documented ways to wait for a busy session.
    # =========================================================================

    @pytest.mark.asyncio
    async def test_waiting_with_empty_string(self, executor):
        """
        PROMPT DOCUMENTATION: Send command="" to wait for a busy session.
        """
        # Start command that takes ~1 second
        await executor.run("sleep 1; echo 'finished'", timeout=0.1)
        assert executor.busy is True
        
        # Wait by sending empty string
        result = await executor.run("", timeout=5.0)
        
        # Should have waited and captured output
        assert result["status"] == "completed"
        assert "finished" in result["content"]
        assert executor.busy is False

    @pytest.mark.asyncio
    async def test_waiting_with_comment(self, executor):
        """
        PROMPT DOCUMENTATION: Send command="# waiting" to wait for a busy session.
        """
        # Start command that takes ~1 second
        await executor.run("sleep 1; echo 'done waiting'", timeout=0.1)
        assert executor.busy is True
        
        # Wait using the documented "# waiting" pattern
        result = await executor.run("# waiting", timeout=5.0)
        
        assert result["status"] == "completed"
        assert "done waiting" in result["content"]

    @pytest.mark.asyncio
    async def test_waiting_timeout_returns_running(self, executor):
        """
        PROMPT DOCUMENTATION: If wait times out, status is still 'running'.
        
        This helps the agent understand it should wait longer or interrupt.
        """
        # Start a long command
        await executor.run("sleep 30", timeout=0.1)
        
        # Wait with a short timeout
        result = await executor.run("", timeout=0.5)
        
        # Should still be running
        assert result["status"] == "running"
        assert executor.busy is True

    # =========================================================================
    # INTERRUPT BEHAVIOR
    # =========================================================================

    @pytest.mark.asyncio
    async def test_ctrl_c_recovers_stuck_session(self, manager):
        """
        PROMPT DOCUMENTATION: Send command="^C" to interrupt a stuck session.
        """
        # Start a command that would run forever
        await manager.execute_command(
            "sleep 60",
            terminal_id="stuck",
            timeout=0.5
        )
        
        # Send ^C to interrupt
        interrupt_result = await manager.execute_command(
            "^C",
            terminal_id="stuck",
            timeout=2.0
        )
        
        assert interrupt_result["status"] == "completed"
        assert interrupt_result["exit_code"] == 130  # SIGINT
        
        # Now session should be usable
        result = await manager.execute_command(
            "echo 'recovered'",
            terminal_id="stuck",
            timeout=5.0
        )
        assert result["status"] == "completed"
        assert "recovered" in result["content"]

    # =========================================================================
    # CHAINED COMMAND BEHAVIOR
    # These tests document behavior of && chained commands.
    # =========================================================================

    @pytest.mark.asyncio
    async def test_chained_command_timeout_shows_running(self, manager):
        """
        PROMPT DOCUMENTATION: Avoid '&&' chaining with long commands.
        
        This test shows WHY: if the first command is long, the chain
        appears as 'running' even though partial output may exist.
        """
        # Chain a long command with a quick one
        result = await manager.execute_command(
            "sleep 5 && echo 'chain complete'",
            terminal_id="chain",
            timeout=0.5
        )
        
        # The chain is still running (sleep hasn't finished)
        assert result["status"] == "running"
        # No output yet because sleep produces nothing
        assert "chain complete" not in result.get("content", "")

    @pytest.mark.asyncio
    async def test_quick_chained_commands_work(self, manager):
        """
        Verify that quick chained commands complete normally.
        """
        result = await manager.execute_command(
            "echo 'one' && echo 'two' && echo 'three'",
            timeout=5.0
        )
        
        assert result["status"] == "completed"
        assert "one" in result["content"]
        assert "two" in result["content"]
        assert "three" in result["content"]

    # =========================================================================
    # REQUIRE_INPUT BEHAVIOR
    # =========================================================================

    @pytest.mark.asyncio
    async def test_require_input_sends_to_busy_session(self, manager):
        """
        PROMPT DOCUMENTATION: Use require_input=true to respond to prompts.
        """
        # Start a process that reads input
        await manager.execute_command(
            "read var; echo \"Got: $var\"",
            terminal_id="input-test",
            timeout=0.5
        )
        
        # Send input using is_input=True
        result = await manager.execute_command(
            "my_input_value",
            terminal_id="input-test",
            is_input=True,
            timeout=2.0
        )
        
        # Should NOT get a busy error
        assert result.get("status") != "error" or "busy" not in result.get("error", "").lower()

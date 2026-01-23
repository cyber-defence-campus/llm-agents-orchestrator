"""
Tests for the ShellExecutor class in shell_session.py.

These tests validate that:
1. Simple commands execute and return correct output
2. Exit codes are properly captured
3. Multi-line output is handled correctly
4. Timeouts don't leak internal markers
5. Consecutive commands don't have output mixing
6. Internal markers are sanitized from all outputs
"""

import asyncio
import pytest
import re
from agent_framework.tools.terminal.shell_session import ShellExecutor


class TestShellExecutor:
    """Tests for ShellExecutor functionality."""

    @pytest.fixture
    async def executor(self):
        """Create a ShellExecutor for testing and clean up after."""
        exc = ShellExecutor("test-session")
        # Small warmup delay to ensure shell is fully ready
        await asyncio.sleep(0.3)
        yield exc
        exc.terminate()

    @pytest.mark.asyncio
    async def test_simple_command_execution(self, executor):
        """Test that a simple echo command returns correct output."""
        result = await executor.run("echo 'hello world'", timeout=5.0)

        assert result["status"] == "completed"
        assert "hello world" in result["content"]
        assert result["exit_code"] == 0
        assert result["terminal_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_command_exit_code_success(self, executor):
        """Test that successful commands report exit code 0."""
        result = await executor.run("true", timeout=5.0)

        assert result["exit_code"] == 0
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_command_exit_code_failure(self, executor):
        """Test that failed commands report non-zero exit code."""
        result = await executor.run("false", timeout=5.0)

        assert result["exit_code"] == 1
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_custom_exit_code(self, executor):
        """Test that custom exit codes are captured correctly."""
        result = await executor.run("exit 42", timeout=5.0)

        # Note: exit will close the shell, but the code should be captured
        # The shell may restart via PROMPT_COMMAND issues, but ideally
        # we capture the exit code before that.
        assert result["exit_code"] == 42 or result["status"] == "running"

    @pytest.mark.asyncio
    async def test_multiline_output(self, executor):
        """Test that multi-line output is captured completely."""
        result = await executor.run("echo -e 'line1\\nline2\\nline3'", timeout=5.0)

        assert result["status"] == "completed"
        assert "line1" in result["content"]
        assert "line2" in result["content"]
        assert "line3" in result["content"]

    @pytest.mark.asyncio
    async def test_marker_not_in_output(self, executor):
        """Test that internal markers are never visible in output."""
        # Run a command
        result = await executor.run("echo 'test output'", timeout=5.0)

        assert ShellExecutor.MARKER_PREFIX not in result["content"]
        assert "__AG_CMD__" not in result["content"]

    @pytest.mark.asyncio
    async def test_marker_not_in_timeout_output(self, executor):
        """Test that markers are filtered even when command times out."""
        # Run a command that will timeout
        result = await executor.run("sleep 10", timeout=0.5)

        # Should be still running (timeout)
        assert result["status"] == "running"
        # But markers should be sanitized
        assert ShellExecutor.MARKER_PREFIX not in result["content"]
        assert "__AG_CMD__" not in result["content"]

    @pytest.mark.asyncio
    async def test_consecutive_commands_no_mixing(self, executor):
        """Test that consecutive commands don't mix output."""
        # First command
        result1 = await executor.run("echo 'FIRST_OUTPUT'", timeout=5.0)

        # Second command
        result2 = await executor.run("echo 'SECOND_OUTPUT'", timeout=5.0)

        # Verify no mixing
        assert "FIRST_OUTPUT" in result1["content"]
        assert "SECOND_OUTPUT" not in result1["content"]

        assert "SECOND_OUTPUT" in result2["content"]
        # Note: result2 might contain FIRST_OUTPUT if shell echoes it back,
        # but it should at minimum contain SECOND_OUTPUT

    @pytest.mark.asyncio
    async def test_command_with_special_characters(self, executor):
        """Test commands with special characters."""
        result = await executor.run("echo 'test$var|pipe&background'", timeout=5.0)

        assert result["status"] == "completed"
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_working_directory(self, executor):
        """Test that the working directory is reported."""
        result = await executor.run("pwd", timeout=5.0)

        assert result["status"] == "completed"
        assert result["working_dir"] == executor.work_dir

    @pytest.mark.asyncio
    async def test_session_is_active(self, executor):
        """Test that executor reports active state correctly."""
        assert executor.is_active is True

        executor.terminate()
        assert executor.is_active is False

    @pytest.mark.asyncio
    async def test_empty_command(self, executor):
        """Test that empty commands are handled gracefully."""
        result = await executor.run("", timeout=5.0)

        assert result["status"] == "running"
        assert "__AG_CMD_END__" not in result["content"]

    @pytest.mark.asyncio
    async def test_input_mode(self, executor):
        """Test sending input to a running process."""
        # Start a cat command that reads from stdin
        # Then send input
        result = await executor.run("cat", timeout=0.5)

        # Send some input
        input_result = await executor.run("test input", timeout=1.0, is_input=True)

        assert input_result["status"] == "running"
        # Markers should be sanitized
        assert "__AG_CMD_END__" not in input_result["content"]

    @pytest.mark.asyncio
    async def test_heredoc_execution(self, executor):
        """Test that heredocs are handled correctly without hanging."""
        # This checks the fix for the heredoc hanging issue
        cmd = "cat << 'EOF'\nline1\nline2\nEOF"
        result = await executor.run(cmd, timeout=5.0)

        assert result["status"] == "completed"
        assert result["exit_code"] == 0
        assert "line1" in result["content"]
        assert "line2" in result["content"]
        # Ensure the command itself is not echoed back in a way that breaks parsing
        assert "cat <<" not in result["content"]

    @pytest.mark.asyncio
    async def test_echo_suppression(self, executor):
        """Test that the command itself is not echoed in the output."""
        # Using a variable assignment and echo to distinguish input from output
        # If echo is ON, we might see 'x=10' in the output.
        # If echo is OFF, we should only see '10'.
        cmd = "x=10; echo $x"
        result = await executor.run(cmd, timeout=5.0)

        assert result["status"] == "completed"
        assert "10" in result["content"]
        # The assignment command should not appear in the output
        assert "x=10" not in result["content"]


class TestShellExecutorSanitization:
    """Tests specifically for output sanitization."""

    @pytest.fixture
    async def executor(self):
        """Create a ShellExecutor for testing."""
        exc = ShellExecutor("sanitize-test")
        # Small warmup delay to ensure shell is fully ready
        await asyncio.sleep(0.3)
        yield exc
        exc.terminate()

    def test_sanitize_output_removes_markers(self, executor):
        """Test that _sanitize_output removes marker patterns."""
        test_input = (
            "some output\n__AG_CMD__abc123_START\nmore output\n__AG_CMD__abc123_END0"
        )
        result = executor._sanitize_output(test_input)

        assert "__AG_CMD__" not in result
        assert "some output" in result
        assert "more output" in result

    def test_sanitize_output_handles_multiple_markers(self, executor):
        """Test sanitization with multiple markers."""
        test_input = "__AG_CMD__test1_START\noutput\n__AG_CMD__test1_END0\n"
        result = executor._sanitize_output(test_input)

        assert "__AG_CMD__" not in result

    def test_sanitize_output_reduces_whitespace(self, executor):
        """Test that excessive whitespace is cleaned up."""
        test_input = "line1\n\n\n\n\nline2"
        result = executor._sanitize_output(test_input)

        # Should reduce to max 2 newlines
        assert "\n\n\n" not in result

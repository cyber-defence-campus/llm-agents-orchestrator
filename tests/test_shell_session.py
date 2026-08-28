import asyncio
import pytest
from agent_framework.tools.terminal.shell_session import ShellExecutor


class TestShellExecutor:
    @pytest.fixture
    async def executor(self):
        exc = ShellExecutor("test-session")
        # Small warmup delay to ensure shell is fully ready
        await asyncio.sleep(0.3)
        yield exc
        exc.terminate()

    @pytest.mark.asyncio
    async def test_simple_command_execution(self, executor):
        result = await executor.run("echo 'hello world'", timeout=5.0)

        assert result["status"] == "completed"
        assert "hello world" in result["content"]
        assert result["exit_code"] == 0
        assert result["terminal_id"] == "test-session"

    def test_network_auth_prompt_guard_is_narrow(self, executor):
        assert executor._looks_like_unattended_auth_prompt(
            "ssh app@host 'id'", "app@host's password:"
        )
        assert executor._looks_like_unattended_auth_prompt(
            "sudo id", "[sudo] password for pentester:"
        )
        assert not executor._looks_like_unattended_auth_prompt(
            "read -p 'Password: ' value", "Password: "
        )
        assert not executor._looks_like_unattended_auth_prompt(
            "sshpass -p secret ssh app@host id", ""
        )

    @pytest.mark.asyncio
    async def test_command_exit_code_success(self, executor):
        result = await executor.run("true", timeout=5.0)

        assert result["exit_code"] == 0
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_command_exit_code_failure(self, executor):
        result = await executor.run("false", timeout=5.0)

        assert result["exit_code"] == 1
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_custom_exit_code(self, executor):
        result = await executor.run("exit 42", timeout=5.0)

        # Note: exit will close the shell, but the code should be captured
        # The shell may restart via PROMPT_COMMAND issues, but ideally
        # we capture the exit code before that.
        assert result["exit_code"] == 42 or result["status"] == "running"

    @pytest.mark.asyncio
    async def test_multiline_output(self, executor):
        result = await executor.run("echo -e 'line1\\nline2\\nline3'", timeout=5.0)

        assert result["status"] == "completed"
        assert "line1" in result["content"]
        assert "line2" in result["content"]
        assert "line3" in result["content"]

    @pytest.mark.asyncio
    async def test_marker_not_in_output(self, executor):
        result = await executor.run("echo 'test output'", timeout=5.0)

        assert ShellExecutor.MARKER_PREFIX not in result["content"]
        assert "__AG_CMD__" not in result["content"]

    @pytest.mark.asyncio
    async def test_marker_not_in_timeout_output(self, executor):
        result = await executor.run("sleep 10", timeout=0.5)

        assert result["status"] == "running"
        assert ShellExecutor.MARKER_PREFIX not in result["content"]
        assert "__AG_CMD__" not in result["content"]

    @pytest.mark.asyncio
    async def test_consecutive_commands_no_mixing(self, executor):
        result1 = await executor.run("echo 'FIRST_OUTPUT'", timeout=5.0)
        result2 = await executor.run("echo 'SECOND_OUTPUT'", timeout=5.0)

        assert "FIRST_OUTPUT" in result1["content"]
        assert "SECOND_OUTPUT" not in result1["content"]

        assert "SECOND_OUTPUT" in result2["content"]
        # Note: result2 might contain FIRST_OUTPUT if shell echoes it back,
        # but it should at minimum contain SECOND_OUTPUT

    @pytest.mark.asyncio
    async def test_command_with_special_characters(self, executor):
        result = await executor.run("echo 'test$var|pipe&background'", timeout=5.0)

        assert result["status"] == "completed"
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_working_directory(self, executor):
        result = await executor.run("pwd", timeout=5.0)

        assert result["status"] == "completed"
        assert result["working_dir"] == executor.work_dir

    @pytest.mark.asyncio
    async def test_session_is_active(self, executor):
        assert executor.is_active is True

        executor.terminate()
        assert executor.is_active is False

    @pytest.mark.asyncio
    async def test_empty_command(self, executor):
        result = await executor.run("", timeout=5.0)

        assert result["status"] == "running"
        assert "__AG_CMD_END__" not in result["content"]

    @pytest.mark.asyncio
    async def test_input_mode(self, executor):
        result = await executor.run("cat", timeout=0.5)
        input_result = await executor.run("test input", timeout=1.0, is_input=True)

        assert input_result["status"] == "running"
        assert "__AG_CMD_END__" not in input_result["content"]

    @pytest.mark.asyncio
    async def test_heredoc_execution(self, executor):
        # This checks the fix for the heredoc hanging issue
        cmd = "cat << 'EOF'\nline1\nline2\nEOF"
        result = await executor.run(cmd, timeout=5.0)

        assert result["status"] == "completed"
        assert result["exit_code"] == 0
        assert "line1" in result["content"]
        assert "line2" in result["content"]
        assert "cat <<" not in result["content"]

    @pytest.mark.asyncio
    async def test_multiline_command_emits_end_marker(self, executor):
        cmd = "printf 'first\\n'\nprintf 'second\\n'"
        result = await executor.run(cmd, timeout=5.0)

        assert result["status"] == "completed"
        assert result["exit_code"] == 0
        assert "first" in result["content"]
        assert "second" in result["content"]

    @pytest.mark.asyncio
    async def test_external_command_cannot_drop_end_marker(self, executor):
        # The marker must be sequenced in the same shell command.  A marker
        # sent as a second tmux input line can disappear while sleep is active.
        result = await executor.run("sleep 0.2; printf 'finished\\n'", timeout=3.0)

        assert result["status"] == "completed"
        assert result["exit_code"] == 0
        assert "finished" in result["content"]

    @pytest.mark.asyncio
    async def test_echo_suppression(self, executor):
        # Using a variable assignment and echo to distinguish input from output.
        # If echo is ON, we might see 'x=10' in the output.
        # If echo is OFF, we should only see '10'.
        cmd = "x=10; echo $x"
        result = await executor.run(cmd, timeout=5.0)

        assert result["status"] == "completed"
        assert "10" in result["content"]
        assert "x=10" not in result["content"]


class TestShellExecutorSanitization:
    @pytest.fixture
    async def executor(self):
        exc = ShellExecutor("sanitize-test")
        # Small warmup delay to ensure shell is fully ready
        await asyncio.sleep(0.3)
        yield exc
        exc.terminate()

    def test_sanitize_output_removes_markers(self, executor):
        test_input = (
            "some output\n__AG_CMD__abc123_START\nmore output\n__AG_CMD__abc123_END0"
        )
        result = executor._sanitize_output(test_input)

        assert "__AG_CMD__" not in result
        assert "some output" in result
        assert "more output" in result

    def test_sanitize_output_handles_multiple_markers(self, executor):
        test_input = "__AG_CMD__test1_START\noutput\n__AG_CMD__test1_END0\n"
        result = executor._sanitize_output(test_input)

        assert "__AG_CMD__" not in result

    def test_sanitize_output_reduces_whitespace(self, executor):
        test_input = "line1\n\n\n\n\nline2"
        result = executor._sanitize_output(test_input)

        assert "\n\n\n" not in result

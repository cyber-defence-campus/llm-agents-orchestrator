from typing import Any, Optional, Dict
from pydantic import BaseModel, Field

from agent_framework.tools import register_tool
from .terminal_manager import get_terminal_manager


class ShellCommandResult(BaseModel):
    """Structured result of a shell command execution."""

    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    command_executed: str
    execution_status: str = "unknown"
    error_details: Optional[str] = None


@register_tool
async def run_shell_command(
    command: str,
    require_input: bool = False,
    exec_timeout: Optional[float] = None,
    session_id: Optional[str] = None,
    terminal_id: Optional[str] = None,
    suppress_newline: bool = False,
    agent_state: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Executes a shell command in the specified terminal session.

    Args:
        command: The command string to execute.
        require_input: Whether this execution provides input to a running process.
        exec_timeout: Maximum time in seconds to wait for completion.
        session_id: Identifier for the terminal session (defaults to 'default').
        terminal_id: Alias for session_id.
        suppress_newline: If True, does not send a newline character at the end.
        agent_state: The agent state object, injected by the executor.
        **kwargs: Additional arguments (ignored).
    """
    tm = get_terminal_manager()

    # Determine the default session ID
    default_session = "default"
    if agent_state and hasattr(agent_state, "agent_id"):
        default_session = agent_state.agent_id

    target_session = session_id or terminal_id or default_session

    try:
        result = await tm.execute_command(
            command=command,
            is_input=require_input,
            timeout=exec_timeout,
            terminal_id=target_session,
            no_enter=suppress_newline,
        )
        return result

    except Exception as ex:
        error_response = {
            "status": "execution_failed",
            "reason": str(ex),
            "command_attempted": command,
            "session": target_session,
            "output_buffer": "",
            "return_code": -1,
            "cwd": None,
        }
        return error_response

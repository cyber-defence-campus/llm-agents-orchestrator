from typing import Any, Optional, Dict

from agent_framework.tools import register_tool
from .terminal_manager import get_terminal_manager

# Unbounded, one wrong number parks an agent for hours and reads exactly like a
# hang: a model that took the unit for milliseconds asked for 15000 and would
# have blocked for four hours on a comment. A command needing longer than this
# belongs in its own session, which is what `status: running` invites.
MAX_EXEC_TIMEOUT = 900.0


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

    default_session = "default"
    if agent_state and hasattr(agent_state, "agent_id"):
        default_session = agent_state.agent_id

    target_session = session_id or terminal_id or default_session

    capped = exec_timeout
    if exec_timeout is not None and exec_timeout > MAX_EXEC_TIMEOUT:
        capped = MAX_EXEC_TIMEOUT

    try:
        result = await tm.execute_command(
            command=command,
            is_input=require_input,
            timeout=capped,
            terminal_id=target_session,
            no_enter=suppress_newline,
        )
        if capped != exec_timeout and isinstance(result, dict):
            result["timeout_note"] = (
                f"exec_timeout {exec_timeout} exceeds the {MAX_EXEC_TIMEOUT}s "
                f"maximum and was capped. It is measured in SECONDS."
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

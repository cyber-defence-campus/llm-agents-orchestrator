import os

from .executor import (
    execute_tool,
    execute_tool_invocation,
    execute_tool_with_validation,
    extract_screenshot_from_result,
    process_tool_invocations,
    remove_screenshot_from_result,
    validate_tool_availability,
)
from .registry import (
    register_tool,
    get_tool_by_name,
    get_tool_names,
    get_tools_prompt,
    needs_agent_state,
)

# Re-export registry for compatibility if needed
from .registry import ToolRegistry

SANDBOX_MODE = os.getenv("AGENT_SANDBOX_MODE", "false").lower() == "true"

# Tool Modules Import
from .agent_management import *  # noqa: F403
from .terminal import *  # noqa: F403

# Load external tools from AGENT_TOOL_PATHS
from .loader import load_external_tools

load_external_tools()

__all__ = [
    "execute_tool",
    "execute_tool_invocation",
    "execute_tool_with_validation",
    "extract_screenshot_from_result",
    "get_tool_by_name",
    "get_tool_names",
    "get_tools_prompt",
    "needs_agent_state",
    "process_tool_invocations",
    "register_tool",
    "remove_screenshot_from_result",
    "validate_tool_availability",
]

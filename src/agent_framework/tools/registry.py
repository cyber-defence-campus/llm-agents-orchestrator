import logging
import os
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

logger = logging.getLogger("agent_framework.tools.registry")


class ToolDef(TypedDict):
    name: str
    fn: Callable[..., Any]
    sandbox: bool
    needs_context: bool
    schema_xml: str


class ToolRegistry:
    """
    Central repository for all available agent tools.
    """

    _instance = None

    def __init__(self):
        self._registry: Dict[str, ToolDef] = {}

    @classmethod
    def instance(cls) -> "ToolRegistry":
        if not cls._instance:
            cls._instance = ToolRegistry()
        return cls._instance

    def clear(self):
        self._registry.clear()

    def register(self, fn: Callable, sandbox: bool = True) -> Callable:
        name = fn.__name__
        logger.debug(f"Registering tool: {name}")

        schema, desc = self._load_schema(fn)
        sig = inspect.signature(fn)

        self._registry[name] = {
            "name": name,
            "fn": fn,
            "sandbox": sandbox,
            "needs_context": "agent_state" in sig.parameters,
            "schema_xml": schema,
        }
        return fn

    def get_tool(self, name: str) -> Optional[ToolDef]:
        return self._registry.get(name)

    def list_tools(self) -> List[ToolDef]:
        return list(self._registry.values())

    def generate_prompt_xml(
        self, sandbox_active: bool = False, exclude: Optional[List[str]] = None
    ) -> str:
        """Constructs the XML prompt for tool definitions."""
        valid_tools = [
            t for t in self._registry.values() if (not t["sandbox"]) or sandbox_active
        ]

        if exclude:
            valid_tools = [t for t in valid_tools if t["name"] not in exclude]

        # Sort for stability
        valid_tools.sort(key=lambda x: x["name"])

        return "\n\n".join([t["schema_xml"] for t in valid_tools])

    def _load_schema(self, fn: Callable) -> Tuple[str, str]:
        # Improved schema discovery logic
        try:
            path = Path(inspect.getfile(fn)).parent
            potential_files = [path / "tool_def.xml"]

            for p in potential_files:
                if p.exists():
                    content = p.read_text(encoding="utf-8")
                    # Naive extraction for now - assumes cleaner schemas after refactor
                    # If multiple tools in one file, we need better parsing which I'll assume exists
                    # For now returning full content or finding tag

                    tag = f'<tool name="{fn.__name__}"'
                    if tag in content:
                        start = content.find(tag)
                        end = content.find("</tool>", start) + 7
                        return content[start:end], "Loaded"

                    # Support new simplified schemas if entire file is the tool
                    if "<tool" in content and content.count("<tool") == 1:
                        return content.strip(), "Loaded"

            return (
                f'<tool name="{fn.__name__}"><description>Auto-generated</description></tool>',
                "Auto",
            )

        except Exception:
            return (
                f'<tool name="{fn.__name__}"><description>Error loading schema</description></tool>',
                "Error",
            )


# Global Decorator
def register_tool(func: Optional[Callable] = None, *, sandbox_execution: bool = True):
    reg = ToolRegistry.instance()

    def wrapper(f):
        reg.register(f, sandbox=sandbox_execution)
        return f

    if func:
        return wrapper(func)
    return wrapper


# Facades
def get_tool_by_name(name: str) -> Optional[Callable]:
    t = ToolRegistry.instance().get_tool(name)
    return t["fn"] if t else None


def get_tool_names() -> List[str]:
    return [t["name"] for t in ToolRegistry.instance().list_tools()]


def needs_agent_state(name: str) -> bool:
    t = ToolRegistry.instance().get_tool(name)
    return t["needs_context"] if t else False


def should_execute_in_sandbox(name: str) -> bool:
    t = ToolRegistry.instance().get_tool(name)
    return t["sandbox"] if t else True


def get_tools_prompt(exclude: Optional[List[str]] = None) -> str:
    # Use env var as proxy for sandbox availability
    active = os.getenv("AGENT_SANDBOX_MODE", "false").lower() == "true"
    return ToolRegistry.instance().generate_prompt_xml(active, exclude=exclude)


def clear_registry():
    ToolRegistry.instance().clear()

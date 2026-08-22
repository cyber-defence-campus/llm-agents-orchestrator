import logging
import os
import inspect
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

logger = logging.getLogger("agent_framework.tools.registry")

# tool_def.xml is not well-formed XML -- `<parameter=name ...>` puts the name in
# the tag itself -- so it is read with regexes rather than an XML parser.
PARAM_RE = re.compile(
    r'<parameter=([\w.\-]+)\s+type="([^"]+)"\s+required="(true|false)"\s*>(.*?)</parameter>',
    re.DOTALL,
)
DESCRIPTION_RE = re.compile(r"<description>(.*?)</description>", re.DOTALL)
NOTE_RE = re.compile(r"<note>(.*?)</note>", re.DOTALL)


ACTION_DEF_RE = re.compile(r"<action_definition>.*?</action_definition>", re.DOTALL)
ACTION_NAME_RE = re.compile(r"<name>\s*([\w.\-]+)\s*</name>")


def _action_definition_for(content: str, tool_name: str) -> Optional[str]:
    for match in ACTION_DEF_RE.finditer(content):
        block = match.group(0)
        declared = ACTION_NAME_RE.search(block)
        if declared and declared.group(1) == tool_name:
            return block
    return None


def _json_type(declared: str) -> str:
    t = declared.strip().lower()
    if t.startswith("bool"):
        return "boolean"
    if t in ("integer", "int"):
        return "integer"
    if t in ("number", "float"):
        return "number"
    if t.startswith("list") or t.startswith("array"):
        return "array"
    if t.startswith("dict") or t.startswith("object") or t.startswith("mapping"):
        return "object"
    return "string"


ARG_LINE_RE = re.compile(r"^\s{2,}([\w.\-]+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def _docstring_parts(fn: Callable) -> Tuple[str, Dict[str, str]]:
    """A tool's summary and per-argument text, read off its docstring.

    The only description a tool written in Python has when no tool_def.xml
    describes it. Without this the schema said "Auto-generated" and named the
    parameters and nothing else -- which is all ten ARENA capabilities were
    ever shown as, and a model handed ten undescribed functions answers in
    prose instead of calling one.
    """
    doc = inspect.getdoc(fn) or ""
    if not doc:
        return "", {}

    body, _, tail = doc.partition("Args:")
    args: Dict[str, str] = {}
    if tail:
        for line in tail.splitlines():
            if line.strip() in ("Returns:", "Raises:") or line.strip().endswith(":") \
                    and not ARG_LINE_RE.match(line):
                break
            if match := ARG_LINE_RE.match(line):
                args[match.group(1)] = match.group(2).strip()
    return body.strip(), args


def _schema_from_signature(fn: Callable) -> Dict[str, Any]:
    """Fallback for a tool with no tool_def.xml entry."""
    _, arg_docs = _docstring_parts(fn)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        if name == "agent_state" or param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue
        annotation = param.annotation
        declared = getattr(annotation, "__name__", str(annotation))
        properties[name] = {"type": _json_type(declared)}
        if described := arg_docs.get(name):
            properties[name]["description"] = described
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def _build_json_schema(name: str, schema_xml: str, fn: Callable) -> Dict[str, Any]:
    """An OpenAI-style function schema for one registered tool."""
    head = schema_xml.split("<spec>")[0]
    head_description = DESCRIPTION_RE.search(head)
    description = head_description.group(1).strip() if head_description else ""

    notes = [n.strip() for n in NOTE_RE.findall(schema_xml)]
    if notes:
        description = (description + "\n\n" + "\n".join(f"- {n}" for n in notes)).strip()

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for match in PARAM_RE.finditer(schema_xml):
        param_name, declared, is_required, body = match.groups()
        if param_name == "agent_state":
            continue
        param_description = DESCRIPTION_RE.search(body)
        prop: Dict[str, Any] = {"type": _json_type(declared)}
        if prop["type"] == "array":
            prop["items"] = {}
        if param_description:
            prop["description"] = param_description.group(1).strip()
        properties[param_name] = prop
        if is_required == "true":
            required.append(param_name)

    parameters = (
        {"type": "object", "properties": properties, "required": required}
        if properties
        else _schema_from_signature(fn)
    )
    if not description or description == "Auto-generated":
        description = _docstring_parts(fn)[0]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Invoke the {name} tool.",
            "parameters": parameters,
        },
    }


class ToolDef(TypedDict):
    name: str
    fn: Callable[..., Any]
    sandbox: bool
    needs_context: bool
    schema_xml: str
    schema_json: Dict[str, Any]


class ToolRegistry:
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
            "schema_json": _build_json_schema(name, schema, fn),
        }
        return fn

    def get_tool(self, name: str) -> Optional[ToolDef]:
        return self._registry.get(name)

    def list_tools(self) -> List[ToolDef]:
        return list(self._registry.values())

    def select_tools(
        self,
        sandbox_active: bool = False,
        exclude: Optional[List[str]] = None,
        only: Optional[List[str]] = None,
    ) -> List[ToolDef]:
        """The tools an agent is offered, in a stable order.

        `only` is an allowlist and takes precedence over everything else,
        including `sandbox_active`. An agent whose contract is a fixed set of
        actions cannot express that with `exclude`: the list has to name every
        tool that must not appear, so a tool added later, or one that appears
        because sandbox mode was switched on, silently joins the contract.
        That happened — an agent meant to have typed capabilities and no shell
        was handed `python_execute` when sandbox mode was enabled, and used it.
        """
        if only is not None:
            wanted = set(only)
            valid_tools = [t for t in self._registry.values() if t["name"] in wanted]
        else:
            valid_tools = [
                t
                for t in self._registry.values()
                if (not t["sandbox"]) or sandbox_active
            ]
            if exclude:
                valid_tools = [t for t in valid_tools if t["name"] not in exclude]

        valid_tools.sort(key=lambda x: x["name"])
        return valid_tools

    def generate_prompt_xml(
        self,
        sandbox_active: bool = False,
        exclude: Optional[List[str]] = None,
        only: Optional[List[str]] = None,
    ) -> str:
        tools = self.select_tools(sandbox_active, exclude=exclude, only=only)
        return "\n\n".join([t["schema_xml"] for t in tools])

    def generate_tool_schemas(
        self,
        sandbox_active: bool = False,
        exclude: Optional[List[str]] = None,
        only: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        tools = self.select_tools(sandbox_active, exclude=exclude, only=only)
        return [t["schema_json"] for t in tools]

    def _load_schema(self, fn: Callable) -> Tuple[str, str]:
        try:
            path = Path(inspect.getfile(fn)).parent
            potential_files = [path / "tool_def.xml"]

            for p in potential_files:
                if p.exists():
                    content = p.read_text(encoding="utf-8")

                    tag = f'<tool name="{fn.__name__}"'
                    if tag in content:
                        start = content.find(tag)
                        end = content.find("</tool>", start) + 7
                        return content[start:end], "Loaded"

                    if "<tool" in content and content.count("<tool") == 1:
                        return content.strip(), "Loaded"

                    # <action_definition><name>x</name> is the other spelling on
                    # disk. Unhandled, it fell through to the stub below, and
                    # run_shell_command -- the most used tool there is -- was
                    # advertised to every model with no parameters and no
                    # description at all.
                    block = _action_definition_for(content, fn.__name__)
                    if block:
                        return block, "Loaded"

            return (
                f'<tool name="{fn.__name__}"><description>Auto-generated</description></tool>',
                "Auto",
            )

        except Exception:
            return (
                f'<tool name="{fn.__name__}"><description>Error loading schema</description></tool>',
                "Error",
            )


def register_tool(func: Optional[Callable] = None, *, sandbox_execution: bool = True):
    reg = ToolRegistry.instance()

    def wrapper(f):
        reg.register(f, sandbox=sandbox_execution)
        return f

    if func:
        return wrapper(func)
    return wrapper


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


def get_tools_prompt(exclude: Optional[List[str]] = None,
                     only: Optional[List[str]] = None) -> str:
    # Use env var as proxy for sandbox availability
    active = os.getenv("AGENT_SANDBOX_MODE", "false").lower() == "true"
    return ToolRegistry.instance().generate_prompt_xml(
        active, exclude=exclude, only=only)


def get_tools_schema(exclude: Optional[List[str]] = None,
                     only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    active = os.getenv("AGENT_SANDBOX_MODE", "false").lower() == "true"
    return ToolRegistry.instance().generate_tool_schemas(
        active, exclude=exclude, only=only)


def clear_registry():
    ToolRegistry.instance().clear()

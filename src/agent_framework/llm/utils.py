import html
import re
from typing import Any, List, Dict, Optional

TOOL_CALL_PATTERN = re.compile(
    r"<function=([^>]+)>\n?(.*?)</function(?:=[^>]+)?>", re.DOTALL
)
PARAM_PATTERN = re.compile(
    r"<parameter(?:=| name=[\"'])([^>\"']+)[\"']?>(.*?)</parameter>", re.DOTALL
)


def parse_tool_invocations(content: str) -> Optional[List[Dict[str, Any]]]:
    normalized_content = _normalize_xml_tags(content)
    invocations = []

    for match in TOOL_CALL_PATTERN.finditer(normalized_content):
        tool_name = match.group(1).strip()
        body = match.group(2)

        args = {}
        for param in PARAM_PATTERN.finditer(body):
            key = param.group(1).strip()
            val = html.unescape(param.group(2).strip())
            args[key] = val

        invocations.append({"toolName": tool_name, "args": args})

    return invocations if invocations else None


def _normalize_xml_tags(text: str) -> str:
    if "<function=" in text and text.count("<function=") == 1:
        s_text = text.rstrip()
        if s_text.endswith("</"):
            return s_text + "function>"
        if not s_text.endswith("</function>"):
            return text + "\n</function>"
    return text


def format_tool_call(tool_name: str, args: Dict[str, Any]) -> str:
    params = "".join([f"\n<parameter={k}>{v}</parameter>" for k, v in args.items()])
    return f"<function={tool_name}>{params}\n</function>"


def clean_content(content: str) -> str:
    if not content:
        return ""

    text = _normalize_xml_tags(content)
    text = TOOL_CALL_PATTERN.sub("", text)

    sensitive_tags = [
        r"<inter_agent_message>.*?</inter_agent_message>",
        r"<task_report>.*?</task_report>",
    ]
    for tag in sensitive_tags:
        text = re.sub(tag, "", text, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

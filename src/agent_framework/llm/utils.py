import html
import re
from typing import Any, List, Dict, Optional

TOOL_CALL_PATTERN = re.compile(
    r"<function=([^>]+)>\n?(.*?)</function(?:=[^>]+)?>", re.DOTALL
)
PARAM_PATTERN = re.compile(
    r"<parameter(?:=|\s+name=)[\"']?([\w.\-]+)[\"']?>(.*?)</parameter>", re.DOTALL
)
# No `=`/`name=` on the tag at all -- key=value sits in the body instead.
BARE_PARAM_PATTERN = re.compile(
    r"<parameter>\s*([A-Za-z_][\w.\-]*)=(.*?)</parameter>", re.DOTALL
)
# `name=` with no closing `>` before the value and </parameter>.
UNCLOSED_PARAM_PATTERN = re.compile(
    r"<parameter\s+name=([A-Za-z_][\w.\-]*)=(.*?)</parameter>", re.DOTALL
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
        for pattern in (BARE_PARAM_PATTERN, UNCLOSED_PARAM_PATTERN):
            for param in pattern.finditer(body):
                key = param.group(1).strip()
                if key not in args:
                    args[key] = html.unescape(param.group(2).strip())

        invocations.append({"toolName": tool_name, "args": args})

    return invocations if invocations else None


# The other dialect a model reaches for: the name inside the tag rather than
# in it. `PARAM_PATTERN` already accepts both parameter spellings, so this
# completes an accommodation that was half made -- and until it did, an agent
# that had decided to run a command lost the call to a tag and simply stopped,
# with nothing in the transcript to say why. Seen on a coordination prompt
# where the format block sits 170 lines from the end; the same model emits the
# expected form on a shorter one.
CALL_TAG_PATTERN = re.compile(
    r"<function_call>\s*([A-Za-z_][\w.\-]*)\s*</function_call>")
# DeepSeek's own special-token markup (full-width `｜`) leaking into plain text ahead of the tag.
SPECIAL_TOKEN_PREFIX_PATTERN = re.compile(r"<｜[^｜<>\n]+｜(?=function=)")


def _normalize_xml_tags(text: str) -> str:
    text = SPECIAL_TOKEN_PREFIX_PATTERN.sub("<", text)

    if CALL_TAG_PATTERN.search(text):
        text = CALL_TAG_PATTERN.sub(r"<function=\1>", text, count=1)
        # whatever closed the block in that dialect, including none at all
        text = text.replace("</function_call>", "</function>")

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

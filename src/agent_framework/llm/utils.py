import html
import re
from typing import Any, List, Dict, Optional

TOOL_CALL_PATTERN = re.compile(
    r"<function=([^>]+)>\n?(.*?)</function(?:=[^>]+)?>", re.DOTALL
)
PARAM_PATTERN = re.compile(
    r"<parameter(?:=|\s+name=)[\"']?([\w.\-]+)[\"']?>(.*?)</parameter>", re.DOTALL
)
NAMED_PARAM_PATTERN = re.compile(
    r"<([A-Za-z_][\w.\-]*)>(.*?)</\1>", re.DOTALL
)
# No `=`/`name=` on the tag at all -- key=value sits in the body instead.
BARE_PARAM_PATTERN = re.compile(
    r"<parameter>\s*([A-Za-z_][\w.\-]*)=(.*?)</parameter>", re.DOTALL
)
# `name=` with no closing `>` before the value and </parameter>.
UNCLOSED_PARAM_PATTERN = re.compile(
    r"<parameter\s+name=([A-Za-z_][\w.\-]*)=(.*?)</parameter>", re.DOTALL
)
# Some OpenAI-compatible routes serialize the argument envelope as
# ``<parameter=key</arg_key><arg_value>value</arg_value>``.  The function tag
# is still present, but the old parser silently discarded the argument and
# dispatched a call with no required parameters.  Normalize this dialect
# before the ordinary parameter patterns run.  Keep the key restricted to the
# same identifier shape as the supported forms so arbitrary prose is not
# turned into a tool argument.
MALFORMED_PARAM_PATTERN = re.compile(
    r"<parameter(?:=|\s+name=)[\"']?([A-Za-z_][\w.\-]*)[\"']?"
    r"\s*(?:</arg_key>)?\s*(?:>)?\s*"
    r"<arg_value>(.*?)</arg_value>"
    r"(?:</parameter>)?",
    re.DOTALL | re.IGNORECASE,
)
# A second spelling puts the key in an explicit child tag.  It is emitted by
# the same family of gateways when a tool call is converted twice.
MALFORMED_CHILD_PARAM_PATTERN = re.compile(
    r"<parameter>\s*<arg_key>\s*([A-Za-z_][\w.\-]*)\s*</arg_key>\s*"
    r"<arg_value>(.*?)</arg_value>\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)
# A gateway can also emit a normal ``<parameter=key>value`` opener but splice
# the child-value closing marker into the parameter close as
# ``</arg_value>parameter>``.  Recover only this exact terminator; broad
# balancing here would risk turning ordinary model prose into arguments.
MALFORMED_PARAM_CLOSE_PATTERN = re.compile(
    r"</arg_value>\s*parameter>", re.DOTALL | re.IGNORECASE
)
# Another gateway leaves the parameter opener intact but uses the child-value
# close marker directly, then emits subsequent key/value pairs without their
# enclosing ``<parameter>`` tag.  Keep both recoveries deliberately narrow.
MALFORMED_OPEN_PARAM_PATTERN = re.compile(
    r"<parameter(?:=|\s+name=)[\"']?([A-Za-z_][\w.\-]*)[\"']?>(.*?)</arg_value>",
    re.DOTALL | re.IGNORECASE,
)
MALFORMED_ORPHAN_CHILD_PATTERN = re.compile(
    r"<arg_key>\s*([A-Za-z_][\w.\-]*)\s*</arg_key>\s*"
    r"<arg_value>(.*?)</arg_value>",
    re.DOTALL | re.IGNORECASE,
)
# A fourth dialect leaves the `=` out of the opener entirely: the tag name is
# `parameter`, and the key and its closing `>` sit in the body as
# ``<parameter>command>id</parameter>``. PARAM_PATTERN needs the `=`/`name=`
# opener and BARE_PARAM_PATTERN needs a literal `=` after the key, so this shape
# matched neither and the argument was dropped silently. For `terminal` that
# dropped `command`, leaving an empty call that the sandbox read as a poll; the
# agent could not correct it while its intended command kept disappearing. The
# terminator also accepts an unclosed final parameter that runs into
# ``</function>``, which the same route emits. Keep the key restricted to an
# identifier so ordinary prose is not turned into an argument.
MALFORMED_ANGLE_PARAM_PATTERN = re.compile(
    r"<parameter>\s*([A-Za-z_][\w.\-]*)\s*>(.*?)(?:</parameter>|(?=</function>))",
    re.DOTALL | re.IGNORECASE,
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

        # A native audit trace used flat named children such as <recipient>
        # and <refs>. Dispatching an empty call discarded otherwise complete
        # evidence publications until the integrator exhausted its budget.
        # Recover only a complete, unambiguous flat argument body. Text outside
        # these children, duplicates and mixed parameter dialects remain invalid.
        if not args and '<parameter' not in body:
            children = list(NAMED_PARAM_PATTERN.finditer(body))
            cursor = 0
            candidate = {}
            for child in children:
                key = child.group(1)
                if body[cursor:child.start()].strip() or key in candidate:
                    candidate = {}
                    break
                candidate[key] = html.unescape(child.group(2).strip())
                cursor = child.end()
            if candidate and not body[cursor:].strip():
                args = candidate

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

# DeepSeek also emits its native DSML tool-call envelope as text instead of as
# parsed tool_calls. The name lives in an `invoke name=` attribute and the
# parameters may carry the special token themselves, so none of the `<function=`
# patterns above ever see it. A real call (delete_finding is registered) was
# read as prose, the transcript answered "No tool call was recognized", and the
# agent retried the same spelling until it gave up. Normalize the envelope to
# the XML shape the rest of the parser already understands. The token is
# rendered with full-width bars and, once de-tokenized, can appear with ASCII
# bars or doubled, so match either bar and any run length.
_DSML_BAR = r"[｜|]+"
DSML_INVOKE_OPEN_PATTERN = re.compile(
    rf"<{_DSML_BAR}DSML{_DSML_BAR}\s*invoke\s+name=[\"']?([A-Za-z_][\w.\-]*)[\"']?\s*>",
    re.IGNORECASE,
)
DSML_PARAM_OPEN_PATTERN = re.compile(
    rf"<{_DSML_BAR}DSML{_DSML_BAR}\s*parameter\s+name=[\"']?([A-Za-z_][\w.\-]*)[\"']?\s*>",
    re.IGNORECASE,
)
DSML_INVOKE_CLOSE_PATTERN = re.compile(
    rf"</{_DSML_BAR}DSML{_DSML_BAR}\s*invoke\s*>", re.IGNORECASE
)
DSML_PARAM_CLOSE_PATTERN = re.compile(
    rf"</{_DSML_BAR}DSML{_DSML_BAR}\s*parameter\s*>", re.IGNORECASE
)
DSML_WRAPPER_PATTERN = re.compile(
    rf"</?{_DSML_BAR}DSML{_DSML_BAR}\s*(?:function_calls|tool_calls)\s*>", re.IGNORECASE
)


def _normalize_xml_tags(text: str) -> str:
    def normalize_parameter(match: re.Match[str]) -> str:
        key = match.group(1)
        value = match.group(2)
        return f"<parameter={key}>{value}</parameter>"

    # DeepSeek's DSML envelope first: it carries the tool name in an attribute
    # rather than in the tag, so every pattern below would miss it.
    text = DSML_WRAPPER_PATTERN.sub("", text)
    text = DSML_PARAM_CLOSE_PATTERN.sub("</parameter>", text)
    text = DSML_INVOKE_CLOSE_PATTERN.sub("</function>", text)
    text = DSML_INVOKE_OPEN_PATTERN.sub(r"<function=\1>", text)
    text = DSML_PARAM_OPEN_PATTERN.sub(r"<parameter=\1>", text)

    # Do this before function extraction: the malformed envelope can contain
    # a literal ``>`` only after the gateway's closing arg-key marker, so it
    # cannot be recovered by PARAM_PATTERN itself.
    text = MALFORMED_CHILD_PARAM_PATTERN.sub(normalize_parameter, text)
    text = MALFORMED_PARAM_PATTERN.sub(normalize_parameter, text)
    text = MALFORMED_OPEN_PARAM_PATTERN.sub(normalize_parameter, text)
    text = MALFORMED_ORPHAN_CHILD_PATTERN.sub(normalize_parameter, text)
    text = MALFORMED_PARAM_CLOSE_PATTERN.sub("</parameter>", text)
    text = MALFORMED_ANGLE_PARAM_PATTERN.sub(normalize_parameter, text)
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

import json
import unittest
from unittest.mock import MagicMock

from agent_framework.llm.llm import _native_tool_invocations
from agent_framework.tools.registry import get_tools_schema, ToolRegistry


def _response(tool_calls):
    msg = MagicMock()
    msg.tool_calls = tool_calls
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _call(name, arguments, call_id="call_1"):
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments
    call = MagicMock()
    call.function = fn
    call.id = call_id
    return call


class TestNativeToolCalls(unittest.TestCase):
    def test_parses_native_tool_call(self):
        resp = _response([_call("run_shell_command",
                                '{"command": "nmap -sn 172.28.0.0/24", "session_id": "scan"}')])
        got = _native_tool_invocations(resp)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["toolName"], "run_shell_command")
        self.assertEqual(got[0]["args"]["command"], "nmap -sn 172.28.0.0/24")
        self.assertEqual(got[0]["args"]["session_id"], "scan")

    def test_typed_arguments_survive(self):
        resp = _response([_call("run_shell_command",
                                '{"command": "x", "exec_timeout": 120, "require_input": true}')])
        args = _native_tool_invocations(resp)[0]["args"]
        self.assertIsInstance(args["exec_timeout"], int)
        self.assertIsInstance(args["require_input"], bool)

    def test_bad_json_drops_only_that_call(self):
        resp = _response([
            _call("run_shell_command", "{not json", "bad"),
            _call("think", '{"thought": "ok"}', "good"),
        ])
        got = _native_tool_invocations(resp)
        self.assertEqual([g["toolName"] for g in got], ["think"])

    def test_no_tool_calls_returns_empty(self):
        self.assertEqual(_native_tool_invocations(_response(None)), [])

    def test_every_schema_is_valid(self):
        schemas = get_tools_schema()
        self.assertGreater(len(schemas), 0)
        valid = {"string", "number", "integer", "boolean", "array", "object"}
        for s in schemas:
            fn = s["function"]
            self.assertEqual(s["type"], "function")
            self.assertTrue(fn["description"].strip())
            params = fn["parameters"]
            self.assertEqual(params["type"], "object")
            for pname, pv in params["properties"].items():
                self.assertIn(pv["type"], valid, f"{fn['name']}.{pname}")
            for req in params.get("required", []):
                self.assertIn(req, params["properties"])

    def test_run_shell_command_schema_is_documented(self):
        """It fell back to a bare stub with no parameters at all."""
        schema = next(s for s in get_tools_schema()
                      if s["function"]["name"] == "run_shell_command")
        fn = schema["function"]
        self.assertNotIn("Auto-generated", fn["description"])
        self.assertIn("command", fn["parameters"]["properties"])
        self.assertIn("session_id", fn["parameters"]["properties"])
        self.assertEqual(fn["parameters"]["required"], ["command"])

    def test_no_real_tool_advertises_a_stub(self):
        """Mock tools registered by other test modules legitimately have none."""
        stubs = [t["name"] for t in ToolRegistry.instance().list_tools()
                 if "Auto-generated" in t["schema_xml"]
                 and not t["name"].startswith("mock_")]
        self.assertEqual(stubs, [])


if __name__ == "__main__":
    unittest.main()


class TestDocstringSchemas(unittest.TestCase):
    """A tool described only by its docstring still reaches the model
    described. Ten ARENA capabilities had no tool_def.xml at all and were
    advertised as "Auto-generated" with bare parameter names."""

    def _schema(self, fn):
        from agent_framework.tools.registry import _build_json_schema
        return _build_json_schema(fn.__name__, "", fn)["function"]

    def test_summary_and_arg_docs_are_used(self):
        def enumerate_hosts(subnet: str = "", agent_state=None):
            """List hosts visible from a network segment.

            Args:
                subnet: CIDR to enumerate. Defaults to the agent's own segment.
            """
        fn = self._schema(enumerate_hosts)
        self.assertIn("List hosts visible", fn["description"])
        self.assertNotIn("Auto-generated", fn["description"])
        self.assertIn("CIDR to enumerate",
                      fn["parameters"]["properties"]["subnet"]["description"])

    def test_agent_state_is_never_offered(self):
        def t(target: str, agent_state=None):
            """Do a thing."""
        self.assertNotIn("agent_state", self._schema(t)["parameters"]["properties"])

    def test_undocumented_tool_still_yields_a_schema(self):
        def bare(x: int):
            pass
        fn = self._schema(bare)
        self.assertTrue(fn["description"])
        self.assertEqual(fn["parameters"]["properties"]["x"]["type"], "integer")
        self.assertEqual(fn["parameters"]["required"], ["x"])

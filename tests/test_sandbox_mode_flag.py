import os
from agent_framework.tools.registry import get_tools_prompt, register_tool


@register_tool(sandbox_execution=True)
def mock_sandbox_tool_flag_test(arg: str):
    """A tool that requires sandbox."""
    return f"Executed {arg}"


class TestSandboxModeFlag:
    def setup_method(self):
        self.original_env = dict(os.environ)

    def teardown_method(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_mode_false(self):
        if "AGENT_SANDBOX_MODE" in os.environ:
            del os.environ["AGENT_SANDBOX_MODE"]

        prompt_xml = get_tools_prompt()

        assert (
            'name="mock_sandbox_tool_flag_test"' not in prompt_xml
        ), "Sandbox tool should be HIDDEN when AGENT_SANDBOX_MODE!=true"

    def test_mode_true(self):
        os.environ["AGENT_SANDBOX_MODE"] = "true"

        prompt_xml = get_tools_prompt()

        assert (
            'name="mock_sandbox_tool_flag_test"' in prompt_xml
        ), "Sandbox tool should be SHOWN when AGENT_SANDBOX_MODE=true"

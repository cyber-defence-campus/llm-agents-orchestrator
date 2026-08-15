import pytest
import os
from agent_framework.tools.registry import (
    should_execute_in_sandbox,
    get_tools_prompt,
    register_tool,
)


@register_tool(sandbox_execution=True)
def mock_sandbox_tool(arg: str):
    """A tool that requires sandbox."""
    return f"Executed {arg}"


@register_tool(sandbox_execution=False)
def mock_local_tool(arg: str):
    """A tool that runs locally."""
    return f"Executed {arg}"


class TestSandboxFallback:
    def setup_method(self):
        self.original_sandbox_url = os.environ.get("AGENT_SANDBOX_URL")
        if "AGENT_SANDBOX_URL" in os.environ:
            del os.environ["AGENT_SANDBOX_URL"]

    def teardown_method(self):
        if self.original_sandbox_url:
            os.environ["AGENT_SANDBOX_URL"] = self.original_sandbox_url

    def test_should_execute_in_sandbox_checks(self):
        assert should_execute_in_sandbox("mock_sandbox_tool") is True
        assert should_execute_in_sandbox("mock_local_tool") is False

    def test_tools_prompt_filters_unavailable_tools(self):
        if "AGENT_SANDBOX_MODE" in os.environ:
            del os.environ["AGENT_SANDBOX_MODE"]

        prompt_xml = get_tools_prompt()

        assert 'name="mock_local_tool"' in prompt_xml, "Local tool should be in prompt"

        assert (
            'name="mock_sandbox_tool"' not in prompt_xml
        ), "Sandbox tool should NOT be in prompt when sandbox is disabled"

        os.environ["AGENT_SANDBOX_MODE"] = "true"
        prompt_xml_enabled = get_tools_prompt()

        assert (
            'name="mock_sandbox_tool"' in prompt_xml_enabled
        ), "Sandbox tool should be in prompt when sandbox is enabled"


if __name__ == "__main__":
    t = TestSandboxFallback()
    t.setup_method()
    try:
        t.test_tools_prompt_filters_unavailable_tools()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed as expected: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        t.teardown_method()

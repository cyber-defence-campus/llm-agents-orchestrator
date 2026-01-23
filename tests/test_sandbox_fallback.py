import pytest
import os
from unittest.mock import patch, MagicMock
from agent_framework.tools.registry import (
    should_execute_in_sandbox,
    get_tools_prompt,
    register_tool,
    clear_registry,
)


# Define a mock tool that requires sandbox
@register_tool(sandbox_execution=True)
def mock_sandbox_tool(arg: str):
    """A tool that requires sandbox."""
    return f"Executed {arg}"


# Define a mock tool that runs locally
@register_tool(sandbox_execution=False)
def mock_local_tool(arg: str):
    """A tool that runs locally."""
    return f"Executed {arg}"


class TestSandboxFallback:
    def setup_method(self):
        # Ensure we are in a clean state regarding environment variables
        self.original_sandbox_url = os.environ.get("AGENT_SANDBOX_URL")
        if "AGENT_SANDBOX_URL" in os.environ:
            del os.environ["AGENT_SANDBOX_URL"]

    def teardown_method(self):
        if self.original_sandbox_url:
            os.environ["AGENT_SANDBOX_URL"] = self.original_sandbox_url

    def test_should_execute_in_sandbox_checks(self):
        """Verify registry correctly identifies sandbox tools."""
        assert should_execute_in_sandbox("mock_sandbox_tool") is True
        assert should_execute_in_sandbox("mock_local_tool") is False

    def test_tools_prompt_filters_unavailable_tools(self):
        """
        The prompt generation should EXCLUDE tools that require sandbox
        if AGENT_SANDBOX_MODE is not true.
        """
        # Ensure sandbox mode is disabled
        if "AGENT_SANDBOX_MODE" in os.environ:
            del os.environ["AGENT_SANDBOX_MODE"]

        prompt_xml = get_tools_prompt()

        # Local tool should be present
        assert 'name="mock_local_tool"' in prompt_xml, "Local tool should be in prompt"

        # Sandbox tool should be ABSENT
        assert (
            'name="mock_sandbox_tool"' not in prompt_xml
        ), "Sandbox tool should NOT be in prompt when sandbox is disabled"

        # Enable sandbox mode
        os.environ["AGENT_SANDBOX_MODE"] = "true"
        prompt_xml_enabled = get_tools_prompt()

        # Sandbox tool should be PRESENT
        assert (
            'name="mock_sandbox_tool"' in prompt_xml_enabled
        ), "Sandbox tool should be in prompt when sandbox is enabled"


if __name__ == "__main__":
    # Manually run setup for local testing if needed
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

import pytest
from agent_framework.tools.registry import ToolRegistry, register_tool, get_tools_prompt


# Helper to clear registry between tests
@pytest.fixture(autouse=True)
def clean_registry():
    ToolRegistry.instance().clear()
    yield
    ToolRegistry.instance().clear()


def test_tool_exclusion():
    @register_tool(sandbox_execution=False)
    def tool_a():
        """Description A"""
        pass

    @register_tool(sandbox_execution=False)
    def tool_b():
        """Description B"""
        pass

    @register_tool(sandbox_execution=False)
    def tool_c():
        """Description C"""
        pass

    # Verify all tools needed are present initially
    prompt_all = get_tools_prompt()
    assert 'name="tool_a"' in prompt_all
    assert 'name="tool_b"' in prompt_all
    assert 'name="tool_c"' in prompt_all

    # Verify exclusion works
    prompt_exclude_b = get_tools_prompt(exclude=["tool_b"])
    assert 'name="tool_a"' in prompt_exclude_b
    assert 'name="tool_b"' not in prompt_exclude_b
    assert 'name="tool_c"' in prompt_exclude_b

    # Verify multiple exclusions
    prompt_exclude_ac = get_tools_prompt(exclude=["tool_a", "tool_c"])
    assert 'name="tool_a"' not in prompt_exclude_ac
    assert 'name="tool_b"' in prompt_exclude_ac
    assert 'name="tool_c"' not in prompt_exclude_ac


def test_generate_prompt_xml_exclusion():
    registry = ToolRegistry.instance()

    def dummy_tool():
        pass

    registry.register(dummy_tool, sandbox=False)

    # Manually check the generate_prompt_xml method directly
    xml = registry.generate_prompt_xml(exclude=["dummy_tool"])
    assert 'name="dummy_tool"' not in xml

    xml_with = registry.generate_prompt_xml()
    assert 'name="dummy_tool"' in xml_with

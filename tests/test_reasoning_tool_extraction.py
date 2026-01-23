import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from agent_framework.llm.llm import LLM, LLMConfig

# Mock data
MOCK_CONFIG = LLMConfig(
    model_name="test-provider/test-model",
    api_key="test-key",
)


@pytest.fixture
def mock_queue():
    with patch("agent_framework.llm.llm.get_shared_queue") as mock:
        queue_instance = AsyncMock()
        mock.return_value = queue_instance
        yield queue_instance


@pytest.fixture
def llm_instance(mock_queue):
    return LLM(config=MOCK_CONFIG, agent_name="TestAgent")


class TestReasoningToolExtraction:
    @pytest.mark.asyncio
    async def test_tool_in_reasoning_content_only(self, llm_instance, mock_queue):
        """
        Test that tools are extracted from reasoning_content when content is empty.
        This simulates behavior seen in deepseek-reasoner models.
        """
        reasoning_xml = """
        Thinking process...
        <function=run_shell_command>
        <parameter=command>ls -la</parameter>
        <parameter=terminal_id>term1</parameter>
        </function>
        """

        # Mock response with empty content but populated reasoning_content
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "\n"  # Empty-ish content
        mock_message.reasoning_content = reasoning_xml
        mock_message.provider_specific_fields = {}

        mock_response.choices = [MagicMock(message=mock_message)]
        mock_queue.make_request.return_value = mock_response

        response = await llm_instance.generate([])

        assert response.tool_invocations is not None
        assert len(response.tool_invocations) == 1
        assert response.tool_invocations[0]["toolName"] == "run_shell_command"
        assert response.tool_invocations[0]["args"]["command"] == "ls -la"
        assert response.tool_invocations[0]["args"]["terminal_id"] == "term1"
        assert response.reasoning_content == reasoning_xml

    @pytest.mark.asyncio
    async def test_tool_in_reasoning_content_via_provider_fields(
        self, llm_instance, mock_queue
    ):
        """
        Test extraction when reasoning_content is in provider_specific_fields
        """
        reasoning_xml = """<function=test_tool><parameter=x>1</parameter></function>"""

        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = ""
        del mock_message.reasoning_content
        mock_message.reasoning_content = None

        mock_message.provider_specific_fields = {"reasoning_content": reasoning_xml}

        mock_response.choices = [MagicMock(message=mock_message)]
        mock_queue.make_request.return_value = mock_response

        response = await llm_instance.generate([])

        assert response.tool_invocations is not None
        assert response.tool_invocations[0]["toolName"] == "test_tool"
        assert response.reasoning_content == reasoning_xml

    @pytest.mark.asyncio
    async def test_mixed_tools_content_and_reasoning(self, llm_instance, mock_queue):
        """
        Test that tools are extracted from BOTH content and reasoning_content and merged.
        """
        content_xml = """<function=tool_A><parameter=a>1</parameter></function>"""
        reasoning_xml = """<function=tool_B><parameter=b>2</parameter></function>"""

        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = content_xml
        mock_message.reasoning_content = reasoning_xml

        mock_response.choices = [MagicMock(message=mock_message)]
        mock_queue.make_request.return_value = mock_response

        response = await llm_instance.generate([])

        assert response.tool_invocations is not None
        assert len(response.tool_invocations) == 2

        tool_names = [t["toolName"] for t in response.tool_invocations]
        assert "tool_A" in tool_names
        assert "tool_B" in tool_names

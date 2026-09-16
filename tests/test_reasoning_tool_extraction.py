import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from agent_framework.llm.llm import LLM, LLMConfig
from agent_framework.llm.utils import parse_tool_invocations

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
    def test_recovers_gateway_parameter_envelope(self):
        content = (
            "<function=run_shell_command>\n"
            "<parameter=command</arg_key><arg_value>id</arg_value>\n"
            "</function>"
        )

        calls = parse_tool_invocations(content)

        assert calls == [{"toolName": "run_shell_command", "args": {"command": "id"}}]

    def test_recovers_child_parameter_envelope_and_empty_value(self):
        content = (
            "<function=run_shell_command>"
            "<parameter><arg_key>command</arg_key>"
            "<arg_value></arg_value></parameter></function>"
        )

        calls = parse_tool_invocations(content)

        assert calls == [{"toolName": "run_shell_command", "args": {"command": ""}}]

    def test_recovers_spliced_arg_value_parameter_close(self):
        content = (
            "<function=run_shell_command>"
            "<parameter=command>printf 'ok'</arg_value>parameter>"
            "</function>"
        )

        calls = parse_tool_invocations(content)

        assert calls == [
            {"toolName": "run_shell_command", "args": {"command": "printf 'ok'"}}
        ]

    def test_recovers_open_parameter_and_orphan_child_pairs(self):
        content = (
            "<function=run_shell_command>"
            "<parameter=command>printf 'ok'</arg_value>"
            "<arg_key>session_id</arg_key><arg_value>main2</arg_value>"
            "</tool_call></function>"
        )

        calls = parse_tool_invocations(content)

        assert calls == [
            {
                "toolName": "run_shell_command",
                "args": {"command": "printf 'ok'", "session_id": "main2"},
            }
        ]

    def test_recovers_angle_bracket_parameter_without_equals(self):
        """The opener omitted the `=`: `<parameter>command>...</parameter>`.

        Without this recovery the stray `>` was not a tag boundary to either
        parameter pattern, so the argument was dropped silently. For `terminal`
        that meant `command` vanished and the call ran as an empty poll, and
        the agent could never correct a call whose arguments kept disappearing.
        """
        content = (
            "<function=run_shell_command>\n"
            "<parameter>command>sleep 90; echo hi</parameter>\n"
            "<parameter>purpose>diagnostic</parameter>\n"
            "<parameter>timeout>120</parameter>\n"
            "</function>"
        )

        calls = parse_tool_invocations(content)

        assert calls == [{
            "toolName": "run_shell_command",
            "args": {"command": "sleep 90; echo hi",
                     "purpose": "diagnostic", "timeout": "120"},
        }]

    def test_recovers_angle_bracket_parameter_unclosed_before_function(self):
        content = (
            "<function=run_shell_command>\n"
            "<parameter>command>printf ok</parameter>\n"
            "<parameter>timeout>20</function>"
        )

        calls = parse_tool_invocations(content)

        assert calls == [{
            "toolName": "run_shell_command",
            "args": {"command": "printf ok", "timeout": "20"},
        }]

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

        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "\n"
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

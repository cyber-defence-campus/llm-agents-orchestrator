import pytest
from unittest.mock import MagicMock, patch, AsyncMock, ANY
from agent_framework.llm.llm import LLM, LLMConfig, RequestStats
from agent_framework.llm.types import LLMRequestFailedError
import litellm

# Mock data
MOCK_CONFIG = LLMConfig(
    model_name="test-provider/test-model",
    api_key="test-key",
    enable_prompt_caching=True,
)


@pytest.fixture
def mock_queue():
    with patch("agent_framework.llm.llm.get_shared_queue") as mock:
        queue_instance = AsyncMock()
        mock.return_value = queue_instance
        yield queue_instance


@pytest.fixture
def mock_redis_manager():
    with patch("agent_framework.llm.llm.state_manager") as mock:
        yield mock


@pytest.fixture
def llm_instance(mock_queue, mock_redis_manager):
    return LLM(config=MOCK_CONFIG, agent_name="TestAgent")


class TestLLMCore:
    def test_initialization(self, llm_instance):
        assert llm_instance.config == MOCK_CONFIG
        assert llm_instance.system_prompt is not None
        assert hasattr(llm_instance, "jinja_env")

    def test_prompt_caching_logic(self, llm_instance):
        # Setup specific config for caching
        llm_instance.config.model_name = "anthropic/claude-3-5-sonnet-20240620"

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Msg 1"},
            {"role": "assistant", "content": "Msg 2"},
            # Add enough messages to trigger caching interval
        ] + [{"role": "user", "content": f"Msg {i}"} for i in range(3, 15)]

        with patch(
            "agent_framework.llm.llm.supports_prompt_caching", return_value=True
        ):
            cached_msgs = llm_instance._prepare_cached_messages(messages)

            # System prompt should be cached
            assert isinstance(cached_msgs[0]["content"], list)
            assert cached_msgs[0]["content"][-1]["cache_control"]["type"] == "ephemeral"

            # Check for other cached messages
            cached_count = sum(
                1
                for msg in cached_msgs[1:]
                if isinstance(msg["content"], list)
                and msg["content"][-1].get("cache_control")
            )
            assert cached_count > 0

    @pytest.mark.asyncio
    async def test_generate_success(self, llm_instance, mock_queue):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="Hello world",
                    provider_specific_fields={},
                    reasoning_content=None,
                )
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_queue.make_request.return_value = mock_response

        history = [{"role": "user", "content": "Hi"}]
        response = await llm_instance.generate(history, job_id="job-1")

        assert response.content == "Hello world"
        assert response.role == "agent"

        # Verify request parameters
        mock_queue.make_request.assert_awaited_once()
        call_args = mock_queue.make_request.call_args[0][0]
        assert call_args["model"] == MOCK_CONFIG.model_name
        assert len(call_args["messages"]) >= 2  # System + User

    @pytest.mark.asyncio
    async def test_usage_stats_update(
        self, llm_instance, mock_queue, mock_redis_manager
    ):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.choices = [
            MagicMock(message=MagicMock(content="OK", reasoning_content=None))
        ]

        # Mock completion_cost
        with patch("agent_framework.llm.llm.completion_cost", return_value=0.002):
            mock_queue.make_request.return_value = mock_response
            llm_instance.job_id = "job-123"

            await llm_instance.generate([], job_id="job-123")

            # Check internal stats
            stats = llm_instance.usage_stats
            assert stats["total"]["input_tokens"] == 100
            assert stats["total"]["output_tokens"] == 50
            assert stats["total"]["cost"] == 0.002

            # Check redis update
            mock_redis_manager.increment_usage_stats.assert_called_once()
            _, kwargs = mock_redis_manager.increment_usage_stats.call_args
            assert kwargs["input_tokens"] == 100
            assert kwargs["cost"] == 0.002

    @pytest.mark.asyncio
    async def test_error_handling_rate_limit(self, llm_instance, mock_queue):
        mock_queue.make_request.side_effect = litellm.RateLimitError(
            message="Rate limit exceeded", model="test", llm_provider="test"
        )

        with pytest.raises(LLMRequestFailedError) as exc:
            await llm_instance.generate([])

        assert "Rate limit exceeded" in str(exc.value)

    @pytest.mark.asyncio
    async def test_gemini_message_formatting(self, llm_instance, mock_queue):
        llm_instance.config.model_name = "gemini/gemini-1.5-pro"
        llm_instance.system_prompt = "SYS"

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="OK", reasoning_content=None))
        ]
        mock_queue.make_request.return_value = mock_response

        history = [{"role": "user", "content": "User msg"}]
        await llm_instance.generate(history)

        call_args = mock_queue.make_request.call_args[0][0]
        messages = call_args["messages"]

        # Gemini specific logic: Merges system prompt into first user message if possible
        assert len(messages) == 1
        assert "SYS" in messages[0]["content"][0]["text"]
        assert "User msg" in messages[0]["content"][0]["text"]

    def test_ensure_list_content(self, llm_instance):
        # String -> List
        res = llm_instance._ensure_list_content("text")
        assert res == [{"type": "text", "text": "text"}]

        # List[str] -> List[dict] (legacy format fix)
        res = llm_instance._ensure_list_content(["line1", "line2"])
        assert res == [{"type": "text", "text": "line1\nline2"}]

        # List[dict] -> List[dict] (no change)
        original = [{"type": "image", "url": "..."}]
        res = llm_instance._ensure_list_content(original)
        assert res == original

    @pytest.mark.asyncio
    async def test_tool_parsing_integration(self, llm_instance, mock_queue):
        content_with_tool = """
        <function=test_tool>
        <parameter name="param">value</parameter>
        </function>
        """
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content=content_with_tool, reasoning_content=None)
            )
        ]
        mock_queue.make_request.return_value = mock_response

        response = await llm_instance.generate([])

        assert response.tool_invocations is not None
        assert len(response.tool_invocations) == 1
        assert response.tool_invocations[0]["toolName"] == "test_tool"
        assert response.tool_invocations[0]["args"] == {"param": "value"}

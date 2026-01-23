import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx
from agent_framework.services.sandbox_client import SandboxClient


@pytest.fixture
def sandbox_client():
    return SandboxClient(base_url="http://test-sandbox")


class TestSandboxClient:
    def test_init_defaults(self):
        client = SandboxClient()
        assert client._explicit_base_url is None

    def test_base_url_env_fallback(self):
        with patch.dict("os.environ", {"AGENT_SANDBOX_URL": "http://env-sandbox"}):
            client = SandboxClient()
            assert client.base_url == "http://env-sandbox"

    def test_base_url_explicit_override(self):
        with patch.dict("os.environ", {"AGENT_SANDBOX_URL": "http://env-sandbox"}):
            client = SandboxClient(base_url="http://explicit-sandbox")
            assert client.base_url == "http://explicit-sandbox"

    def test_is_available(self):
        client = SandboxClient(base_url="http://test")
        assert client.is_available is True

        client_empty = SandboxClient()
        with patch.dict("os.environ", {}, clear=True):
            assert client_empty.is_available is False

    @pytest.mark.asyncio
    async def test_ensure_sandbox_success(self, sandbox_client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "sandbox_id": "job-123"}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await sandbox_client.ensure_sandbox("job-123")

            assert result == {"status": "ok", "sandbox_id": "job-123"}
            mock_post.assert_called_once()
            assert mock_post.call_args[0][0] == "http://test-sandbox/sandboxes"
            assert mock_post.call_args[1]["json"] == {"session_id": "job-123"}

    @pytest.mark.asyncio
    async def test_execute_tool_no_url(self):
        client = SandboxClient()
        with patch.dict("os.environ", {}, clear=True):
            result = await client.execute_tool("job-1", "agent-1", "tool-1", {})
            assert "error" in result
            assert "AGENT_SANDBOX_URL not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, sandbox_client):
        # Mock ensure_sandbox to return success
        sandbox_client.ensure_sandbox = AsyncMock(return_value={"status": "ok"})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await sandbox_client.execute_tool(
                session_id="job-1",
                agent_id="agent-1",
                tool_name="test_tool",
                kwargs={"arg": 1},
            )

            assert result == {"result": "success"}
            # First call is ensure_sandbox (mocked method), so we check httpx call
            # Wait, we mocked ensure_sandbox method on the instance, but execute_tool creates a NEW httpx client
            # The httpx client usage is inside execute_tool.
            # We need to be careful: execute_tool calls self.ensure_sandbox FIRST.

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://test-sandbox/execute"
            payload = call_args[1]["json"]
            assert payload["tool_name"] == "test_tool"
            assert payload["session_id"] == "job-1"

    @pytest.mark.asyncio
    async def test_execute_tool_http_error(self, sandbox_client):
        sandbox_client.ensure_sandbox = AsyncMock(return_value={"status": "ok"})

        # Simulate HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=mock_response
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            mock_post.side_effect = httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=mock_response
            )

            result = await sandbox_client.execute_tool(
                session_id="job-1", agent_id="agent-1", tool_name="fail_tool", kwargs={}
            )

            assert "error" in result
            assert "Status 500" in result["error"]

    @pytest.mark.asyncio
    async def test_inject_file_success(self, sandbox_client):
        sandbox_client.ensure_sandbox = AsyncMock(return_value={"status": "ok"})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "uploaded"}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await sandbox_client.inject_file(
                "job-1", "local.txt", "remote.txt"
            )

            assert result == {"status": "uploaded"}
            mock_post.assert_called_once()
            assert (
                mock_post.call_args[0][0] == "http://test-sandbox/sandboxes/job-1/files"
            )

    @pytest.mark.asyncio
    async def test_destroy_sandbox_success(self, sandbox_client):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.delete", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = mock_response

            result = await sandbox_client.destroy_sandbox("job-1")

            assert result == {"success": True}
            mock_delete.assert_called_once_with("http://test-sandbox/sandboxes/job-1")

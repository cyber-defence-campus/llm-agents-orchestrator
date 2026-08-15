import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_redis():
    with patch("main.state_manager", autospec=True) as mock_state_main:
        with patch(
            "agent_framework.services.agent_service.redis_manager", autospec=True
        ) as mock_state_service:
            mock_state_main.get_agent_status.return_value = None
            mock_state_service.get_agent_status.return_value = None

            # Crucially, we link them so test assertions work regardless of which one was called
            # But with autospec, they are distinct objects.
            # To fix this, we can use the 'spec' argument on a shared MagicMock instead of autospec on patch
            # if we want a SINGLE object.

            from agent_framework.state.redis_manager import RedisStateManager

            shared_mock = MagicMock(spec=RedisStateManager)
            shared_mock.get_agent_status.return_value = None

            p1 = patch("main.state_manager", new=shared_mock)
            p2 = patch(
                "agent_framework.services.agent_service.redis_manager", new=shared_mock
            )

            with p1, p2:
                yield shared_mock


@pytest.fixture
def mock_agent():
    with patch("main.DefaultAgent", autospec=True) as mock_agent_class:
        mock_instance = mock_agent_class.return_value

        mock_instance.run_job = AsyncMock(return_value={"status": "completed"})
        mock_instance.start_lifecycle = AsyncMock(return_value={"status": "completed"})

        yield mock_agent_class


@pytest.fixture
def app_with_mocks(mock_redis, mock_agent):
    from main import app, active_agents

    active_agents.clear()
    return app


@pytest.fixture
def test_client(app_with_mocks):
    with TestClient(app_with_mocks) as client:
        yield client


@pytest.fixture
def sample_agent_request():
    return {
        "agent_config": {
            "llm_config": {"model": "gpt-4", "temperature": 0.7},
            "state": {
                "agent_id": "test-agent-001",
                "task": "Test task",
                "parent_id": None,
            },
            "agent_hierarchy": [],
        },
        "job_config": {"automatic": True, "aggressive": False},
    }

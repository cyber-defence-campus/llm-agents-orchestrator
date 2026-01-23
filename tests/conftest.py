"""
Pytest configuration and fixtures for agent-orchestrator tests.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_redis():
    """Mock Redis state manager to avoid requiring a Redis connection."""
    # We patch the *module level* import in main.py and agent_service.py
    # using autospec=True ensures the mock has the same attributes/methods as the real object

    with patch("main.state_manager", autospec=True) as mock_state_main:
        with patch(
            "agent_framework.services.agent_service.redis_manager", autospec=True
        ) as mock_state_service:
            # We want both patches to point to the same mock object logic-wise,
            # but autospec creates specific mocks for where they are patched.
            # However, for simplicity in tests, we can configure them similarly or
            # just return one of them if they are used interchangeably.

            # Common configuration
            mock_state_main.get_agent_status.return_value = None
            mock_state_service.get_agent_status.return_value = None

            # Return the main one for tests to configure further
            # (In a real scenario we might need to sync them or use side_effect to link them)
            # For these tests, we mostly check calls or set return values.

            # Crucially, we link them so test assertions work regardless of which one was called
            # But with autospec, they are distinct objects.
            # To fix this, we can use the 'spec' argument on a shared MagicMock instead of autospec on patch
            # if we want a SINGLE object.

            from agent_framework.state.redis_manager import RedisStateManager

            # Create a shared mock that strictly follows RedisStateManager
            shared_mock = MagicMock(spec=RedisStateManager)
            shared_mock.get_agent_status.return_value = None

            # Apply this shared mock to the patch locations
            p1 = patch("main.state_manager", new=shared_mock)
            p2 = patch(
                "agent_framework.services.agent_service.redis_manager", new=shared_mock
            )

            with p1, p2:
                yield shared_mock


@pytest.fixture
def mock_agent():
    """Mock DefaultAgent to avoid actual LLM calls."""
    # Use autospec=True to strictly follow DefaultAgent structure
    with patch("main.DefaultAgent", autospec=True) as mock_agent_class:
        # The return value of the class (the instance) needs to be configured
        mock_instance = mock_agent_class.return_value

        # Configure the methods to be AsyncMocks since they are async in reality
        # Autospec creates them as MagicMocks, but we need to ensure they behave like async functions
        mock_instance.run_job = AsyncMock(return_value={"status": "completed"})
        mock_instance.start_lifecycle = AsyncMock(return_value={"status": "completed"})

        yield mock_agent_class


@pytest.fixture
def app_with_mocks(mock_redis, mock_agent):
    """Create the app with all mocks applied and cleared active_agents."""
    from main import app, active_agents

    # Clear any leftover state
    active_agents.clear()
    return app


@pytest.fixture
def test_client(app_with_mocks):
    """Create a test client with mocked dependencies."""
    with TestClient(app_with_mocks) as client:
        yield client


@pytest.fixture
def sample_agent_request():
    """Sample request payload for creating an agent."""
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

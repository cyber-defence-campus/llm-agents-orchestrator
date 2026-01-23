"""
Tests for agent lifecycle: create, status, stop.

These tests focus on API contract validation. For full integration testing
with actual agent execution, use the docker-compose setup.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio


class TestAgentCreation:
    """Tests for POST /agents endpoint."""

    def test_create_agent_success(self, test_client, sample_agent_request):
        """Test successful agent creation returns 202 with agent_id."""
        response = test_client.post("/agents", json=sample_agent_request)

        assert response.status_code == 202
        data = response.json()
        assert "agent_id" in data
        assert data["agent_id"] == "test-agent-001"
        assert data["message"] == "Agent creation initiated."

    def test_create_agent_invalid_payload(self, test_client):
        """Test that invalid payload returns 422."""
        response = test_client.post("/agents", json={"invalid": "data"})

        assert response.status_code == 422

    def test_create_agent_missing_llm_config(self, test_client):
        """Test that missing llm_config returns error."""
        request = {
            "agent_config": {"state": {"agent_id": "test", "task": "test"}},
            "job_config": {},
        }
        response = test_client.post("/agents", json=request)

        # Should fail during processing
        assert response.status_code in [422, 500]


class TestSimpleAgentCreation:
    """Tests for POST /agents/simple endpoint."""

    def test_create_agent_simple_success(self, test_client, mock_redis):
        """Test successful simple agent creation."""
        request = {
            "name": "SimpleAgent",
            "task": "Simple task",
            "model": "gemini/gemini-3-flash-preview",
            "automatic": True,
        }

        # We need to mock get_all_agent_nodes to avoid errors in get_agent_hierarchy
        # Note: get_all_agent_nodes might not be used anymore, but keeping just in case
        mock_redis.get_paginated_agent_nodes.return_value = ([], 0)
        # Mock methods needed for get_agent_hierarchy
        mock_redis.get_all_agent_nodes.return_value = {}
        mock_redis.get_agent_nodes_by_job_id.return_value = {}
        mock_redis.get_all_edges.return_value = []

        response = test_client.post("/agents/simple", json=request)

        assert response.status_code == 202
        data = response.json()
        assert "agent_id" in data
        assert data["name"] == "SimpleAgent"

        # Verify that add_agent_node was called (this ensures we are using the correct method)
        assert mock_redis.add_agent_node.called


class TestAgentStatus:
    """Tests for GET /agents/{agent_id}/status endpoint."""

    def test_get_status_not_found_not_in_redis(self, test_client, mock_redis):
        """Test getting status of non-existent agent (not in memory or Redis)."""
        mock_redis.get_agent_status.return_value = None

        response = test_client.get("/agents/nonexistent/status")

        assert response.status_code == 404
        assert response.json()["detail"] == "Agent not found"

    def test_get_status_from_redis(self, test_client, mock_redis):
        """Test getting status from Redis for completed agent."""
        mock_redis.get_agent_status.return_value = "finished"

        response = test_client.get("/agents/completed-agent/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "finished"
        assert data["agent_id"] == "completed-agent"


class TestAgentStop:
    """Tests for POST /agents/{agent_id}/stop endpoint."""

    def test_stop_agent_not_found(self, test_client, mock_redis):
        """Test stopping non-existent agent returns 404."""
        # Ensure it's not found in Redis either
        mock_redis.get_agent_status.return_value = None

        response = test_client.post("/agents/nonexistent/stop")

        assert response.status_code == 404
        assert response.json()["detail"] == "Agent not found"

    def test_stop_agent_not_running(self, test_client, mock_redis):
        """Test stopping an agent that exists but is not running."""
        mock_redis.get_agent_status.return_value = "finished"

        response = test_client.post("/agents/completed-agent/stop")

        assert response.status_code == 200
        assert "is not running" in response.json()["message"]


class TestAgentListing:
    """Tests for GET /agents endpoint."""

    def test_list_agents_success(self, test_client, mock_redis):
        """Test listing agents with pagination."""
        mock_agents = [
            {"id": "agent-1", "name": "Agent 1", "status": "running", "task": "Task 1"},
            {
                "id": "agent-2",
                "name": "Agent 2",
                "status": "finished",
                "task": "Task 2",
            },
        ]
        mock_redis.get_paginated_agent_nodes.return_value = (mock_agents, 10)

        response = test_client.get("/agents?limit=5&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert len(data["agents"]) == 2
        assert data["total"] == 10
        assert data["agents"][0]["agent_id"] == "agent-1"
        mock_redis.get_paginated_agent_nodes.assert_called_with(limit=5, offset=0)


class TestAgentDeletion:
    """Tests for DELETE /agents/{agent_id} endpoint."""

    def test_delete_agent_success(self, test_client, mock_redis):
        """Test deleting an agent."""
        # Ensure agent is not running
        mock_redis.get_agent_status.return_value = "finished"

        response = test_client.delete("/agents/test-agent")

        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
        mock_redis.delete_agent.assert_called_with("test-agent")


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_json(self, test_client):
        """Test that OpenAPI spec is available."""
        response = test_client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Agent Manager Service"
        assert "/agents" in data["paths"]

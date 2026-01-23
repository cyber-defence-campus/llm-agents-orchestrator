"""
Tests for model selection logic, ensuring AGENT_MODEL is respected.
"""

import os
from unittest.mock import patch
import pytest


class TestModelSelection:
    """Tests for model selection in create_agent_simple."""

    def test_create_agent_uses_env_model(self, test_client, mock_redis):
        """Test that AGENT_MODEL env var is used when model is not provided."""
        request = {
            "name": "EnvModelAgent",
            "task": "Test env model",
            # No 'model' field provided
            "automatic": True,
        }

        # Mock Redis calls
        mock_redis.get_paginated_agent_nodes.return_value = ([], 0)
        mock_redis.get_all_agent_nodes.return_value = {}
        mock_redis.get_agent_nodes_by_job_id.return_value = {}
        mock_redis.get_all_edges.return_value = []

        # We need to mock os.getenv to return a specific model
        # We wrap the original getenv to fallback for other vars
        original_getenv = os.getenv

        def side_effect(key, default=None):
            if key == "AGENT_MODEL":
                return "provider/env-defined-model"
            return original_getenv(key, default)

        with patch("os.getenv", side_effect=side_effect):
            response = test_client.post("/agents/simple", json=request)

        assert response.status_code == 202

        # Verify that add_agent_node was called with the correct model
        # We need to inspect the arguments passed to add_agent_node
        assert mock_redis.add_agent_node.called

        # Get the arguments of the first call
        args, _ = mock_redis.add_agent_node.call_args
        node_data = args[0]

        assert node_data["model"] == "provider/env-defined-model"

    def test_create_agent_override_model(self, test_client, mock_redis):
        """Test that provided model overrides AGENT_MODEL env var."""
        request = {
            "name": "OverrideModelAgent",
            "task": "Test override model",
            "model": "provider/explicit-model",
            "automatic": True,
        }

        # Mock Redis calls
        mock_redis.get_paginated_agent_nodes.return_value = ([], 0)
        mock_redis.get_all_agent_nodes.return_value = {}
        mock_redis.get_agent_nodes_by_job_id.return_value = {}
        mock_redis.get_all_edges.return_value = []

        with patch.dict(os.environ, {"AGENT_MODEL": "provider/env-defined-model"}):
            response = test_client.post("/agents/simple", json=request)

        assert response.status_code == 202

        assert mock_redis.add_agent_node.called
        args, _ = mock_redis.add_agent_node.call_args
        node_data = args[0]

        assert node_data["model"] == "provider/explicit-model"

import os
from unittest.mock import patch
import pytest


class TestModelSelection:
    def test_create_agent_uses_env_model(self, test_client, mock_redis):
        request = {
            "name": "EnvModelAgent",
            "task": "Test env model",
            "automatic": True,
        }

        mock_redis.get_paginated_agent_nodes.return_value = ([], 0)
        mock_redis.get_all_agent_nodes.return_value = {}
        mock_redis.get_agent_nodes_by_job_id.return_value = {}
        mock_redis.get_all_edges.return_value = []

        original_getenv = os.getenv

        def side_effect(key, default=None):
            if key == "AGENT_MODEL":
                return "provider/env-defined-model"
            return original_getenv(key, default)

        with patch("os.getenv", side_effect=side_effect):
            response = test_client.post("/agents/simple", json=request)

        assert response.status_code == 202

        assert mock_redis.add_agent_node.called

        args, _ = mock_redis.add_agent_node.call_args
        node_data = args[0]

        assert node_data["model"] == "provider/env-defined-model"

    def test_create_agent_override_model(self, test_client, mock_redis):
        request = {
            "name": "OverrideModelAgent",
            "task": "Test override model",
            "model": "provider/explicit-model",
            "automatic": True,
        }

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

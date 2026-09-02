class TestAgentCreation:
    def test_create_agent_success(self, test_client, sample_agent_request):
        response = test_client.post("/agents", json=sample_agent_request)

        assert response.status_code == 202
        data = response.json()
        assert "agent_id" in data
        assert data["agent_id"] == "test-agent-001"
        assert data["message"] == "Agent creation initiated."

    def test_create_agent_invalid_payload(self, test_client):
        response = test_client.post("/agents", json={"invalid": "data"})

        assert response.status_code == 422

    def test_create_agent_missing_llm_config(self, test_client):
        request = {
            "agent_config": {"state": {"agent_id": "test", "task": "test"}},
            "job_config": {},
        }
        response = test_client.post("/agents", json=request)

        assert response.status_code in [422, 500]


class TestSimpleAgentCreation:
    def test_create_agent_simple_success(self, test_client, mock_redis):
        request = {
            "name": "SimpleAgent",
            "task": "Simple task",
            "model": "gemini/gemini-3-flash-preview",
            "automatic": True,
        }

        # We need to mock get_all_agent_nodes to avoid errors in get_agent_hierarchy
        mock_redis.get_paginated_agent_nodes.return_value = ([], 0)
        mock_redis.get_all_agent_nodes.return_value = {}
        mock_redis.get_agent_nodes_by_job_id.return_value = {}
        mock_redis.get_all_edges.return_value = []

        response = test_client.post("/agents/simple", json=request)

        assert response.status_code == 202
        data = response.json()
        assert "agent_id" in data
        assert data["name"] == "SimpleAgent"

        assert mock_redis.add_agent_node.called

    def test_a_mapping_context_reaches_the_prompt_not_the_task_prose(
        self, test_client, mock_redis
    ):
        """`context` is declared a dict and must be usable as one.

        It was typed `dict` on the request and `str` where it was consumed, so
        a caller's mapping was f-stringed into the task and `context_data` --
        which the field itself aliases to `context`, and which the system
        prompt renders its tool schemas from -- stayed empty. A campaign that
        told each agent which capabilities its range carries out was talking to
        nobody: the model read the names in its task prose, had no schema for
        any of them, and called them by guessing the argument shape.
        """
        from agent_framework.services import agent_service

        mock_redis.get_paginated_agent_nodes.return_value = ([], 0)
        mock_redis.get_all_agent_nodes.return_value = {}
        mock_redis.get_agent_nodes_by_job_id.return_value = {}
        mock_redis.get_all_edges.return_value = []

        _, state = agent_service.create_agent_config(
            name="WithCapabilities", task="Reach the domain controller.",
            context={"capabilities": ["enumerate_hosts", "read_file"]},
        )
        assert state.context_data == {
            "capabilities": ["enumerate_hosts", "read_file"]
        }
        assert "capabilities" not in state.task, (
            "a mapping is template context, not prose to paste into the task")

        # a string still prepends, because callers depend on that
        _, prose = agent_service.create_agent_config(
            name="WithPreamble", task="Reach the domain controller.",
            context="You are on an internal segment.",
        )
        assert prose.task.startswith("You are on an internal segment.")
        assert prose.context_data == {}


class TestAgentStatus:
    def test_get_status_not_found_not_in_redis(self, test_client, mock_redis):
        mock_redis.get_agent_status.return_value = None

        response = test_client.get("/agents/nonexistent/status")

        assert response.status_code == 404
        assert response.json()["detail"] == "Agent not found"

    def test_get_status_from_redis(self, test_client, mock_redis):
        mock_redis.get_agent_status.return_value = "finished"

        response = test_client.get("/agents/completed-agent/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "finished"
        assert data["agent_id"] == "completed-agent"


class TestAgentStop:
    def test_stop_agent_not_found(self, test_client, mock_redis):
        mock_redis.get_agent_status.return_value = None

        response = test_client.post("/agents/nonexistent/stop")

        assert response.status_code == 404
        assert response.json()["detail"] == "Agent not found"

    def test_stop_agent_not_running(self, test_client, mock_redis):
        mock_redis.get_agent_status.return_value = "finished"

        response = test_client.post("/agents/completed-agent/stop")

        assert response.status_code == 200
        assert "is not running" in response.json()["message"]


class TestAgentListing:
    def test_list_agents_success(self, test_client, mock_redis):
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
    def test_delete_agent_success(self, test_client, mock_redis):
        mock_redis.get_agent_status.return_value = "finished"

        response = test_client.delete("/agents/test-agent")

        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
        mock_redis.delete_agent.assert_called_with("test-agent")


class TestAPIDocumentation:
    def test_openapi_json(self, test_client):
        response = test_client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Agent Manager Service"
        assert "/agents" in data["paths"]


def test_supervision_polls_do_not_trip_the_stale_action_breaker():
    """A coordinator waiting on a worker reads the same tree and the same
    findings every time. Counted as stale actions, four such reads killed the
    root while its child was still working."""
    from agent_framework.agents.state import AgentContext

    context = AgentContext(agent_name="root")
    same_tree = {"hierarchy_view": "Root -> Worker", "count": 2}
    for _ in range(AgentContext.STALE_ACTION_STOP_LIMIT + 3):
        context.record_tool_result("inspect_agent_tree", {}, same_tree, False)
        context.record_tool_result("get_findings", {}, [{"id": "f1"}], False)

    assert not context.stop_requested, (
        "supervising a working child tripped the loop breaker"
    )


def test_a_repeated_failing_action_asks_for_a_pivot_but_does_not_kill():
    """Repeating a failed command is ordinary work -- an exploit attempt, a
    login, a scan that timed out. The breaker says so and lets the agent
    decide; killing it ended runs mid-chain."""
    from agent_framework.agents.state import AgentContext

    context = AgentContext(agent_name="worker")
    for _ in range(AgentContext.STALE_ACTION_STOP_LIMIT + 2):
        context.record_tool_result(
            "terminal", {"command": "curl http://x/"}, {"error": "refused"}, True
        )

    assert context.tactical_memory.get("pivot_required") is True
    assert context.tactical_memory.get("stale_circuit_breaker")
    assert not context.stop_requested

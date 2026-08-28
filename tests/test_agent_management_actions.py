from types import SimpleNamespace
from unittest.mock import patch

from agent_framework.tools.agent_management.actions import dispatch_agent_msg


def test_dispatch_agent_msg_rejects_self_alias_without_queueing():
    state = SimpleNamespace(agent_id="agent-1")

    with patch.object(dispatch_agent_msg.__globals__["db"], "get_agent_node") as lookup:
        result = dispatch_agent_msg(state, "self", "continue")

    assert result == {
        "status": "failed",
        "reason": "Cannot dispatch a message to yourself; continue the task directly",
    }
    lookup.assert_not_called()


def test_dispatch_agent_msg_rejects_own_id_without_queueing():
    state = SimpleNamespace(agent_id="agent-1")

    with patch.object(dispatch_agent_msg.__globals__["db"], "get_agent_node") as lookup:
        result = dispatch_agent_msg(state, "agent-1", "continue")

    assert result["status"] == "failed"
    lookup.assert_not_called()

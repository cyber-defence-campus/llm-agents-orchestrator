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
AUTONOMOUS_COORDINATOR_WAIT_SECONDS = 30
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
            if wait_count >= AUTONOMOUS_COORDINATOR_WAIT_LIMIT:
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

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Literal, Optional, List, Dict

from agent_framework.state import redis_manager as db
from agent_framework.tools import register_tool

logger = logging.getLogger(__name__)


@register_tool(sandbox_execution=False)
def spawn_sub_agent(
    agent_state: Any,
    task_description: str,
    agent_name: str,
    ui_summary: Optional[str] = None,
    share_history: bool = True,
    capabilities: Optional[str] = None,
    model_override: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Deploys a new autonomous agent worker.
    """
    creator_id = agent_state.agent_id
    logger.info(f"Agent {creator_id} is spawning sub-agent '{agent_name}'")

    # Module parsing logic
    requested_modules = []
    if capabilities:
        if isinstance(capabilities, str):
            # Split by comma and filter empty
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

        # Delegate to service
        result = spawn_agent(
            parent_state=agent_state,
            name=agent_name,
            task=task_description,
            prompt_modules=requested_modules,
            model=model_override,
            inherit_context=share_history,
        )
        # Add helpful hint about waiting option
        if result.get("success"):
            result["hint"] = (
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
    """
    Signals the completion of the assigned objective.
    """
    current_id = agent_state.agent_id
    supervisor_id = getattr(agent_state, "parent_id", None)

    # Handle legacy 'discovered_items' alias
    if discovered_items:
        if artifacts is None:
            artifacts = discovered_items
        else:
            artifacts.extend(discovered_items)

    logger.info(f"Task completion for {current_id}. Success={is_success}")

    # ALWAYS Mark local state as completed
    agent_state.mark_completed()

    # Update DB status
    status = "completed" if is_success else "failed"
    db.update_agent_status(current_id, status)

    if not supervisor_id:
        return {
            "status": "complete",
            "message": "Task completed. No supervisor to notify.",
        }

    if notify_supervisor:
        try:
            # Build report structure
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

            # Convert to XML-like format for compatibility if needed,
            # or just send a structured JSON-like text if the receiver understands it.
            # We will use a custom format distinct from the original.

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
            # We wrap it in a system tag so the parent parser picks it up?
            # Or we just send it as content.
            # Original used <agent_completion_report>.
            # We should probably stick to a structural tag for the parent to parse it mechanically if logic depends on it.
            # Let's use <task_report> to differentiate.

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

    return {"status": "complete", "supervisor_notified": notify_supervisor}


@register_tool(sandbox_execution=False)
def dispatch_agent_msg(
    agent_state: Any,
    recipient_id: str,
    body: str,
    category: Literal["query", "instruction", "info"] = "info",
    urgency: Literal["low", "normal", "high", "critical"] = "normal",
) -> Dict[str, Any]:
    """
    Sends a message to another agent in the graph.
    """
    sender = agent_state.agent_id
    logger.info(f"Message dispatch: {sender} -> {recipient_id}")

    # Verify recipient
    target_node = db.get_agent_node(recipient_id)
    if not target_node:
        return {"status": "failed", "reason": "Recipient ID unknown"}

    msg_uuid = f"msg_{uuid.uuid4().hex[:12]}"

    # Map urgency/category to standard format if needed, or use new ones
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
        # We record the interaction in the graph
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
    """
    Pauses execution to await external events.
    """
    agent_id = agent_state.agent_id
    logger.info(f"Agent {agent_id} entering sleep: {wait_reason}")

    try:
        agent_state.set_waiting(timeout=max_wait_seconds)
        db.update_agent_status(agent_id, "waiting")
        db.update_agent_node_fields(agent_id, {"wait_reason": wait_reason})
        return {"status": "paused", "mode": "waiting"}
    except Exception as e:
        return {"status": "error", "details": str(e)}


@register_tool(sandbox_execution=False)
def inspect_agent_tree(agent_state: Any) -> Dict[str, Any]:
    """
    Retrieves the current agent hierarchy structure.
    """
    try:
        # 1. Filter by job_id if available
        job_id = None
        if agent_state.sandbox_info:
            job_id = agent_state.sandbox_info.get("job_id")

        if job_id:
            nodes_map = db.get_agent_nodes_by_job_id(job_id)
        else:
            nodes_map = db.get_all_agent_nodes()

        all_nodes = list(nodes_map.values())

        # 2. Build tree structure
        children_map = {}
        roots = []
        node_ids = set(n.get("id") for n in all_nodes)

        for node in all_nodes:
            parent_id = node.get("parent_id")
            # A node is a root if it has no parent OR its parent is not in the current set
            if not parent_id or parent_id not in node_ids:
                roots.append(node)
            else:
                children_map.setdefault(parent_id, []).append(node)

        # Helper to sort nodes
        def sort_key(n):
            return n.get("created_at", "")

        roots.sort(key=sort_key)
        for pid in children_map:
            children_map[pid].sort(key=sort_key)

        # 3. Render tree
        output_lines = ["Agent System Overview:", "======================"]

        def render_node(node, depth=0):
            nid = node.get("id")
            name = node.get("name", "Unknown")
            status = node.get("status", "unknown").upper()
            role = node.get("agent_type", "General")

            # Indentation
            indent = "  " * depth
            marker = "└─ " if depth > 0 else "• "

            output_lines.append(f"{indent}{marker}{name} [{nid}]")
            output_lines.append(f"{indent}   Status: {status} | Type: {role}")

            # Recurse
            if nid in children_map:
                for child in children_map[nid]:
                    render_node(child, depth + 1)

        if not roots and all_nodes:
            # Fallback if circle or weirdness: just dump all
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
    """
    Stops or removes an agent from the system.
    """
    try:
        # Check existence
        node = db.get_agent_node(target_id)
        if not node:
            return {"status": "error", "message": "Agent does not exist"}

        current_status = node.get("status")

        if current_status in ("running", "waiting", "initializing"):
            # Soft stop
            db.update_agent_status(target_id, "stopping")
            logger.info(f"Stop signal sent to {target_id}: {justification}")
            return {"status": "signaled", "action": "stopping"}
        else:
            # Hard delete
            db.delete_agent(target_id)
            logger.info(f"Agent {target_id} deleted: {justification}")
            return {"status": "deleted", "action": "removed"}

    except Exception as e:
        return {"status": "failed", "error": str(e)}

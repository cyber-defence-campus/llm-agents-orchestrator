import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

from agent_framework.agents.state import AgentContext
from agent_framework.llm.config import LLMConfig
from agent_framework.state import redis_manager

logger = logging.getLogger("agent_framework.services.agent_service")


def _resolve_api_key(model_name: str) -> str | None:
    if not model_name:
        return None
    model_lower = model_name.lower()
    if "openai" in model_lower or "gpt" in model_lower:
        return os.getenv("OPENAI_API_KEY")
    elif "gemini" in model_lower or "google" in model_lower:
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    elif "deepseek" in model_lower:
        return os.getenv("DEEPSEEK_API_KEY")
    elif "anthropic" in model_lower or "claude" in model_lower:
        return os.getenv("ANTHROPIC_API_KEY")
    return None


def get_agent_hierarchy(job_id: str | None = None) -> list[dict[str, Any]]:
    if job_id:
        agent_nodes = redis_manager.get_agent_nodes_by_job_id(job_id)
    else:
        agent_nodes = redis_manager.get_all_agent_nodes()

    if not agent_nodes:
        return []

    edges = redis_manager.get_all_edges()

    children_map: dict[str, list[str]] = {}
    child_ids = set()

    for edge in edges:
        if edge.get("type") == "delegation":
            parent_id = edge.get("from")
            child_id = edge.get("to")
            if parent_id and child_id:
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(child_id)
                child_ids.add(child_id)

    def build_tree(agent_id: str, depth: int = 0) -> dict[str, Any] | None:
        if agent_id not in agent_nodes or depth > 10:
            return None

        node = agent_nodes[agent_id].copy()
        node["children"] = []

        for child_id in children_map.get(agent_id, []):
            child_node = build_tree(child_id, depth + 1)
            if child_node:
                node["children"].append(child_node)

        return node

    root_nodes = []
    for agent_id in agent_nodes:
        if agent_id not in child_ids:
            tree = build_tree(agent_id)
            if tree:
                root_nodes.append(tree)

    return root_nodes


def format_agent_hierarchy(agents: list[dict], level: int = 0) -> str:
    result = ""
    for agent in agents:
        indent = "  " * level
        status = agent.get("status", "unknown")
        result += f"{indent}- {agent.get('name', 'Unknown')} ({agent.get('id', 'N/A')}) - {status}\n"
        result += f"{indent}  Task: {agent.get('task', 'N/A')}\n"
        if agent.get("children"):
            result += format_agent_hierarchy(agent["children"], level + 1)
    return result


def create_agent_config(
    name: str,
    task: str,
    job_id: str | None = None,
    parent_id: str | None = None,
    prompt_modules: list[str] | None = None,
    model: str | None = None,
    context: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> tuple[dict[str, Any], AgentContext]:
    full_task = task
    if context:
        full_task = f"{context}\n\nYour assigned task is as follows:\n{task}"

    module_list = list(prompt_modules) if prompt_modules else []

    if parent_id is not None:
        if "coordination/sub_agent" not in module_list:
            module_list.append("coordination/sub_agent")
        if "coordination/root_agent" in module_list:
            module_list.remove("coordination/root_agent")
    else:
        if "coordination/root_agent" not in module_list and not module_list:
            module_list.append("coordination/root_agent")

    agent_state = AgentContext(
        task=full_task,
        original_task=task,
        agent_name=name,
        parent_id=parent_id,
        sandbox_info={"job_id": job_id} if job_id else {},
    )
    agent_id = agent_state.agent_id

    platform_llm_name = model or os.getenv(
        "AGENT_MODEL", "gemini/gemini-3-flash-preview"
    )
    effective_api_key = api_key or _resolve_api_key(platform_llm_name)

    agent_hierarchy = get_agent_hierarchy(job_id)

    llm_config = LLMConfig(
        model_name=platform_llm_name,
        prompt_modules=module_list,
        api_key=effective_api_key,
        reasoning_effort=reasoning_effort,
    )

    display_model = platform_llm_name
    if reasoning_effort:
        display_model = f"{platform_llm_name} ({reasoning_effort})"

    node_data = {
        "id": agent_id,
        "name": name,
        "status": "initializing",
        "task": task,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": display_model,
    }

    agent_config = {
        "llm_config": llm_config.model_dump(),
        "state": agent_state.model_dump(mode="json"),
        "agent_hierarchy": agent_hierarchy,
    }

    job_config = {
        "job_id": job_id,
        "model": platform_llm_name,
        "api_key": effective_api_key,
        "reasoning_effort": reasoning_effort,
    }

    return {
        "agent_config": agent_config,
        "job_config": job_config,
        "node_data": node_data,
    }, agent_state


def register_agent_in_graph(
    agent_state: AgentContext,
    node_data: dict[str, Any],
    job_id: str | None = None,
) -> None:
    agent_id = agent_state.agent_id
    parent_id = agent_state.parent_id

    redis_manager.add_agent_node(node_data)

    if parent_id is None:
        redis_manager.set_root_agent_id(agent_id)
    else:
        redis_manager.add_edge(
            from_id=parent_id,
            to_id=agent_id,
            edge_type="delegation",
        )

    if job_id:
        redis_manager.publish_event(job_id, "graph_node_added", {"node": node_data})

    logger.info(f"Registered agent '{node_data.get('name')}' ({agent_id}) in graph")


def dispatch_agent_msg(
    target_agent_id: str,
    message: str,
    sender: str = "user",
    job_id: str | None = None,
) -> dict[str, Any]:
    agent_node = redis_manager.get_agent_node(target_agent_id)
    if not agent_node:
        return {"success": False, "error": f"Agent '{target_agent_id}' not found."}

    message_id = f"msg_{uuid.uuid4().hex[:8]}"
    message_data = {
        "id": message_id,
        "role": "user",
        "from": sender,
        "content": message,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    redis_manager.add_message_to_queue(target_agent_id, message_data)

    if job_id:
        redis_manager.publish_event(
            job_id,
            "new_message",
            {
                "agent_id": target_agent_id,
                "sender": sender,
                "content": message,
                "timestamp": message_data["timestamp"],
            },
        )

    logger.info(f"Sent message to agent {target_agent_id}")
    return {"success": True, "message_id": message_id}

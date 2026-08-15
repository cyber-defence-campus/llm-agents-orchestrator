import logging
from typing import Any, Callable

from agent_framework.services import agent_service

logger = logging.getLogger("agent_framework.services.agent_spawner")
_agent_starter: Callable[[dict, dict], tuple[str, Any]] | None = None
_routing_func: Callable[[str, str, str | None], tuple[str, str]] | None = None


def set_agent_starter(starter_func: Callable[[dict, dict], tuple[str, Any]]) -> None:
    global _agent_starter
    _agent_starter = starter_func
    logger.info("Agent spawner initialized with starter function")


def set_routing_function(
    routing_func: Callable[[str, str, str | None], tuple[str, str]] | None
) -> None:
    global _routing_func
    _routing_func = routing_func
    if routing_func:
        logger.info("Routing function registered")
    else:
        logger.info("Routing function cleared")


async def spawn_agent(
    parent_state: Any,
    name: str,
    task: str,
    short_task: str | None = None,
    prompt_modules: list[str] | None = None,
    model: str | None = None,
    inherit_context: bool = True,
) -> dict[str, Any]:
    if _agent_starter is None:
        return {
            "success": False,
            "error": "Agent spawner not initialized. Running in standalone mode without spawner.",
        }

    try:
        job_id = None
        inherited_model = None
        inherited_api_key = None
        inherited_reasoning_effort = None
        provider = None
        routing_mode = False

        if hasattr(parent_state, "sandbox_info") and parent_state.sandbox_info:
            job_id = parent_state.sandbox_info.get("job_id")
            inherited_model = parent_state.sandbox_info.get("model")
            inherited_api_key = parent_state.sandbox_info.get("api_key")
            inherited_reasoning_effort = parent_state.sandbox_info.get("reasoning_effort")
            provider = parent_state.sandbox_info.get("provider")
            routing_mode = parent_state.sandbox_info.get("routing_mode", False)

        effective_model = model or inherited_model
        effective_reasoning_effort = inherited_reasoning_effort

        if not model and provider and _routing_func and routing_mode:
            try:
                effective_model, effective_reasoning_effort = await _routing_func(
                    provider, task, inherited_api_key
                )
                logger.info(
                    f"Routing: Assigned model '{effective_model}' ({effective_reasoning_effort}) for '{name}'"
                )
            except Exception as e:
                logger.error(f"Routing failed: {e}. Falling back to inherited model.")
                effective_model = model or inherited_model
                effective_reasoning_effort = inherited_reasoning_effort

        context = None
        if inherit_context and job_id:
            hierarchy = agent_service.get_agent_hierarchy(job_id)
            if hierarchy:
                context = "Current Agent Hierarchy:\n"
                context += agent_service.format_agent_hierarchy(hierarchy)

        sandbox_info = {"job_id": job_id} if job_id else {}
        if inherited_model:
            sandbox_info["model"] = inherited_model
        if inherited_api_key:
            sandbox_info["api_key"] = inherited_api_key
        if effective_reasoning_effort:
            sandbox_info["reasoning_effort"] = effective_reasoning_effort
        if provider:
            sandbox_info["provider"] = provider
        sandbox_info["routing_mode"] = routing_mode

        config_result, agent_state = agent_service.create_agent_config(
            name=name,
            task=task,
            job_id=job_id,
            parent_id=parent_state.agent_id,
            prompt_modules=prompt_modules,
            model=effective_model,
            context=context,
            api_key=inherited_api_key,
            reasoning_effort=effective_reasoning_effort,
        )

        agent_state.sandbox_info = sandbox_info

        agent_service.register_agent_in_graph(
            agent_state,
            config_result["node_data"],
            job_id,
        )

        agent_id, _ = _agent_starter(
            config_result["agent_config"],
            config_result["job_config"],
        )

        logger.info(
            f"Successfully spawned sub-agent '{name}' ({agent_id}) from parent {parent_state.agent_id}"
        )

        return {
            "success": True,
            "agent_id": agent_id,
            "message": f"Sub-agent '{name}' created and started.",
            "hint": "You can call enter_wait_mode to wait for this agent to complete.",
        }

    except Exception as e:
        logger.exception(f"Failed to spawn agent '{name}': {e}")
        return {
            "success": False,
            "error": f"Failed to create agent: {str(e)}",
        }


def is_spawner_available() -> bool:
    return _agent_starter is not None


def is_routing_available() -> bool:
    return _routing_func is not None

"""
Agent Spawner - Internal module for spawning agents from within the framework.

This module provides a way for agents to spawn sub-agents without HTTP calls
when running in standalone mode.
"""

import logging
from typing import Any, Callable

from agent_framework.services import agent_service

logger = logging.getLogger("agent_framework.services.agent_spawner")
_agent_starter: Callable[[dict, dict], tuple[str, Any]] | None = None
_routing_func: Callable[[str, str, str | None], tuple[str, str]] | None = None


def set_agent_starter(starter_func: Callable[[dict, dict], tuple[str, Any]]) -> None:
    """
    Register the agent starter function from main.py.
    Called during application startup.
    """
    global _agent_starter
    _agent_starter = starter_func
    logger.info("Agent spawner initialized with starter function")


def set_routing_function(
    routing_func: Callable[[str, str, str | None], tuple[str, str]] | None
) -> None:
    """
    Register an optional routing function for model selection.
    
    The routing function should accept (provider: str, task: str, api_key: str | None)
    and return (model: str, reasoning_effort: str).
    
    This allows domain-specific layers to inject model selection logic without
    the orchestrator knowing about domain specifics.
    """
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
    """
    Spawn a new sub-agent from within an existing agent.

    Args:
        parent_state: The parent agent's state object
        name: Name for the new agent
        task: Task description for the new agent
        prompt_modules: Optional list of prompt modules
        model: Optional model override
        inherit_context: Whether to inherit context from parent

    Returns:
        dict with success status and agent_id or error
    """
    if _agent_starter is None:
        return {
            "success": False,
            "error": "Agent spawner not initialized. Running in standalone mode without spawner.",
        }

    try:
        # Get job_id and settings from parent's sandbox_info
        job_id = None
        inherited_model = None
        inherited_api_key = None
        inherited_reasoning_effort = None
        provider = None

        if hasattr(parent_state, "sandbox_info") and parent_state.sandbox_info:
            job_id = parent_state.sandbox_info.get("job_id")
            inherited_model = parent_state.sandbox_info.get("model")
            inherited_api_key = parent_state.sandbox_info.get("api_key")
            inherited_reasoning_effort = parent_state.sandbox_info.get("reasoning_effort")
            provider = parent_state.sandbox_info.get("provider")

        # Determine effective model
        effective_model = model or inherited_model
        effective_reasoning_effort = inherited_reasoning_effort

        # Apply routing if available and no explicit model override
        if not model and provider and _routing_func:
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

        # Build context if inheriting
        context = None
        if inherit_context and job_id:
            hierarchy = agent_service.get_agent_hierarchy(job_id)
            if hierarchy:
                context = "Current Agent Hierarchy:\n"
                context += agent_service.format_agent_hierarchy(hierarchy)

        # Build sandbox_info for nested spawns
        sandbox_info = {"job_id": job_id} if job_id else {}
        if inherited_model:
            sandbox_info["model"] = inherited_model
        if inherited_api_key:
            sandbox_info["api_key"] = inherited_api_key
        if effective_reasoning_effort:
            sandbox_info["reasoning_effort"] = effective_reasoning_effort
        if provider:
            sandbox_info["provider"] = provider

        # Create agent config
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
        
        # Update the agent_state's sandbox_info
        agent_state.sandbox_info = sandbox_info

        # Register in graph
        agent_service.register_agent_in_graph(
            agent_state,
            config_result["node_data"],
            job_id,
        )

        # Start the agent using the registered starter
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
    """Check if the agent spawner is available."""
    return _agent_starter is not None


def is_routing_available() -> bool:
    """Check if the routing function is available."""
    return _routing_func is not None

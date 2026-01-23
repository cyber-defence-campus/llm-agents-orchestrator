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


def set_agent_starter(starter_func: Callable[[dict, dict], tuple[str, Any]]) -> None:
    """
    Register the agent starter function from main.py.
    Called during application startup.
    """
    global _agent_starter
    _agent_starter = starter_func
    logger.info("Agent spawner initialized with starter function")


def spawn_agent(
    parent_state: Any,
    name: str,
    task: str,
    short_task: str
    | None = None,  # TODO: In the future, short task is better than task to show in CLI/UI.
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

        if hasattr(parent_state, "sandbox_info") and parent_state.sandbox_info:
            job_id = parent_state.sandbox_info.get("job_id")
            # Inherit model configuration from parent (which comes from job_config)
            inherited_model = parent_state.sandbox_info.get("model")
            inherited_api_key = parent_state.sandbox_info.get("api_key")
            inherited_reasoning_effort = parent_state.sandbox_info.get(
                "reasoning_effort"
            )

        # Use explicitly provided model, or inherit from parent
        effective_model = model or inherited_model

        # Build context if inheriting
        context = None
        if inherit_context and job_id:
            hierarchy = agent_service.get_agent_hierarchy(job_id)
            if hierarchy:
                context = "Current Agent Hierarchy:\n"
                context += agent_service.format_agent_hierarchy(hierarchy)

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
            reasoning_effort=inherited_reasoning_effort,
        )

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

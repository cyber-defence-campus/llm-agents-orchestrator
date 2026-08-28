import logging
from typing import Any, Callable

from agent_framework.services import agent_service
from agent_framework.tools.registry import get_tool_names

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

        # `capabilities` historically doubled as a prompt-module argument.
        # INTENTS also uses it to name the tools a child may use. Keep the two
        # namespaces separate: passing a tool name as a prompt module produced
        # a warning, while omitting the actual context allowlist exposed every
        # globally registered tool to the child.
        tool_names = set(get_tool_names())
        requested = list(prompt_modules or ())
        requested_tools = [name for name in requested if name in tool_names]
        requested_prompts = [name for name in requested if name not in tool_names]

        # A child has no implicit foothold or beacon binding. Do not inherit
        # the parent's target-facing tools just because it shares history:
        # that would make a fresh child appear to be on the compromised host.
        # The parent must explicitly grant a tool, and the child always keeps
        # only the coordination primitives needed to report back. Autonomous
        # INTENTS campaigns opt out of the generic wait primitive: no beacon
        # or target failure should park a run indefinitely.
        parent_context = getattr(parent_state, "context_data", None) or {}
        autonomous_no_wait = bool(parent_context.get("autonomous_no_wait"))
        coordination_tools = [
            name for name in (
                "complete_assignment", "dispatch_agent_msg", "enter_wait_mode"
            ) if name in tool_names
            and not (name == "enter_wait_mode" and autonomous_no_wait)
        ]

        context = None
        if inherit_context and job_id:
            hierarchy = agent_service.get_agent_hierarchy(job_id)
            if hierarchy:
                context = "Current Agent Hierarchy:\n"
                context += agent_service.format_agent_hierarchy(hierarchy)

        child_task = task
        if context:
            child_task = f"{context}\n\nYour assigned task is as follows:\n{task}"

        child_tools = coordination_tools + requested_tools
        # Legacy/general TACTICS jobs do not carry a target capability
        # allowlist: their operator surface is the terminal itself. Preserve
        # that one explicit operator capability for children, while keeping
        # INTENTS's allowlist strict once a beacon contract exists. This avoids
        # a child that is told to scan or SSH but can only send coordination
        # messages.
        if "capabilities" not in parent_context and "run_shell_command" in tool_names:
            child_tools.append("run_shell_command")
        child_context = {"capabilities": list(dict.fromkeys(child_tools))}
        if autonomous_no_wait:
            child_context["autonomous_no_wait"] = True

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
            task=child_task,
            job_id=job_id,
            parent_id=parent_state.agent_id,
            prompt_modules=requested_prompts,
            model=effective_model,
            context=child_context,
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
            "hint": (
                "The child will notify you via inter_agent_message; continue "
                "with other evidence and do not park this run."
                if autonomous_no_wait else
                "You can call enter_wait_mode to wait for this agent to complete."
            ),
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

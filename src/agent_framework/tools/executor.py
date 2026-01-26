import asyncio
import inspect
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any
from agent_framework.utils.id_utils import generate_ulid

import httpx
from pydantic import ValidationError

from agent_framework.state import redis_manager as state_manager
from .registry import (
    get_tool_by_name,
    get_tool_names,
    needs_agent_state,
    should_execute_in_sandbox,
)
from .argument_parser import convert_arguments, ArgumentConversionError


# AGENT_SANDBOX_MODE: whether sandbox tools are enabled (for prompts/registration)
# AGENT_IS_SANDBOX_RUNTIME: whether we're running INSIDE the sandbox container
#   - Agent-manager with sandbox: SANDBOX_MODE=true, IS_SANDBOX_RUNTIME=false -> delegate
#   - Sandbox container (tool_server): SANDBOX_MODE=true, IS_SANDBOX_RUNTIME=true -> local
sandbox_mode_enabled = os.getenv("AGENT_SANDBOX_MODE", "false").lower() == "true"
is_sandbox_runtime = os.getenv("AGENT_IS_SANDBOX_RUNTIME", "false").lower() == "true"

logger = logging.getLogger("agent_framework.tools")


async def execute_tool(
    tool_name: str, agent_state: Any | None = None, **kwargs: Any
) -> Any:
    """
    Executes a tool.
    - If the tool is non-sandboxed, it's executed locally.
    - If the tool is sandboxed, routes to sandbox-runtime (directly or via Core API).
    - If the tool is an orchestration tool (like create_agent), tries local spawner first.
    """
    is_orchestration_tool = tool_name in ["spawn_sub_agent"]
    is_sandboxed = should_execute_in_sandbox(tool_name)

    agent_id_log = agent_state.agent_id if agent_state else "N/A"
    logger.info(
        f"Executing tool '{tool_name}' for agent {agent_id_log}. "
        f"Sandbox mode: {sandbox_mode_enabled}, Sandboxed: {is_sandboxed}, Orchestration: {is_orchestration_tool}"
    )

    # If running inside sandbox CONTAINER (tool_server), execute locally
    if is_sandbox_runtime:
        return await _execute_tool_locally(tool_name, agent_state, **kwargs)

    # Handle orchestration tools (like spawn_sub_agent)
    if is_orchestration_tool:
        result = await _try_local_orchestration(tool_name, agent_state, **kwargs)
        if result is not None:
            return result
        # Fall through to Core API delegation if local fails

    # Handle sandboxed tools
    if is_sandboxed:
        result = await _try_direct_sandbox_execution(tool_name, agent_state, **kwargs)
        if result is not None:
            logger.info(f"Direct sandbox execution of '{tool_name}' succeeded")
            return result
        logger.info(
            f"Direct sandbox execution of '{tool_name}' returned None, falling through to Core API"
        )
        # Fall through to Core API delegation if direct fails

    # If tool needs remote execution but local/direct failed, try Core API
    if is_sandboxed or is_orchestration_tool:
        core_api_url = os.getenv("CORE_API_URL")
        if core_api_url:
            logger.info(f"Delegating '{tool_name}' to Core API at {core_api_url}")
            return await _delegate_tool_to_core_api(tool_name, agent_state, **kwargs)
        else:
            # Diagnostics for the error message
            sandbox_url = os.getenv("AGENT_SANDBOX_URL")
            job_id_status = "missing"
            if (
                agent_state
                and hasattr(agent_state, "sandbox_info")
                and agent_state.sandbox_info
            ):
                if agent_state.sandbox_info.get("job_id"):
                    job_id_status = "present"

            error_details = f"AGENT_SANDBOX_URL='{sandbox_url}', job_id={job_id_status}"

            logger.error(
                f"No execution path available for sandboxed tool '{tool_name}'. Details: {error_details}"
            )
            return {
                "error": f"Tool '{tool_name}' requires remote execution but no execution path available. "
                f"Set AGENT_SANDBOX_URL for sandbox tools or configure external API delegation. (Debug: {error_details})"
            }

    # Non-sandboxed, non-orchestration tools execute locally
    return await _execute_tool_locally(tool_name, agent_state, **kwargs)


async def _try_local_orchestration(
    tool_name: str, agent_state: Any, **kwargs: Any
) -> Any | None:
    """
    Try to execute orchestration tools locally using the agent spawner.
    Returns None if local execution is not available.
    """
    if tool_name != "spawn_sub_agent":
        return None

    try:
        from agent_framework.services.agent_spawner import (
            spawn_agent,
            is_spawner_available,
        )

        if is_spawner_available():
            logger.info(f"Executing '{tool_name}' via local agent spawner")

            # Parse capabilities
            prompt_modules = kwargs.get("capabilities")
            if isinstance(prompt_modules, str):
                prompt_modules = [
                    m.strip() for m in prompt_modules.split(",") if m.strip()
                ]

            # Extract IDs for event publishing
            tool_call_id = kwargs.get("tool_call_id")
            tool_result_id = kwargs.get("tool_result_id")

            if agent_state:
                _publish_tool_event(
                    agent_state.sandbox_info.get("job_id"),
                    "tool_call",
                    agent_state.agent_id,
                    tool_name,
                    kwargs,
                    id=tool_call_id,
                )

            try:
                result = spawn_agent(
                    parent_state=agent_state,
                    name=kwargs.get("agent_name", "Sub-Agent"),
                    task=kwargs.get("task_description", ""),
                    short_task=kwargs.get("ui_summary"),
                    prompt_modules=prompt_modules,
                    model=kwargs.get("model_override"),
                    inherit_context=kwargs.get("share_history", True),
                )

                if agent_state:
                    _publish_tool_event(
                        agent_state.sandbox_info.get("job_id"),
                        "tool_result",
                        agent_state.agent_id,
                        tool_name,
                        kwargs,
                        result=result,
                        id=tool_result_id,
                    )
                return result
            except Exception as e:
                error_msg = str(e)
                if agent_state:
                    _publish_tool_event(
                        agent_state.sandbox_info.get("job_id"),
                        "tool_error",
                        agent_state.agent_id,
                        tool_name,
                        kwargs,
                        error=error_msg,
                        id=tool_result_id,
                    )
                raise e
    except ImportError:
        pass

    return None


async def _try_direct_sandbox_execution(
    tool_name: str, agent_state: Any, **kwargs: Any
) -> Any | None:
    """
    Try to execute sandboxed tools directly via sandbox-runtime.
    Returns None if direct execution is not available.
    """
    sandbox_url = os.getenv("AGENT_SANDBOX_URL")
    if not sandbox_url:
        logger.info(
            f"Sandbox URL not configured, skipping direct sandbox for '{tool_name}'"
        )
        return None

    if (
        not agent_state
        or not hasattr(agent_state, "sandbox_info")
        or not agent_state.sandbox_info
    ):
        logger.warning(
            f"Agent state missing sandbox_info for direct sandbox execution of '{tool_name}'"
        )
        return None

    job_id = agent_state.sandbox_info.get("job_id")
    if not job_id:
        logger.warning(
            f"No job_id in sandbox_info for direct sandbox execution of '{tool_name}'"
        )
        return None

    try:
        from agent_framework.services.sandbox_client import sandbox_client

        if not sandbox_client.is_available:
            logger.info(
                f"Sandbox client not available (base_url='{sandbox_client.base_url}'), skipping direct sandbox for '{tool_name}'"
            )
            return None

        logger.info(
            f"Executing '{tool_name}' via direct sandbox connection (job_id={job_id})"
        )

        # Remove IDs from kwargs
        remote_kwargs = kwargs.copy()
        remote_kwargs.pop("tool_call_id", None)
        remote_kwargs.pop("tool_result_id", None)

        if agent_state:
            _publish_tool_event(
                job_id,
                "tool_call",
                agent_state.agent_id,
                tool_name,
                kwargs,
                id=kwargs.get("tool_call_id"),
            )

        result = await sandbox_client.execute_tool(
            session_id=job_id,
            agent_id=agent_state.agent_id,
            tool_name=tool_name,
            kwargs=remote_kwargs,
        )

        if agent_state:
            # Check for error in result dict
            is_error = False
            error_msg = None
            if isinstance(result, dict) and result.get("error"):
                is_error = True
                error_msg = result["error"]

            if is_error:
                _publish_tool_event(
                    job_id,
                    "tool_error",
                    agent_state.agent_id,
                    tool_name,
                    kwargs,
                    error=error_msg,
                    id=kwargs.get("tool_result_id"),
                )
            else:
                _publish_tool_event(
                    job_id,
                    "tool_result",
                    agent_state.agent_id,
                    tool_name,
                    kwargs,
                    result=result,
                    id=kwargs.get("tool_result_id"),
                )
        return result

    except ImportError:
        logger.warning(f"Could not import sandbox_client for '{tool_name}'")
        return None
    except Exception as e:
        logger.warning(f"Direct sandbox execution failed for '{tool_name}': {e}")
        return None


async def _delegate_tool_to_core_api(
    tool_name: str, agent_state: Any, **kwargs: Any
) -> Any:
    """
    Called by an Agent Worker to delegate tool execution to the Core API.
    """
    core_api_url = os.getenv("CORE_API_URL")
    if not core_api_url:
        raise RuntimeError("CORE_API_URL environment variable is not set for worker.")

    if (
        not agent_state
        or not hasattr(agent_state, "sandbox_info")
        or not agent_state.sandbox_info
    ):
        raise ValueError(
            f"Agent state missing sandbox_info for tool '{tool_name}'. State: {agent_state}"
        )

    job_id = agent_state.sandbox_info.get("job_id")
    if not job_id:
        raise ValueError(
            f"job_id not found in agent_state.sandbox_info for tool '{tool_name}'."
        )

    request_url = f"{core_api_url}/execute_tool"
    correlation_id = f"corr-{generate_ulid()[:12]}"

    # STRIP IDs from kwargs to prevent crashes in older sandboxes
    remote_kwargs = kwargs.copy()
    t_call_id = remote_kwargs.pop("tool_call_id", None)
    t_result_id = remote_kwargs.pop("tool_result_id", None)

    request_data = {
        "job_id": job_id,
        "agent_id": agent_state.agent_id,
        "tool_name": tool_name,
        "kwargs": remote_kwargs,
        "correlation_id": correlation_id,
        "tool_call_id": t_call_id,
        "tool_result_id": t_result_id,
    }
    headers = {"X-Correlation-ID": correlation_id}

    logger.info(
        f"--- DELEGATING TOOL TO CORE API (Agent: {agent_state.agent_id}, Tool: {tool_name}, CorrID: {correlation_id}) ---"
    )
    logger.info(f"URL: {request_url}")
    logger.debug(f"DELEGATION DATA: {request_data}")

    async with httpx.AsyncClient() as client:
        try:
            # Publish tool_call event BEFORE waiting for response
            _publish_tool_event(
                job_id,
                "tool_call",
                agent_state.agent_id,
                tool_name,
                kwargs,
                id=t_call_id,
            )

            response = await client.post(
                request_url, json=request_data, headers=headers, timeout=None
            )
            response.raise_for_status()
            response_data = response.json()

            logger.debug(f"Raw RESPONSE FROM CORE API: {response_data}")
            logger.info(
                f"--- END DELEGATING TOOL (Agent: {agent_state.agent_id}, Tool: {tool_name}) ---"
            )

            # Publish tool_result event
            _publish_tool_event(
                job_id,
                "tool_result",
                agent_state.agent_id,
                tool_name,
                kwargs,
                result=response_data,
                id=t_result_id,
            )

            return response_data

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            logger.exception(
                f"HTTP error calling Core API for tool '{tool_name}': {e.response.status_code} - {error_detail}"
            )
            error_data = {
                "error": f"Error executing tool '{tool_name}': HTTP error calling Core API: {e.response.status_code} - {error_detail}",
                "type": "HTTPError",
            }
            _publish_tool_event(
                job_id,
                "tool_error",
                agent_state.agent_id,
                tool_name,
                kwargs,
                error=error_data["error"],
                id=t_result_id,
            )
            return error_data
        except httpx.RequestError as e:
            logger.exception(
                f"Request error calling Core API for tool '{tool_name}': {e}"
            )
            error_data = {
                "error": f"Error executing tool '{tool_name}': Request error calling Core API: {e}",
                "type": "RequestError",
            }
            _publish_tool_event(
                job_id,
                "tool_error",
                agent_state.agent_id,
                tool_name,
                kwargs,
                error=error_data["error"],
                id=t_result_id,
            )
            return error_data
        except Exception as e:
            logger.exception(
                f"Unexpected error delegating tool '{tool_name}' to Core API: {e}"
            )
            error_data = {
                "error": f"Error executing tool '{tool_name}': Unexpected error during tool delegation: {e}",
                "type": "UnexpectedError",
            }
            _publish_tool_event(
                job_id,
                "tool_error",
                agent_state.agent_id,
                tool_name,
                kwargs,
                error=error_data["error"],
                id=t_result_id,
            )
            return error_data


def _publish_tool_event(
    job_id: str | None,
    event_type: str,
    agent_id: str,
    tool_name: str,
    kwargs: dict[str, Any],
    result: Any | None = None,
    error: str | None = None,
    id: str | None = None,
) -> None:
    """Helper to publish tool-related events to Redis safely."""
    if is_sandbox_runtime or not job_id:
        return

    payload = {
        "agent_id": agent_id,
        "tool_name": tool_name,
        "kwargs": kwargs,
    }
    if id:
        payload["id"] = id
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error

    try:
        state_manager.publish_event(job_id, event_type, payload)
    except Exception as e:
        logger.error(f"Failed to publish {event_type} event: {e}")


async def _execute_tool_locally(
    tool_name: str, agent_state: Any | None, **kwargs: Any
) -> Any:
    # Executes a tool directly in the current process (Worker or Sandbox).
    agent_id = agent_state.agent_id if agent_state else "sandbox"

    # Extract IDs if present, ensuring they don't get passed to the tool
    tool_call_id = kwargs.pop("tool_call_id", None)
    tool_result_id = kwargs.pop("tool_result_id", None)

    logger.info(f"--- TOOL EXECUTION START (Agent: {agent_id}, Tool: {tool_name}) ---")
    logger.info(f"ARGS: {kwargs}")

    if agent_state:
        _publish_tool_event(
            agent_state.sandbox_info.get("job_id"),
            "tool_call",
            agent_id,
            tool_name,
            kwargs,
            id=tool_call_id,
        )

    tool_func = get_tool_by_name(tool_name)
    if not tool_func:
        error_msg = f"Tool '{tool_name}' not found locally."
        logger.error(error_msg)
        if agent_state:
            _publish_tool_event(
                agent_state.sandbox_info.get("job_id"),
                "tool_error",
                agent_id,
                tool_name,
                kwargs,
                error=error_msg,
                id=tool_result_id,
            )
        return {
            "error": f"Error executing tool '{tool_name}': Tool not found locally.",
            "type": "NotFoundError",
        }

    try:
        converted_kwargs = convert_arguments(tool_func, kwargs)
        logger.debug(f"Converted Kwargs for '{tool_name}': {converted_kwargs}")

        if tool_name == "add_action":
            sig = inspect.signature(tool_func)
            if "tool_name" in sig.parameters:
                converted_kwargs["tool_name"] = tool_name

    except (ArgumentConversionError, ValidationError, ValueError) as e:
        error_msg = f"Error executing tool '{tool_name}': Invalid arguments - {e}"
        logger.warning(f"Argument conversion failed for tool '{tool_name}': {e}")
        if agent_state:
            _publish_tool_event(
                agent_state.sandbox_info.get("job_id"),
                "tool_error",
                agent_id,
                tool_name,
                kwargs,
                error=str(e),
                id=tool_result_id,
            )
        return {"error": error_msg, "type": "ArgumentError"}

    try:
        if needs_agent_state(tool_name):
            if agent_state is None and not sandbox_mode_enabled:
                raise ValueError(
                    f"Tool '{tool_name}' requires agent_state but none was provided."
                )
            result = tool_func(agent_state=agent_state, **converted_kwargs)
        else:
            filtered_kwargs = {
                k: v for k, v in converted_kwargs.items() if k != "agent_state"
            }
            if inspect.iscoroutinefunction(tool_func):
                result = tool_func(**filtered_kwargs)
            else:
                result = asyncio.to_thread(tool_func, **filtered_kwargs)

        final_result = await result if inspect.isawaitable(result) else result

        result_for_log = str(final_result)
        if len(result_for_log) > 1000:
            result_for_log = result_for_log[:1000] + " ... (truncated)"
        logger.info(f"RESULT: {result_for_log}")
        logger.info(
            f"--- TOOL EXECUTION END (Agent: {agent_id}, Tool: {tool_name}) ---"
        )

        if agent_state:
            event_result = final_result
            if not isinstance(
                event_result, (dict, list, str, int, float, bool, type(None))
            ):
                event_result = str(event_result)

            _publish_tool_event(
                agent_state.sandbox_info.get("job_id"),
                "tool_result",
                agent_id,
                tool_name,
                kwargs,
                result=event_result,
                id=tool_result_id,
            )
        return final_result
    except Exception as e:
        logger.exception(f"Error during local execution of tool '{tool_name}'")
        error_str = str(e)
        if len(error_str) > 500:
            error_str = error_str[:497] + "..."

        if agent_state:
            _publish_tool_event(
                agent_state.sandbox_info.get("job_id"),
                "tool_error",
                agent_id,
                tool_name,
                kwargs,
                error=error_str,
                id=tool_result_id,
            )
        return {
            "error": f"Error executing tool '{tool_name}': {error_str}",
            "type": "ExecutionError",
        }


def validate_tool_availability(tool_name: str | None) -> tuple[bool, str]:
    # Checks if a tool name is valid and registered.
    if tool_name is None:
        return False, "Tool name is missing"

    if tool_name not in get_tool_names():
        return False, f"Tool '{tool_name}' is not available"

    return True, ""


async def execute_tool_with_validation(
    target_tool_name: str | None, agent_state: Any | None = None, **kwargs: Any
) -> Any:
    # Validates tool name before attempting execution.
    is_valid, error_msg = validate_tool_availability(target_tool_name)
    if not is_valid:
        return {"error": error_msg, "type": "ValidationError"}

    assert target_tool_name is not None

    return await execute_tool(target_tool_name, agent_state, **kwargs)


async def execute_tool_invocation(
    tool_inv: dict[str, Any], agent_state: Any | None = None
) -> Any:
    # Executes a tool invocation dictionary.
    tool_name = tool_inv.get("toolName")
    tool_args = tool_inv.get("args", {})

    return await execute_tool_with_validation(tool_name, agent_state, **tool_args)


def _check_error_result(result: Any) -> tuple[bool, Any]:
    # Checks if the result indicates an error.
    is_error = False
    error_payload: Any = None

    if (isinstance(result, dict) and result.get("error")) or (
        isinstance(result, str) and result.strip().lower().startswith("error:")
    ):
        is_error = True
        error_payload = result

    return is_error, error_payload


def extract_screenshot_from_result(result: Any) -> str | None:
    # Extracts base64 screenshot data if present in the result dict.
    if not isinstance(result, dict):
        return None
    screenshot = result.get("screenshot")
    if isinstance(screenshot, str) and screenshot:
        return screenshot
    return None


def remove_screenshot_from_result(result: Any) -> Any:
    # Removes or replaces screenshot data in the result dict.
    if not isinstance(result, dict):
        return result
    if "screenshot" not in result:
        return result

    result_copy = result.copy()
    result_copy["screenshot"] = "[Image data extracted - see attached image]"
    return result_copy


def _format_tool_result(
    tool_name: str, result: Any
) -> tuple[str, list[dict[str, Any]]]:
    # Formats the tool result into XML for the agent, extracting images.
    images: list[dict[str, Any]] = []
    result_for_llm: Any

    screenshot_data = extract_screenshot_from_result(result)
    if screenshot_data:
        images.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_data}"},
            }
        )
        result_for_llm = remove_screenshot_from_result(result)
    else:
        result_for_llm = result

    if isinstance(result_for_llm, dict):
        if "result" in result_for_llm:
            result_for_llm = result_for_llm["result"]

    if isinstance(result_for_llm, dict):
        if result_for_llm.get("agent_completed"):
            result_for_llm = result_for_llm.get(
                "message", "Agent task marked as completed."
            )
        elif result_for_llm.get("job_completed"):
            result_for_llm = result_for_llm.get("message", "Job marked as completed.")
        elif result_for_llm.get("error"):
            result_for_llm = f"Error: {result_for_llm['error']}"
        elif result_for_llm.get("agent_should_wait"):
            result_for_llm = result_for_llm.get("message", "Agent is waiting.")

    if result_for_llm is None:
        result_for_llm = (
            f"Tool '{tool_name}' executed successfully with no explicit return value."
        )

    try:
        final_result_str = json.dumps(result_for_llm, indent=None)
    except TypeError:
        final_result_str = json.dumps(str(result_for_llm), indent=None)

    max_len = 10000
    if len(final_result_str) > max_len:
        start_part = final_result_str[: max_len // 2 - 50]
        end_part = final_result_str[-max_len // 2 + 50 :]
        final_result_str = (
            start_part + "\n\n... [tool result truncated] ...\n\n" + end_part
        )

    observation_xml = (
        f"<tool_result>\n<tool_name>{tool_name}</tool_name>\n"
        f"<result>{final_result_str}</result>\n</tool_result>"
    )

    return observation_xml, images


async def process_tool_invocations(
    tool_invocations: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]],
    agent_state: Any | None = None,
) -> bool:
    logger.info(f"Processing {len(tool_invocations)} tool invocation(s) in parallel.")
    logger.debug(f"Raw tool_invocations: {tool_invocations}")

    observation_parts: list[str] = []
    all_images: list[dict[str, Any]] = []
    should_agent_finish_overall = False

    agent_id = agent_state.agent_id if agent_state else "unknown_agent"

    tasks = []
    for tool_inv in tool_invocations:
        tool_name = tool_inv.get("toolName", "unknown")
        tool_args = tool_inv.get("args", {})

        tool_call_id = f"call_{generate_ulid()}"
        tool_result_id = f"result_{generate_ulid()}"

        # Inject IDs into args so they flow down to execution
        tool_args["tool_call_id"] = tool_call_id
        tool_args["tool_result_id"] = tool_result_id

        tool_call_timestamp = datetime.now(UTC).isoformat()
        conversation_history.append(
            {
                "id": tool_call_id,
                "role": "tool_call",
                "content": {
                    "tool_name": tool_name,
                    "kwargs": tool_args,
                },
                "timestamp": tool_call_timestamp,
                "iteration": agent_state.iteration if agent_state else -1,
            }
        )
        logger.debug(
            f"Appended structured tool_call to history for '{tool_name}' (ID: {tool_call_id})"
        )

        tasks.append(execute_tool_invocation(tool_inv, agent_state))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        tool_inv = tool_invocations[i]
        tool_name = tool_inv.get("toolName", "unknown")

        if isinstance(result, Exception):
            logger.error(
                f"Critical error processing tool invocation for '{tool_name}': {result}",
                exc_info=result,
            )
            error_msg = f"Critical error during execution: {result}"
            result = {"error": error_msg, "type": "Crash"}

        logger.debug(
            f"Processing result for tool '{tool_name}': {str(result)[:500]}..."
        )

        is_error, error_payload = _check_error_result(result)

        tool_result_timestamp = datetime.now(UTC).isoformat()
        result_content_for_history = {
            "tool_name": tool_name,
            "isError": is_error,
        }

        if is_error:
            try:
                json.dumps(error_payload)
                result_content_for_history["error"] = error_payload
            except TypeError:
                result_content_for_history["error"] = str(error_payload)
        else:
            try:
                json.dumps(result)
                result_content_for_history["result"] = result
            except TypeError:
                result_content_for_history["result"] = str(result)

        conversation_history.append(
            {
                "id": tool_inv.get("args", {}).get(
                    "tool_result_id"
                ),  # Retrieve the ID we generated earlier
                "role": "tool_result",
                "content": result_content_for_history,
                "timestamp": tool_result_timestamp,
                "iteration": agent_state.iteration if agent_state else -1,
            }
        )
        logger.debug(f"Appended structured tool_result to history for '{tool_name}'")

        tool_signals_finish = False
        tool_signals_wait = False
        if isinstance(result, dict):
            if result.get("agent_completed") or result.get("job_completed"):
                tool_signals_finish = True
                should_agent_finish_overall = True
            elif result.get("agent_should_wait"):
                tool_signals_wait = True
                if agent_state:
                    logger.info(
                        f"Agent {agent_id} entering waiting state due to tool {tool_name}."
                    )
                    agent_state.set_waiting()

        observation_xml, images = _format_tool_result(tool_name, result)
        observation_parts.append(observation_xml)
        all_images.extend(images)

    if observation_parts:
        if all_images:
            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": "Tool Results:\n\n" + "\n\n".join(observation_parts),
                }
            ]
            content.extend(all_images)
            conversation_history.append({"role": "user", "content": content})
        else:
            observation_content = "Tool Results:\n\n" + "\n\n".join(observation_parts)
            conversation_history.append(
                {"role": "user", "content": observation_content}
            )
        logger.debug("Appended XML tool results to conversation history for LLM.")

    logger.info("Tool invocation processing finished.")
    return should_agent_finish_overall

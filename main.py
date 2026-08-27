import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

_script_dir = os.path.dirname(__file__)
_extensions_path = os.path.abspath(os.path.join(_script_dir, ".."))
if os.path.isdir(os.path.join(_extensions_path, "rt_automation_tactics")):
    sys.path.insert(0, _extensions_path)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent_framework.agents.default import DefaultAgent
from agent_framework.agents.state import AgentContext
from agent_framework.llm.config import LLMConfig
from agent_framework.state import redis_manager as state_manager
from agent_framework.services import agent_service
from agent_framework.services.agent_spawner import set_routing_function
from agent_framework.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

active_agents: dict[str, dict[str, Any]] = {}


class AgentCreationRequest(BaseModel):
    """Full agent creation request (used by external callers like Core API)."""

    agent_config: dict[str, Any]
    job_config: dict[str, Any]


class SimpleAgentRequest(BaseModel):
    """Simplified agent creation request for standalone usage."""

    name: str
    task: str
    job_id: str | None = None
    parent_id: str | None = None
    prompt_modules: list[str] | None = None
    model: str | None = None
    context: dict[str, Any] | None = None


class MessageRequest(BaseModel):
    """Request to send a message to an agent."""

    message: str
    sender: str = "user"


class CapabilitiesRequest(BaseModel):
    """Capabilities that are currently available to an agent."""

    capabilities: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Agent Manager Service starting up.")

    from agent_framework.services import agent_spawner

    agent_spawner.set_agent_starter(start_agent_task)

    try:
        from rt_automation_tactics.core_api.routing import get_model_and_reasoning_effort_for_task
        set_routing_function(get_model_and_reasoning_effort_for_task)
        logger.info("Registered model routing function")
    except ImportError:
        logger.debug("No routing extension available (running in standalone mode)")

    yield
    logger.info("Agent Manager Service shutting down.")
    for agent_id, agent_data in list(active_agents.items()):
        task = agent_data.get("task")
        if task and not task.done():
            task.cancel()
            logger.info(f"Cancelled task for agent {agent_id}")


app = FastAPI(
    title="Agent Manager Service",
    description="Manages the lifecycle of LLM agents. Can run standalone or be extended.",
    lifespan=lifespan,
)


async def run_agent(agent: DefaultAgent, agent_state: AgentContext):
    agent_id = agent_state.agent_id
    job_id = (
        agent_state.sandbox_info.get("job_id") if agent_state.sandbox_info else None
    )

    try:
        logger.info(f"Starting execution for agent {agent_id}")
        state_manager.update_agent_status(agent_id, "running")

        if agent_state.parent_id is None:
            await agent.run_job(agent_state.sandbox_info or {})
        else:
            await agent.start_lifecycle(primary_task=agent_state.task)

        logger.info(f"Agent {agent_id} finished execution successfully.")
        state_manager.update_agent_status(
            agent_id, "finished", "Agent completed its task."
        )

    except asyncio.CancelledError:
        logger.info(f"Agent {agent_id} was cancelled.")
        state_manager.update_agent_status(agent_id, "stopped", "Agent was stopped.")
        raise

    except Exception as e:
        logger.exception(f"CRITICAL ERROR during execution for agent {agent_id}: {e}")
        state_manager.update_agent_status(agent_id, "error", str(e))

    finally:
        if agent_id in active_agents:
            del active_agents[agent_id]
            logger.info(f"Removed agent {agent_id} from active list.")


def start_agent_task(
    agent_config: dict[str, Any],
    job_config: dict[str, Any],
) -> tuple[str, AgentContext]:
    llm_config = LLMConfig(**agent_config["llm_config"])

    state_params = agent_config["state"]
    agent_state = AgentContext(**state_params)
    agent_id = agent_state.agent_id

    if agent_id in active_agents:
        raise ValueError(f"Agent with ID {agent_id} is already running.")

    agent_hierarchy = agent_config.get("agent_hierarchy", [])
    config = {
        "llm_config": llm_config,
        "state": agent_state,
        "agent_hierarchy": agent_hierarchy,
    }

    logger.info(f"Creating agent {agent_id} (Parent: {agent_state.parent_id})")

    if agent_state.sandbox_info:
        agent_state.sandbox_info.update(job_config)
    else:
        agent_state.sandbox_info = job_config

    if not agent_state.sandbox_info.get("job_id"):
        import uuid

        generated_job_id = f"job_{uuid.uuid4().hex[:8]}"
        agent_state.sandbox_info["job_id"] = generated_job_id
        logger.info(
            f"Generated missing job_id for agent {agent_id}: {generated_job_id}"
        )

    agent = DefaultAgent(config)
    loop = asyncio.get_event_loop()
    task = loop.create_task(run_agent(agent, agent_state))

    active_agents[agent_id] = {"agent": agent, "task": task, "state": agent_state}

    return agent_id, agent_state


@app.post("/agents", status_code=202)
async def create_agent_endpoint(request: AgentCreationRequest):
    """
    Creates and starts a new agent (full config mode).
    Used by external callers like Core API.
    """
    try:
        agent_id, _ = start_agent_task(request.agent_config, request.job_config)
        return {"message": "Agent creation initiated.", "agent_id": agent_id}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@app.post("/agents/simple", status_code=202)
async def create_agent_simple(request: SimpleAgentRequest):
    """
    Creates and starts a new agent (simplified mode).
    Used for standalone operation without Core API.
    """
    try:
        config_result, agent_state = agent_service.create_agent_config(
            name=request.name,
            task=request.task,
            job_id=request.job_id,
            parent_id=request.parent_id,
            prompt_modules=request.prompt_modules,
            model=request.model,
            context=request.context,
        )

        agent_service.register_agent_in_graph(
            agent_state,
            config_result["node_data"],
            request.job_id,
        )

        agent_id, _ = start_agent_task(
            config_result["agent_config"],
            config_result["job_config"],
        )

        return {
            "message": "Agent creation initiated.",
            "agent_id": agent_id,
            "name": request.name,
        }

    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@app.post("/agents/{agent_id}/stop", status_code=200)
async def stop_agent(agent_id: str):
    """
    Forces a running agent to stop by cancelling its task.
    """
    if agent_id not in active_agents:
        status = state_manager.get_agent_status(agent_id)
        if not status:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"message": f"Agent {agent_id} is not running (Status: {status})."}

    agent_data = active_agents[agent_id]
    task = agent_data["task"]

    logger.info(f"Force stopping agent {agent_id}...")

    if not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.error(f"Error during forced stop of agent {agent_id}: {e}")

    if agent_id in active_agents:
        del active_agents[agent_id]
        logger.info(f"Agent {agent_id} removed from active list.")

    return {"message": f"Agent {agent_id} stopped successfully."}


@app.post("/agents/{agent_id}/capabilities", status_code=200)
async def update_agent_capabilities(
    agent_id: str, request: CapabilitiesRequest
):
    """Replace the tools exposed to a live agent.

    Some deployments have a staged foothold: an agent starts with a shell and
    must land a beacon before the typed capabilities become real. Updating the
    live LLM object as well as Redis keeps both native tool schemas and the
    persisted agent state on the same side of that handshake.
    """
    agent_data = active_agents.get(agent_id)
    if not agent_data:
        raise HTTPException(status_code=404, detail="Agent not running")

    capabilities = list(dict.fromkeys(request.capabilities))
    agent = agent_data["agent"]
    state = agent_data["state"]
    state.context_data["capabilities"] = capabilities
    state.touch()
    agent.llm.update_capabilities(capabilities)
    state_manager.add_agent_state(agent_id, state)
    return {"agent_id": agent_id, "capabilities": capabilities}


@app.delete("/agents/{agent_id}", status_code=200)
async def delete_agent(agent_id: str):
    """
    Stops the agent (if running) and deletes its data (history, state) from Redis.
    """
    if agent_id in active_agents:
        await stop_agent(agent_id)

    try:
        state_manager.delete_agent(agent_id)
        return {"message": f"Agent {agent_id} deleted successfully."}
    except Exception as e:
        logger.exception(f"Error deleting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {e}")


@app.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """
    Gets the status of a running agent.
    """
    if agent_id not in active_agents:
        status = state_manager.get_agent_status(agent_id)
        if status:
            return {"agent_id": agent_id, "status": status}
        raise HTTPException(status_code=404, detail="Agent not found")

    task = active_agents[agent_id]["task"]
    status = "running"
    if task.done():
        if task.cancelled():
            status = "cancelled"
        elif task.exception():
            status = "error"
        else:
            status = "finished"

    return {"agent_id": agent_id, "status": status}


@app.get("/agents/{agent_id}")
async def get_agent_details(agent_id: str):
    """
    Gets the full details (state) of an agent, including history.
    """
    if agent_id in active_agents:
        state = active_agents[agent_id].get("state")
        if state:
            return state.model_dump()

    state = state_manager.get_agent_state(agent_id)
    if state:
        return state.model_dump()

    raise HTTPException(status_code=404, detail="Agent not found")


@app.post("/agents/{agent_id}/message", status_code=200)
async def send_message(agent_id: str, request: MessageRequest):
    """
    Sends a message to an agent.
    """
    job_id = None
    if agent_id in active_agents:
        state = active_agents[agent_id].get("state")
        if state and state.sandbox_info:
            job_id = state.sandbox_info.get("job_id")

    result = agent_service.dispatch_agent_msg(
        target_agent_id=agent_id,
        message=request.message,
        sender=request.sender,
        job_id=job_id,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=404, detail=result.get("error", "Failed to send message")
        )

    return result


from fastapi import Query


@app.get("/agents")
async def list_agents(
    limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0)
):
    """
    Lists agents (active and historical) from the graph, paginated.
    """
    agents_list, total_count = state_manager.get_paginated_agent_nodes(
        limit=limit, offset=offset
    )

    results = []
    for node in agents_list:
        results.append(
            {
                "agent_id": node.get("id"),
                "status": node.get("status", "unknown"),
                "name": node.get("name", "Unknown"),
                "parent_id": node.get("parent_id"),
                "task": node.get("task", ""),
            }
        )

    return {
        "agents": results,
        "count": len(results),
        "total": total_count,
        "limit": limit,
        "offset": offset,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "active_agents": len(active_agents)}


class CompletionRequest(BaseModel):
    """One completion, with no agent wrapped around it."""

    prompt: str
    system: str | None = None
    model: str | None = None
    max_tokens: int = 4096
    reasoning_effort: str | None = None
    reasoning_max_tokens: int | None = None
    temperature: float | None = None


@app.post("/completions")
async def create_completion(request: CompletionRequest):
    """Ask the configured model one question and return its answer.

    The agent endpoints render a system prompt of tool definitions, an XML
    call protocol and wait-mode rules, which is right for an autonomous
    agent and wrong for a caller that wants a single structured document
    back. Asking for JSON through an agent makes the model choose between
    the two contracts, and a reasoning model can spend its whole budget on
    that choice and return empty content.

    finish_reason and the token counts come back with the content so a
    caller can distinguish "the model declined" from "the budget ran out
    during reasoning", which are the same empty string otherwise.
    """
    from agent_framework.llm.llm import complete

    config = LLMConfig(
        model_name=request.model,
        reasoning_effort=request.reasoning_effort,
        temperature=request.temperature,
    )
    try:
        return await complete(
            prompt=request.prompt,
            system=request.system,
            config=config,
            max_tokens=request.max_tokens,
            reasoning_max_tokens=request.reasoning_max_tokens,
        )
    except Exception as e:
        logger.exception("Completion failed")
        raise HTTPException(status_code=502, detail=str(e))


class ToolExecutionRequest(BaseModel):
    """Request to execute a tool."""

    job_id: str
    agent_id: str | None = (
        None  # Optional now as path param takes precedence in new endpoint
    )
    tool_name: str
    kwargs: dict[str, Any] = {}
    correlation_id: str | None = None


@app.post("/agents/{agent_id}/tool-executions")
async def execute_agent_tool(agent_id: str, request: ToolExecutionRequest):
    """
    Executes a tool for a specific agent.
    """
    from agent_framework.tools.registry import (
        should_execute_in_sandbox,
        get_tool_by_name,
    )
    from agent_framework.services.sandbox_client import sandbox_client
    from agent_framework.services import agent_spawner

    tool_name = request.tool_name
    job_id = request.job_id
    kwargs = request.kwargs
    correlation_id = request.correlation_id

    logger.info(f"Executing tool '{tool_name}' for agent {agent_id}")

    if tool_name == "spawn_sub_agent":
        parent_state = None
        if agent_id in active_agents:
            parent_state = active_agents[agent_id].get("state")

        if parent_state is None:
            from agent_framework.agents.state import AgentContext

            parent_state = AgentContext(
                agent_id=agent_id,
                task="",
                sandbox_info={"job_id": job_id},
            )

        if agent_spawner.is_spawner_available():
            result = await agent_spawner.spawn_agent(
                parent_state=parent_state,
                name=kwargs.get("agent_name", "Sub-Agent"),
                task=kwargs.get("task_description", ""),
                prompt_modules=kwargs.get("capabilities"),
                model=kwargs.get("model_override"),
                inherit_context=kwargs.get("share_history", True),
            )
            return {"result": result}
        else:
            return {"error": "Agent spawner not available"}

    if should_execute_in_sandbox(tool_name):
        if not sandbox_client.is_available:
            return {
                "error": f"AGENT_SANDBOX_URL not configured. Cannot execute '{tool_name}'."
            }

        result = await sandbox_client.execute_tool(
            session_id=job_id,
            agent_id=agent_id,
            tool_name=tool_name,
            kwargs=kwargs,
            correlation_id=correlation_id,
        )
        return result

    tool_func = get_tool_by_name(tool_name)
    if not tool_func:
        return {"error": f"Tool '{tool_name}' not found"}

    try:
        from agent_framework.tools.executor import _execute_tool_locally
        from agent_framework.agents.state import AgentContext

        agent_state = None
        if agent_id in active_agents:
            agent_state = active_agents[agent_id].get("state")

        if agent_state is None:
            agent_state = AgentContext(
                agent_id=agent_id,
                task="",
                sandbox_info={"job_id": job_id},
            )

        result = await _execute_tool_locally(tool_name, agent_state, **kwargs)
        return {"result": result}

    except Exception as e:
        logger.exception(f"Tool execution failed: {e}")
        return {"error": f"Tool execution failed: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("AGENT_MANAGER_PORT", "8083"))
    uvicorn.run(app, host="0.0.0.0", port=port)

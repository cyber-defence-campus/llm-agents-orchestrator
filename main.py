"""
Agent Manager Service - Standalone Agent Orchestrator

This service manages the lifecycle of LLM agents, including creation,
execution, and termination. It can run independently or be extended
by integration layers for specialized use cases.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent_framework.agents.default import DefaultAgent
from agent_framework.agents.state import AgentContext
from agent_framework.llm.config import LLMConfig
from agent_framework.state import redis_manager as state_manager
from agent_framework.services import agent_service
from agent_framework.utils.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# In-memory storage for active agents and their tasks
active_agents: dict[str, dict[str, Any]] = {}


# =============================================================================
# Request/Response Models
# =============================================================================


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


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Agent Manager Service starting up.")

    # Register the agent spawner for internal agent creation
    from agent_framework.services import agent_spawner

    agent_spawner.set_agent_starter(start_agent_task)

    yield
    logger.info("Agent Manager Service shutting down.")
    # Clean up any running agent tasks
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


# =============================================================================
# Agent Execution
# =============================================================================


async def run_agent(agent: DefaultAgent, agent_state: AgentContext):
    """
    Wrapper to run the agent's execution logic and handle completion/errors.
    """
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
        # Remove the agent from the active list once it's done
        if agent_id in active_agents:
            del active_agents[agent_id]
            logger.info(f"Removed agent {agent_id} from active list.")


def start_agent_task(
    agent_config: dict[str, Any],
    job_config: dict[str, Any],
) -> tuple[str, AgentContext]:
    """
    Internal function to start an agent. Can be called from within the framework.

    Returns:
        Tuple of (agent_id, agent_state)
    """
    llm_config = LLMConfig(**agent_config["llm_config"])

    # Combine state from agent config with job-level settings
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

    # Merge sandbox_info with job_config
    if agent_state.sandbox_info:
        agent_state.sandbox_info.update(job_config)
    else:
        agent_state.sandbox_info = job_config

    # Ensure job_id exists
    if not agent_state.sandbox_info.get("job_id"):
        import uuid

        generated_job_id = f"job_{uuid.uuid4().hex[:8]}"
        agent_state.sandbox_info["job_id"] = generated_job_id
        logger.info(
            f"Generated missing job_id for agent {agent_id}: {generated_job_id}"
        )

    agent = DefaultAgent(config)
    # Run the agent's main loop as a background task
    loop = asyncio.get_event_loop()
    task = loop.create_task(run_agent(agent, agent_state))

    # Store the agent and its task
    active_agents[agent_id] = {"agent": agent, "task": task, "state": agent_state}

    return agent_id, agent_state


# =============================================================================
# API Endpoints
# =============================================================================


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
        # Use agent service to build config
        config_result, agent_state = agent_service.create_agent_config(
            name=request.name,
            task=request.task,
            job_id=request.job_id,
            parent_id=request.parent_id,
            prompt_modules=request.prompt_modules,
            model=request.model,
            context=request.context,
        )

        # Register in graph
        agent_service.register_agent_in_graph(
            agent_state,
            config_result["node_data"],
            request.job_id,
        )

        # Start the agent
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
        # Check if it exists in state to avoid 404 if just not running
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


@app.delete("/agents/{agent_id}", status_code=200)
async def delete_agent(agent_id: str):
    """
    Stops the agent (if running) and deletes its data (history, state) from Redis.
    """
    # 1. Stop if running
    if agent_id in active_agents:
        await stop_agent(agent_id)

    # 2. Delete data
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
        # If not in memory, check Redis for a final status
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
    # 1. Try active agents (most fresh)
    if agent_id in active_agents:
        state = active_agents[agent_id].get("state")
        if state:
            return state.model_dump()

    # 2. Try Redis (persisted state)
    state = state_manager.get_agent_state(agent_id)
    if state:
        return state.model_dump()

    raise HTTPException(status_code=404, detail="Agent not found")


@app.post("/agents/{agent_id}/message", status_code=200)
async def send_message(agent_id: str, request: MessageRequest):
    """
    Sends a message to an agent.
    """
    # Get job_id from active agent or Redis
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


# =============================================================================
# Tool Execution
# =============================================================================


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

    # Handle orchestration tools locally
    if tool_name == "spawn_sub_agent":
        # Get parent agent state from active agents
        parent_state = None
        if agent_id in active_agents:
            parent_state = active_agents[agent_id].get("state")

        if parent_state is None:
            # Try to reconstruct minimal state
            from agent_framework.agents.state import AgentContext

            parent_state = AgentContext(
                agent_id=agent_id,
                task="",
                sandbox_info={"job_id": job_id},
            )

        if agent_spawner.is_spawner_available():
            result = agent_spawner.spawn_agent(
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

    # Route sandboxed tools to sandbox-runtime
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

    # Execute non-sandboxed tools locally
    tool_func = get_tool_by_name(tool_name)
    if not tool_func:
        return {"error": f"Tool '{tool_name}' not found"}

    try:
        from agent_framework.tools.executor import _execute_tool_locally
        from agent_framework.agents.state import AgentContext

        # Create minimal agent state for context
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


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("AGENT_MANAGER_PORT", "8083"))
    uvicorn.run(app, host="0.0.0.0", port=port)

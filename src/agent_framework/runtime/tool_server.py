from __future__ import annotations

import argparse
import asyncio
import contextvars
import logging
import os
import sys
import uuid
import multiprocessing as mp
from typing import Any, Dict, Optional, Tuple, List

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# Context Management
correlation_id_ctx = contextvars.ContextVar("correlation_id", default=None)


class ContextAwareLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        cid = correlation_id_ctx.get()
        if cid:
            kwargs.setdefault("extra", {})["correlation_id"] = cid
        return msg, kwargs


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s",
)
logger = ContextAwareLogger(logging.getLogger("tool_server"), {})

# Environment checks
IS_SANDBOX = os.getenv("AGENT_SANDBOX_MODE", "false").lower() == "true"
if not IS_SANDBOX:
    raise SystemError("Tool Server must run in SANDBOX_MODE=true configuration.")

os.environ["AGENT_IS_SANDBOX_RUNTIME"] = "true"

# Argument Parsing
parser = argparse.ArgumentParser(description="Agent Tool Execution Server")
parser.add_argument("--token", required=True, help="Auth Bearer Token")
parser.add_argument("--host", default="0.0.0.0", help="Bind Host")
parser.add_argument("--port", type=int, required=True, help="Bind Port")
if "pytest" not in sys.modules and not os.getenv("TESTING"):  # Hack for tests
    SERVER_ARGS = parser.parse_args()
else:

    class MockArgs:
        token = "test"
        host = "localhost"
        port = 8080

    SERVER_ARGS = MockArgs()

AUTH_TOKEN = SERVER_ARGS.token

# App Setup
app = FastAPI(title="Agent Tool Server")
security_scheme = HTTPBearer()


# Middleware
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    cid = request.headers.get("X-Correlation-ID") or f"req-{uuid.uuid4().hex[:8]}"
    token = correlation_id_ctx.set(cid)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    correlation_id_ctx.reset(token)
    return response


# Dependency Injection
def validate_auth(
    creds: HTTPAuthorizationCredentials = Security(security_scheme),
) -> str:
    if creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer token required")
    if creds.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return creds.credentials


# Models
class ExecutionRequest(BaseModel):
    agent_id: str
    tool_name: str
    kwargs: Dict[str, Any]


class ExecutionResponse(BaseModel):
    result: Optional[Any] = None
    error: Optional[str] = None


# Process Management
active_workers: Dict[str, Dict[str, Any]] = {}


def _worker_process(agent_id: str, inbox: mp.Queue, outbox: mp.Queue):
    """Isolated process running the agent's tool execution logic."""
    # Reset logging handlers for child process
    logging.getLogger().handlers = []

    # Import locally to avoid side effects in parent
    from agent_framework.tools.registry import get_tool_by_name
    from agent_framework.tools.argument_parser import (
        convert_arguments,
    )
    import agent_framework.tools  # Trigger registration

    logger.info(f"Worker process started for {agent_id}")

    while True:
        try:
            req = inbox.get()
            if req is None:  # Poison pill
                break

            tool_name = req["tool_name"]
            raw_kwargs = req["kwargs"]

            try:
                tool_fn = get_tool_by_name(tool_name)
                if not tool_fn:
                    raise ValueError(f"Unknown tool: {tool_name}")

                # Convert and Validate
                kwargs = convert_arguments(tool_fn, raw_kwargs)

                # Execute
                if asyncio.iscoroutinefunction(tool_fn):
                    # We are in a sync process, need new loop
                    res = asyncio.run(tool_fn(**kwargs))
                else:
                    res = tool_fn(**kwargs)

                outbox.put({"result": res})

            except Exception as e:
                logger.error(f"Execution failed: {e}")
                outbox.put({"error": str(e)})

        except Exception as e:
            logger.critical(f"Worker loop crash: {e}")
            outbox.put({"error": "Worker crash"})


def get_or_create_worker(agent_id: str) -> Tuple[mp.Queue, mp.Queue]:
    if agent_id not in active_workers:
        inbox = mp.Queue()
        outbox = mp.Queue()

        proc = mp.Process(
            target=_worker_process,
            args=(agent_id, inbox, outbox),
            daemon=True,
            name=f"worker-{agent_id}",
        )
        proc.start()
        active_workers[agent_id] = {"proc": proc, "inbox": inbox, "outbox": outbox}

    return active_workers[agent_id]["inbox"], active_workers[agent_id]["outbox"]


# Endpoints
@app.post("/execute", response_model=ExecutionResponse)
async def execute_endpoint(
    payload: ExecutionRequest, _auth: str = Depends(validate_auth)
):
    inbox, outbox = get_or_create_worker(payload.agent_id)

    # Send request
    await asyncio.to_thread(
        inbox.put, {"tool_name": payload.tool_name, "kwargs": payload.kwargs}
    )

    # Wait for response
    # Using run_in_executor to avoid blocking event loop
    resp = await asyncio.get_running_loop().run_in_executor(None, outbox.get)

    if "error" in resp:
        return ExecutionResponse(error=resp["error"])
    return ExecutionResponse(result=resp["result"])


@app.post("/register_agent")
async def register_agent_endpoint(agent_id: str, _auth: str = Depends(validate_auth)):
    get_or_create_worker(agent_id)
    return {"status": "registered", "agent_id": agent_id}


@app.get("/health")
async def health_endpoint(_auth: str = Depends(validate_auth)):
    return {"status": "healthy", "workers": len(active_workers), "mode": "sandbox"}


# Lifecycle
@app.on_event("shutdown")
def shutdown_workers():
    logger.info("Shutting down workers...")
    for aid, meta in active_workers.items():
        try:
            meta["inbox"].put(None)
            meta["proc"].join(timeout=1)
            if meta["proc"].is_alive():
                meta["proc"].terminate()
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_ARGS.host, port=SERVER_ARGS.port)

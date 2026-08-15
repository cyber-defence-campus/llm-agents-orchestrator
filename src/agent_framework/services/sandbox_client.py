import logging
import os
import uuid
from typing import Any

import httpx

logger = logging.getLogger("agent_framework.services.sandbox_client")


class SandboxClient:
    def __init__(self, base_url: str | None = None):
        self._explicit_base_url = base_url

    @property
    def base_url(self) -> str:
        if self._explicit_base_url:
            return self._explicit_base_url
        return os.getenv("AGENT_SANDBOX_URL", "")

    @property
    def is_available(self) -> bool:
        return bool(self.base_url)

    async def ensure_sandbox(self, session_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=300.0) as client:
            logger.debug(f"Ensuring sandbox for run {session_id}")
            payload = {"session_id": session_id}
            response = await client.post(f"{self.base_url}/sandboxes", json=payload)
            response.raise_for_status()
            return response.json()

    async def execute_tool(
        self,
        session_id: str,
        agent_id: str,
        tool_name: str,
        kwargs: dict[str, Any],
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.base_url:
            return {
                "error": "AGENT_SANDBOX_URL not configured. Cannot execute sandboxed tools."
            }

        if not correlation_id:
            correlation_id = f"corr-{uuid.uuid4().hex[:12]}"

        payload = {
            "session_id": session_id,
            "agent_id": agent_id,
            "tool_name": tool_name,
            "kwargs": kwargs,
            "correlation_id": correlation_id,
        }
        headers = {"X-Correlation-ID": correlation_id}

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                sandbox_info = await self.ensure_sandbox(session_id)
                if sandbox_info.get("error"):
                    return sandbox_info

                logger.debug(
                    f"Executing tool '{tool_name}' in sandbox for agent {agent_id}"
                )
                response = await client.post(
                    f"{self.base_url}/execute",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                error_text = f"Status {e.response.status_code}"
                try:
                    error_detail = e.response.json()
                    error_text += f" - {error_detail}"
                except Exception:
                    error_text += f" - {e.response.text}"
                logger.error(f"Sandbox service error: {error_text}")
                return {"error": f"Sandbox service error: {error_text}"}

            except httpx.RequestError as e:
                logger.exception(f"Failed to execute tool in sandbox: {e}")
                return {"error": f"Sandbox execution failed: {e}"}

    async def inject_file(
        self, session_id: str, src_path: str, dest_path: str
    ) -> dict[str, Any]:
        if not self.base_url:
            return {"error": "AGENT_SANDBOX_URL not configured"}

        payload = {"src_path": src_path, "dest_path": dest_path}

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                await self.ensure_sandbox(session_id)

                logger.info(
                    f"Injecting file {src_path} -> {dest_path} for session {session_id}"
                )
                response = await client.post(
                    f"{self.base_url}/sandboxes/{session_id}/files", json=payload
                )
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                logger.error(f"Failed to inject file to sandbox: {e}")
                return {"error": f"Failed to inject file: {e}"}
            except httpx.HTTPStatusError as e:
                logger.error(f"Sandbox injection failed: {e.response.text}")
                return {"error": f"Sandbox injection failed: {e.response.text}"}

    async def destroy_sandbox(self, session_id: str) -> dict[str, Any]:
        if not self.base_url:
            return {"error": "AGENT_SANDBOX_URL not configured"}

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                logger.info(f"Destroying sandbox for session {session_id}")
                response = await client.delete(
                    f"{self.base_url}/sandboxes/{session_id}"
                )
                if response.status_code >= 300:
                    return {
                        "warning": f"Sandbox destruction returned {response.status_code}"
                    }
                return {"success": True}
            except httpx.RequestError as e:
                logger.error(f"Failed to destroy sandbox: {e}")
                return {"error": f"Failed to destroy sandbox: {e}"}


sandbox_client = SandboxClient()

from abc import ABC, abstractmethod
from typing import TypedDict, Optional


class RuntimeConfig(TypedDict):
    workspace_id: str
    api_url: str
    auth_token: Optional[str]
    tool_server_port: int
    agent_id: str


class AgentRuntime(ABC):
    @abstractmethod
    async def create_sandbox(
        self,
        agent_id: str,
        existing_token: Optional[str] = None,
        local_source_path: Optional[str] = None,
    ) -> RuntimeConfig:
        pass

    @abstractmethod
    async def get_sandbox_url(self, container_id: str, port: int) -> str:
        pass

    @abstractmethod
    async def destroy_sandbox(self, container_id: str) -> None:
        pass

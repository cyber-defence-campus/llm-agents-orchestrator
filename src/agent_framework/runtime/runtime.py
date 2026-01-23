from abc import ABC, abstractmethod
from typing import TypedDict, Optional


class RuntimeConfig(TypedDict):
    """Configuration for the agent's execution environment."""

    workspace_id: str
    api_url: str
    auth_token: Optional[str]
    tool_server_port: int
    agent_id: str


class AgentRuntime(ABC):
    """Abstract base class defining the contract for agent runtime environments."""

    @abstractmethod
    async def create_sandbox(
        self,
        agent_id: str,
        existing_token: Optional[str] = None,
        local_source_path: Optional[str] = None,
    ) -> RuntimeConfig:
        """Provisions a new isolated environment for an agent."""
        pass

    @abstractmethod
    async def get_sandbox_url(self, container_id: str, port: int) -> str:
        """Resolves the external URL for a service running within the sandbox."""
        pass

    @abstractmethod
    async def destroy_sandbox(self, container_id: str) -> None:
        """Tears down the specified sandbox environment."""
        pass

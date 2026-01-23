import uuid
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from agent_framework.utils.id_utils import generate_ulid


def _new_agent_id() -> str:
    return f"agent_{generate_ulid()[:12]}"


class AgentContext(BaseModel):
    """
    Represents the operational state and memory of an agent.
    """

    model_config = ConfigDict(populate_by_name=True)

    agent_id: str = Field(default_factory=_new_agent_id)
    agent_name: str = "Assistant"
    parent_id: Optional[str] = None

    # Workflow Status
    status: str = "initializing"
    task: str = ""
    short_task: Optional[str] = None
    original_task: Optional[str] = None

    # Execution Metrics
    iteration: int = 0
    max_iterations: int = 1000
    start_time: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Flags
    completed: bool = False
    stop_requested: bool = False
    waiting_for_input: bool = False
    waiting_since: Optional[datetime] = None
    wait_timeout: Optional[int] = None
    llm_failed: bool = False

    # Memory
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context_data: Dict[str, Any] = Field(default_factory=dict, alias="context")
    tool_history: List[Dict[str, Any]] = Field(default_factory=list)
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    # Sandbox
    sandbox_id: Optional[str] = None
    sandbox_token: Optional[str] = None
    sandbox_info: Optional[Dict[str, Any]] = None

    # Results
    final_result: Optional[Dict[str, Any]] = None
    consecutive_empty_responses: int = 0

    def touch(self) -> None:
        self.last_updated = datetime.now(UTC).isoformat()

    def append_message(self, role: str, content: Any) -> Dict[str, Any]:
        msg = {
            "id": generate_ulid(),
            "role": role,
            "content": content,
            "iteration": self.iteration,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.messages.append(msg)
        self.touch()
        return msg

    def record_tool_use(self, tool_call: Dict[str, Any]) -> None:
        self.tool_history.append(
            {
                "iteration": self.iteration,
                "timestamp": datetime.now(UTC).isoformat(),
                "tool_call": tool_call,
            }
        )
        self.touch()

    def record_observation(self, observation: Dict[str, Any]) -> None:
        self.observations.append(
            {
                "iteration": self.iteration,
                "timestamp": datetime.now(UTC).isoformat(),
                "observation": observation,
            }
        )
        self.touch()

    def record_error(self, error: str) -> None:
        self.errors.append(f"Iter {self.iteration}: {error}")
        self.touch()

    def set_kv(self, key: str, value: Any) -> None:
        self.context_data[key] = value
        self.touch()

    def mark_completed(self, result: Optional[Dict[str, Any]] = None) -> None:
        self.completed = True
        self.final_result = result
        self.touch()

    def signal_stop(self) -> None:
        self.stop_requested = True
        self.touch()

    def should_terminate(self) -> bool:
        return (
            self.stop_requested
            or self.completed
            or self.iteration >= self.max_iterations
        )

    def set_waiting(
        self, timeout: Optional[int] = None, error_state: bool = False
    ) -> None:
        self.waiting_for_input = True
        self.stop_requested = False
        self.llm_failed = error_state
        self.waiting_since = datetime.now(UTC)
        self.wait_timeout = timeout
        self.touch()

    def resume(self, new_task_text: Optional[str] = None) -> None:
        self.waiting_for_input = False
        self.stop_requested = False
        self.completed = False
        self.llm_failed = False
        self.waiting_since = None
        self.wait_timeout = None
        if new_task_text:
            self.task = new_task_text
        self.touch()

    def get_history_for_llm(self) -> List[Dict[str, Any]]:
        """Returns messages formatted for LLM consumption."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages
            if m.get("content") is not None
            and m["role"] not in ["tool_call", "tool_result"]
        ]

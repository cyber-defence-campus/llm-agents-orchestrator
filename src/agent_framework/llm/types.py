from dataclasses import dataclass
from enum import Enum
from typing import Any


class LLMRequestFailedError(Exception):
    def __init__(self, message: str, details: str | None = None):
        super().__init__(message)
        self.message = message
        self.details = details


class StepRole(str, Enum):
    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"


@dataclass
class LLMResponse:
    content: str
    tool_invocations: list[dict[str, Any]] | None = None
    job_id: str | None = None
    step_number: int = 1
    role: StepRole = StepRole.AGENT
    reasoning_content: str | None = None


@dataclass
class RequestStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    cost: float = 0.0
    requests: int = 0
    failed_requests: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cost": round(self.cost, 4),
            "requests": self.requests,
            "failed_requests": self.failed_requests,
        }

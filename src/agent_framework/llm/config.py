import os
from typing import Any
from pydantic import BaseModel, Field, model_validator


class LLMConfig(BaseModel):
    model_name: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    enable_prompt_caching: bool = True
    prompt_modules: list[str] = Field(default_factory=list)
    max_concurrent: int = 6
    delay_between_requests: float = 1.0
    batching_enabled: bool = Field(
        default=True, description="Whether to enable request batching"
    )
    batch_size: int = Field(
        default=10, description="Maximum number of requests in a batch"
    )
    batch_max_wait_time: float = Field(
        default=0.1, description="Maximum time to wait for a batch to fill (seconds)"
    )
    request_timeout: int = Field(
        default=180, description="Timeout for LLM requests in seconds"
    )

    @model_validator(mode="before")
    @classmethod
    def check_env_vars(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("reasoning_effort"):
                # Deployment-level cap. An uncapped reasoning model can spend
                # the whole max_tokens budget thinking and return empty
                # content, which presents as a slow call rather than as a
                # malformed request.
                env_val = os.getenv("LLM_REASONING_EFFORT")
                if env_val:
                    data["reasoning_effort"] = env_val.strip().strip("'")

            if "batching_enabled" not in data:
                env_val = os.getenv("LLM_BATCHING_ENABLED")
                if env_val is not None:
                    data["batching_enabled"] = env_val.lower() in ("true", "1", "yes")

            if "batch_size" not in data:
                env_val = os.getenv("LLM_BATCH_SIZE")
                if env_val is not None:
                    try:
                        data["batch_size"] = int(env_val)
                    except ValueError:
                        pass

            if "batch_max_wait_time" not in data:
                env_val = os.getenv("LLM_BATCH_MAX_WAIT_TIME")
                if env_val is not None:
                    try:
                        data["batch_max_wait_time"] = float(env_val)
                    except ValueError:
                        pass
        return data

    @model_validator(mode="after")
    def complete_config(self) -> "LLMConfig":
        model_name = self.model_name or os.getenv(
            "AGENT_MODEL", "openrouter/z-ai/glm-5.3-flash"
        )

        if not model_name:
            raise ValueError(
                "AGENT_MODEL environment variable must be set or model_name provided and not empty"
            )

        self.model_name = model_name.strip("'")

        if self.temperature is not None:
            self.temperature = max(0.0, min(1.0, self.temperature))
        return self

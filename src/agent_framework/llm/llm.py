import json
import logging
import os
from pathlib import Path
from typing import Any

import litellm
from jinja2 import (
    Environment,
    FileSystemLoader,
    select_autoescape,
)
from litellm import ModelResponse, completion_cost
from litellm.utils import supports_prompt_caching

_DEEPSEEK_V4_PRICES = {
    "input_cost_per_token": 0.14e-6,
    "output_cost_per_token": 0.28e-6,
    "cache_read_input_token_cost": 0.028e-6,
    "litellm_provider": "deepseek",
    "mode": "chat",
}
_DEEPSEEK_V4_PRO_PRICES = {
    "input_cost_per_token": 1.74e-6,
    "output_cost_per_token": 3.48e-6,
    "cache_read_input_token_cost": 0.145e-6,
    "litellm_provider": "deepseek",
    "mode": "chat",
}
def _reported_cost(response) -> float | None:
    """Cost the provider itself reported, if any.

    Preferred over litellm's own table because it is always right and never
    goes stale. OpenRouter returns usage.cost per call; a dated or
    provider-prefixed model id that litellm has no entry for would otherwise
    be silently accounted at zero, which is how every run so far has shown
    $0.0000 despite spending real tokens.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    for attr in ("cost", "total_cost"):
        value = getattr(usage, attr, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(attr)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _price_variants(base: str, prices: dict) -> dict:
    """Register a model under the shapes a model id actually arrives in.

    Dated releases (…-0731) and provider prefixes (openrouter/…) are both
    common and neither matches a bare name, so registering only the bare one
    prices nothing in practice.
    """
    names = {base, f"deepseek/{base}"}
    for name in tuple(names):
        names.add(f"openrouter/{name}")
    return {name: prices for name in names}


try:
    registry: dict = {}
    for _base, _prices in (
        ("deepseek-v4-flash", _DEEPSEEK_V4_PRICES),
        ("deepseek-v4-flash-0731", _DEEPSEEK_V4_PRICES),
        ("deepseek-v4-pro", _DEEPSEEK_V4_PRO_PRICES),
    ):
        registry.update(_price_variants(_base, _prices))
    litellm.register_model(registry)
except Exception:  # pragma: no cover - pricing is best-effort
    pass

from agent_framework.agents.state import AgentContext
from agent_framework.llm.config import LLMConfig
from agent_framework.llm.request_queue import get_shared_queue
from agent_framework.llm.utils import parse_tool_invocations
from agent_framework.prompts import load_prompt_modules, get_all_module_names
from agent_framework.tools import get_tools_prompt
from agent_framework.state import redis_manager as state_manager

logger = logging.getLogger("agent_framework.llm")


api_base = (
    os.getenv("LLM_API_BASE")
    or os.getenv("OPENAI_API_BASE")
    or os.getenv("LITELLM_BASE_URL")
    or os.getenv("OLLAMA_API_BASE")
)
if api_base:
    litellm.api_base = api_base

from agent_framework.llm.types import (
    LLMRequestFailedError,
    LLMResponse,
    StepRole,
    RequestStats,
)

MODELS_WITHOUT_STOP_WORDS = ["o1", "o1-preview", "o1-mini", "o3-mini"]
REASONING_EFFORT_SUPPORTED_MODELS = ["o1", "o1-preview", "o1-mini", "o3-mini"]



def openrouter_routing() -> dict[str, Any] | None:
    """OpenRouter provider pin, from OPENROUTER_PROVIDER_ORDER.

    Left unpinned, OpenRouter picks a provider per request and they do not
    behave alike: for the same model one upstream returned reasoning tokens
    and another returned none, which turned a reasoning cap into a coin
    flip. Pinning also keeps prompt caching effective, since a cache lives
    with one provider.
    """
    order = os.getenv("OPENROUTER_PROVIDER_ORDER", "").strip()
    if not order:
        return None
    providers = [p.strip() for p in order.split(",") if p.strip()]
    if not providers:
        return None
    allow_fallbacks = os.getenv(
        "OPENROUTER_ALLOW_FALLBACKS", "false").lower() in ("1", "true", "yes")
    return {"order": providers, "allow_fallbacks": allow_fallbacks}


class LLM:
    def __init__(
        self,
        config: LLMConfig,
        agent_name: str | None = None,
        agent_type: str | None = None,
        agent_hierarchy: list[dict[str, Any]] | None = None,
        agent_state: AgentContext | None = None,
    ):
        logger.debug(f"[LLM for {agent_name}] Initializing...")
        self.config = config
        self.queue = get_shared_queue(config)

        self.agent_name = agent_name
        self.job_id = None
        self.agent_id = None
        if agent_state and agent_state.sandbox_info:
            self.job_id = agent_state.sandbox_info.get("job_id")
        if agent_state:
            self.agent_id = agent_state.agent_id

        self._total_stats = RequestStats()
        self._last_request_stats = RequestStats()

        if agent_name:
            agent_template_name = agent_type or agent_name
            logger.debug(
                f"[LLM for {agent_name}] Setting up Jinja environment for prompts using agent type '{agent_template_name}'."
            )
            prompt_dir = Path(__file__).parent.parent / "prompts" / agent_template_name
            prompts_dir = Path(__file__).parent.parent / "prompts"

            search_paths = []

            # 1. Add external prompt paths (highest priority)
            extra_paths_env = os.getenv("AGENT_PROMPT_PATHS", "")
            if extra_paths_env:
                for path in extra_paths_env.split(os.pathsep):
                    if path:
                        p = Path(path)
                        search_paths.append(p)

                        agent_specific_path = p / agent_template_name
                        if agent_specific_path.is_dir():
                            search_paths.append(agent_specific_path)

                        for subdir in p.rglob(""):
                            if subdir.is_dir() and not subdir.name.startswith("__"):
                                search_paths.append(subdir)

            # 2. Add local paths
            search_paths.append(prompt_dir)
            search_paths.append(prompts_dir)

            for p in prompts_dir.rglob(""):
                if p.is_dir():
                    search_paths.append(p)

            loader = FileSystemLoader(search_paths)
            self.jinja_env = Environment(
                loader=loader,
                autoescape=select_autoescape(
                    enabled_extensions=(), default_for_string=False
                ),
            )

            try:
                logger.debug(
                    f"[LLM for {agent_name}] Loading prompt modules: {self.config.prompt_modules}"
                )
                prompt_module_content = load_prompt_modules(
                    self.config.prompt_modules or [], self.jinja_env
                )
                logger.debug(f"[LLM for {agent_name}] Prompt modules loaded.")

                def get_module(name: str) -> str:
                    return prompt_module_content.get(name, "")

                self.jinja_env.globals["get_module"] = get_module

                logger.debug(
                    f"[LLM for {agent_name}] Rendering system prompt template..."
                )
                render_params = {
                    "get_tools_prompt": get_tools_prompt,
                    "loaded_module_names": self.config.prompt_modules or [],
                    "available_prompt_modules": sorted(list(get_all_module_names())),
                    "agent_hierarchy": agent_hierarchy,
                    "context": {},
                }
                if agent_state:
                    render_params["context"] = agent_state.context_data
                    # Unpack context values as top-level template variables
                    # so templates can use {% if varname %} directly
                    if agent_state.context_data:
                        render_params.update(agent_state.context_data)

                logger.info(
                    f"Rendering Jinja template with context={render_params.get('context')}"
                )

                self.system_prompt = self.jinja_env.get_template(
                    "system_prompt.jinja"
                ).render(**render_params)
                logger.debug(
                    f"[LLM for {agent_name}] System prompt rendered successfully."
                )
                logger.debug(
                    f"Rendered System Prompt (first 1000 chars for {agent_name}):\n{self.system_prompt[:1000]}..."
                )

                try:
                    logs_dir = Path(__file__).parent.parent.parent.parent / "logs"
                    logs_dir.mkdir(exist_ok=True)
                    sanitized_agent_name = agent_name.replace(" ", "_")
                    prompt_file = logs_dir / f"agent_{sanitized_agent_name}_prompt.txt"
                    prompt_file.write_text(self.system_prompt)
                    logger.info(
                        f"Saved full prompt for agent '{agent_name}' to {prompt_file}"
                    )
                except Exception as e:
                    logger.error(f"Failed to save prompt for agent '{agent_name}': {e}")

            except Exception:
                logger.exception(
                    f"[LLM for {agent_name}] CRITICAL FAILURE during prompt loading/rendering."
                )
                self.system_prompt = "You are a helpful AI assistant."
        else:
            self.system_prompt = "You are a helpful AI assistant."
        logger.debug(f"[LLM for {agent_name}] Initialization complete.")

    def _add_cache_control_to_content(
        self,
        content: str | list[dict[str, Any]],
    ) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if isinstance(content, list) and content:
            last_item = content[-1]
            if isinstance(last_item, dict) and last_item.get("type") == "text":
                return content[:-1] + [
                    {**last_item, "cache_control": {"type": "ephemeral"}}
                ]
        return content

    def _ensure_list_content(
        self, content: str | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Ensures message content is in the list[dict] format."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            if content and isinstance(content[0], str):
                return [{"type": "text", "text": "\n".join(content)}]
            return content
        return [{"type": "text", "text": str(content)}]

    def _is_anthropic_model(self) -> bool:
        if not self.config.model_name:
            return False
        model_lower = self.config.model_name.lower()
        return any(provider in model_lower for provider in ["anthropic/", "claude"])

    def _calculate_cache_interval(self, total_messages: int) -> int:
        if total_messages <= 1:
            return 10

        max_cached_messages = 3
        non_system_messages = total_messages - 1
        interval = 10
        while non_system_messages // interval > max_cached_messages:
            interval += 10
        return interval

    def _prepare_cached_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if (
            not self.config.enable_prompt_caching
            or not supports_prompt_caching(self.config.model_name)
            or not messages
        ):
            return messages

        if not self._is_anthropic_model():
            return messages

        cached_messages = list(messages)

        if cached_messages and cached_messages[0].get("role") == "system":
            system_message = cached_messages[0].copy()
            system_message["content"] = self._add_cache_control_to_content(
                system_message["content"]
            )
            cached_messages[0] = system_message

        total_messages = len(cached_messages)
        if total_messages > 1:
            interval = self._calculate_cache_interval(total_messages)
            cached_count = 0
            for i in range(interval, total_messages, interval):
                if cached_count >= 3:
                    break
                if i < len(cached_messages):
                    message = cached_messages[i].copy()
                    message["content"] = self._add_cache_control_to_content(
                        message["content"]
                    )
                    cached_messages[i] = message
                    cached_count += 1
            return cached_messages
        return messages

    async def generate(
        self,
        conversation_history: list[dict[str, Any]],
        job_id: str | None = None,
        step_number: int = 1,
    ) -> LLMResponse:
        messages = []
        model_name = self.config.model_name or ""
        is_gemini_model = "gemini" in model_name.lower()

        if is_gemini_model:
            history_copy = list(conversation_history)
            if history_copy and history_copy[0].get("role") == "user":
                first_message = history_copy[0].copy()
                original_content = first_message.get("content", "")
                is_string_content = isinstance(original_content, str)
                if not is_string_content or not original_content.startswith(
                    self.system_prompt
                ):
                    first_message["content"] = (
                        f"{self.system_prompt}\n\n---\n\n{original_content}"
                    )
                messages = [first_message] + history_copy[1:]
            else:
                messages = [
                    {"role": "system", "content": self.system_prompt}
                ] + history_copy
        else:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(conversation_history)

        normalized_messages = []
        for msg in messages:
            new_msg = msg.copy()
            new_msg["content"] = self._ensure_list_content(new_msg["content"])
            normalized_messages.append(new_msg)

        cached_messages = self._prepare_cached_messages(normalized_messages)

        logger.info(
            f"--- LLM REQUEST (Agent: {self.agent_name}, Model: {self.config.model_name}) ---"
        )
        try:
            log_payload = json.dumps(cached_messages, indent=2)
            if len(log_payload) > 5000:
                log_payload = log_payload[:5000] + "\n... (payload truncated)"
            logger.debug(f"MESSAGES (for {self.agent_name}):\n{log_payload}")
        except Exception as e:
            logger.warning(f"Could not serialize messages for logging: {e}")

        try:
            response = await self._make_request(cached_messages)
            self._update_usage_stats(response)

            content = ""
            raw_content = None
            reasoning_content = None
            if response.choices and response.choices[0].message:
                if response.choices[0].message.content is not None:
                    content = response.choices[0].message.content
                    raw_content = content

                # specific handling for provider reasoning content
                reasoning_content = getattr(
                    response.choices[0].message, "reasoning_content", None
                )
                if not reasoning_content and hasattr(
                    response.choices[0].message, "provider_specific_fields"
                ):
                    provider_fields = response.choices[
                        0
                    ].message.provider_specific_fields
                    if provider_fields:
                        reasoning_content = provider_fields.get("reasoning_content")

            if "<function=" in content and not content.rstrip().endswith("</function>"):
                last_func_start = content.rfind("<function=")
                if content.find("</function>", last_func_start) == -1:
                    content = content + "\n</function>"

            tool_invocations = parse_tool_invocations(content) or []

            if reasoning_content and isinstance(reasoning_content, str):
                reasoning_tools = parse_tool_invocations(reasoning_content)
                if reasoning_tools:
                    logger.info(
                        f"Extracted {len(reasoning_tools)} tool(s) from REASONING_CONTENT for {self.agent_name}"
                    )
                    tool_invocations.extend(reasoning_tools)

            if not tool_invocations:
                tool_invocations = None
            logger.debug(
                f"Raw LLM Response Content (for {self.agent_name}):\n{raw_content}"
            )
            logger.debug(
                f"Parsed Tool Invocations (for {self.agent_name}): {tool_invocations}"
            )

            logger.info(
                f"--- LLM RESPONSE (Agent: {self.agent_name}, Model: {self.config.model_name}) ---"
            )
            logger.info(f"CONTENT:\n{content}")
            if reasoning_content:
                logger.info(f"REASONING CONTENT:\n{reasoning_content}")
            if tool_invocations:
                logger.info(
                    f"TOOL INVOCATIONS:\n{json.dumps(tool_invocations, indent=2)}"
                )
            logger.info(f"USAGE: {self.usage_stats['last_request']}")
            logger.info("--- END LLM RESPONSE ---")

            return LLMResponse(
                job_id=job_id,
                step_number=step_number,
                role=StepRole.AGENT,
                content=content,
                tool_invocations=tool_invocations if tool_invocations else None,
                reasoning_content=reasoning_content,
            )

        except litellm.RateLimitError as e:
            raise LLMRequestFailedError(
                "LLM request failed: Rate limit exceeded", str(e)
            ) from e
        except litellm.AuthenticationError as e:
            raise LLMRequestFailedError(
                "LLM request failed: Invalid API key", str(e)
            ) from e
        except litellm.NotFoundError as e:
            raise LLMRequestFailedError(
                "LLM request failed: Model not found", str(e)
            ) from e
        except litellm.ContextWindowExceededError as e:
            raise LLMRequestFailedError(
                "LLM request failed: Context too long", str(e)
            ) from e
        except litellm.ContentPolicyViolationError as e:
            raise LLMRequestFailedError(
                "LLM request failed: Content policy violation", str(e)
            ) from e
        except litellm.ServiceUnavailableError as e:
            raise LLMRequestFailedError(
                "LLM request failed: Service unavailable", str(e)
            ) from e
        except litellm.Timeout as e:
            raise LLMRequestFailedError(
                "LLM request failed: Request timed out", str(e)
            ) from e
        except Exception as e:
            raise LLMRequestFailedError(
                f"LLM request failed: {type(e).__name__}", str(e)
            ) from e

    @property
    def usage_stats(self) -> dict[str, dict[str, int | float]]:
        return {
            "total": self._total_stats.to_dict(),
            "last_request": self._last_request_stats.to_dict(),
        }

    def _should_include_stop_param(self) -> bool:
        if not self.config.model_name:
            return True
        model_lower = self.config.model_name.lower()
        return not any(
            model_lower.endswith(unsupported)
            for unsupported in MODELS_WITHOUT_STOP_WORDS
        )

    def _should_include_reasoning_effort(self) -> bool:
        if not self.config.model_name:
            return False
        model_lower = self.config.model_name.lower()
        return any(
            model_lower.endswith(supported)
            for supported in REASONING_EFFORT_SUPPORTED_MODELS
        )

    def _update_usage_stats(self, response: ModelResponse) -> None:
        try:
            cost = _reported_cost(response)
            if cost is None:
                try:
                    cost = completion_cost(response) or 0.0
                except Exception as ce:
                    # A model litellm has no price for must NOT discard the
                    # token counts below — record tokens with cost 0.
                    logger.debug(
                        f"completion_cost unavailable ({ce}); recording tokens with cost=0"
                    )
                    cost = 0.0
            usage = response.usage
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", 0)
                output_tokens = getattr(usage, "completion_tokens", 0)

                cached_tokens = 0
                cache_creation_tokens = 0

                prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
                if prompt_tokens_details:
                    if isinstance(prompt_tokens_details, dict):
                        cached_tokens = prompt_tokens_details.get("cached_tokens") or 0
                        cache_creation_tokens = (
                            prompt_tokens_details.get("cache_creation_input_tokens")
                            or 0
                        )
                    else:
                        cached_tokens = (
                            getattr(prompt_tokens_details, "cached_tokens", 0) or 0
                        )
                        cache_creation_tokens = (
                            getattr(
                                prompt_tokens_details,
                                "cache_creation_input_tokens",
                                0,
                            )
                            or 0
                        )

                self._total_stats.input_tokens += input_tokens
                self._total_stats.output_tokens += output_tokens
                self._total_stats.cost += cost
                self._last_request_stats = RequestStats(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                )
                logger.info(
                    f"Usage stats updated: Cost=${cost:.4f}, In={input_tokens}, Out={output_tokens}, Cached={cached_tokens}"
                )

                if self.job_id:
                    state_manager.increment_usage_stats(
                        job_id=self.job_id,
                        agent_id=self.agent_id,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=cost,
                        requests=1,
                        cached_tokens=cached_tokens,
                        cache_creation_tokens=cache_creation_tokens,
                        model_name=self.config.model_name,
                    )
        except Exception as e:
            logger.warning(f"Failed to update usage stats: {e}")

    async def _make_request(
        self,
        messages: list[dict[str, Any]],
    ) -> ModelResponse:
        completion_args: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "timeout": self.config.request_timeout,
            "metadata": {"job_id": self.job_id},
            "max_tokens": 8192,
            "headers": {},
        }

        if self.config.temperature is not None:
            completion_args["temperature"] = self.config.temperature

        provider = None
        if self.config.model_name and "/" in self.config.model_name:
            provider = self.config.model_name.split("/")[0]
            completion_args["custom_llm_provider"] = provider

        if self.config.api_key:
            completion_args["api_key"] = self.config.api_key
        elif provider:
            api_key_env = f"{provider.upper()}_API_KEY"
            api_key = os.getenv(api_key_env)
            if api_key:
                completion_args["api_key"] = api_key.strip("'")

        if self.config.api_base:
            completion_args["api_base"] = self.config.api_base
        else:
            api_base = os.getenv("LLM_API_BASE") or os.getenv("OLLAMA_API_BASE")
            if api_base:
                completion_args["api_base"] = api_base

        if self._should_include_stop_param():
            completion_args["stop"] = ["</function>"]

        effort = self.config.reasoning_effort
        if not effort and self._should_include_reasoning_effort():
            effort = "high"
        if effort:
            if provider == "openrouter":
                # litellm rejects the normalised reasoning_effort on the
                # OpenRouter route (UnsupportedParamsError). OpenRouter takes
                # its own `reasoning` object instead, which has to travel in
                # extra_body. Without this, capping reasoning is impossible
                # here, and an uncapped reasoning model can spend the whole
                # max_tokens budget thinking and return empty content.
                completion_args.setdefault("extra_body", {})["reasoning"] = {
                    "effort": effort
                }
            else:
                completion_args["reasoning_effort"] = effort

        if provider == "openrouter" and (routing := openrouter_routing()):
            completion_args.setdefault("extra_body", {})["provider"] = routing

        logger.debug(
            f"Completion Args for LiteLLM (Agent: {self.agent_name}): {completion_args}"
        )

        response = await self.queue.make_request(completion_args)

        self._total_stats.requests += 1
        self._last_request_stats = RequestStats(requests=1)

        logger.debug(
            f"Raw LiteLLM Response Object (Agent: {self.agent_name}): {response}"
        )

        return response


async def complete(
    prompt: str,
    system: str | None = None,
    config: LLMConfig | None = None,
    max_tokens: int = 4096,
    reasoning_max_tokens: int | None = None,
) -> dict[str, Any]:
    """One completion, with no agent loop around it.

    The LLM class exists to drive a conversational agent: it renders a system
    prompt full of tool definitions, an XML call protocol and wait-mode
    rules. That is the wrong vehicle for asking a model to emit one
    structured document. A caller that wants JSON back ends up fighting the
    agent contract, and a reasoning model can spend its whole budget
    deciding whether to answer or to emit a tool call, returning empty
    content after several minutes.

    This shares the provider, key and reasoning resolution with
    _make_request so there is one place that knows how to reach a model, and
    returns the raw fields a caller needs to tell an empty answer from a
    truncated one.
    """
    config = config or LLMConfig()
    model_name = config.model_name or ""

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    args: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "timeout": config.request_timeout,
    }

    provider = model_name.split("/")[0] if "/" in model_name else None
    if provider:
        args["custom_llm_provider"] = provider

    if config.api_key:
        args["api_key"] = config.api_key
    elif provider and (key := os.getenv(f"{provider.upper()}_API_KEY")):
        args["api_key"] = key.strip("'")

    if config.api_base:
        args["api_base"] = config.api_base
    elif api_base_env := (os.getenv("LLM_API_BASE") or os.getenv("OLLAMA_API_BASE")):
        args["api_base"] = api_base_env

    if config.temperature is not None:
        args["temperature"] = config.temperature

    # A token budget beats an effort level. "low" is advisory and a
    # reasoning model can still spend the entire max_tokens thinking and
    # return empty content with finish_reason=length, which is a silent
    # failure that costs the full budget. Capping reasoning by tokens
    # guarantees headroom for an answer.
    if provider == "openrouter" and reasoning_max_tokens:
        args.setdefault("extra_body", {})["reasoning"] = {
            "max_tokens": reasoning_max_tokens
        }
    elif config.reasoning_effort:
        if provider == "openrouter":
            args.setdefault("extra_body", {})["reasoning"] = {
                "effort": config.reasoning_effort
            }
        else:
            args["reasoning_effort"] = config.reasoning_effort

    if provider == "openrouter" and (routing := openrouter_routing()):
        args.setdefault("extra_body", {})["provider"] = routing

    response = await litellm.acompletion(**args)
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    usage = getattr(response, "usage", None)

    return {
        "content": (getattr(message, "content", None) or ""),
        "reasoning": (getattr(message, "reasoning_content", None)
                      or getattr(message, "reasoning", None) or ""),
        "finish_reason": getattr(choice, "finish_reason", None),
        "model": model_name,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
    }

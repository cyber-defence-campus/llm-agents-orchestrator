import asyncio
import logging
import re
import threading
import time
from typing import Any

import litellm
from litellm import ModelResponse, acompletion
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tenacity.wait import wait_base

from .config import LLMConfig


logger = logging.getLogger(__name__)


class LLMRequestQueue:
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        # Use asyncio.Semaphore for async concurrency control
        self._semaphore = asyncio.Semaphore(llm_config.max_concurrent)
        self._last_request_time = 0.0
        self._lock = threading.Lock()

        logger.info(
            f"Initializing LLMRequestQueue for model {llm_config.model_name}. "
            f"Max Concurrent: {llm_config.max_concurrent}, "
            f"Delay: {llm_config.delay_between_requests}s"
        )

    async def make_request(self, completion_args: dict[str, Any]) -> ModelResponse:
        return await self._reliable_request(completion_args)

    def handle_rate_limit(self, delay: float) -> None:
        """
        Updates the last request time to enforce a global pause for this queue.
        """
        with self._lock:
            now = time.time()
            target_time = now + delay
            # If the current last_request_time is already further out, keep it.
            # Otherwise, push it to target_time.
            if target_time > self._last_request_time:
                logger.warning(
                    f"Updating global queue pause. Pausing for {delay:.2f}s."
                )
                self._last_request_time = target_time

    @retry(  # type: ignore[misc]
        stop=stop_after_attempt(10),
        retry=retry_if_exception(lambda e: should_retry_exception(e)),
        reraise=True,
    )
    async def _reliable_request(self, completion_args: dict[str, Any]) -> ModelResponse:
        async with self._semaphore:
            # Enforce rate limiting (delay between requests)
            sleep_needed = 0.0
            with self._lock:
                now = time.time()
                # We want to wait until _last_request_time + delay_between_requests
                next_slot = (
                    self._last_request_time + self.llm_config.delay_between_requests
                )
                sleep_needed = max(0, next_slot - now)
                self._last_request_time = now + sleep_needed

            if sleep_needed > 0:
                if sleep_needed > 1.0:
                    logger.info(
                        f"Rate limiting: Sleeping for {sleep_needed:.2f}s before request."
                    )
                await asyncio.sleep(sleep_needed)

            response = await acompletion(**completion_args, stream=False)
        if isinstance(response, ModelResponse):
            return response
        self._raise_unexpected_response()
        raise RuntimeError("Unreachable code")

    def _raise_unexpected_response(self) -> None:
        raise RuntimeError("Unexpected response type")


class wait_respecting_quota(wait_base):
    def __init__(self, fallback_wait):
        self.fallback_wait = fallback_wait

    def __call__(self, retry_state) -> float:
        exc = retry_state.outcome.exception()
        queue_instance: LLMRequestQueue = retry_state.args[
            0
        ]  # Get 'self' from method args

        if exc:
            error_str = str(exc)
            delay = None

            # Pattern 1: "retry in Xs" (Gemini often says "Please retry in 16.22s.")
            match = re.search(r"retry in (\d+(\.\d+)?)s", error_str, re.IGNORECASE)
            if match:
                try:
                    delay = float(match.group(1))
                except ValueError:
                    pass

            # Pattern 2: JSON "retryDelay": "16s"
            if delay is None:
                match = re.search(r'"retryDelay":\s*"(\d+(\.\d+)?)s?"', error_str)
                if match:
                    try:
                        delay = float(match.group(1))
                    except ValueError:
                        pass

            if delay is not None:
                # Add a small buffer
                adjusted_delay = delay + 1.0
                logger.warning(
                    f"Rate limit hit. Pausing queue for {adjusted_delay:.2f}s."
                )
                # Update the queue so other agents/requests also wait
                queue_instance.handle_rate_limit(adjusted_delay)
                return adjusted_delay

        return self.fallback_wait(retry_state)


LLMRequestQueue._reliable_request.retry.wait = wait_respecting_quota(
    wait_exponential(multiplier=2, min=1, max=60)
)


def should_retry_exception(exception: Exception) -> bool:
    # Check for Quota Exceeded / Resource Exhausted errors
    error_msg = str(exception).lower()
    if (
        "unable to get json response" in error_msg
        or "expecting value: line 1 column 1" in error_msg
        or "gateway timeout" in error_msg
        or "504" in error_msg
    ):
        logger.warning(
            f"Transient error detected (JSON/Timeout). Retrying. Error: {exception}"
        )
        return True

    if (
        "quota" in error_msg and "exceeded" in error_msg
    ) or "resource exhausted" in error_msg:
        logger.warning(
            f"Quota exceeded error detected. Retrying after delay... Error: {exception}"
        )
        return True

    status_code = None

    if hasattr(exception, "status_code"):
        status_code = exception.status_code
    elif hasattr(exception, "response") and hasattr(exception.response, "status_code"):
        status_code = exception.response.status_code

    if status_code is not None:
        return bool(litellm._should_retry(status_code))
    return True


_queue_registry: dict[str, LLMRequestQueue] = {}
_registry_lock = threading.Lock()


def get_shared_queue(llm_config: LLMConfig) -> LLMRequestQueue:
    """
    Returns a shared LLMRequestQueue instance for the given model.
    """
    global _queue_registry
    key = llm_config.model_name or "default"

    with _registry_lock:
        if key not in _queue_registry:
            _queue_registry[key] = LLMRequestQueue(llm_config)
        return _queue_registry[key]

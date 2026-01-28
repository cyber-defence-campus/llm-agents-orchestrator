import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List, Type

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agent_framework.llm.config import LLMConfig
from agent_framework.llm.types import LLMRequestFailedError, LLMResponse
from agent_framework.state import redis_manager as db
from agent_framework.tools import process_tool_invocations

from .state import AgentContext

logger = logging.getLogger("agent_framework.core_agent")


def _resolve_template_paths(agent_type_name: str) -> List[Path]:
    """Helper to determine search paths for Jinja templates."""
    paths = []

    # 1. External Overrides (ENV)
    if env_paths := os.getenv("AGENT_PROMPT_PATHS"):
        for p_str in env_paths.split(os.pathsep):
            if not p_str.strip():
                continue
            p = Path(p_str)
            paths.append(p)
            if (p / agent_type_name).is_dir():
                paths.append(p / agent_type_name)

    # 2. Built-in Defaults
    root = Path(__file__).parent.parent
    builtin_agent_dir = root / "prompts" / agent_type_name

    if builtin_agent_dir.is_dir():
        paths.append(builtin_agent_dir)

    return paths


class BaseAgent:
    """
    The primary execution unit for the agent system.
    """

    MAX_ITERATION_LIMIT = 200
    agent_name: str = "GenericAgent"
    jinja_env: Environment

    # Default configuration
    _default_llm_conf = LLMConfig()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Avoid running setup for the abstract base class itself
        if cls.__name__ == "BaseAgent":
            return

        cls._configure_template_engine()

    @classmethod
    def _configure_template_engine(cls):
        """Sets up the Jinja2 environment for the agent subclass."""
        cls.agent_name = cls.__name__
        search_paths = _resolve_template_paths(cls.agent_name)

        cls.jinja_env = Environment(
            loader=FileSystemLoader(search_paths),
            autoescape=select_autoescape(
                enabled_extensions=[], default_for_string=False
            ),
        )

    def __init__(self, configuration: Dict[str, Any]):
        self.configuration = configuration
        self._setup_llm_config(configuration)
        self._setup_context(configuration)

        # Register in Redis
        self._announce_presence()

        # Initialize LLM Interface
        from agent_framework.llm import LLM

        self.llm = LLM(
            self.llm_config,
            agent_name=self.context.agent_name,
            agent_type=self.__class__.__name__,
            agent_hierarchy=configuration.get("agent_hierarchy", []),
            agent_state=self.context,
        )

        self._active_tool_task: Optional[asyncio.Task] = None
        logger.info(
            f"BaseAgent initialized: {self.context.agent_name} [{self.context.agent_id}]"
        )

    @property
    def context(self) -> AgentContext:
        """Access the agent's state context."""
        return self.state

    def _setup_llm_config(self, config: Dict[str, Any]):
        base_conf = self._default_llm_conf.model_copy()

        if overrides := config.get("llm_config"):
            if isinstance(overrides, dict):
                overrides = LLMConfig(**overrides)
            base_conf = base_conf.model_copy(
                update=overrides.model_dump(exclude_unset=True)
            )

        self.llm_config = base_conf
        self.config_profile_name = config.get("llm_config_name", "standard")

    def _setup_context(self, config: Dict[str, Any]):
        raw_state = config.get("state")
        if isinstance(raw_state, AgentContext):
            self.state = raw_state
        elif isinstance(raw_state, dict):
            self.state = AgentContext(**raw_state)
        else:
            self.state = AgentContext(
                agent_name=self.agent_name, max_iterations=self.MAX_ITERATION_LIMIT
            )

    def _announce_presence(self):
        """Registers the agent node in the distributed graph."""
        model_display = self.llm_config.model_name
        if self.llm_config.reasoning_effort:
            model_display = (
                f"{self.llm_config.model_name} ({self.llm_config.reasoning_effort})"
            )

        details = {
            "id": self.context.agent_id,
            "name": self.context.agent_name,
            "task": self.context.original_task or self.context.task,
            "status": "booting",
            "parent_id": self.context.parent_id,
            "created_at": self.context.start_time,
            "agent_type": self.__class__.__name__,
            "model": model_display,
        }
        db.add_agent_node(details)
        db.add_agent_state(self.context.agent_id, self.context)

        if self.context.parent_id:
            db.add_edge(self.context.parent_id, self.context.agent_id, "delegation")
        else:
            db.set_root_agent_id(self.context.agent_id)

    async def start_lifecycle(self, primary_task: str) -> Dict[str, Any]:
        """Begins the agent's main execution loop."""
        await self._initialize_task(primary_task)
        self.context.status = "running"
        self._persist_state()

        # Start background monitor for stop signals
        monitor_task = asyncio.create_task(self._watch_status())

        try:
            while not self.context.should_terminate():
                # Check for external stop request
                if self._is_stop_requested():
                    break

                # Handle incoming messages
                await self._check_messages()
                if self.context.should_terminate():
                    break

                # Handle Waiting State
                if self.context.waiting_for_input:
                    await self._wait_cycle()
                    continue

                # Iteration Check
                self.context.iteration += 1
                if self.context.iteration > 0 and self.context.iteration % 50 == 0:
                    await self._compact_memory()

                # Core Execution Step
                try:
                    is_done = await self._execute_cycle()
                    if is_done:
                        self.context.mark_completed()
                        break

                except Exception as ex:
                    self._handle_runtime_error(ex)
                    if self.context.should_terminate():
                        break

                self._persist_state()

        finally:
            monitor_task.cancel()
            final_status = "stopped" if not self.context.completed else "completed"
            self._persist_state(status_override=final_status)

        return {"status": self.context.status}

    async def _initialize_task(self, task_desc: str):
        self.context.task = task_desc
        formatted_task = f"Current Objective:\n<task>\n{task_desc}\n</task>"

        # Idempotency check
        existing = any(
            m["role"] == "user" and task_desc in m.get("content", "")
            for m in self.context.messages
        )
        if not existing:
            self.context.append_message("user", formatted_task)

    async def _execute_cycle(self) -> bool:
        """Performs one cognitive cycle: Think -> Act."""
        prompt_history = self.context.get_history_for_llm()

        try:
            response: LLMResponse = await self.llm.generate(prompt_history)
        except LLMRequestFailedError as e:
            logger.warning(f"Generation failure: {e}")
            self.context.record_error(str(e))
            self.context.set_waiting(error_state=True)
            return False

        llm_content = response.content
        tool_calls = response.tool_invocations

        # Fallback: If content is empty but we have reasoning (e.g. DeepSeek), use it. // TODO: We should really standardize this.
        if not llm_content and response.reasoning_content:
            llm_content = f"Thinking Process:\n{response.reasoning_content}"

        new_msg = self.context.append_message("assistant", llm_content)
        self._emit_message_event(new_msg)

        # Dispatch Tools
        if tool_calls:
            return await self._dispatch_tools(tool_calls)

        # Handling Empty Responses (Loop Detection)
        if not llm_content.strip():
            self.context.consecutive_empty_responses += 1
            if self.context.consecutive_empty_responses >= 5:
                raise RuntimeError("Agent stuck in empty response loop.")
            self.context.append_message(
                "user", "System: Received empty response. Please proceed."
            )
            return False

        self.context.consecutive_empty_responses = 0

        self.context.set_waiting()
        return False

    async def _dispatch_tools(self, tool_list: List[Dict[str, Any]]) -> bool:
        """Runs the requested tools and updates history."""
        for t in tool_list:
            self.context.record_tool_use(t)

        # We invoke the tool processor
        current_history = (
            self.context.messages
        )  # Direct reference modification by processor

        task = asyncio.create_task(
            process_tool_invocations(tool_list, current_history, self.context)
        )
        self._active_tool_task = task

        try:
            should_terminate = await task
        except asyncio.CancelledError:
            logger.warning("Tool execution interrupted.")
            raise
        finally:
            self._active_tool_task = None

        return should_terminate

    async def _check_messages(self):
        """Reads and processes the agent's inbox."""
        inbox = db.pop_all_messages_for_agent(self.context.agent_id)
        for msg in inbox:
            if msg.get("type") == "control":
                if msg.get("content") == "stop":
                    self.context.signal_stop()
            else:
                # Standard inter-agent message
                sender = msg.get("from", "unknown")
                body = msg.get("content", "")
                formatted = (
                    f"<inter_agent_message>\n"
                    f"  <from>{sender}</from>\n"
                    f"  <content>{body}</content>\n"
                    f"</inter_agent_message>"
                )
                self.context.append_message("user", formatted)

                # Wake up if sleeping
                if self.context.waiting_for_input:
                    self.context.resume()
                    self.context.status = "running"
                    db.update_agent_status(self.context.agent_id, "running")

    def _persist_state(self, status_override: Optional[str] = None):
        if status_override:
            self.context.status = status_override
            db.update_agent_status(self.context.agent_id, status_override)
        db.add_agent_state(self.context.agent_id, self.context)

    def _is_stop_requested(self) -> bool:
        status = db.get_agent_status(self.context.agent_id)
        if status == "stopping":
            self.context.signal_stop()
            return True
        return False

    def _emit_message_event(self, msg: Dict[str, Any]):
        if not self.context.sandbox_info:
            return
        try:
            db.publish_event(
                self.context.sandbox_info.get("job_id"),
                "new_message",
                {
                    "agent_id": self.context.agent_id,
                    "sender": "agent",
                    "content": msg.get("content"),
                    "id": msg.get("id"),
                    "timestamp": msg.get("timestamp"),
                },
            )
        except Exception:
            pass

    async def _watch_status(self):
        """Background task to poll for kill signals."""
        while True:
            await asyncio.sleep(2)
            if self._is_stop_requested():
                if self._active_tool_task:
                    self._active_tool_task.cancel()
                break

    async def _wait_cycle(self):
        # Placeholder for more complex wait logic (e.g., event listeners)
        await asyncio.sleep(1)

    def _handle_runtime_error(self, err: Exception):
        logger.exception(f"Runtime Error: {err}")
        self.context.record_error(str(err))
        db.update_agent_status(self.context.agent_id, "error", str(err))
        self.context.signal_stop()

    async def _compact_memory(self):
        # Future implementation: Context summarization
        pass

import asyncio
import logging
import threading
from typing import Dict, Any, Optional

from .shell_session import ShellExecutor

logger = logging.getLogger(__name__)


class TerminalToolManager:
    """
    Registry for managing multiple ShellExecutor sessions.
    Allows executing commands across different isolated terminals.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TerminalToolManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._executors: Dict[str, ShellExecutor] = {}
        self._manager_lock = threading.Lock()
        self._default_id = "default"
        self._initialized = True

    def get_executor(self, session_id: str) -> ShellExecutor:
        with self._manager_lock:
            if session_id not in self._executors:
                self._executors[session_id] = ShellExecutor(session_id)

            executor = self._executors[session_id]

            # Auto-heal if dead
            if not executor.is_active:
                self._executors[session_id] = ShellExecutor(session_id)
                executor = self._executors[session_id]

            return executor

    async def execute_command(
        self,
        command: str,
        is_input: bool = False,
        timeout: float | None = None,
        terminal_id: str | None = None,
        no_enter: bool = False,
    ) -> Dict[str, Any]:
        """
        Main entry point for executing terminal commands.
        """
        tid = terminal_id or self._default_id
        t_out = timeout if timeout is not None else 30.0

        executor = self.get_executor(tid)

        try:
            return await executor.run(
                command, timeout=t_out, is_input=is_input, no_enter=no_enter
            )
        except Exception as e:
            logger.exception(f"Terminal execution error in session {tid}")
            return {
                "error": str(e),
                "terminal_id": tid,
                "status": "error",
                "content": "",
            }

    def close_session(self, terminal_id: str) -> Dict[str, Any]:
        with self._manager_lock:
            if terminal_id in self._executors:
                self._executors[terminal_id].terminate()
                del self._executors[terminal_id]
                return {"status": "closed", "terminal_id": terminal_id}

        return {"status": "not_found", "terminal_id": terminal_id}

    def shutdown_all(self):
        with self._manager_lock:
            for exc in self._executors.values():
                exc.terminate()
            self._executors.clear()


# Singleton Accessor
_terminal_manager = TerminalToolManager()


def get_terminal_manager() -> TerminalToolManager:
    return _terminal_manager

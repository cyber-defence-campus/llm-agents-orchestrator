"""
ShellExecutor: Async wrapper for persistent tmux-based shell session.

Provides reliable command execution with proper output capture and exit code tracking.
Uses a unique marker system to detect command completion.
"""

import asyncio
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import libtmux

logger = logging.getLogger(__name__)


class ShellExecutor:
    """
    Async wrapper for a persistent tmux-based shell session.
    Provides methods to execute commands, handle input, and manage the session lifecycle.
    """

    # Unique marker prefix to identify command boundaries
    MARKER_PREFIX = "__AG_CMD__"

    def __init__(self, session_id: str, work_dir: str = "/workspace"):
        self.id = session_id
        self.work_dir_path = Path(work_dir).resolve()
        self.work_dir = str(self.work_dir_path)
        self.server = libtmux.Server()
        self.tmux_session: Optional[libtmux.Session] = None
        self.pane: Optional[libtmux.Pane] = None
        self.active = False
        self.busy = False

        self._initialize()

    def _initialize(self):
        """Sets up the tmux session and pane."""
        session_name = f"ag-exec-{self.id}-{uuid.uuid4().hex[:4]}"

        # Kill if exists (shouldn't happen with unique names, but safety first)
        if self.server.has_session(session_name):
            self.server.kill_session(session_name)

        self.tmux_session = self.server.new_session(
            session_name=session_name,
            start_directory=self.work_dir,
            window_name="shell",
            x=200,
            y=50,
        )
        self.tmux_session.set_option("history-limit", "50000")

        self.pane = self.tmux_session.active_window.active_pane

        # Configure minimal prompt - we'll handle markers per-command
        # Send each command with a small delay to ensure shell processes them
        init_cmds = [
            "exec bash --noprofile --norc",
            "export TERM=xterm",
            "stty -echo",
            "export PS1=''",
            "export PS2=''",
            "unset PROMPT_COMMAND",
            "unset PROMPT",
            "unset RPROMPT",
        ]

        for cmd in init_cmds:
            self.pane.send_keys(cmd, enter=True)
            time.sleep(0.1)

        # Final clear and reset history
        time.sleep(0.2)
        self.pane.send_keys("clear", enter=True)
        time.sleep(0.2)

        # Clear history multiple times to ensure it's clean
        self.pane.cmd("clear-history")
        time.sleep(0.1)
        self.pane.cmd("clear-history")

        self.active = True

    @property
    def is_active(self) -> bool:
        return self.active and self.tmux_session is not None

    def _generate_marker(self) -> str:
        """Generate a unique marker for command tracking."""
        return f"{self.MARKER_PREFIX}{uuid.uuid4().hex[:8]}"

    def _sanitize_output(self, content: str) -> str:
        """Remove internal markers and clean up output for display."""
        # Remove all marker lines (START/END markers with any suffix)
        content = re.sub(rf"{re.escape(self.MARKER_PREFIX)}[^\n]*\n?", "", content)
        # Clean up excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()

    async def run(
        self,
        cmd: str,
        timeout: float = 30.0,
        is_input: bool = False,
        no_enter: bool = False,
    ) -> dict:
        """
        Runs a command or sends input to the shell.
        """
        if not self.is_active:
            return {"error": "Session inactive", "status": "error"}

        # Handle interrupt request to unblock a busy session
        if cmd.strip() == "^C":
            self.pane.send_keys("C-c")
            # Wait briefly for the process to terminate
            await asyncio.sleep(0.5)
            self.busy = False
            return {
                "content": "^C (Interrupted)",
                "status": "completed",
                "exit_code": 130,
                "working_dir": self.work_dir,
                "terminal_id": self.id,
            }

        # If session is busy, only allow input to the running process
        if self.busy and not is_input:
            return {
                "error": "Session is busy with a running command (e.g., blocking call like 'top' or 'nmap'). "
                "Use 'require_input=True' to interact with it, send '^C' to interrupt, "
                "or use a different terminal_id.",
                "status": "error",
                "terminal_id": self.id,
            }

        if is_input:
            self.pane.send_keys(cmd, enter=not no_enter)
            await asyncio.sleep(0.2)
            content = self._sanitize_output(self._read_buffer())
            return {
                "content": content,
                "status": "running",
                "exit_code": None,
                "working_dir": self.work_dir,
                "terminal_id": self.id,
            }

        return await self._execute_command(cmd, timeout)

    async def _execute_command(self, cmd: str, timeout: float) -> dict:
        """Execute a command and wait for completion with marker detection."""

        # Mark session as busy
        self.busy = True

        try:
            if not cmd.strip():
                self.busy = False
                return {
                    "content": self._sanitize_output(self._read_buffer()),
                    "status": "running",
                    "exit_code": None,
                    "working_dir": self.work_dir,
                    "terminal_id": self.id,
                }

            # Generate unique marker for this command
            marker = self._generate_marker()
            start_marker = f"{marker}_START"
            end_marker = f"{marker}_END"

            # Clear history to avoid stale data
            self.pane.cmd("clear-history")
            await asyncio.sleep(0.05)

            # Build wrapped command: echo start marker, run command, echo end marker with exit code
            # Using a compound command ensures we capture the actual exit code
            # We add a newline before the end marker to handle heredocs correctly
            wrapped_cmd = f"echo '{start_marker}'; {cmd}\necho '{end_marker}'$?"
            self.pane.send_keys(wrapped_cmd, enter=True)

            start_ts = time.time()

            while (time.time() - start_ts) < timeout:
                await asyncio.sleep(0.15)
                output = self._read_buffer()

                # Look for our end marker with exit code
                end_pattern = rf"{re.escape(end_marker)}(\d+)"
                match = re.search(end_pattern, output)

                if match:
                    exit_code = int(match.group(1))

                    # Extract content between start and end markers
                    start_pattern = rf"{re.escape(start_marker)}\n?"
                    start_match = re.search(start_pattern, output)

                    if start_match:
                        # Content is between end of start marker and start of end marker
                        content_start = start_match.end()
                        content_end = match.start()
                        content = output[content_start:content_end]

                        # Clean up the content
                        content = content.strip()
                        content = self._sanitize_output(content)

                        self.busy = False
                        return {
                            "content": content,
                            "status": "completed",
                            "exit_code": exit_code,
                            "working_dir": self.work_dir,
                            "terminal_id": self.id,
                        }

            # Timeout - sanitize output to hide any markers
            # NOTE: We leave self.busy = True because the command is likely still running
            return {
                "content": self._sanitize_output(self._read_buffer()),
                "status": "running",
                "exit_code": None,
                "working_dir": self.work_dir,
                "terminal_id": self.id,
            }
        except Exception:
            # If a crash happens, we reset busy state to allow recovery attempts
            self.busy = False
            raise

    def _read_buffer(self) -> str:
        """Read the current tmux pane buffer."""
        if not self.pane:
            return ""
        return "\n".join(self.pane.cmd("capture-pane", "-p", "-J", "-S", "-").stdout)

    def terminate(self):
        """Kills the tmux session."""
        if self.tmux_session:
            try:
                self.server.kill_session(self.tmux_session.name)
            except Exception:
                pass
        self.active = False

import asyncio
import base64
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import libtmux


class ShellExecutor:
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
        # `run()` is normally entered by one asyncio loop, but tool calls can
        # also arrive from worker threads. Reserve the persistent tmux pane
        # before the first await so two new commands cannot both pass the
        # busy check and overwrite the marker context.
        self._state_lock = threading.Lock()

        self._initialize()

    def _initialize(self):
        session_name = f"ag-exec-{self.id}-{uuid.uuid4().hex[:4]}"

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

        init_cmds = [
            "exec bash --noprofile --norc",
            "export TERM=xterm",
            "stty -echo",
            "export PS1=''",
            "export PS2=''",
            "unset PROMPT_COMMAND",
            "unset PROMPT",
            "unset RPROMPT",
            # This pane is a real tty, so anything that pages on isatty(stdout)
            # (git log/show/diff, man...) reaches for one by default even when
            # a caller forgot --no-pager on one call out of many -- and a
            # pager with no one to send it a keypress can wedge the session
            # for the rest of the run. Disabling it here means a single
            # forgotten flag costs nothing instead of the whole budget.
            "export PAGER=cat",
            "export GIT_PAGER=cat",
        ]

        for cmd in init_cmds:
            self.pane.send_keys(cmd, enter=True)
            time.sleep(0.1)

        time.sleep(0.2)
        self.pane.send_keys("clear", enter=True)
        time.sleep(0.2)

        self.pane.cmd("clear-history")
        time.sleep(0.1)
        self.pane.cmd("clear-history")

        self.active = True

    @property
    def is_active(self) -> bool:
        return self.active and self.tmux_session is not None

    def _generate_marker(self) -> str:
        return f"{self.MARKER_PREFIX}{uuid.uuid4().hex[:8]}"

    def _sanitize_output(self, content: str) -> str:
        content = re.sub(rf"{re.escape(self.MARKER_PREFIX)}[^\n]*\n?", "", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()

    async def run(
        self,
        cmd: str,
        timeout: float = 30.0,
        is_input: bool = False,
        no_enter: bool = False,
    ) -> dict:
        if not self.is_active:
            return {"error": "Session inactive", "status": "error"}

        if cmd.strip() == "^C":
            self.pane.send_keys("C-c")
            await asyncio.sleep(0.5)
            with self._state_lock:
                self.busy = False
            return {
                "content": "^C (Interrupted)",
                "status": "completed",
                "exit_code": 130,
                "working_dir": self.work_dir,
                "terminal_id": self.id,
            }

        is_wait_command = not cmd.strip() or cmd.strip().startswith("#")

        with self._state_lock:
            busy = self.busy

        if busy:
            if is_wait_command and not is_input:
                return await self._wait_for_marker(timeout)
            elif not is_input:
                return {
                    "error": "Session is busy with a running command (e.g., blocking call like 'top'). "
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

        with self._state_lock:
            if self.busy:
                return {
                    "error": "Session is busy with a running command (e.g., blocking call like 'top'). "
                    "Use 'require_input=True' to interact with it, send '^C' to interrupt, "
                    "or use a different terminal_id.",
                    "status": "error",
                    "terminal_id": self.id,
                }
            self.busy = True

        return await self._execute_command(cmd, timeout)

    async def _execute_command(self, cmd: str, timeout: float) -> dict:
        try:
            if not cmd.strip():
                # Should not happen here due to check in run(), but safe fallback
                with self._state_lock:
                    self.busy = False
                return {
                    "content": self._sanitize_output(self._read_buffer()),
                    "status": "running",
                    "exit_code": None,
                    "working_dir": self.work_dir,
                    "terminal_id": self.id,
                }

            marker = self._generate_marker()
            self.current_start_marker = f"{marker}_START"
            self.current_end_marker = f"{marker}_END"

            self.pane.cmd("clear-history")
            await asyncio.sleep(0.05)

            # tmux send-keys cannot reliably queue a second Enter while an
            # external process is consuming the pane.  If the completion
            # marker is sent as a later line, commands such as nmap can finish
            # successfully while the marker is dropped, which makes callers
            # retry an already-completed command.  Encode the exact script and
            # send one shell line whose marker is sequenced by the shell after
            # the script returns.  Sourcing a temporary file preserves
            # multiline commands, heredocs, working-directory changes, and
            # shell state between terminal calls.
            encoded_command = base64.b64encode(cmd.encode("utf-8")).decode("ascii")
            script_path = f"/tmp/.ag_cmd_{marker}.sh"
            wrapped_cmd = (
                f"echo '{self.current_start_marker}'; "
                f"printf '%s' '{encoded_command}' | base64 -d > '{script_path}'; "
                f"source '{script_path}'; "
                f"__ag_exit_code=$?; rm -f '{script_path}'; "
                f"echo '{self.current_end_marker}'$__ag_exit_code"
            )
            self.pane.send_keys(wrapped_cmd, enter=True)

            return await self._wait_for_marker(timeout)

        except Exception:
            with self._state_lock:
                self.busy = False
            raise

    async def _wait_for_marker(self, timeout: float) -> dict:
        start_ts = time.time()

        start_marker = getattr(self, "current_start_marker", "")
        end_marker = getattr(self, "current_end_marker", "")

        if not start_marker or not end_marker:
            with self._state_lock:
                self.busy = False
            return {"error": "No active command context to wait for", "status": "error"}

        while (time.time() - start_ts) < timeout:
            await asyncio.sleep(0.15)
            output = self._read_buffer()

            end_pattern = rf"{re.escape(end_marker)}(\d+)"
            match = re.search(end_pattern, output)

            if match:
                exit_code = int(match.group(1))

                start_pattern = rf"{re.escape(start_marker)}\n?"
                start_match = re.search(start_pattern, output)

                if start_match:
                    content_start = start_match.end()
                    content_end = match.start()
                    content = output[content_start:content_end]

                    content = content.strip()
                    content = self._sanitize_output(content)

                    with self._state_lock:
                        self.busy = False
                    return {
                        "content": content,
                        "status": "completed",
                        "exit_code": exit_code,
                        "working_dir": self.work_dir,
                        "terminal_id": self.id,
                    }

        # We leave self.busy = True: the command is likely still running.
        return {
            "content": self._sanitize_output(self._read_buffer()),
            "status": "running",
            "exit_code": None,
            "working_dir": self.work_dir,
            "terminal_id": self.id,
        }

    def _read_buffer(self) -> str:
        if not self.pane:
            return ""
        return "\n".join(self.pane.cmd("capture-pane", "-p", "-J", "-S", "-").stdout)

    def terminate(self):
        if self.tmux_session:
            try:
                self.server.kill_session(self.tmux_session.name)
            except Exception:
                pass
        self.active = False

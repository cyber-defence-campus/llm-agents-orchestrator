from datetime import UTC, datetime
import hashlib
import json
import re
from typing import Any, ClassVar, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from agent_framework.utils.id_utils import generate_ulid


def _new_agent_id() -> str:
    return f"agent_{generate_ulid()[:12]}"


class AgentContext(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    agent_id: str = Field(default_factory=_new_agent_id)
    agent_name: str = "Assistant"
    parent_id: Optional[str] = None

    status: str = "initializing"
    task: str = ""
    short_task: Optional[str] = None
    original_task: Optional[str] = None

    iteration: int = 0
    max_iterations: int = 1000
    start_time: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    completed: bool = False
    stop_requested: bool = False
    waiting_for_input: bool = False
    waiting_since: Optional[datetime] = None
    wait_timeout: Optional[int] = None
    llm_failed: bool = False

    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context_data: Dict[str, Any] = Field(default_factory=dict, alias="context")
    tool_history: List[Dict[str, Any]] = Field(default_factory=list)
    observations: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    # Durable, bounded operational memory. Unlike the transcript, this keeps
    # exact artifacts such as a returned install command, a known URL and the
    # last terminal status across compaction.
    tactical_memory: Dict[str, Any] = Field(default_factory=dict)

    # A bounded action-health checkpoint. It is intentionally based on action
    # outcomes and exact repeats, not on target-specific strings or guessed
    # exploitation artefacts.
    STALE_ACTION_LIMIT: ClassVar[int] = 5
    # A pivot reminder is useful once, but an autonomous run must not rely on
    # the model obeying it forever. Stop after a small grace window so a
    # repeated failed avenue cannot consume the whole lease or leave a run
    # effectively hung behind tool calls.
    STALE_ACTION_STOP_LIMIT: ClassVar[int] = 4

    sandbox_id: Optional[str] = None
    sandbox_token: Optional[str] = None
    sandbox_info: Optional[Dict[str, Any]] = None

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
        tool_name = str(tool_call.get("toolName") or tool_call.get("tool_name") or "")
        args = dict(tool_call.get("args") or tool_call.get("kwargs") or {})
        args.pop("tool_call_id", None)
        args.pop("tool_result_id", None)
        checkpoint = {"tool": tool_name, "args": self._clip_memory(args)}
        self._tactical_set("last_tool_call", checkpoint)
        self._tactical_set("pending_tool", checkpoint)
        if tool_name == "terminal" and args.get("command"):
            self._remember("recent_commands", str(args["command"]), limit=8)
        self.touch()

    @staticmethod
    def _clip_memory(value: Any, limit: int = 2400) -> Any:
        try:
            encoded = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            encoded = str(value)
        if len(encoded) <= limit:
            return value
        return encoded[:limit] + "…"

    def _tactical_set(self, key: str, value: Any) -> None:
        self.tactical_memory[key] = self._clip_memory(value)

    def _remember(self, key: str, value: str, limit: int = 12) -> None:
        values = self.tactical_memory.setdefault(key, [])
        if not isinstance(values, list):
            values = []
            self.tactical_memory[key] = values
        if value in values:
            values.remove(value)
        values.append(value)
        del values[:-limit]

    def record_tool_result(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Keep a bounded, model-readable checkpoint for the next compact.

        The normal transcript stores tool_call/tool_result records, but those
        roles are intentionally omitted from the LLM history. The old
        compactor therefore saw only a lossy XML rendering and dropped the
        exact command/URL relationship that the next decision needed.
        """
        clean_args = dict(args or {})
        clean_args.pop("tool_call_id", None)
        clean_args.pop("tool_result_id", None)
        event: Dict[str, Any] = {
            "tool": str(tool_name),
            "args": self._clip_memory(clean_args),
            "ok": not is_error,
            "result": self._clip_memory(result),
        }
        self._tactical_set("last_tool_result", event)
        self.tactical_memory.pop("pending_tool", None)
        self._remember("recent_tool_names", str(tool_name), limit=12)

        failed = self._result_failed(result, is_error)
        duplicate_suppressed = (
            isinstance(result, dict) and
            bool(result.get("duplicate_suppressed")))
        action_signature = self._action_signature(tool_name, clean_args)
        evidence_signature = self._evidence_signature(tool_name, result)
        recent_evidence = self.tactical_memory.setdefault(
            "recent_evidence", [])
        if not isinstance(recent_evidence, list):
            recent_evidence = []
            self.tactical_memory["recent_evidence"] = recent_evidence
        evidence_is_new = (
            evidence_signature is not None and
            evidence_signature not in recent_evidence)
        if evidence_signature is not None:
            self._remember("recent_evidence", evidence_signature, limit=8)

        previous_streak = int(
            self.tactical_memory.get("stale_action_streak", 0) or 0)
        running_poll = (
            tool_name == "terminal" and isinstance(result, dict) and
            str(result.get("status", "")).lower() == "running")
        stale = failed or (
            evidence_signature is not None and not evidence_is_new)
        stale_streak = (
            previous_streak if running_poll else
            previous_streak + 1 if stale else 0
        )
        self._tactical_set("last_action_signature", action_signature)
        self._tactical_set("last_action_failed", failed)
        self._tactical_set("last_evidence_new", evidence_is_new)
        self._tactical_set("stale_action_streak", stale_streak)
        if stale_streak >= self.STALE_ACTION_LIMIT:
            self._tactical_set("pivot_required", True)
        elif not failed:
            self._tactical_set("pivot_required", False)
            self._tactical_set("pivot_reminder_sent", False)
        if stale_streak >= self.STALE_ACTION_STOP_LIMIT:
            self._tactical_set("stale_circuit_breaker", "repeated_failed_actions")
            self.signal_stop()
        elif failed and duplicate_suppressed:
            # The executor has already proved this is the same failed action
            # as the immediately preceding one. One such suppressed retry is
            # enough evidence that the model is looping; do not spend the
            # remaining stale-action grace window replaying it.
            self._tactical_set("stale_circuit_breaker", "duplicate_action")
            self.signal_stop()

        # Phase is a generic operational checkpoint. It is deliberately
        # inferred from interface state, not from names of files, products or
        # vulnerabilities that happen to exist in one lab.
        if not failed and tool_name == "install_beacon":
            self._tactical_set("phase", "access_pending")
        elif not failed and tool_name not in {
                "terminal", "install_beacon", "complete_assignment",
                "dispatch_agent_msg", "enter_wait_mode", "spawn_sub_agent"}:
            self._tactical_set("phase", "post_access")
        elif "phase" not in self.tactical_memory:
            self._tactical_set("phase", "discovery")

        text = json.dumps(event, ensure_ascii=False, default=str)
        self._remember_matches("known_urls", r"https?://[^\s'\"<>]+", text)
        self._remember_matches(
            "known_addresses",
            r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])",
            text,
        )
        self._remember_matches(
            "known_paths",
            r"(?<![\w])/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+",
            text,
        )
        if tool_name == "install_beacon" and isinstance(result, dict):
            if clean_args.get("address"):
                self._tactical_set(
                    "install_address", str(clean_args["address"]))
            command = result.get("command")
            if command:
                self._tactical_set("install_command", str(command))
                self._tactical_set(
                    "install_next_action",
                    "run install_command verbatim through terminal",
                )
        if tool_name == "terminal" and isinstance(result, dict):
            self._tactical_set(
                "terminal_status",
                {
                    "status": result.get("status"),
                    "exit_code": result.get("exit_code"),
                    "terminal_id": result.get("terminal_id"),
                },
            )
        self.touch()

    @staticmethod
    def _result_failed(result: Any, is_error: bool) -> bool:
        if is_error:
            return True
        if not isinstance(result, dict):
            return False
        if result.get("ok") is False or result.get("success") is False:
            return True
        return str(result.get("status", "")).lower() in {
            "error", "failed", "failure",
        }

    @staticmethod
    def _action_signature(tool_name: str, args: Dict[str, Any]) -> str:
        return json.dumps(
            {"tool": str(tool_name), "args": args},
            ensure_ascii=False, sort_keys=True, default=str,
            separators=(",", ":"),
        )

    @staticmethod
    def _evidence_signature(tool_name: str, result: Any) -> str | None:
        """Fingerprint observable evidence without target-specific parsing.

        Transport metadata changes on every terminal call and is not evidence
        of progress. The remaining payload is hashed so the checkpoint stays
        bounded even when a tool returns a large record. A running terminal
        poll is neutral: asking whether a command has finished is not a new
        hypothesis and must not consume the stale-action budget.
        """
        if (tool_name == "terminal" and isinstance(result, dict) and
                str(result.get("status", "")).lower() == "running"):
            return None
        if isinstance(result, dict):
            value = dict(result)
            for key in (
                    "ok", "success", "capability", "status", "terminal_id",
                    "exit_code", "duplicate_suppressed"):
                value.pop(key, None)
            for key in list(value):
                if key.startswith("previous_"):
                    value.pop(key, None)
            if "stdout" in value:
                value = value["stdout"]
            elif "content" in value:
                value = value["content"]
        elif result is None:
            return None
        else:
            value = result

        if isinstance(value, str):
            value = " ".join(value.split())
        try:
            encoded = json.dumps(
                value, ensure_ascii=False, sort_keys=True, default=str,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            encoded = str(value)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def consume_pivot_reminder(self) -> bool:
        """Return true once when a stale action streak needs a model nudge."""
        if (self.tactical_memory.get("pivot_required") and
                not self.tactical_memory.get("pivot_reminder_sent")):
            self._tactical_set("pivot_reminder_sent", True)
            self.touch()
            return True
        return False

    def _remember_matches(self, key: str, pattern: str, text: str) -> None:
        for match in re.findall(pattern, text):
            self._remember(key, str(match).rstrip(".,);"), limit=16)

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
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages
            if m.get("content") is not None
            and m["role"] not in ["tool_call", "tool_result", "reasoning"]
        ]

    # Compaction. The prompt a long-running agent sends grows with every
    # exchange until it sits at the model's own ceiling, where reasoning
    # degrades and the next oversized request fails outright. `_compact_memory`
    # was called every 50 iterations as an empty `pass`, so nothing ever
    # shrank: a 67-minute single-agent run rode 235k tokens of history.
    COMPACT_TRIGGER: ClassVar[int] = 60
    COMPACT_KEEP_TAIL: ClassVar[int] = 16
    COMPACT_DIGEST_BUDGET: ClassVar[int] = 6000
    COMPACT_FINDING_MARKERS: ClassVar[tuple] = (
        "password", "passwd", "credential", "token", "secret", "flag",
        ".env", "authorized_keys", "ssh", "login", "http://", "https://",
        "/root", "open port",
    )

    def compact(self) -> int:
        """Fold old exchanges into a short digest; return how many folded.

        Keeps the head (system + task) and a recent tail verbatim -- the model
        needs its own last words far more than its first ones -- and replaces
        everything between with one synthetic user note: first lines of what
        the assistant said it found, plus high-signal lines from tool output.
        Deterministic on purpose; a summarizer call can fail and compaction
        runs precisely when the session is already fragile.
        """
        msgs = self.messages
        n = len(msgs)
        if n <= self.COMPACT_TRIGGER:
            return 0

        task_idx = None
        for i, m in enumerate(msgs[:6]):
            if m.get("role") == "user" and "<task>" in str(m.get("content", "")):
                task_idx = i
                break
        head_end = (task_idx + 1) if task_idx is not None else 0
        tail_start = n - self.COMPACT_KEEP_TAIL
        # Never leave a compacted tail beginning with only a tool result or a
        # rendered observation. Keep the complete most recent action/result
        # group even when that makes the tail a few messages longer.
        if tail_start > head_end:
            tail_content = str(msgs[tail_start].get("content") or "")
            if msgs[tail_start].get("role") in ("tool_call", "tool_result"):
                while (tail_start > head_end and
                       msgs[tail_start].get("role") in ("tool_call", "tool_result")):
                    tail_start -= 1
            elif (msgs[tail_start].get("role") == "user" and
                  tail_content.startswith("Tool Results:")):
                tail_start = max(head_end, tail_start - 2)
        if tail_start <= head_end:
            return 0

        middle = msgs[head_end:tail_start]
        digest: List[str] = []
        budget = self.COMPACT_DIGEST_BUDGET
        if self.tactical_memory:
            priority = (
                "phase", "stale_action_streak", "pivot_required",
                "last_evidence_new",
                "install_address", "install_command", "install_next_action",
                "terminal_status", "known_urls", "known_addresses",
                "known_paths", "recent_commands", "last_tool_call",
                "last_tool_result",
            )
            checkpoint = {
                key: self.tactical_memory[key]
                for key in priority
                if key in self.tactical_memory
            }
            checkpoint_text = json.dumps(
                checkpoint, ensure_ascii=False, default=str,
                separators=(",", ":"))
            if len(checkpoint_text) > 5200:
                # Keep the executable artifact and the current terminal state
                # intact; trim the navigational history before trimming a
                # command the model must copy verbatim.
                checkpoint = {
                    key: self.tactical_memory[key]
                    for key in (
                        "phase", "stale_action_streak", "pivot_required",
                        "install_address", "install_command",
                        "install_next_action", "terminal_status",
                        "known_urls", "known_addresses",
                    )
                    if key in self.tactical_memory
                }
                checkpoint_text = json.dumps(
                    checkpoint, ensure_ascii=False, default=str,
                    separators=(",", ":"))
            digest.append(
                "[TACTICAL MEMORY — authoritative checkpoint] " +
                checkpoint_text
            )
        seen: set[str] = set(digest)
        for m in middle:
            role = m.get("role")
            raw_content = m.get("content")
            content = str(raw_content or "")
            if role == "assistant" and content.strip():
                line = "[assistant] " + content.strip().split("\n")[0][:220]
                if line not in seen:
                    digest.append(line)
                    seen.add(line)
            elif role == "tool_call" and isinstance(raw_content, dict):
                name = raw_content.get("tool_name", "")
                if name in ("terminal", "install_beacon"):
                    line = "[tool_call] " + json.dumps(
                        {"tool": name, "kwargs": raw_content.get("kwargs", {})},
                        ensure_ascii=False, default=str,
                    )[:700]
                    if line not in seen:
                        digest.append(line)
                        seen.add(line)
            elif role == "tool_result" and isinstance(raw_content, dict):
                line = "[tool_result] " + json.dumps(
                    raw_content, ensure_ascii=False, default=str,
                )[:1000]
                if (raw_content.get("isError") or
                        raw_content.get("tool_name") in ("terminal", "install_beacon")):
                    if line not in seen:
                        digest.append(line)
                        seen.add(line)
            elif "<tool_result>" in content:
                for line in content.splitlines():
                    low = line.lower()
                    if any(mark in low
                           for mark in self.COMPACT_FINDING_MARKERS):
                        finding = "[finding] " + line.strip()[:220]
                        if finding not in seen:
                            digest.append(finding)
                            seen.add(finding)
            used = sum(len(d) for d in digest)
            if used >= budget:
                break

        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        summary_msg = {
            "id": generate_ulid(),
            "role": "user",
            "content": (
                f"[Context compacted at iteration {self.iteration}: "
                f"{len(middle)} earlier exchanges folded into these notes]\n"
                + "\n".join(digest)
            ),
            "iteration": self.iteration,
            "timestamp": stamp,
        }
        self.messages[:] = msgs[:head_end] + [summary_msg] + msgs[tail_start:]
        return len(middle)

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4

logger = logging.getLogger("agent.telemetry")


class TelemetryEvent(TypedDict):
    id: int
    type: str
    timestamp: str
    payload: Dict[str, Any]


class TelemetryService:
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or f"run-{uuid4().hex[:8]}"
        self.start_time = datetime.now(timezone.utc).isoformat()
        self._events: List[TelemetryEvent] = []
        self._next_event_id = 1
        self._run_dir: Optional[Path] = None
        self._final_result: Optional[Dict[str, Any]] = None

    def record_event(self, event_type: str, payload: Dict[str, Any]) -> int:
        event_id = self._next_event_id
        self._next_event_id += 1

        event: TelemetryEvent = {
            "id": event_id,
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        self._events.append(event)
        return event_id

    def update_event_payload(self, event_id: int, updates: Dict[str, Any]) -> None:
        for event in self._events:
            if event["id"] == event_id:
                event["payload"].update(updates)
                return

    def set_result(self, result: Any, success: bool = True) -> None:
        self._final_result = {
            "content": result,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"Job finished. Success: {success}")

    def save(self) -> None:
        if not self._run_dir:
            self._run_dir = Path.cwd() / "logs" / "runs" / self.run_id

        try:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            report_path = self._run_dir / "report.md"

            with report_path.open("w", encoding="utf-8") as f:
                f.write(f"# Run Report: {self.run_id}\n\n")
                f.write(f"Start: {self.start_time}\n")
                if self._final_result:
                    f.write(f"End: {self._final_result['timestamp']}\n")
                    f.write(f"Success: {self._final_result['success']}\n\n")
                    f.write("## Result\n")
                    f.write(str(self._final_result["content"]))

            logger.info(f"Telemetry saved to {self._run_dir}")
        except Exception as e:
            logger.error(f"Failed to save telemetry: {e}")

    # Backward compatibility helpers / Projections
    def get_tool_executions(self, agent_id: str) -> List[Dict[str, Any]]:
        executions = []
        for e in self._events:
            if (
                e["type"] == "tool_execution"
                and e["payload"].get("agent_id") == agent_id
            ):
                executions.append(e["payload"])
        return executions


_global_service: Optional[TelemetryService] = None


def get_telemetry() -> Optional[TelemetryService]:
    return _global_service


def set_telemetry(service: TelemetryService) -> None:
    global _global_service
    _global_service = service

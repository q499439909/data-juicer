"""Normalize Data-Juicer operation events into Plan Explorer step states."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso(timestamp: Any) -> str | None:
    if not isinstance(timestamp, (int, float)):
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _value(event: dict[str, Any], name: str) -> Any:
    value = event.get(name)
    if value is not None:
        return value
    metadata = event.get("metadata")
    return metadata.get(name) if isinstance(metadata, dict) else None


def read_run_steps(run_path: Path, process: list[Any], run_status: str) -> dict[str, Any]:
    steps = []
    for index, item in enumerate(process):
        name = str(next(iter(item))) if isinstance(item, dict) and len(item) == 1 else ""
        steps.append({"process_index": index, "operator_name": name, "status": "pending"})

    diagnostics: list[str] = []
    event_count = 0
    for event_path in sorted((run_path / "work").glob("**/events_*.jsonl")):
        try:
            lines = event_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            diagnostics.append(f"Cannot read {event_path.name}: {exc}")
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                diagnostics.append(f"Ignored malformed event in {event_path.name}")
                continue
            event_type = event.get("event_type")
            if event_type not in {"op_start", "op_complete", "op_failed"}:
                continue
            index = event.get("operation_idx")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(steps):
                diagnostics.append("Ignored operation event without a valid operation_idx")
                continue
            step = steps[index]
            if event.get("operation_name") != step["operator_name"]:
                diagnostics.append(f"Ignored operation {index} because its name does not match the plan")
                continue
            event_count += 1
            timestamp = _iso(event.get("timestamp"))
            if event_type == "op_start":
                step["status"] = "running"
                if timestamp:
                    step["started_at"] = timestamp
            elif event_type == "op_complete":
                step["status"] = "succeeded"
                if timestamp:
                    step["finished_at"] = timestamp
                duration = _value(event, "duration_seconds")
                if isinstance(duration, (int, float)):
                    step["duration_ms"] = round(duration * 1000)
                for field in ("input_rows", "output_rows"):
                    value = _value(event, field)
                    if isinstance(value, int):
                        step[field] = value
            else:
                step["status"] = "failed"
                if timestamp:
                    step["finished_at"] = timestamp
                error = event.get("error_message") or _value(event, "error_message")
                if error:
                    step["error"] = str(error)

    if run_status == "cancelled":
        for step in steps:
            if step["status"] in {"pending", "running"}:
                step["status"] = "cancelled"
    elif run_status == "failed" and any(step["status"] == "failed" for step in steps):
        failed_index = min(step["process_index"] for step in steps if step["status"] == "failed")
        for step in steps:
            if step["process_index"] > failed_index and step["status"] == "pending":
                step["status"] = "skipped"

    terminal = {"succeeded", "failed", "skipped", "cancelled"}
    return {
        "steps": steps,
        "event_count": event_count,
        "mapping_complete": bool(steps) and all(step["status"] in terminal for step in steps),
        "diagnostics": diagnostics,
    }

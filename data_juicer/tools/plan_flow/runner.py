"""Approved-plan execution and run lifecycle management."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .common import (
    FileLock,
    PlanFlowError,
    now_iso,
    read_json,
    read_yaml,
    write_json_atomic,
    write_text_atomic,
    write_yaml_atomic,
)
from .execution import ExecutionBackend, LocalProcessBackend, RunHandle, RuntimeSpec
from .store import PlanStore


def _replace(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("${" + key + "}", replacement)
        return value
    if isinstance(value, list):
        return [_replace(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, variables) for key, item in value.items()}
    return value


class PlanRunner:
    def __init__(self, workspace_root: str | Path, backend: ExecutionBackend | None = None):
        self.store = PlanStore(workspace_root)
        self.backend = backend or LocalProcessBackend(self.store.workspace)

    def start(self, task_id: str, plan_version: str, *, timeout_seconds: int | None = None) -> dict[str, Any]:
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0
        ):
            raise PlanFlowError("INVALID_TIMEOUT", "timeout_seconds must be a positive integer")
        content_hash = self.store.verify_bundle(task_id, plan_version)
        plan_info = self.store.get_plan(task_id, plan_version)
        approval = plan_info.get("approval")
        if not approval or approval.get("content_hash") != content_hash:
            raise PlanFlowError("APPROVAL_REQUIRED", "Approve this exact plan version before running it")
        task_path = self.store.task_path(task_id)
        with FileLock(task_path / ".lock"):
            runs_root = task_path / "runs"
            run_id = f"run_r{self.store._next_number(runs_root, 'run_r'):03d}"
            run_path = runs_root / run_id
            run_path.mkdir(parents=True)
            task = read_yaml(task_path / "task.yaml")
            output = self.store.outputs_root / task["task_slug"] / task_id / plan_version / run_id
            output.mkdir(parents=True, exist_ok=False)
            (run_path / "logs").mkdir()
            recipe = self._materialize(plan_info["plan"]["recipe"], run_path, output)
            write_yaml_atomic(run_path / "materialized-recipe.yaml", recipe)
            created_at = datetime.now(timezone.utc)
            deadline = created_at + timedelta(seconds=timeout_seconds) if timeout_seconds else None
            state = {
                "task_id": task_id,
                "plan_version": plan_version,
                "run_id": run_id,
                "status": "starting",
                "created_at": created_at.isoformat(),
                "updated_at": now_iso(),
                "output_dir": str(output),
                "content_hash": content_hash,
            }
            write_json_atomic(run_path / "run.json", state)
            spec = RuntimeSpec(
                task_id=task_id,
                plan_version=plan_version,
                run_id=run_id,
                workspace_root=str(self.store.workspace),
                run_dir=str(run_path),
                output_dir=str(output),
                recipe_path=str(run_path / "materialized-recipe.yaml"),
                stdout_log=str(run_path / "logs" / "stdout.log"),
                stderr_log=str(run_path / "logs" / "stderr.log"),
                content_hash=content_hash,
                created_at=created_at,
                deadline=deadline,
            )
            try:
                handle = self.backend.start(spec)
            except Exception as exc:
                state.update(
                    {
                        "status": "failed",
                        "updated_at": now_iso(),
                        "error_code": getattr(exc, "code", "BACKEND_START_FAILED"),
                        "error": str(exc),
                    }
                )
                write_json_atomic(run_path / "run.json", state)
                raise
            state.update({"status": "running", "handle": handle.to_dict(), "updated_at": now_iso()})
            write_json_atomic(run_path / "run.json", state)
            write_text_atomic(run_path / ".started", "ready\n")
            current = read_json(task_path / "current.json")
            current["latest_run"] = run_id
            write_json_atomic(task_path / "current.json", current)
        return state

    @staticmethod
    def _materialize(recipe: dict[str, Any], run_path: Path, output: Path) -> dict[str, Any]:
        result = _replace(copy.deepcopy(recipe), {"RUN_OUTPUT": str(output), "RUN_DIR": str(run_path)})
        # Avoid Data-Juicer's CLI-style dataset_path parser treating Windows
        # backslashes as shell escapes. The persisted plan stays human-friendly;
        # only the executor-specific recipe uses the structured local form.
        if result.get("dataset_path"):
            result["dataset"] = {"configs": [{"type": "local", "path": result.pop("dataset_path")}]}
        result["work_dir"] = str(run_path / "work")
        result["temp_dir"] = str(run_path / "tmp")
        # Plan Explorer requires real per-operation telemetry. This only changes
        # the materialized runtime recipe, never the immutable approved plan.
        result["use_dag"] = True
        return result

    def get(self, task_id: str, run_id: str | None = None) -> dict[str, Any]:
        task_path = self.store.task_path(task_id)
        run_id = run_id or read_json(task_path / "current.json").get("latest_run")
        run_path = task_path / "runs" / str(run_id)
        if not (run_path / "run.json").is_file():
            raise PlanFlowError("RUN_NOT_FOUND", f"Unknown run: {run_id}")
        state = read_json(run_path / "run.json")
        if state.get("status") in {"starting", "running"}:
            handle = self._active_handle(state)
            observed = self.backend.inspect(handle)
            refreshed = read_json(run_path / "run.json")
            if refreshed.get("status") not in {"starting", "running"}:
                state = refreshed
            elif observed.status == "lost":
                state.update(
                    {
                        "status": "failed",
                        "updated_at": now_iso(),
                        "error_code": "RUNNER_LOST",
                        "error": observed.message or "Execution backend stopped without a final result",
                    }
                )
                write_json_atomic(run_path / "run.json", state)
            elif observed.terminal:
                result = self.backend.collect(handle)
                state.update(
                    {
                        "status": "failed" if result.status == "lost" else result.status,
                        "updated_at": now_iso(),
                    }
                )
                if result.error_code:
                    state["error_code"] = result.error_code
                if result.error:
                    state["error"] = result.error
                if result.provenance:
                    state["runtime_provenance"] = result.provenance
                write_json_atomic(run_path / "run.json", state)
        state["stdout_log"] = str(run_path / "logs" / "stdout.log")
        state["stderr_log"] = str(run_path / "logs" / "stderr.log")
        report = run_path / "report.md"
        if report.is_file():
            state["report_path"] = str(report)
        from .run_status import read_run_steps

        plan = self.store.get_plan(task_id, state["plan_version"])["plan"]
        telemetry = read_run_steps(run_path, plan.get("recipe", {}).get("process", []), state["status"])
        state["steps"] = telemetry.pop("steps")
        state["step_telemetry"] = telemetry
        return state

    def cancel(self, task_id: str, run_id: str) -> dict[str, Any]:
        state = self.get(task_id, run_id)
        if state["status"] not in {"starting", "running"}:
            return state
        handle = self._active_handle(state)
        self.backend.cancel(handle)
        run_path = self.store.task_path(task_id) / "runs" / run_id
        state.update({"status": "cancelled", "updated_at": now_iso()})
        write_json_atomic(run_path / "run.json", state)
        return state

    def cleanup(self, task_id: str, run_id: str) -> dict[str, Any]:
        state = self.get(task_id, run_id)
        if state.get("cleaned_at"):
            return state
        if state["status"] in {"starting", "running"}:
            raise PlanFlowError("RUN_ACTIVE", f"Cannot clean up an active run: {run_id}")
        raw = state.get("handle")
        if not isinstance(raw, dict):
            raise PlanFlowError("RUNNER_LOST", "Run has no backend handle for cleanup")
        self.backend.cleanup(RunHandle.from_dict(raw))
        state["cleaned_at"] = now_iso()
        run_path = self.store.task_path(task_id) / "runs" / run_id
        write_json_atomic(run_path / "run.json", state)
        return state

    def _active_handle(self, state: dict[str, Any]) -> RunHandle:
        raw = state.get("handle")
        if not isinstance(raw, dict):
            raise PlanFlowError(
                "RUNNER_LOST",
                "Active run predates the execution backend handle format and cannot be recovered",
            )
        handle = RunHandle.from_dict(raw)
        if handle.backend != self.backend.name:
            raise PlanFlowError(
                "BACKEND_MISMATCH",
                f"Run belongs to backend {handle.backend}, not {self.backend.name}",
            )
        return handle

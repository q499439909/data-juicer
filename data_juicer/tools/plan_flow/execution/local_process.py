"""Local subprocess adapter for the execution backend seam."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common import FileLock, PlanFlowError, is_within, read_json, require_workspace, write_json_atomic
from .spec import RunHandle, RunResult, RunStatus, RuntimeSpec

_BACKEND_REF = re.compile(r"[0-9a-f]{32}\Z")
_TERMINAL = {"succeeded", "failed", "cancelled"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LocalProcessBackend:
    """Run an approved RuntimeSpec as a detached local Python worker."""

    name = "local-process"

    def __init__(self, workspace_root: str | Path):
        self.workspace = require_workspace(workspace_root)
        self._state_root = self.workspace / ".dj" / "execution" / self.name

    def start(self, spec: RuntimeSpec) -> RunHandle:
        self._validate_spec_paths(spec)
        backend_ref = uuid.uuid4().hex
        record_path = self._record_path(backend_ref)
        record = {
            "schema_version": 1,
            "backend": self.name,
            "backend_ref": backend_ref,
            "run_id": spec.run_id,
            "status": "starting",
            "pid": None,
            "pid_create_time": None,
            "exit_code": None,
            "created_at": spec.created_at.isoformat(),
            "finished_at": None,
            "run_state_path": str(Path(spec.run_dir) / "run.json"),
            "stdout_log": spec.stdout_log,
            "stderr_log": spec.stderr_log,
            "runtime_spec": spec.to_dict(),
        }
        write_json_atomic(record_path, record)

        command = [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "data_juicer.tools.plan_flow.execution.local_worker",
            "--backend-record",
            str(record_path),
            spec.workspace_root,
            spec.task_id,
            spec.plan_version,
            spec.run_id,
        ]
        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[4])
        environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        stdout_path = Path(spec.stdout_log)
        stderr_path = Path(spec.stderr_log)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = stdout_path.open("ab")
        stderr = stderr_path.open("ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=spec.workspace_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=flags,
            )
        except Exception:
            record_path.unlink(missing_ok=True)
            raise
        finally:
            stdout.close()
            stderr.close()

        record.update({"status": "running", "pid": process.pid})
        try:
            import psutil

            record["pid_create_time"] = psutil.Process(process.pid).create_time()
        except Exception:
            pass
        try:
            write_json_atomic(record_path, record)
        except Exception:
            process.terminate()
            raise
        return RunHandle(
            backend=self.name,
            run_id=spec.run_id,
            created_at=spec.created_at,
            deadline=spec.deadline,
            backend_ref=backend_ref,
        )

    def inspect(self, handle: RunHandle) -> RunStatus:
        record = self._load_record(handle, missing_ok=True)
        if record is None:
            return RunStatus("lost", _now(), "Local process state is missing")
        state = self._read_run_state(record)
        if state and state.get("status") in _TERMINAL:
            return RunStatus(str(state["status"]), _now(), state.get("error"))
        if record.get("status") == "cancelled":
            return RunStatus("cancelled", _now())
        if self._same_process(record):
            return RunStatus("running", _now())
        return RunStatus("lost", _now(), "Local worker exited without a terminal run state")

    def cancel(self, handle: RunHandle) -> None:
        record = self._load_record(handle)
        if not self._same_process(record):
            raise PlanFlowError("RUNNER_LOST", f"Local worker is not running: {handle.run_id}")
        try:
            import psutil

            process = psutil.Process(int(record["pid"]))
            for child in process.children(recursive=True):
                child.terminate()
            process.terminate()
        except Exception as exc:
            raise PlanFlowError("CANCEL_FAILED", f"Could not stop run {handle.run_id}: {exc}") from exc
        record_path = self._record_path(handle.backend_ref)
        with FileLock(record_path.with_suffix(".lock")):
            current = read_json(record_path)
            current.update({"status": "cancelled", "finished_at": _now().isoformat()})
            write_json_atomic(record_path, current)

    def collect(self, handle: RunHandle) -> RunResult:
        record = self._load_record(handle, missing_ok=True)
        if record is None:
            return RunResult("lost", _now(), error_code="RUNNER_LOST", error="Local process state is missing")
        status = self.inspect(handle)
        if not status.terminal:
            raise PlanFlowError("RUN_NOT_FINISHED", f"Run is still active: {handle.run_id}")
        if record.get("exit_code") is None and status.status in {"succeeded", "failed"}:
            for _ in range(40):
                time.sleep(0.05)
                refreshed = self._load_record(handle, missing_ok=True)
                if refreshed is None:
                    break
                record = refreshed
                if record.get("exit_code") is not None:
                    break
        state = self._read_run_state(record) or {}
        error_code = state.get("error_code")
        if status.status == "lost":
            error_code = "RUNNER_LOST"
        return RunResult(
            status=status.status,
            collected_at=_now(),
            exit_code=record.get("exit_code"),
            error_code=error_code,
            error=state.get("error") or status.message,
        )

    def cleanup(self, handle: RunHandle) -> None:
        self._validate_handle(handle)
        record = self._load_record(handle, missing_ok=True)
        if record is None:
            return
        if not self.inspect(handle).terminal:
            raise PlanFlowError("RUN_ACTIVE", f"Cannot clean up an active run: {handle.run_id}")
        self._record_path(handle.backend_ref).unlink(missing_ok=True)

    def _validate_spec_paths(self, spec: RuntimeSpec) -> None:
        workspace = Path(spec.workspace_root).resolve()
        if workspace != self.workspace:
            raise PlanFlowError("BACKEND_MISMATCH", "RuntimeSpec workspace does not match LocalProcessBackend")
        for field in ("run_dir", "output_dir", "recipe_path", "stdout_log", "stderr_log"):
            path = Path(getattr(spec, field)).resolve()
            if not is_within(path, workspace):
                raise PlanFlowError("PATH_NOT_ALLOWED", f"RuntimeSpec {field} escaped workspace: {path}")

    def _record_path(self, backend_ref: str) -> Path:
        if not _BACKEND_REF.fullmatch(str(backend_ref or "")):
            raise PlanFlowError("INVALID_BACKEND_REF", "Local backend_ref is invalid")
        return self._state_root / f"{backend_ref}.json"

    def _validate_handle(self, handle: RunHandle) -> None:
        if handle.backend != self.name:
            raise PlanFlowError(
                "BACKEND_MISMATCH", f"Handle belongs to {handle.backend}, not {self.name}"
            )
        self._record_path(handle.backend_ref)

    def _load_record(self, handle: RunHandle, *, missing_ok: bool = False) -> dict[str, Any] | None:
        self._validate_handle(handle)
        path = self._record_path(handle.backend_ref)
        if not path.is_file():
            if missing_ok:
                return None
            raise PlanFlowError("RUNNER_LOST", f"Local process state is missing: {handle.run_id}")
        record = read_json(path)
        if (
            record.get("schema_version") != 1
            or record.get("backend") != self.name
            or record.get("backend_ref") != handle.backend_ref
            or record.get("run_id") != handle.run_id
        ):
            raise PlanFlowError("INVALID_BACKEND_STATE", "Local process state does not match RunHandle")
        return record

    @staticmethod
    def _read_run_state(record: dict[str, Any]) -> dict[str, Any] | None:
        path = Path(str(record.get("run_state_path") or ""))
        try:
            return read_json(path) if path.is_file() else None
        except Exception:
            return None

    @staticmethod
    def _same_process(record: dict[str, Any]) -> bool:
        try:
            import psutil

            process = psutil.Process(int(record["pid"]))
            expected = record.get("pid_create_time")
            return process.is_running() and (expected is None or abs(process.create_time() - float(expected)) < 0.01)
        except Exception:
            return False

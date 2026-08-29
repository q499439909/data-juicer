"""Runtime-neutral execution values shared across backend adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from typing import Any

from ..common import PlanFlowError

_BACKEND = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_RUN_ID = re.compile(r"run[_-][A-Za-z0-9][A-Za-z0-9._-]{0,126}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ACTIVE_STATUSES = {"starting", "running"}
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "lost"}
_STATUSES = _ACTIVE_STATUSES | _TERMINAL_STATUSES


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise PlanFlowError("INVALID_RUNTIME_SPEC", f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse_time(value: Any, field: str, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise PlanFlowError("INVALID_RUNTIME_SPEC", f"{field} must be an ISO-8601 string")
    try:
        return _utc(datetime.fromisoformat(value), field)
    except ValueError as exc:
        raise PlanFlowError("INVALID_RUNTIME_SPEC", f"{field} must be a valid ISO-8601 datetime") from exc


def _required_text(value: str, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise PlanFlowError("INVALID_RUNTIME_SPEC", f"{field} must not be empty")
    return result


@dataclass(frozen=True)
class RuntimeSpec:
    """Immutable inputs required by any execution backend for one approved run."""

    task_id: str
    plan_version: str
    run_id: str
    workspace_root: str
    run_dir: str
    output_dir: str
    recipe_path: str
    stdout_log: str
    stderr_log: str
    content_hash: str
    created_at: datetime
    deadline: datetime | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PlanFlowError("INVALID_RUNTIME_SPEC", f"Unsupported RuntimeSpec version: {self.schema_version}")
        for field in (
            "task_id",
            "plan_version",
            "workspace_root",
            "run_dir",
            "output_dir",
            "recipe_path",
            "stdout_log",
            "stderr_log",
            "content_hash",
        ):
            object.__setattr__(self, field, _required_text(getattr(self, field), field))
        if not _RUN_ID.fullmatch(str(self.run_id or "")):
            raise PlanFlowError("INVALID_RUNTIME_SPEC", f"Invalid run_id: {self.run_id}")
        if not _SHA256.fullmatch(self.content_hash):
            raise PlanFlowError("INVALID_RUNTIME_SPEC", "content_hash must be a lowercase SHA-256 value")
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        if self.deadline is not None:
            deadline = _utc(self.deadline, "deadline")
            if deadline <= self.created_at:
                raise PlanFlowError("INVALID_RUNTIME_SPEC", "deadline must be later than created_at")
            object.__setattr__(self, "deadline", deadline)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "plan_version": self.plan_version,
            "run_id": self.run_id,
            "workspace_root": self.workspace_root,
            "run_dir": self.run_dir,
            "output_dir": self.output_dir,
            "recipe_path": self.recipe_path,
            "stdout_log": self.stdout_log,
            "stderr_log": self.stderr_log,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuntimeSpec":
        if not isinstance(value, dict):
            raise PlanFlowError("INVALID_RUNTIME_SPEC", "RuntimeSpec must be an object")
        expected = {
            "schema_version",
            "task_id",
            "plan_version",
            "run_id",
            "workspace_root",
            "run_dir",
            "output_dir",
            "recipe_path",
            "stdout_log",
            "stderr_log",
            "content_hash",
            "created_at",
            "deadline",
        }
        if set(value) != expected:
            raise PlanFlowError(
                "INVALID_RUNTIME_SPEC",
                "RuntimeSpec fields do not match schema",
                details={"missing": sorted(expected - set(value)), "unknown": sorted(set(value) - expected)},
            )
        payload = dict(value)
        payload["created_at"] = _parse_time(payload["created_at"], "created_at")
        payload["deadline"] = _parse_time(payload["deadline"], "deadline", optional=True)
        return cls(**payload)


@dataclass(frozen=True)
class RunHandle:
    """Opaque routing envelope; backend-specific process identity stays private."""

    backend: str
    run_id: str
    created_at: datetime
    deadline: datetime | None
    backend_ref: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PlanFlowError("INVALID_RUN_HANDLE", f"Unsupported RunHandle version: {self.schema_version}")
        if not _BACKEND.fullmatch(str(self.backend or "")):
            raise PlanFlowError("INVALID_RUN_HANDLE", f"Invalid backend: {self.backend}")
        if not _RUN_ID.fullmatch(str(self.run_id or "")):
            raise PlanFlowError("INVALID_RUN_HANDLE", f"Invalid run_id: {self.run_id}")
        backend_ref = _required_text(self.backend_ref, "backend_ref")
        if len(backend_ref) > 256:
            raise PlanFlowError("INVALID_RUN_HANDLE", "backend_ref is too long")
        object.__setattr__(self, "backend_ref", backend_ref)
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        if self.deadline is not None:
            deadline = _utc(self.deadline, "deadline")
            if deadline <= self.created_at:
                raise PlanFlowError("INVALID_RUN_HANDLE", "deadline must be later than created_at")
            object.__setattr__(self, "deadline", deadline)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "backend_ref": self.backend_ref,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunHandle":
        if not isinstance(value, dict):
            raise PlanFlowError("INVALID_RUN_HANDLE", "RunHandle must be an object")
        expected = {"schema_version", "backend", "run_id", "created_at", "deadline", "backend_ref"}
        if set(value) != expected:
            raise PlanFlowError(
                "INVALID_RUN_HANDLE",
                "RunHandle fields do not match schema",
                details={"missing": sorted(expected - set(value)), "unknown": sorted(set(value) - expected)},
            )
        payload = dict(value)
        payload["created_at"] = _parse_time(payload["created_at"], "created_at")
        payload["deadline"] = _parse_time(payload["deadline"], "deadline", optional=True)
        return cls(**payload)


@dataclass(frozen=True)
class RunStatus:
    status: str
    observed_at: datetime
    message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise PlanFlowError("INVALID_RUN_STATUS", f"Unsupported run status: {self.status}")
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@dataclass(frozen=True)
class RunResult:
    status: str
    collected_at: datetime
    exit_code: int | None = None
    error_code: str | None = None
    error: str | None = None
    provenance: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _TERMINAL_STATUSES:
            raise PlanFlowError("INVALID_RUN_RESULT", f"RunResult must be terminal: {self.status}")
        object.__setattr__(self, "collected_at", _utc(self.collected_at, "collected_at"))
        if not isinstance(self.provenance, dict):
            raise PlanFlowError("INVALID_RUN_RESULT", "RunResult provenance must be an object")
        object.__setattr__(self, "provenance", dict(self.provenance))

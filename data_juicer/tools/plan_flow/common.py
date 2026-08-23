"""Shared filesystem, hashing, and error helpers for plan-flow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class PlanFlowError(ValueError):
    """Caller-visible failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str, *, details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload = {"ok": False, "error": {"code": self.code, "message": self.message}}
        if self.details is not None:
            payload["error"]["details"] = self.details
        return payload


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def require_workspace(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise PlanFlowError(
            "WORKSPACE_NOT_ABSOLUTE",
            "workspace_root must be the absolute workspace selected in DSH; it is never inferred from the MCP server cwd",
        )
    workspace = resolve_path(path)
    if not workspace.is_dir():
        raise PlanFlowError("WORKSPACE_NOT_FOUND", f"Workspace is not a directory: {workspace}")
    if not os.access(workspace, os.R_OK | os.W_OK):
        raise PlanFlowError("WORKSPACE_NOT_WRITABLE", f"Workspace is not readable and writable: {workspace}")
    return workspace


def resolve_workspace_path(path: str | Path, workspace: Path) -> Path:
    """Resolve a caller path relative to its declared workspace, never server cwd."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return candidate.resolve()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def require_within(path: str | Path, root: Path, *, code: str = "PATH_NOT_ALLOWED") -> Path:
    resolved = resolve_workspace_path(path, root)
    if not is_within(resolved, root):
        raise PlanFlowError(code, f"Path must be inside {root}: {resolved}")
    return resolved


def slugify(value: str) -> str:
    value = str(value or "").strip().lower()
    ascii_value = value.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
    return (slug or "data-task")[:48]


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_yaml_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PlanFlowError("NOT_FOUND", f"File does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PlanFlowError("INVALID_STATE", f"Expected an object in {path}")
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PlanFlowError("NOT_FOUND", f"File does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PlanFlowError("INVALID_STATE", f"Expected a mapping in {path}")
    return value


class FileLock(AbstractContextManager):
    """Small cross-process lock with stale-owner recovery."""

    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, canonical_json({"pid": os.getpid(), "created_at": now_iso()}))
                return self
            except FileExistsError:
                try:
                    owner = read_json(self.path)
                    pid = int(owner.get("pid", 0))
                    if pid and _pid_exists(pid):
                        raise PlanFlowError("TASK_BUSY", f"Task is being modified: {self.path.parent.name}")
                    self.path.unlink(missing_ok=True)
                except PlanFlowError:
                    raise
                except Exception:
                    self.path.unlink(missing_ok=True)
        raise PlanFlowError("TASK_BUSY", f"Could not acquire task lock: {self.path}")

    def __exit__(self, exc_type, exc, traceback):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)
        return False


def _pid_exists(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

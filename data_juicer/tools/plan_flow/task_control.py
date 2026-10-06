"""Trusted workspace and task ownership; approval authority stays outside tools."""

from __future__ import annotations

import os
import sqlite3
from contextvars import ContextVar
from pathlib import Path

from .common import PlanFlowError
from .user_operator_store import current_user

trusted_decision: ContextVar[str | None] = ContextVar("trusted_decision", default=None)


def controlled():
    if os.environ.get("DJ_PLAN_FLOW_SINGLE_USER"):
        return False
    return bool(os.environ.get("DSH_DJ_INTERNAL_TOKEN") or current_user.get())


def connection():
    root = os.environ.get("DSH_USER_DATA_ROOT")
    if not root:
        raise PlanFlowError("CONTROL_NOT_CONFIGURED", "DSH_USER_DATA_ROOT is required")
    path = Path(root).resolve().parent / "dj-control.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("CREATE TABLE IF NOT EXISTS workspaces (path TEXT PRIMARY KEY, owner TEXT NOT NULL)")
    return db


def workspace_key(path):
    return os.path.normcase(str(Path(path).resolve()))


def register_workspace(path, owner):
    """Explicit local administration only; never an agent tool."""
    key = workspace_key(path)
    allowed = {workspace_key(p) for p in os.environ.get("DSH_DJ_ALLOWED_WORKSPACES", "").split(os.pathsep) if p}
    if key not in allowed:
        raise PlanFlowError("WORKSPACE_NOT_REGISTERED", "Administrator must allow this workspace before use")
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        for other, account in db.execute("SELECT path, owner FROM workspaces"):
            if account != owner and (Path(key).is_relative_to(Path(other)) or Path(other).is_relative_to(Path(key))):
                raise PlanFlowError("WORKSPACE_FORBIDDEN", "Overlapping workspaces cannot have different owners")
        db.execute("INSERT OR IGNORE INTO workspaces(path,owner) VALUES (?,?)", (key, owner))
        if db.execute("SELECT owner FROM workspaces WHERE path=?", (key,)).fetchone()[0] != owner:
            raise PlanFlowError("WORKSPACE_FORBIDDEN", "Workspace belongs to another account")
    return {"ok": True, "workspace_root": str(Path(path).resolve())}


def authorize_workspace(path):
    if not controlled():
        return
    owner = current_user.get()
    if not owner:
        raise PlanFlowError("ACCOUNT_REQUIRED", "Authenticated account context is required")
    with connection() as db:
        row = db.execute("SELECT owner FROM workspaces WHERE path=?", (workspace_key(path),)).fetchone()
    if not row or row[0] != owner:
        raise PlanFlowError("WORKSPACE_FORBIDDEN", "Workspace is not granted to this account")


def authorize_task(store, task_id):
    from .common import read_yaml

    authorize_workspace(store.workspace)
    if controlled():
        task = read_yaml(store.task_path(task_id) / "task.yaml")
        if task.get("owner") != current_user.get():
            raise PlanFlowError(
                "TASK_FORBIDDEN",
                "Task owner is missing or belongs to another account; migrate legacy ownership explicitly",
            )


def list_owned_runs(offset=0, limit=100):
    from .common import read_json, read_yaml

    user = current_user.get()
    if not user:
        raise PlanFlowError("ACCOUNT_REQUIRED", "Authenticated account required")
    with connection() as db:
        roots = [Path(row[0]) for row in db.execute("SELECT path FROM workspaces WHERE owner=? ORDER BY path", (user,))]
    items = []
    for root in roots:
        for task in sorted((root / ".dj" / "tasks").glob("task_*")):
            if not (task / "task.yaml").is_file() or read_yaml(task / "task.yaml").get("owner") != user:
                continue
            for run_path in sorted((task / "runs").glob("run_*/run.json")):
                state = read_json(run_path)
                items.append(
                    {
                        "workspaceRoot": str(root),
                        "taskId": task.name,
                        "planVersion": state["plan_version"],
                        "resultRef": state["run_id"],
                        "status": state["status"],
                    }
                )
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    return {
        "ok": True,
        "items": items[offset : offset + limit],
        "nextCursor": str(offset + limit) if offset + limit < len(items) else None,
    }

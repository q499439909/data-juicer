"""Immutable task/plan/run persistence for plan-flow."""

from __future__ import annotations

import copy
import shutil
import uuid
from pathlib import Path
from typing import Any

from .common import (
    FileLock,
    PlanFlowError,
    canonical_json,
    is_within,
    now_iso,
    read_json,
    read_yaml,
    require_workspace,
    resolve_workspace_path,
    sha256_bytes,
    sha256_file,
    slugify,
    write_json_atomic,
    write_text_atomic,
    write_yaml_atomic,
)


class PlanStore:
    """Owns immutable version allocation and artifact snapshots."""

    def __init__(self, workspace_root: str | Path):
        self.workspace = require_workspace(workspace_root)
        self.root = self.workspace / ".dj"
        self.tasks_root = self.root / "tasks"
        self.outputs_root = self.workspace / "outputs"

    def create_task(self, title: str, task_id: str | None = None) -> tuple[str, Path]:
        task_id = str(task_id or "").strip() or f"task_{uuid.uuid4().hex[:12]}"
        if not task_id.startswith("task_") or not task_id.replace("_", "").isalnum():
            raise PlanFlowError("INVALID_TASK_ID", f"Invalid task_id: {task_id}")
        task_path = self.tasks_root / task_id
        if task_path.exists():
            return task_id, task_path
        task_path.mkdir(parents=True, exist_ok=False)
        write_yaml_atomic(
            task_path / "task.yaml",
            {
                "schema_version": 1,
                "task_id": task_id,
                "title": str(title or "Data processing task").strip(),
                "task_slug": slugify(title),
                "workspace_root": str(self.workspace),
                "created_at": now_iso(),
            },
        )
        write_json_atomic(task_path / "current.json", {"latest_plan": None, "approved_plan": None, "latest_run": None})
        return task_id, task_path

    def task_path(self, task_id: str) -> Path:
        if not str(task_id).startswith("task_") or not str(task_id).replace("_", "").isalnum():
            raise PlanFlowError("INVALID_TASK_ID", f"Invalid task_id: {task_id}")
        path = self.tasks_root / task_id
        if not (path / "task.yaml").is_file():
            raise PlanFlowError("TASK_NOT_FOUND", f"Unknown task_id: {task_id}")
        return path

    @staticmethod
    def _next_number(parent: Path, prefix: str) -> int:
        highest = 0
        if parent.is_dir():
            for child in parent.iterdir():
                if child.is_dir() and child.name.startswith(prefix):
                    suffix = child.name[len(prefix) :]
                    if suffix.isdigit():
                        highest = max(highest, int(suffix))
        return highest + 1

    def save_plan(
        self,
        *,
        task_id: str,
        plan: dict[str, Any],
        validation: dict[str, Any],
        artifact_paths: list[str],
        base_plan_version: str | None,
    ) -> dict[str, Any]:
        task_path = self.task_path(task_id)
        with FileLock(task_path / ".lock"):
            plans_root = task_path / "plans"
            number = self._next_number(plans_root, "plan_v")
            version = f"plan_v{number:03d}"
            version_path = plans_root / version
            version_path.mkdir(parents=True, exist_ok=False)

            saved_plan = copy.deepcopy(plan)
            saved_plan.update(
                {
                    "schema_version": 1,
                    "task_id": task_id,
                    "plan_id": f"{task_id}/{version}",
                    "plan_version": version,
                    "based_on": base_plan_version,
                    "created_at": now_iso(),
                }
            )
            saved_plan.pop("status", None)
            saved_plan.pop("content_hash", None)
            copied = self._copy_artifacts(artifact_paths, version_path / "artifacts")
            saved_plan["artifacts"] = copied
            self._rewrite_postprocess_artifacts(saved_plan, copied)

            content_hash = self._bundle_hash(saved_plan, version_path, copied)
            write_yaml_atomic(version_path / "plan.yaml", saved_plan)
            write_json_atomic(version_path / "validation.json", validation)
            write_text_atomic(version_path / "content-hash.txt", content_hash + "\n")

            diff = self._diff_from_base(task_path, base_plan_version, saved_plan)
            write_json_atomic(version_path / "diff.json", {"changes": diff})
            current = read_json(task_path / "current.json")
            current["latest_plan"] = version
            write_json_atomic(task_path / "current.json", current)
        return {
            "task_id": task_id,
            "plan_version": version,
            "plan_path": str(version_path / "plan.yaml"),
            "content_hash": content_hash,
            "changes": diff,
            "valid": bool(validation.get("ok")),
        }

    def _copy_artifacts(self, artifact_paths: list[str], destination: Path) -> list[dict[str, Any]]:
        copied: list[dict[str, Any]] = []
        names: set[str] = set()
        for raw in artifact_paths:
            source = resolve_workspace_path(raw, self.workspace)
            if not is_within(source, self.workspace):
                raise PlanFlowError("PATH_NOT_ALLOWED", f"Artifact must be inside workspace: {source}")
            if not source.is_file():
                raise PlanFlowError("ARTIFACT_NOT_FOUND", f"Artifact does not exist: {source}")
            if source.name in names:
                raise PlanFlowError("ARTIFACT_NAME_CONFLICT", f"Duplicate artifact name: {source.name}")
            names.add(source.name)
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / source.name
            shutil.copy2(source, target)
            copied.append({"path": f"artifacts/{source.name}", "sha256": sha256_file(target)})
        return copied

    @staticmethod
    def _rewrite_postprocess_artifacts(plan: dict[str, Any], artifacts: list[dict[str, Any]]) -> None:
        by_name = {Path(item["path"]).name: item["path"] for item in artifacts}
        for step in plan.get("postprocess", []) or []:
            if not isinstance(step, dict) or not step.get("script"):
                continue
            name = Path(str(step["script"])).name
            if name in by_name:
                step["script"] = by_name[name]

    @staticmethod
    def _bundle_hash(plan: dict[str, Any], version_path: Path, artifacts: list[dict[str, Any]]) -> str:
        digest_parts = [canonical_json(plan)]
        for artifact in sorted(artifacts, key=lambda item: item["path"]):
            digest_parts.append((version_path / artifact["path"]).read_bytes())
        return sha256_bytes(b"\0".join(digest_parts))

    @staticmethod
    def _diff_from_base(task_path: Path, base: str | None, current: dict[str, Any]) -> list[dict[str, Any]]:
        if not base:
            return []
        base_path = task_path / "plans" / base / "plan.yaml"
        previous = read_yaml(base_path)
        ignored = {"plan_id", "plan_version", "based_on", "created_at", "task_id"}
        changes: list[dict[str, Any]] = []

        def walk(before: Any, after: Any, path: str) -> None:
            if isinstance(before, dict) and isinstance(after, dict):
                for key in sorted(set(before) | set(after)):
                    if key in ignored:
                        continue
                    walk(before.get(key), after.get(key), f"{path}.{key}" if path else key)
            elif isinstance(before, list) and isinstance(after, list):
                for index in range(max(len(before), len(after))):
                    old = before[index] if index < len(before) else None
                    new = after[index] if index < len(after) else None
                    walk(old, new, f"{path}[{index}]")
            elif before != after:
                changes.append({"path": path, "before": before, "after": after})

        walk(previous, current, "")
        return changes

    def plan_path(self, task_id: str, version: str | None = None) -> Path:
        task_path = self.task_path(task_id)
        if version is None:
            version = read_json(task_path / "current.json").get("latest_plan")
        if not version or not str(version).startswith("plan_v") or not str(version)[6:].isdigit():
            raise PlanFlowError("PLAN_NOT_FOUND", f"Invalid or missing plan version: {version}")
        path = task_path / "plans" / str(version)
        if not (path / "plan.yaml").is_file():
            raise PlanFlowError("PLAN_NOT_FOUND", f"Unknown plan version: {version}")
        return path

    def get_plan(self, task_id: str, version: str | None = None) -> dict[str, Any]:
        path = self.plan_path(task_id, version)
        approval = read_json(path / "approval.json") if (path / "approval.json").is_file() else None
        return {
            "plan": read_yaml(path / "plan.yaml"),
            "validation": read_json(path / "validation.json"),
            "changes": read_json(path / "diff.json").get("changes", []),
            "content_hash": (path / "content-hash.txt").read_text(encoding="utf-8").strip(),
            "approval": approval,
            "status": "approved" if approval else "proposed",
        }

    def list_plans(self, task_id: str) -> list[dict[str, Any]]:
        task_path = self.task_path(task_id)
        result = []
        for path in sorted((task_path / "plans").glob("plan_v*")):
            if not (path / "plan.yaml").is_file():
                continue
            plan = read_yaml(path / "plan.yaml")
            result.append(
                {
                    "plan_version": plan["plan_version"],
                    "based_on": plan.get("based_on"),
                    "created_at": plan.get("created_at"),
                    "valid": read_json(path / "validation.json").get("ok", False),
                    "approved": (path / "approval.json").is_file(),
                }
            )
        return result

    def approve(self, task_id: str, version: str, expected_hash: str, note: str) -> dict[str, Any]:
        task_path = self.task_path(task_id)
        with FileLock(task_path / ".lock"):
            plan_path = self.plan_path(task_id, version)
            self.verify_bundle(task_id, version)
            validation = read_json(plan_path / "validation.json")
            if not validation.get("ok"):
                raise PlanFlowError(
                    "PLAN_INVALID", "Only a valid plan can be approved", details=validation.get("errors")
                )
            actual = (plan_path / "content-hash.txt").read_text(encoding="utf-8").strip()
            if expected_hash != actual:
                raise PlanFlowError("CONTENT_CHANGED", "The plan content does not match the version shown to the user")
            approval_path = plan_path / "approval.json"
            if approval_path.exists():
                approval = read_json(approval_path)
                if approval.get("content_hash") != actual:
                    raise PlanFlowError("APPROVAL_CONFLICT", "Existing approval references different content")
                return approval
            approval = {
                "task_id": task_id,
                "plan_version": version,
                "content_hash": actual,
                "approved_at": now_iso(),
                "note": note or "",
            }
            write_json_atomic(approval_path, approval)
            current = read_json(task_path / "current.json")
            current["approved_plan"] = version
            write_json_atomic(task_path / "current.json", current)
            return approval

    def verify_bundle(self, task_id: str, version: str) -> str:
        """Verify that an immutable plan and its copied artifacts were not edited."""
        path = self.plan_path(task_id, version)
        plan = read_yaml(path / "plan.yaml")
        artifacts = plan.get("artifacts", []) or []
        for artifact in artifacts:
            artifact_path = path / str(artifact.get("path", ""))
            if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.get("sha256"):
                raise PlanFlowError("PLAN_TAMPERED", f"Plan artifact changed after preparation: {artifact_path}")
        actual = self._bundle_hash(plan, path, artifacts)
        expected = (path / "content-hash.txt").read_text(encoding="utf-8").strip()
        if actual != expected:
            raise PlanFlowError("PLAN_TAMPERED", "Plan content changed after preparation")
        return actual

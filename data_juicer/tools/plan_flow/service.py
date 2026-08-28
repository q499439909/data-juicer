"""Small application service used by MCP tools and ordinary Python callers."""

from __future__ import annotations

from typing import Any

from .discovery import capability_schemas as load_capability_schemas
from .discovery import inspect_input as inspect_local_input
from .discovery import runtime_capabilities
from .discovery import search_capabilities as discover_capabilities
from .runner import PlanRunner
from .store import PlanStore
from .validation import normalize_and_validate


class PlanFlowService:
    """Coordinates validation, immutable persistence, approval, and runs."""

    def inspect_input(self, workspace_root: str, input: dict[str, Any], sample_size: int = 20) -> dict[str, Any]:
        return inspect_local_input(workspace_root, input, sample_size)

    def search_capabilities(
        self, requirements: list[str], modality: str | None = None, executor_type: str = "default", top_k: int = 3
    ) -> dict[str, Any]:
        return discover_capabilities(requirements, modality, executor_type, top_k)

    def get_capability_schemas(self, operator_names: list[str]) -> dict[str, Any]:
        return load_capability_schemas(operator_names)

    def prepare_plan(
        self,
        workspace_root: str,
        plan: dict[str, Any],
        task_id: str | None = None,
        base_plan_version: str | None = None,
    ) -> dict[str, Any]:
        store = PlanStore(workspace_root)
        if task_id is None:
            task_id, _ = store.create_task(str(plan.get("user_intent", "Data processing task")))
        else:
            store.task_path(task_id)
        if base_plan_version:
            store.plan_path(task_id, base_plan_version)
        normalized, validation, artifacts = normalize_and_validate(str(store.workspace), plan)
        saved = store.save_plan(
            task_id=task_id,
            plan=normalized,
            validation=validation,
            artifact_paths=artifacts,
            base_plan_version=base_plan_version,
        )
        return {
            "ok": True,
            "workspace_root": str(store.workspace),
            **saved,
            "validation": validation,
            "runtime": runtime_capabilities(),
            "plan": store.get_plan(task_id, saved["plan_version"])["plan"],
        }

    def get_plan(
        self, workspace_root: str, task_id: str, plan_version: str | None = None, include_versions: bool = False
    ) -> dict[str, Any]:
        store = PlanStore(workspace_root)
        result = {
            "ok": True,
            "workspace_root": str(store.workspace),
            "task_id": task_id,
            **store.get_plan(task_id, plan_version),
        }
        if include_versions:
            result["versions"] = store.list_plans(task_id)
        return result

    def approve_plan(
        self, workspace_root: str, task_id: str, plan_version: str, content_hash: str, note: str = ""
    ) -> dict[str, Any]:
        store = PlanStore(workspace_root)
        return {
            "ok": True,
            "workspace_root": str(store.workspace),
            "approval": store.approve(task_id, plan_version, content_hash, note),
        }

    def run_plan(self, workspace_root: str, task_id: str, plan_version: str) -> dict[str, Any]:
        runner = PlanRunner(workspace_root)
        return {"ok": True, "workspace_root": str(runner.store.workspace), "run": runner.start(task_id, plan_version)}

    def get_run(self, workspace_root: str, task_id: str, run_id: str | None = None) -> dict[str, Any]:
        runner = PlanRunner(workspace_root)
        return {"ok": True, "workspace_root": str(runner.store.workspace), "run": runner.get(task_id, run_id)}

    def cancel_run(self, workspace_root: str, task_id: str, run_id: str) -> dict[str, Any]:
        runner = PlanRunner(workspace_root)
        return {"ok": True, "workspace_root": str(runner.store.workspace), "run": runner.cancel(task_id, run_id)}

    def preview_plan(self, workspace_root: str, task_id: str, plan_version: str) -> dict[str, Any]:
        """Return a safe preflight preview; it deliberately does not execute an unapproved plan."""
        store = PlanStore(workspace_root)
        info = store.get_plan(task_id, plan_version)
        recipe = info["plan"]["recipe"]
        return {
            "ok": True,
            "workspace_root": str(store.workspace),
            "task_id": task_id,
            "plan_version": plan_version,
            "validation": info["validation"],
            "content_hash": info["content_hash"],
            "execution_preview": {
                "input": recipe.get("dataset_path") or recipe.get("dataset") or recipe.get("generated_dataset_config"),
                "dj_operators": [next(iter(step)) for step in recipe.get("process", [])],
                "postprocess": info["plan"].get("postprocess", []),
                "executor_type": recipe.get("executor_type", "default"),
                "np": recipe.get("np", 1),
                "output_template": recipe.get("export_path"),
            },
        }

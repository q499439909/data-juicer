"""Small application service used by MCP tools and ordinary Python callers."""

from __future__ import annotations

from urllib.parse import urlparse
from typing import Any

import httpx

from .discovery import capability_schemas as load_capability_schemas
from .discovery import inspect_input as inspect_local_input
from .discovery import operator_catalog as load_operator_catalog
from .discovery import operator_detail as load_operator_detail
from .discovery import search_capabilities as discover_capabilities
from .runner import PlanRunner
from .runtime_preflight import RuntimePreflight
from .store import PlanStore
from .validation import normalize_and_validate
from .common import PlanFlowError
from . import operator_catalog_service as unified_catalog
from .user_operator_store import UserOperatorStore, current_user, resolve_bindings
from .capability_schema import CapabilityCatalog, CapabilityDescriptor, OperatorArtifact


class BrokerHttpClient:
    """Small loopback-only client for the execution broker public interface."""

    def __init__(self, base_url: str):
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise PlanFlowError("BROKER_LOOPBACK_REQUIRED", "Execution broker URL must be loopback HTTP")
        self.base_url = base_url.rstrip("/")

    def start(self, *, task_id: str, plan_version: str, runtime_id: str, profile: str) -> dict[str, Any]:
        return self._request("POST", "/v1/runs", {"task_id": task_id, "plan_version": plan_version, "runtime_id": runtime_id, "profile": profile})

    def get(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/runs/{run_id}")

    def cancel(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/runs/{run_id}:cancel")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = httpx.request(method, self.base_url + path, json=payload, timeout=30)
            data = response.json()
        except Exception as exc:
            raise PlanFlowError("BROKER_UNAVAILABLE", "Local execution broker is unavailable") from exc
        if response.status_code >= 400:
            error = data.get("error", {}) if isinstance(data, dict) else {}
            raise PlanFlowError(str(error.get("code") or "BROKER_REJECTED"), str(error.get("message") or "Broker rejected request"))
        return data


class PlanFlowService:
    """Coordinates validation, immutable persistence, approval, and runs."""

    _EXECUTION_MODES = frozenset({"broker", "native"})

    def __init__(
        self,
        *,
        runtime_resolver=None,
        broker_client=None,
        capability_lifecycle=None,
        worker_root=None,
        execution_mode: str = "broker",
        runtime_preflight: RuntimePreflight | None = None,
    ):
        normalized_mode = str(execution_mode).strip().casefold()
        if normalized_mode not in self._EXECUTION_MODES:
            choices = ", ".join(sorted(self._EXECUTION_MODES))
            raise ValueError(f"Unsupported plan-flow execution mode {execution_mode!r}; expected one of: {choices}")
        self.runtime_resolver = runtime_resolver
        self.broker_client = broker_client
        self.capability_lifecycle = capability_lifecycle
        self.capability_catalog = CapabilityCatalog(worker_root) if worker_root is not None else None
        self.execution_mode = normalized_mode
        self.runtime_preflight = runtime_preflight or RuntimePreflight()

    @classmethod
    def native(cls, **kwargs) -> "PlanFlowService":
        """Use the approved-plan runner with its constrained local-process adapter."""
        return cls(execution_mode="native", **kwargs)

    @classmethod
    def local_for_tests(cls) -> "PlanFlowService":
        """Compatibility factory for tests that exercise the native adapter."""
        return cls.native()

    def inspect_input(self, workspace_root: str, input: dict[str, Any], sample_size: int = 20) -> dict[str, Any]:
        return inspect_local_input(workspace_root, input, sample_size)

    def operator_catalog(self) -> dict[str, Any]:
        return unified_catalog.catalog()

    def operator_detail(self, name: str) -> dict[str, Any]:
        return unified_catalog.detail(name)

    def search_capabilities(
        self, requirements: list[str], modality: str | None = None, executor_type: str = "default", top_k: int = 3
    ) -> dict[str, Any]:
        return unified_catalog.search(requirements, modality, executor_type, top_k)

    def get_capability_schemas(self, operator_names: list[str]) -> dict[str, Any]:
        return unified_catalog.schemas(operator_names)

    def resolve_capabilities(self, requirements: list[str]) -> dict[str, Any]:
        if self.capability_catalog is None:
            raise PlanFlowError("CAPABILITY_CONTROL_NOT_CONFIGURED", "Capability catalog is not configured")
        required = {str(item).casefold() for item in requirements if str(item).strip()}
        matches = []
        for path in sorted(self.capability_catalog.root.glob("*.json")):
            descriptor = self.capability_catalog.resolve(path.stem)
            contracts = {item.casefold() for item in descriptor.implements}
            if not required or any(any(term in contract for contract in contracts) for term in required):
                matches.append(descriptor.to_dict())
        return {"ok": True, "requirements": requirements, "capabilities": matches}

    def prepare_capability(self, capability: dict[str, Any], operator_artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        if self.capability_lifecycle is None:
            raise PlanFlowError("CAPABILITY_CONTROL_NOT_CONFIGURED", "Capability lifecycle is not configured")
        descriptor = CapabilityDescriptor.from_dict(capability)
        artifacts = tuple(OperatorArtifact.from_dict(item) for item in operator_artifacts)
        proposal = self.capability_lifecycle.prepare(descriptor, artifacts)
        if proposal.status != "available":
            proposal = self.capability_lifecycle.build_and_validate(proposal.proposal_id)
        return {"ok": True, "proposal": self._proposal_dict(proposal)}

    def get_capability(self, capability_id: str) -> dict[str, Any]:
        if self.capability_catalog is None:
            raise PlanFlowError("CAPABILITY_CONTROL_NOT_CONFIGURED", "Capability catalog is not configured")
        return {"ok": True, "capability": self.capability_catalog.resolve(capability_id).to_dict()}

    def approve_capability(self, proposal_id: str, content_hash: str, note: str = "") -> dict[str, Any]:
        if self.capability_lifecycle is None:
            raise PlanFlowError("CAPABILITY_CONTROL_NOT_CONFIGURED", "Capability lifecycle is not configured")
        approved = self.capability_lifecycle.approve(proposal_id, content_hash, note=note)
        available = self.capability_lifecycle.publish(proposal_id)
        return {
            "ok": True,
            "approval": self._proposal_dict(approved),
            "capability": self.capability_catalog.resolve(available.capability_id).to_dict(),
        }

    @staticmethod
    def _proposal_dict(proposal) -> dict[str, Any]:
        return {
            "proposal_id": proposal.proposal_id,
            "capability_id": proposal.capability_id,
            "content_hash": proposal.content_hash,
            "status": proposal.status,
            "reused": proposal.reused,
        }

    def prepare_plan(
        self,
        workspace_root: str,
        plan: dict[str, Any],
        task_id: str | None = None,
        base_plan_version: str | None = None,
        view_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        store = PlanStore(workspace_root)
        if task_id is None:
            task_id, _ = store.create_task(str(plan.get("user_intent", "Data processing task")))
        else:
            store.task_path(task_id)
            if store.list_plans(task_id):
                self._authorize_personal_plan(store.get_plan(task_id)["plan"])
        if base_plan_version:
            store.plan_path(task_id, base_plan_version)
        import copy
        plan = copy.deepcopy(plan)
        personal = resolve_bindings(plan)
        if personal:
            plan["operator_owner"] = UserOperatorStore().user_id
            plan["operator_validation"] = {name: item["validation_summary"] for name, item in personal.items()}
            plan["risk_notes"] = [*plan.get("risk_notes", []), *[
                f"Experimental personal operator {name}: task quality is not established"
                for name, item in personal.items() if item["status"] == "experimental"
            ]]
        elif plan.get("operator_owner"):
            raise PlanFlowError("INVALID_OPERATOR_OWNER", "operator_owner is server-controlled")
        bindings = plan.get("capability_bindings", []) if isinstance(plan, dict) else []
        if bindings and self.execution_mode == "native":
            raise PlanFlowError("UNVERIFIED_EXTERNAL_OPERATORS", "Native personal operators require verified operator_bindings")
        external_names = frozenset(
            str(name)
            for binding in bindings
            if isinstance(binding, dict)
            for name in binding.get("operators", [])
        )
        normalized, validation, artifacts = normalize_and_validate(
            str(store.workspace), plan, external_operator_names=external_names, operator_schemas=personal
        )
        saved = store.save_plan(
            task_id=task_id,
            plan=normalized,
            validation=validation,
            artifact_paths=artifacts,
            base_plan_version=base_plan_version,
            view_spec=view_spec,
        )
        stored = store.get_plan(task_id, saved["plan_version"])
        runtime_assessment = self._runtime_assessment(stored["plan"])
        return {
            "ok": True,
            "workspace_root": str(store.workspace),
            **saved,
            "validation": validation,
            "plan": stored["plan"],
            "view": stored["view"],
            "execution_preview": self._execution_preview(stored["plan"]),
            "runtime_assessment": runtime_assessment,
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
        self._authorize_personal_plan(result["plan"])
        result["execution_preview"] = self._execution_preview(result["plan"])
        result["runtime_assessment"] = self._runtime_assessment(result["plan"])
        return result

    def approve_plan(
        self, workspace_root: str, task_id: str, plan_version: str, content_hash: str, note: str = ""
    ) -> dict[str, Any]:
        store = PlanStore(workspace_root)
        self._authorize_personal_plan(store.get_plan(task_id, plan_version)["plan"])
        return {
            "ok": True,
            "workspace_root": str(store.workspace),
            "approval": store.approve(task_id, plan_version, content_hash, note),
        }

    def run_plan(self, workspace_root: str, task_id: str, plan_version: str) -> dict[str, Any]:
        store = PlanStore(workspace_root)
        content_hash = store.verify_bundle(task_id, plan_version)
        info = store.get_plan(task_id, plan_version)
        self._authorize_personal_plan(info["plan"])
        if not info.get("approval") or info["approval"].get("content_hash") != content_hash:
            raise PlanFlowError("APPROVAL_REQUIRED", "Approve this exact plan version before running it")
        runtime_assessment = self._runtime_assessment(info["plan"])
        hardware_blockers = [
            item for item in runtime_assessment["blocking_issues"] if item.get("code") == "GPU_REQUIRED"
        ]
        if hardware_blockers:
            raise PlanFlowError(
                "RUNTIME_PREFLIGHT_FAILED",
                "Runtime preflight found blocking hardware requirements",
                details=hardware_blockers,
            )
        if self.execution_mode == "native":
            runner = PlanRunner(workspace_root)
            run = runner.start(task_id, plan_version)
            run["result_ref"] = run["run_id"]
            return {"ok": True, "workspace_root": str(runner.store.workspace), "run": run}
        if self.runtime_resolver is None or self.broker_client is None:
            raise PlanFlowError("BROKER_REQUIRED", "Production run_plan requires Runtime Resolver and loopback Broker")
        plan = info["plan"]
        bindings = plan.get("capability_bindings", [])
        capability_ids = tuple(str(item["capability_id"]) for item in bindings)
        external_operators = tuple(
            str(name) for item in bindings for name in item.get("operators", [])
        )
        runtime = self.runtime_resolver.resolve(
            capability_ids=capability_ids,
            operator_names=external_operators,
            profile_family=str(plan.get("execution_profile", "local-cpu")).removeprefix("local-"),
        )
        run = self.broker_client.start(
            task_id=task_id,
            plan_version=plan_version,
            runtime_id=runtime.runtime_id,
            profile=str(plan.get("execution_profile", "local-cpu")),
        )
        return {"ok": True, "workspace_root": str(store.workspace), "run": run}

    def get_run(self, workspace_root: str, task_id: str, run_id: str | None = None) -> dict[str, Any]:
        if self.execution_mode == "native":
            runner = PlanRunner(workspace_root)
            run = runner.get(task_id, run_id)
            self._authorize_personal_plan(runner.store.get_plan(task_id, run["plan_version"])["plan"])
            run["result_ref"] = run["run_id"]
            return {"ok": True, "workspace_root": str(runner.store.workspace), "run": run}
        if self.broker_client is None or run_id is None:
            raise PlanFlowError("BROKER_REQUIRED", "A broker and public run_id are required")
        store = PlanStore(workspace_root)
        run = self.broker_client.get(run_id)
        if run.get("task_id") != task_id:
            raise PlanFlowError("RUN_NOT_FOUND", "Broker run does not belong to this task")
        return {"ok": True, "workspace_root": str(store.workspace), "run": run}

    def cancel_run(self, workspace_root: str, task_id: str, run_id: str) -> dict[str, Any]:
        if self.execution_mode == "native":
            runner = PlanRunner(workspace_root)
            existing = runner.get(task_id, run_id)
            self._authorize_personal_plan(runner.store.get_plan(task_id, existing["plan_version"])["plan"])
            return {"ok": True, "workspace_root": str(runner.store.workspace), "run": runner.cancel(task_id, run_id)}
        if self.broker_client is None:
            raise PlanFlowError("BROKER_REQUIRED", "A broker is required")
        store = PlanStore(workspace_root)
        current = self.broker_client.get(run_id)
        if current.get("task_id") != task_id:
            raise PlanFlowError("RUN_NOT_FOUND", "Broker run does not belong to this task")
        return {"ok": True, "workspace_root": str(store.workspace), "run": self.broker_client.cancel(run_id)}

    @staticmethod
    def _authorize_personal_plan(plan):
        if plan.get("operator_bindings"):
            if not current_user.get() or plan.get("operator_owner") != current_user.get():
                raise PlanFlowError("OPERATOR_OWNER_FORBIDDEN", "This personal-operator plan belongs to another account")
            resolved = resolve_bindings(plan)
            from .user_operator_runtime import runtime_python
            runtime_python(UserOperatorStore(), [requirement for item in resolved.values() for requirement in item["_manifest"].get("dependencies", [])])

    @staticmethod
    def _execution_preview(plan: dict[str, Any]) -> dict[str, Any]:
        recipe = plan.get("recipe", {})
        return {
            "input": recipe.get("dataset_path") or recipe.get("dataset") or recipe.get("generated_dataset_config"),
            "dj_operators": [next(iter(step)) for step in recipe.get("process", []) if isinstance(step, dict) and step],
            "postprocess": plan.get("postprocess", []),
            "executor_type": recipe.get("executor_type", "default"),
            "np": recipe.get("np", 1),
            "output_template": recipe.get("export_path"),
        }

    def _runtime_assessment(self, plan: dict[str, Any]) -> dict[str, Any]:
        return self.runtime_preflight.assess(
            plan,
            execution_mode=self.execution_mode,
            runtime_resolver_configured=self.runtime_resolver is not None,
            broker_configured=self.broker_client is not None,
        )

"""Loopback execution broker with a small approved-plan interface."""

from __future__ import annotations

import argparse
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .capability import CapabilityDescriptor, LocalCapabilityCatalog
from .common import (
    FileLock,
    PlanFlowError,
    is_within,
    read_json,
    require_workspace,
    write_json_atomic,
)
from .execution import DockerBackend, DockerResourceLimits, ExecutionBackend
from .model_store import LocalModelStore
from .runner import PlanRunner
from .store import PlanStore

_BROKER_RUN_ID = re.compile(r"run_[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class BrokerProfile:
    name: str
    limits: DockerResourceLimits
    timeout_seconds: int
    max_concurrent: int | None = None


PROFILES = {
    "local-tiny": BrokerProfile(
        "local-tiny",
        DockerResourceLimits(cpus=1, memory_bytes=2 * 1024**3, pids_limit=64, tmpfs_bytes=256 * 1024**2),
        timeout_seconds=300,
    ),
    "local-cpu": BrokerProfile(
        "local-cpu",
        DockerResourceLimits(cpus=2, memory_bytes=8 * 1024**3, pids_limit=256, tmpfs_bytes=1024**3),
        timeout_seconds=1800,
        max_concurrent=1,
    ),
}


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    plan_version: str
    capability_id: str
    profile: str


BackendFactory = Callable[[CapabilityDescriptor, BrokerProfile], ExecutionBackend]


class ExecutionBroker:
    """Turns approved plan references into constrained backend runs."""

    def __init__(
        self,
        workspace_root: str | Path,
        worker_root: str | Path,
        *,
        allowed_capabilities: tuple[str, ...],
        tenant_id: str = "local-test",
        backend_factory: BackendFactory | None = None,
    ):
        self.workspace = require_workspace(workspace_root)
        self.worker_root = Path(worker_root).resolve()
        if not self.worker_root.is_dir() or self.worker_root == self.workspace or is_within(self.worker_root, self.workspace):
            raise PlanFlowError("INVALID_WORKER_ROOT", "Broker worker root must exist outside the workspace")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", tenant_id):
            raise PlanFlowError("INVALID_TENANT", "Broker tenant_id contains unsupported characters")
        if not allowed_capabilities:
            raise PlanFlowError("EMPTY_IMAGE_ALLOWLIST", "Broker requires at least one allowed capability")
        self.tenant_id = tenant_id
        self.allowed_capabilities = frozenset(allowed_capabilities)
        self.catalog = LocalCapabilityCatalog(self.worker_root)
        for capability_id in self.allowed_capabilities:
            self.catalog.resolve(capability_id)
        self.state_root = self.worker_root / "broker-state" / "runs"
        self.backend_factory = backend_factory or self._docker_backend

    def start(self, request: StartRunRequest) -> dict[str, Any]:
        descriptor = self._allowed_capability(request.capability_id)
        profile = PROFILES.get(request.profile)
        if profile is None:
            raise PlanFlowError("PROFILE_NOT_ALLOWED", f"Unknown broker profile: {request.profile}")
        self._validate_capability_models(descriptor, request.task_id, request.plan_version)
        with FileLock(self.state_root / ".start.lock"):
            self._enforce_capacity(profile)
            backend = self.backend_factory(descriptor, profile)
            state = PlanRunner(self.workspace, backend=backend).start(
                request.task_id,
                request.plan_version,
                timeout_seconds=profile.timeout_seconds,
            )
            public_run_id = "run_" + uuid.uuid4().hex
            record = {
                "schema_version": 1,
                "run_id": public_run_id,
                "tenant_id": self.tenant_id,
                "task_id": request.task_id,
                "plan_version": request.plan_version,
                "internal_run_id": state["run_id"],
                "capability_id": descriptor.capability_id,
                "profile": profile.name,
                "created_at": state["created_at"],
            }
            write_json_atomic(self.state_root / f"{public_run_id}.json", record)
        return self._project(record, state)

    def reconcile(self) -> tuple[str, ...]:
        """Adopt untracked, policy-matching backend runs after a broker crash."""
        candidates: dict[tuple[str, str], list[tuple[CapabilityDescriptor, BrokerProfile, Any]]] = {}
        for capability_id in sorted(self.allowed_capabilities):
            descriptor = self._allowed_capability(capability_id)
            for profile in PROFILES.values():
                backend = self.backend_factory(descriptor, profile)
                discover = getattr(backend, "discover_managed_runs", None)
                if discover is None:
                    continue
                for managed in discover():
                    key = (managed.task_id, managed.handle.run_id)
                    candidates.setdefault(key, []).append((descriptor, profile, managed))

        adopted: list[str] = []
        # Share the admission lock so recovered local-cpu runs cannot race a new start.
        with FileLock(self.state_root / ".start.lock"):
            tracked = {
                (record.get("task_id"), record.get("internal_run_id"))
                for path in self.state_root.glob("run_*.json")
                for record in (read_json(path),)
                if record.get("tenant_id") == self.tenant_id
            }
            for key, matches in candidates.items():
                if key in tracked or len(matches) != 1:
                    continue
                descriptor, profile, managed = matches[0]
                self._validate_capability_models(descriptor, managed.task_id, managed.plan_version)
                run_path = (
                    PlanStore(self.workspace).task_path(managed.task_id)
                    / "runs" / managed.handle.run_id / "run.json"
                )
                if not run_path.is_file():
                    continue
                state = read_json(run_path)
                if (
                    state.get("task_id") != managed.task_id
                    or state.get("plan_version") != managed.plan_version
                    or state.get("handle") != managed.handle.to_dict()
                ):
                    continue
                public_run_id = "run_" + uuid.uuid4().hex
                record = {
                    "schema_version": 1,
                    "run_id": public_run_id,
                    "tenant_id": self.tenant_id,
                    "task_id": managed.task_id,
                    "plan_version": managed.plan_version,
                    "internal_run_id": managed.handle.run_id,
                    "capability_id": descriptor.capability_id,
                    "profile": profile.name,
                    "created_at": state["created_at"],
                    "reconciled": True,
                }
                write_json_atomic(self.state_root / f"{public_run_id}.json", record)
                tracked.add(key)
                adopted.append(public_run_id)
        return tuple(adopted)

    def _validate_capability_models(
        self, descriptor: CapabilityDescriptor, task_id: str, plan_version: str
    ) -> None:
        plan = PlanStore(self.workspace).get_plan(task_id, plan_version)["plan"]
        raw_models = plan.get("models", [])
        declared = {
            str(item.get("artifact_id"))
            for item in raw_models
            if isinstance(item, dict) and set(item) == {"artifact_id"}
        }
        expected = {str(item.get("artifact_id")): str(item.get("sha256")) for item in descriptor.model_refs}
        if declared != set(expected):
            raise PlanFlowError(
                "CAPABILITY_MODEL_MISMATCH",
                "Approved plan models do not match the allowlisted capability contract",
                details={"declared": sorted(declared), "required": sorted(expected)},
            )
        store = LocalModelStore(self.worker_root)
        for artifact_id, expected_hash in expected.items():
            actual = store.verify(artifact_id)
            if actual.sha256 != expected_hash:
                raise PlanFlowError(
                    "CAPABILITY_MODEL_MISMATCH",
                    f"Published model hash does not match capability contract: {artifact_id}",
                )

    def _enforce_capacity(self, profile: BrokerProfile) -> None:
        if profile.max_concurrent is None:
            return
        active = 0
        for path in self.state_root.glob("run_*.json"):
            record = read_json(path)
            if (
                record.get("tenant_id") != self.tenant_id
                or record.get("profile") != profile.name
                or record.get("cleaned_at")
            ):
                continue
            state = self._runner(record).get(record["task_id"], record["internal_run_id"])
            if state["status"] in {"starting", "running"}:
                active += 1
        if active >= profile.max_concurrent:
            raise PlanFlowError(
                "PROFILE_CAPACITY_EXCEEDED",
                f"Profile {profile.name} allows at most {profile.max_concurrent} active run(s)",
            )

    def get(self, run_id: str) -> dict[str, Any]:
        record = self._record(run_id)
        descriptor = self._allowed_capability(record["capability_id"])
        profile = PROFILES[record["profile"]]
        state = PlanRunner(self.workspace, backend=self.backend_factory(descriptor, profile)).get(
            record["task_id"], record["internal_run_id"]
        )
        return self._project(record, state)

    def cancel(self, run_id: str) -> dict[str, Any]:
        record = self._record(run_id)
        runner = self._runner(record)
        state = runner.cancel(record["task_id"], record["internal_run_id"])
        return self._project(record, state)

    def cleanup(self, run_id: str) -> dict[str, Any]:
        record = self._record(run_id)
        if record.get("cleaned_at"):
            state = self._runner(record).get(record["task_id"], record["internal_run_id"])
            return self._project(record, state)
        state = self._runner(record).cleanup(record["task_id"], record["internal_run_id"])
        record["cleaned_at"] = state["cleaned_at"]
        write_json_atomic(self.state_root / f"{run_id}.json", record)
        return self._project(record, state)

    def _runner(self, record: dict[str, Any]) -> PlanRunner:
        descriptor = self._allowed_capability(record["capability_id"])
        profile = PROFILES[record["profile"]]
        return PlanRunner(self.workspace, backend=self.backend_factory(descriptor, profile))

    def _docker_backend(self, descriptor: CapabilityDescriptor, profile: BrokerProfile) -> DockerBackend:
        if descriptor.backend != "docker" or set(descriptor.backend_ref) != {"image_id"}:
            raise PlanFlowError("CAPABILITY_BACKEND_NOT_ALLOWED", "Broker only accepts immutable Docker capabilities")
        return DockerBackend(
            self.workspace,
            self.worker_root,
            descriptor.backend_ref["image_id"],
            tenant_id=self.tenant_id,
            limits=profile.limits,
            model_store=LocalModelStore(self.worker_root),
        )

    def _allowed_capability(self, capability_id: str) -> CapabilityDescriptor:
        if capability_id not in self.allowed_capabilities:
            raise PlanFlowError("IMAGE_NOT_ALLOWED", f"Capability is not in the broker allowlist: {capability_id}")
        return self.catalog.resolve(capability_id)

    def _record(self, run_id: str) -> dict[str, Any]:
        if not _BROKER_RUN_ID.fullmatch(str(run_id or "")):
            raise PlanFlowError("INVALID_RUN_ID", f"Invalid broker run id: {run_id}")
        path = self.state_root / f"{run_id}.json"
        if not path.is_file():
            raise PlanFlowError("RUN_NOT_FOUND", f"Unknown broker run: {run_id}")
        record = read_json(path)
        if record.get("tenant_id") != self.tenant_id or record.get("run_id") != run_id:
            raise PlanFlowError("INVALID_BROKER_STATE", "Broker run identity does not match its persisted record")
        return record

    @staticmethod
    def _project(record: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "run_id": record["run_id"],
            "task_id": record["task_id"],
            "plan_version": record["plan_version"],
            "capability_id": record["capability_id"],
            "profile": record["profile"],
            "status": state["status"],
            "created_at": state["created_at"],
            "deadline": (state.get("handle") or {}).get("deadline"),
            "cleaned": bool(record.get("cleaned_at") or state.get("cleaned_at")),
        }
        for key in ("updated_at", "error_code", "error"):
            if state.get(key) is not None:
                payload[key] = state[key]
        return payload


def create_broker_app(broker: ExecutionBroker) -> FastAPI:
    broker.reconcile()
    app = FastAPI(title="Data-Juicer Local Execution Broker", docs_url=None, redoc_url=None)

    @app.exception_handler(PlanFlowError)
    async def plan_flow_error(_request: Request, error: PlanFlowError):
        status = 404 if error.code in {"RUN_NOT_FOUND", "TASK_NOT_FOUND", "CAPABILITY_MISSING"} else 409
        return JSONResponse(status_code=status, content=error.to_dict())

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, error: RequestValidationError):
        details = [
            {"path": ".".join(str(item) for item in issue["loc"]), "type": issue["type"]}
            for issue in error.errors()
        ]
        failure = PlanFlowError("INVALID_REQUEST", "Request fields do not match the broker interface", details=details)
        return JSONResponse(status_code=422, content=failure.to_dict())

    @app.post("/v1/runs", status_code=201)
    def start_run(request: StartRunRequest):
        return broker.start(request)

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str):
        return broker.get(run_id)

    @app.post("/v1/runs/{run_id}:cancel")
    def cancel_run(run_id: str):
        return broker.cancel(run_id)

    @app.post("/v1/runs/{run_id}:cleanup")
    def cleanup_run(run_id: str):
        return broker.cleanup(run_id)

    return app


def serve_broker(broker: ExecutionBroker, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    try:
        address = ip_address(host)
    except ValueError as error:
        raise PlanFlowError("BROKER_LOOPBACK_REQUIRED", "Broker host must be a numeric loopback address") from error
    if not address.is_loopback:
        raise PlanFlowError("BROKER_LOOPBACK_REQUIRED", "Broker may only listen on a loopback address")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise PlanFlowError("INVALID_BROKER_PORT", "Broker port must be between 1 and 65535")
    import uvicorn

    uvicorn.run(create_broker_app(broker), host=host, port=port, access_log=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the loopback Data-Juicer execution broker")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--worker-root", required=True)
    parser.add_argument("--allow-capability", action="append", required=True)
    parser.add_argument("--tenant-id", default="local-test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    broker = ExecutionBroker(
        args.workspace,
        args.worker_root,
        allowed_capabilities=tuple(args.allow_capability),
        tenant_id=args.tenant_id,
    )
    serve_broker(broker, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

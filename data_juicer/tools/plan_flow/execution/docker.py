"""Docker adapter with a generic public handle and private container state."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..common import (
    FileLock,
    PlanFlowError,
    is_within,
    read_json,
    read_yaml,
    require_workspace,
    write_json_atomic,
    write_yaml_atomic,
)
from ..store import PlanStore
from ..model_store import LocalModelStore
from .spec import RunHandle, RunResult, RunStatus, RuntimeSpec

_BACKEND_REF = re.compile(r"[0-9a-f]{32}\Z")
_TERMINAL = {"succeeded", "failed", "cancelled", "lost"}
_EXIT_ERRORS = {
    10: "INVALID_RUN_SPEC",
    11: "RECIPE_HASH_MISMATCH",
    12: "PATH_NOT_ALLOWED",
    13: "MOUNT_POLICY_VIOLATION",
    20: "EXECUTION_FAILED",
    21: "RESULT_MANIFEST_FAILED",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class DockerResourceLimits:
    cpus: float = 2.0
    memory_bytes: int = 8 * 1024**3
    pids_limit: int = 256
    tmpfs_bytes: int = 1024**3
    stop_grace_seconds: int = 10

    def __post_init__(self) -> None:
        if self.cpus <= 0 or self.memory_bytes <= 0 or self.pids_limit <= 0 or self.tmpfs_bytes <= 0:
            raise PlanFlowError("INVALID_RESOURCE_LIMIT", "Docker resource limits must be positive")
        if self.stop_grace_seconds < 0:
            raise PlanFlowError("INVALID_RESOURCE_LIMIT", "Docker stop grace period must not be negative")


class DockerBackend:
    """Execute a RuntimeSpec in a locked-down Docker container.

    Container and image identities live only in the backend record.  The public
    RunHandle remains an opaque routing envelope shared with future backends.
    """

    name = "docker"

    def __init__(
        self,
        workspace_root: str | Path,
        worker_root: str | Path,
        image: str,
        *,
        tenant_id: str = "local-test",
        limits: DockerResourceLimits | None = None,
        model_store: LocalModelStore | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ):
        self.workspace = require_workspace(workspace_root)
        self.worker_root = Path(worker_root).resolve()
        if self.worker_root.parent == self.worker_root or any(character in str(self.worker_root) for character in (",", "\n", "\r")):
            raise PlanFlowError("INVALID_WORKER_ROOT", "Docker worker root must be a concrete mount-safe directory")
        self.worker_root.mkdir(parents=True, exist_ok=True)
        if self.worker_root == self.workspace or is_within(self.worker_root, self.workspace):
            raise PlanFlowError("INVALID_WORKER_ROOT", "Docker worker root must be separate from the plan workspace")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", tenant_id):
            raise PlanFlowError("INVALID_TENANT", "Docker tenant_id contains unsupported characters")
        if not str(image).strip():
            raise PlanFlowError("INVALID_IMAGE", "Docker image must not be empty")
        self.image = str(image).strip()
        self.tenant_id = tenant_id
        self.limits = limits or DockerResourceLimits()
        if model_store is not None and model_store.worker_root != self.worker_root:
            raise PlanFlowError("BACKEND_MISMATCH", "DockerBackend and ModelStore must use the same worker root")
        self.model_store = model_store
        self._command_runner = command_runner or subprocess.run
        self._state_root = self.workspace / ".dj" / "execution" / self.name

    def start(self, spec: RuntimeSpec) -> RunHandle:
        self._validate_spec_paths(spec)
        backend_ref = uuid.uuid4().hex
        run_root = (self.worker_root / "runs" / f"run-docker-{backend_ref}").resolve()
        if not is_within(run_root, self.worker_root):
            raise PlanFlowError("PATH_NOT_ALLOWED", "Generated Docker run root escaped worker root")
        image_id = self._resolve_image_id()
        container_name = f"dj-run-{backend_ref[:20]}"
        staged = self._stage_run(spec, run_root)
        record_path = self._record_path(backend_ref)
        record = {
            "schema_version": 1,
            "backend": self.name,
            "backend_ref": backend_ref,
            "run_id": spec.run_id,
            "status": "starting",
            "container_id": None,
            "container_name": container_name,
            "image_id": image_id,
            "created_at": spec.created_at.isoformat(),
            "finished_at": None,
            "exit_code": None,
            "oom_killed": False,
            "cancellation_requested": False,
            "timed_out": False,
            "worker_run_root": str(run_root),
            "runtime_spec": spec.to_dict(),
            "limits": asdict(self.limits),
            "models": staged["model_records"],
        }
        write_json_atomic(record_path, record)
        create_args = self._create_args(spec, record, staged)
        container_id: str | None = None
        try:
            created = self._docker(create_args)
            container_id = created.stdout.strip()
            if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
                raise PlanFlowError("DOCKER_PROTOCOL_ERROR", "docker create returned an invalid container ID")
            record.update({"container_id": container_id, "status": "created"})
            write_json_atomic(record_path, record)
            self._docker(["start", container_id])
            record["status"] = "running"
            write_json_atomic(record_path, record)
        except Exception:
            if container_id:
                self._docker(["rm", "--force", container_id], check=False)
            record_path.unlink(missing_ok=True)
            raise
        return RunHandle(self.name, spec.run_id, spec.created_at, spec.deadline, backend_ref)

    def inspect(self, handle: RunHandle) -> RunStatus:
        record = self._load_record(handle, missing_ok=True)
        if record is None:
            return RunStatus("lost", _now(), "Docker backend state is missing")
        if record.get("status") in _TERMINAL:
            return RunStatus(str(record["status"]), _now(), record.get("error"))
        if handle.deadline is not None and _now() >= handle.deadline:
            self._stop(record, timed_out=True)
            return RunStatus("failed", _now(), "Docker run exceeded its deadline")
        inspected = self._container_inspect(record)
        if inspected is None:
            return RunStatus("lost", _now(), "Managed Docker container is missing")
        state = inspected.get("State") or {}
        status = str(state.get("Status") or "")
        if status in {"created", "restarting"}:
            return RunStatus("starting", _now())
        if status in {"running", "paused"}:
            return RunStatus("running", _now())
        if status in {"exited", "dead", "removing"}:
            self._capture_terminal(record, state)
            terminal = "cancelled" if record.get("cancellation_requested") else "succeeded" if record["exit_code"] == 0 else "failed"
            record["status"] = terminal
            record["finished_at"] = _now().isoformat()
            self._save_record(handle.backend_ref, record)
            message = "Docker container was OOM-killed" if record["oom_killed"] else None
            return RunStatus(terminal, _now(), message)
        return RunStatus("lost", _now(), f"Unsupported Docker container state: {status or 'unknown'}")

    def cancel(self, handle: RunHandle) -> None:
        record = self._load_record(handle)
        if record.get("status") in _TERMINAL:
            return
        record["cancellation_requested"] = True
        self._save_record(handle.backend_ref, record)
        self._stop(record, timed_out=False)

    def collect(self, handle: RunHandle) -> RunResult:
        record = self._load_record(handle, missing_ok=True)
        if record is None:
            return RunResult("lost", _now(), error_code="RUNNER_LOST", error="Docker backend state is missing")
        status = self.inspect(handle)
        if not status.terminal:
            raise PlanFlowError("RUN_NOT_FINISHED", f"Run is still active: {handle.run_id}")
        record = self._load_record(handle)
        self._capture_logs(record)
        provenance = self._write_provenance(record)
        if status.status == "succeeded":
            try:
                self._collect_outputs(record)
            except PlanFlowError as exc:
                record.update({"status": "failed", "error_code": exc.code, "error": exc.message})
                self._save_record(handle.backend_ref, record)
                return RunResult(
                    "failed", _now(), exit_code=record.get("exit_code"), error_code=exc.code,
                    error=exc.message, provenance=provenance,
                )
        error_code = record.get("error_code")
        error = record.get("error") or status.message
        if status.status == "lost":
            error_code = "RUNNER_LOST"
        elif record.get("timed_out"):
            error_code, error = "RUN_TIMED_OUT", "Docker run exceeded its deadline"
        elif record.get("oom_killed"):
            error_code, error = "RUN_OOM", "Docker container was OOM-killed"
        elif status.status == "failed":
            error_code = error_code or _EXIT_ERRORS.get(record.get("exit_code"), "EXECUTION_FAILED")
        return RunResult(
            status.status, _now(), exit_code=record.get("exit_code"), error_code=error_code,
            error=error, provenance=provenance,
        )

    def cleanup(self, handle: RunHandle) -> None:
        record = self._load_record(handle, missing_ok=True)
        if record is None:
            return
        status = self.inspect(handle)
        if not status.terminal:
            raise PlanFlowError("RUN_ACTIVE", f"Cannot clean up an active run: {handle.run_id}")
        result = self.collect(handle)
        container_id = record.get("container_id")
        if container_id:
            self._docker(["rm", str(container_id)], check=False)
        run_root = self._trusted_run_root(record)
        if result.status == "succeeded":
            work = (run_root / "work").resolve()
            if is_within(work, run_root) and work.is_dir():
                shutil.rmtree(work)
        cleanup = {
            "schema_version": 1,
            "run_id": handle.run_id,
            "status": result.status,
            "container_removed": True,
            "work_retained": result.status != "succeeded",
            "cleaned_at": _now().isoformat(),
        }
        write_json_atomic(run_root / "cleanup.json", cleanup)
        self._record_path(handle.backend_ref).unlink(missing_ok=True)

    def _validate_spec_paths(self, spec: RuntimeSpec) -> None:
        if Path(spec.workspace_root).resolve() != self.workspace:
            raise PlanFlowError("BACKEND_MISMATCH", "RuntimeSpec workspace does not match DockerBackend")
        for field in ("run_dir", "output_dir", "recipe_path", "stdout_log", "stderr_log"):
            path = Path(getattr(spec, field)).resolve()
            if not is_within(path, self.workspace):
                raise PlanFlowError("PATH_NOT_ALLOWED", f"RuntimeSpec {field} escaped workspace: {path}")

    def _stage_run(self, spec: RuntimeSpec, run_root: Path) -> dict[str, Any]:
        plan = PlanStore(self.workspace).get_plan(spec.task_id, spec.plan_version)["plan"]
        if plan.get("postprocess"):
            raise PlanFlowError("DOCKER_POSTPROCESS_UNSUPPORTED", "Docker backend does not yet support postprocess scripts")
        recipe = read_yaml(Path(spec.recipe_path))
        if recipe.get("executor_type", "default") != "default":
            raise PlanFlowError("DOCKER_EXECUTOR_UNSUPPORTED", "Docker backend currently supports executor_type=default")
        if recipe.get("custom_operator_paths"):
            raise PlanFlowError("DOCKER_CUSTOM_OPERATOR_UNSUPPORTED", "Custom operators are not mounted yet")
        model_records = self._resolve_models(plan)
        recipe = self._materialize_model_uris(recipe, model_records)
        dirs = {name: run_root / name for name in ("input", "bundle", "output", "work", "logs")}
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=False)
        configs = (recipe.get("dataset") or {}).get("configs")
        if not isinstance(configs, list) or not configs:
            raise PlanFlowError("INVALID_DATASET", "Docker recipe must contain local dataset configs")
        for index, config in enumerate(configs):
            if not isinstance(config, dict) or config.get("type") != "local" or not config.get("path"):
                raise PlanFlowError("NON_LOCAL_DATASET", "Docker backend only accepts local dataset configs")
            source = self._trusted_input(Path(str(config["path"])))
            destination = dirs["input"] / f"dataset-{index}{source.suffix if source.is_file() else ''}"
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            config["path"] = f"/workspace/input/{destination.name}"
        host_output = Path(spec.output_dir).resolve()
        export = Path(str(recipe.get("export_path") or "")).resolve()
        if not is_within(export, host_output):
            raise PlanFlowError("PATH_NOT_ALLOWED", "Recipe export_path escaped RuntimeSpec output_dir")
        relative_export = export.relative_to(host_output).as_posix()
        recipe["export_path"] = f"/workspace/output/{relative_export}"
        recipe["work_dir"] = "/run/work"
        recipe["temp_dir"] = "/tmp"
        materialized = dirs["bundle"] / "materialized-recipe.yaml"
        write_yaml_atomic(materialized, recipe)
        run_spec = {
            "schema_version": 1,
            "run_id": spec.run_id,
            "tenant_id": self.tenant_id,
            "recipe": {"path": "/run/bundle/materialized-recipe.yaml", "sha256": _sha256_file(materialized)},
            "mounts": {
                "input": "/workspace/input", "bundle": "/run/bundle", "output": "/workspace/output",
                "work": "/run/work", "temp": "/tmp",
            },
            "models": [
                {"artifact_id": item["artifact_id"], "path": f"/models/{item['artifact_id']}"}
                for item in model_records
            ],
        }
        write_json_atomic(dirs["bundle"] / "run-spec.json", run_spec)
        dirs["model_records"] = model_records
        return dirs

    def _resolve_models(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        raw_models = plan.get("models", [])
        if not isinstance(raw_models, list):
            raise PlanFlowError("INVALID_MODELS", "Plan models must be an array")
        if raw_models and self.model_store is None:
            raise PlanFlowError("MODEL_STORE_REQUIRED", "Docker run declares models but no ModelStore is configured")
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_models):
            if not isinstance(item, dict) or set(item) != {"artifact_id"}:
                raise PlanFlowError("INVALID_MODEL_REF", f"Plan models[{index}] must contain exactly artifact_id")
            artifact_id = str(item.get("artifact_id") or "")
            if artifact_id.casefold() in seen:
                raise PlanFlowError("INVALID_MODEL_REF", f"Duplicate model artifact: {artifact_id}")
            seen.add(artifact_id.casefold())
            artifact = self.model_store.resolve(artifact_id)  # type: ignore[union-attr]
            records.append(
                {
                    "artifact_id": artifact_id,
                    "host_path": str(artifact.path),
                    "manifest": artifact.manifest.to_provenance(),
                    "files": [entry.path for entry in artifact.manifest.files],
                }
            )
        return records

    @staticmethod
    def _materialize_model_uris(recipe: dict[str, Any], models: list[dict[str, Any]]) -> dict[str, Any]:
        by_id = {item["artifact_id"].casefold(): item for item in models}

        def replace(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            if isinstance(value, list):
                return [replace(item) for item in value]
            if not isinstance(value, str) or not value.startswith("model-store://"):
                return value
            remainder = value.removeprefix("model-store://")
            artifact_id, separator, relative_text = remainder.partition("/")
            relative = Path(relative_text)
            if (
                not separator
                or not relative_text
                or relative.is_absolute()
                or ".." in relative.parts
                or ":" in relative_text
                or "\\" in relative_text
            ):
                raise PlanFlowError("INVALID_MODEL_URI", f"Invalid model URI: {value}")
            record = by_id.get(artifact_id.casefold())
            if record is None:
                raise PlanFlowError("MODEL_NOT_DECLARED", f"Model artifact is not declared: {artifact_id}")
            normalized = relative.as_posix()
            if normalized not in record["files"]:
                raise PlanFlowError("MODEL_FILE_NOT_DECLARED", f"Model file is not declared: {value}")
            return f"/models/{record['artifact_id']}/{normalized}"

        return replace(recipe)

    def _trusted_input(self, source: Path) -> Path:
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise PlanFlowError("DATASET_NOT_FOUND", f"Dataset path is unavailable: {source}") from exc
        if not is_within(resolved, self.workspace):
            raise PlanFlowError("PATH_NOT_ALLOWED", f"Dataset escaped workspace: {resolved}")
        current = source.absolute()
        while is_within(current, self.workspace):
            if current.is_symlink():
                raise PlanFlowError("PATH_NOT_ALLOWED", f"Dataset path contains a symbolic link: {source}")
            if current == self.workspace:
                break
            current = current.parent
        return resolved

    def _create_args(self, spec: RuntimeSpec, record: dict[str, Any], dirs: dict[str, Any]) -> list[str]:
        memory = str(self.limits.memory_bytes)
        args = [
            "create", "--name", record["container_name"],
            "--label", "dj.managed=true", "--label", f"dj.run-id={spec.run_id}",
            "--label", f"dj.backend-ref={record['backend_ref']}", "--label", f"dj.task-id={spec.task_id}",
            "--user", "10001:10001", "--read-only", "--network", "none", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true", "--pids-limit", str(self.limits.pids_limit),
            "--cpus", str(self.limits.cpus), "--memory", memory, "--memory-swap", memory,
            "--tmpfs", f"/tmp:rw,noexec,nosuid,size={self.limits.tmpfs_bytes}",
        ]
        mounts = (("input", "/workspace/input", True), ("bundle", "/run/bundle", True),
                  ("output", "/workspace/output", False), ("work", "/run/work", False))
        for name, destination, readonly in mounts:
            source = dirs[name].resolve()
            if not is_within(source, self.worker_root):
                raise PlanFlowError("PATH_NOT_ALLOWED", f"Docker mount escaped worker root: {source}")
            value = f"type=bind,src={source},dst={destination}" + (",readonly" if readonly else "")
            args.extend(["--mount", value])
        for model in record["models"]:
            source = Path(model["host_path"]).resolve()
            expected_root = (self.worker_root / "models").resolve()
            if source.parent != expected_root or source.name != model["artifact_id"]:
                raise PlanFlowError("INVALID_BACKEND_STATE", "Published model path escaped ModelStore")
            args.extend(
                ["--mount", f"type=bind,src={source},dst=/models/{model['artifact_id']},readonly"]
            )
        args.extend([record["image_id"], "--run-spec", "/run/bundle/run-spec.json"])
        return args

    def _resolve_image_id(self) -> str:
        result = self._docker(["image", "inspect", self.image])
        try:
            image_id = json.loads(result.stdout)[0]["Id"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise PlanFlowError("DOCKER_PROTOCOL_ERROR", "Could not parse Docker image identity") from exc
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(image_id)):
            raise PlanFlowError("DOCKER_PROTOCOL_ERROR", "Docker returned an invalid image identity")
        return str(image_id)

    def _container_inspect(self, record: dict[str, Any]) -> dict[str, Any] | None:
        container_id = record.get("container_id")
        if not container_id:
            return {"State": {"Status": "created"}}
        result = self._docker(["inspect", str(container_id)], check=False)
        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout)[0]
        except (IndexError, TypeError, json.JSONDecodeError) as exc:
            raise PlanFlowError("DOCKER_PROTOCOL_ERROR", "Could not parse docker inspect output") from exc

    def _capture_terminal(self, record: dict[str, Any], state: dict[str, Any]) -> None:
        record["exit_code"] = int(state.get("ExitCode", -1))
        record["oom_killed"] = bool(state.get("OOMKilled"))

    def _stop(self, record: dict[str, Any], *, timed_out: bool) -> None:
        container_id = record.get("container_id")
        if container_id:
            stopped = self._docker(
                ["stop", "--time", str(self.limits.stop_grace_seconds), str(container_id)], check=False
            )
            if stopped.returncode != 0:
                self._docker(["kill", str(container_id)], check=False)
        inspected = self._container_inspect(record)
        if inspected:
            self._capture_terminal(record, inspected.get("State") or {})
        record.update({
            "status": "failed" if timed_out else "cancelled",
            "timed_out": timed_out,
            "finished_at": _now().isoformat(),
        })
        self._save_record(record["backend_ref"], record)

    def _capture_logs(self, record: dict[str, Any]) -> None:
        container_id = record.get("container_id")
        if not container_id:
            return
        result = self._docker(["logs", str(container_id)], check=False)
        spec = RuntimeSpec.from_dict(record["runtime_spec"])
        stdout = Path(spec.stdout_log)
        stderr = Path(spec.stderr_log)
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text(result.stdout or "", encoding="utf-8")
        stderr.write_text(result.stderr or "", encoding="utf-8")
        run_root = self._trusted_run_root(record)
        (run_root / "logs" / "stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (run_root / "logs" / "stderr.log").write_text(result.stderr or "", encoding="utf-8")

    def _collect_outputs(self, record: dict[str, Any]) -> None:
        run_root = self._trusted_run_root(record)
        source_root = (run_root / "output").resolve()
        manifest_path = source_root / "result-manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise PlanFlowError("RESULT_MANIFEST_MISSING", "Successful container did not produce a result manifest")
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != 1 or manifest.get("run_id") != record["run_id"] or manifest.get("status") != "succeeded":
            raise PlanFlowError("INVALID_RESULT_MANIFEST", "Result manifest identity or status is invalid")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list) or manifest.get("output_count") != len(outputs):
            raise PlanFlowError("INVALID_RESULT_MANIFEST", "Result manifest output inventory is invalid")
        spec = RuntimeSpec.from_dict(record["runtime_spec"])
        destination_root = Path(spec.output_dir).resolve()
        total = 0
        for item in outputs:
            if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256"}:
                raise PlanFlowError("INVALID_RESULT_MANIFEST", "Result manifest contains an invalid output entry")
            relative = Path(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise PlanFlowError("INVALID_RESULT_MANIFEST", "Result manifest contains path traversal")
            source = (source_root / relative).resolve()
            if not is_within(source, source_root) or not source.is_file() or source.is_symlink():
                raise PlanFlowError("INVALID_RESULT_MANIFEST", f"Result output is unsafe or missing: {relative}")
            size = source.stat().st_size
            if size != item["size_bytes"] or _sha256_file(source) != item["sha256"]:
                raise PlanFlowError("RESULT_INTEGRITY_FAILED", f"Result output failed integrity validation: {relative}")
            destination = (destination_root / relative).resolve()
            if not is_within(destination, destination_root):
                raise PlanFlowError("PATH_NOT_ALLOWED", "Result destination escaped output root")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".docker-part")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            total += size
        if manifest.get("output_size_bytes") != total:
            raise PlanFlowError("INVALID_RESULT_MANIFEST", "Result manifest total size is invalid")
        shutil.copy2(manifest_path, destination_root / "result-manifest.json")

    def _write_provenance(self, record: dict[str, Any]) -> dict[str, Any]:
        provenance = {
            "schema_version": 1,
            "backend": self.name,
            "run_id": record["run_id"],
            "image_id": record["image_id"],
            "container_id": record.get("container_id"),
            "container_name": record["container_name"],
            "sandbox": {"read_only": True, "network": "none", "cap_drop": ["ALL"], "no_new_privileges": True},
            "resources": record["limits"],
            "models": [item["manifest"] for item in record.get("models", [])],
            "collected_at": _now().isoformat(),
        }
        spec = RuntimeSpec.from_dict(record["runtime_spec"])
        write_json_atomic(Path(spec.run_dir) / "runtime-provenance.json", provenance)
        write_json_atomic(self._trusted_run_root(record) / "runtime-provenance.json", provenance)
        return provenance

    def _trusted_run_root(self, record: dict[str, Any]) -> Path:
        run_root = Path(str(record.get("worker_run_root") or "")).resolve()
        if not is_within(run_root, self.worker_root) or run_root.parent != (self.worker_root / "runs").resolve():
            raise PlanFlowError("INVALID_BACKEND_STATE", "Docker worker run root escaped controlled storage")
        return run_root

    def _record_path(self, backend_ref: str) -> Path:
        if not _BACKEND_REF.fullmatch(str(backend_ref or "")):
            raise PlanFlowError("INVALID_BACKEND_REF", "Docker backend_ref is invalid")
        return self._state_root / f"{backend_ref}.json"

    def _load_record(self, handle: RunHandle, *, missing_ok: bool = False) -> dict[str, Any] | None:
        if handle.backend != self.name:
            raise PlanFlowError("BACKEND_MISMATCH", f"Handle belongs to {handle.backend}, not {self.name}")
        path = self._record_path(handle.backend_ref)
        if not path.is_file():
            if missing_ok:
                return None
            raise PlanFlowError("RUNNER_LOST", f"Docker backend state is missing: {handle.run_id}")
        record = read_json(path)
        if (record.get("schema_version") != 1 or record.get("backend") != self.name
                or record.get("backend_ref") != handle.backend_ref or record.get("run_id") != handle.run_id):
            raise PlanFlowError("INVALID_BACKEND_STATE", "Docker backend state does not match RunHandle")
        spec = RuntimeSpec.from_dict(record.get("runtime_spec"))
        if handle.created_at != spec.created_at or handle.deadline != spec.deadline:
            raise PlanFlowError("INVALID_BACKEND_STATE", "Docker RunHandle timing does not match private state")
        self._trusted_run_root(record)
        return record

    def _save_record(self, backend_ref: str, record: dict[str, Any]) -> None:
        path = self._record_path(backend_ref)
        with FileLock(path.with_suffix(".lock")):
            write_json_atomic(path, record)

    def _docker(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            result = self._command_runner(
                ["docker", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
            )
        except OSError as exc:
            raise PlanFlowError("DOCKER_UNAVAILABLE", f"Could not invoke Docker: {exc}") from exc
        if check and result.returncode != 0:
            message = (result.stderr or result.stdout or "Docker command failed").strip()
            raise PlanFlowError("DOCKER_COMMAND_FAILED", message)
        return result

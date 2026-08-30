"""Approval-gated builder for immutable, backend-specific capabilities."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from data_juicer.tools.plan_flow.common import (
    FileLock,
    PlanFlowError,
    canonical_json,
    is_within,
    now_iso,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)


@dataclass(frozen=True)
class CapabilityModelRef:
    artifact_id: str
    sha256: str


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    operator_name: str
    import_module: str
    source_dir: Path
    wheelhouse_dir: Path
    model_refs: tuple[CapabilityModelRef, ...] = ()


@dataclass(frozen=True)
class CapabilityProposal:
    proposal_id: str
    capability_id: str
    content_hash: str
    status: str
    path: Path


@dataclass(frozen=True)
class CapabilityDescriptor:
    capability_id: str
    operator_name: str
    content_hash: str
    backend: str
    backend_ref: dict[str, str]
    base_image_id: str
    created_at: str
    source_hash: str = ""
    dependency_lock_hash: str = ""
    model_refs: tuple[dict[str, str], ...] = ()


class LocalCapabilityCatalog:
    def __init__(self, worker_root: str | Path):
        self.root = Path(worker_root).resolve() / "broker-state" / "capabilities"

    def resolve(self, capability_id: str) -> CapabilityDescriptor:
        _require_identifier(capability_id, "capability id")
        path = self.root / f"{capability_id}.json"
        if not path.is_file():
            raise PlanFlowError("CAPABILITY_MISSING", f"Capability is not registered: {capability_id}")
        payload = read_json(path)
        try:
            payload["model_refs"] = tuple(payload.get("model_refs", ()))
            return CapabilityDescriptor(**payload)
        except TypeError as error:
            raise PlanFlowError("INVALID_CAPABILITY", f"Invalid capability descriptor: {path}") from error

    def register(self, descriptor: CapabilityDescriptor) -> CapabilityDescriptor:
        _require_identifier(descriptor.capability_id, "capability id")
        path = self.root / f"{descriptor.capability_id}.json"
        with FileLock(self.root / f".{descriptor.capability_id}.lock"):
            if path.exists():
                existing = self.resolve(descriptor.capability_id)
                if existing.content_hash != descriptor.content_hash:
                    raise PlanFlowError(
                        "CAPABILITY_CONFLICT",
                        f"Capability id already refers to different content: {descriptor.capability_id}",
                    )
                return existing
            write_json_atomic(path, {
                "capability_id": descriptor.capability_id,
                "operator_name": descriptor.operator_name,
                "content_hash": descriptor.content_hash,
                "backend": descriptor.backend,
                "backend_ref": descriptor.backend_ref,
                "base_image_id": descriptor.base_image_id,
                "created_at": descriptor.created_at,
                "source_hash": descriptor.source_hash,
                "dependency_lock_hash": descriptor.dependency_lock_hash,
                "model_refs": list(descriptor.model_refs),
            })
        return descriptor


class CapabilityBuilder:
    """Snapshots a proposal and refuses to build content that was not approved."""

    def __init__(
        self,
        worker_root: str | Path,
        *,
        base_image_ref: str,
        base_image_id: str,
        command_runner: Callable = subprocess.run,
    ):
        self.worker_root = Path(worker_root).resolve()
        self.fixture_root = self.worker_root / "fixtures" / "capabilities"
        self.proposal_root = self.worker_root / "broker-state" / "capability-proposals"
        self.build_root = self.worker_root / "build-contexts"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:-]{0,255}", base_image_ref):
            raise PlanFlowError("INVALID_IMAGE", "Base image reference contains unsupported characters")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", base_image_id):
            raise PlanFlowError("INVALID_IMAGE", "Base image id must be an immutable lowercase SHA-256 value")
        self.base_image_ref = base_image_ref
        self.base_image_id = base_image_id
        self.command_runner = command_runner
        self.catalog = LocalCapabilityCatalog(self.worker_root)

    def prepare(self, spec: CapabilitySpec) -> CapabilityProposal:
        _require_identifier(spec.capability_id, "capability id")
        _require_identifier(spec.operator_name, "operator name")
        _require_module(spec.import_module)
        model_refs = []
        seen_models = set()
        for model in spec.model_refs:
            _require_identifier(model.artifact_id, "model artifact id")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", model.sha256):
                raise PlanFlowError("INVALID_CAPABILITY", f"Invalid model hash: {model.sha256}")
            if model.artifact_id.casefold() in seen_models:
                raise PlanFlowError("INVALID_CAPABILITY", f"Duplicate model artifact: {model.artifact_id}")
            seen_models.add(model.artifact_id.casefold())
            model_refs.append({"artifact_id": model.artifact_id, "sha256": model.sha256})
        source = Path(spec.source_dir).resolve()
        wheelhouse = Path(spec.wheelhouse_dir).resolve()
        for candidate in (source, wheelhouse):
            if not candidate.is_dir() or not is_within(candidate, self.fixture_root):
                raise PlanFlowError(
                    "CAPABILITY_SOURCE_NOT_ALLOWED",
                    f"Capability inputs must be directories inside {self.fixture_root}: {candidate}",
                )
            if any(path.is_symlink() for path in candidate.rglob("*")):
                raise PlanFlowError("CAPABILITY_SOURCE_NOT_ALLOWED", f"Capability input contains a symlink: {candidate}")
        source_files = self._files(source)
        wheel_files = self._files(wheelhouse, suffix=".whl")
        if not source_files or not wheel_files:
            raise PlanFlowError("INVALID_CAPABILITY", "Capability source and wheelhouse must not be empty")
        source_manifest = self._manifest(source, source_files)
        wheel_manifest = self._manifest(wheelhouse, wheel_files)
        manifest = {
            "builder_format": 2,
            "capability_id": spec.capability_id,
            "operator_name": spec.operator_name,
            "import_module": spec.import_module,
            "base_image_id": self.base_image_id,
            "source": source_manifest,
            "source_hash": sha256_bytes(canonical_json(source_manifest)),
            "wheels": wheel_manifest,
            "dependency_lock_hash": sha256_bytes(canonical_json(wheel_manifest)),
            "model_refs": model_refs,
        }
        content_hash = sha256_bytes(canonical_json(manifest))
        proposal_id = "capability-" + content_hash.removeprefix("sha256:")[:24]
        target = self.proposal_root / proposal_id
        if target.exists():
            existing = read_json(target / "proposal.json")
            if existing.get("content_hash") != content_hash:
                raise PlanFlowError("CAPABILITY_CONTENT_CHANGED", "Proposal id collision")
        else:
            (target / "snapshot").mkdir(parents=True)
            shutil.copytree(source, target / "snapshot" / "source")
            shutil.copytree(wheelhouse, target / "snapshot" / "wheelhouse")
            write_json_atomic(
                target / "proposal.json",
                {
                    "proposal_id": proposal_id,
                    "content_hash": content_hash,
                    "created_at": now_iso(),
                    "status": "pending-approval",
                    "manifest": manifest,
                },
            )
        return CapabilityProposal(proposal_id, spec.capability_id, content_hash, "pending-approval", target)

    def approve(self, proposal_id: str, expected_content_hash: str, *, note: str) -> None:
        proposal = self._load(proposal_id)
        if proposal["content_hash"] != expected_content_hash:
            raise PlanFlowError(
                "CAPABILITY_CONTENT_CHANGED",
                "Approval hash does not match the immutable proposal content",
            )
        write_json_atomic(
            self._proposal_path(proposal_id) / "approval.json",
            {"content_hash": expected_content_hash, "note": note, "approved_at": now_iso()},
        )

    def publish(self, proposal_id: str) -> CapabilityDescriptor:
        proposal = self._load(proposal_id)
        approval_path = self._proposal_path(proposal_id) / "approval.json"
        if not approval_path.is_file():
            raise PlanFlowError("CAPABILITY_APPROVAL_REQUIRED", "Capability proposal has not been approved")
        approval = read_json(approval_path)
        if approval.get("content_hash") != proposal["content_hash"]:
            raise PlanFlowError("CAPABILITY_CONTENT_CHANGED", "Approved content no longer matches proposal")
        manifest = proposal["manifest"]
        try:
            registered = self.catalog.resolve(manifest["capability_id"])
        except PlanFlowError as error:
            if error.code != "CAPABILITY_MISSING":
                raise
        else:
            if registered.content_hash != proposal["content_hash"]:
                raise PlanFlowError("CAPABILITY_CONFLICT", "Registered capability has different content")
            return registered

        self._verify_snapshot(proposal_id, manifest)
        base = self._run(["docker", "image", "inspect", "--format", "{{.Id}}", self.base_image_ref])
        if base.stdout.strip() != self.base_image_id:
            raise PlanFlowError("CAPABILITY_BASE_CHANGED", "Base image reference no longer matches approved image id")
        context = self._write_build_context(proposal_id, proposal)
        image_tag = "dj-capability-" + proposal["content_hash"].removeprefix("sha256:")[:24]
        self._run(["docker", "build", "--pull=false", "--network=none", "--tag", image_tag, str(context)])
        inspected = self._run(["docker", "image", "inspect", "--format", "{{.Id}}", image_tag])
        image_id = inspected.stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise PlanFlowError("CAPABILITY_BUILD_FAILED", "Docker did not return an immutable image id")
        check = (
            "import importlib,sys;sys.path.insert(0,'/opt/dj-capabilities/source');"
            f"importlib.import_module({manifest['import_module']!r});"
            "from data_juicer.ops.base_op import OPERATORS;"
            f"assert OPERATORS.get({manifest['operator_name']!r}) is not None"
        )
        self._run([
            "docker", "run", "--rm", "--network", "none", "--entrypoint",
            "/opt/dj-venv/bin/python", image_id, "-c", check,
        ])
        descriptor = CapabilityDescriptor(
            capability_id=manifest["capability_id"],
            operator_name=manifest["operator_name"],
            content_hash=proposal["content_hash"],
            backend="docker",
            backend_ref={"image_id": image_id},
            base_image_id=self.base_image_id,
            created_at=now_iso(),
            source_hash=manifest["source_hash"],
            dependency_lock_hash=manifest["dependency_lock_hash"],
            model_refs=tuple(manifest["model_refs"]),
        )
        return self.catalog.register(descriptor)

    def _run(self, argv: list[str]):
        try:
            return self.command_runner(argv, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            stderr = getattr(error, "stderr", "")
            raise PlanFlowError("CAPABILITY_BUILD_FAILED", "Capability Docker build failed", details=stderr) from error

    def _verify_snapshot(self, proposal_id: str, manifest: dict) -> None:
        snapshot = self._proposal_path(proposal_id) / "snapshot"
        for key, directory in (("source", "source"), ("wheels", "wheelhouse")):
            expected = manifest[key]
            actual_files = self._files(snapshot / directory, suffix=".whl" if key == "wheels" else None)
            actual = self._manifest(snapshot / directory, actual_files)
            if actual != expected:
                raise PlanFlowError("CAPABILITY_CONTENT_CHANGED", "Capability snapshot changed after proposal")

    def _write_build_context(self, proposal_id: str, proposal: dict) -> Path:
        target = self.build_root / proposal_id
        marker = target / "content-hash.txt"
        if target.exists():
            if marker.is_file() and marker.read_text(encoding="utf-8").strip() == proposal["content_hash"]:
                return target
            raise PlanFlowError("CAPABILITY_CONTENT_CHANGED", f"Unexpected existing build context: {target}")
        target.mkdir(parents=True)
        snapshot = self._proposal_path(proposal_id) / "snapshot"
        shutil.copytree(snapshot / "source", target / "source")
        shutil.copytree(snapshot / "wheelhouse", target / "wheelhouse")
        manifest = proposal["manifest"]
        bootstrap = (
            "import importlib, os, sys\n"
            "os.environ.setdefault('HOME', '/tmp/dj-home')\n"
            "os.makedirs(os.environ['HOME'], exist_ok=True)\n"
            "sys.path.insert(0, '/opt/dj-capabilities/source')\n"
            f"importlib.import_module({manifest['import_module']!r})\n"
            "from data_juicer.tools.plan_flow.container_entry import main\n"
            "raise SystemExit(main())\n"
        )
        (target / "bootstrap.py").write_text(bootstrap, encoding="utf-8")
        dockerfile = (
            f"FROM {self.base_image_ref}\n"
            "USER root\n"
            "COPY wheelhouse /tmp/capability-wheelhouse\n"
            "RUN /usr/local/bin/uv pip install --python /opt/dj-venv/bin/python --no-index "
            "--find-links /tmp/capability-wheelhouse /tmp/capability-wheelhouse/*.whl "
            "&& rm -rf /tmp/capability-wheelhouse\n"
            "COPY source /opt/dj-capabilities/source\n"
            "COPY bootstrap.py /opt/dj-capabilities/bootstrap.py\n"
            "USER 10001:10001\n"
            'ENTRYPOINT ["/opt/dj-venv/bin/python", "/opt/dj-capabilities/bootstrap.py"]\n'
        )
        (target / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        marker.write_text(proposal["content_hash"] + "\n", encoding="utf-8")
        return target

    def _proposal_path(self, proposal_id: str) -> Path:
        if not re.fullmatch(r"capability-[0-9a-f]{24}", proposal_id):
            raise PlanFlowError("INVALID_CAPABILITY", "Invalid capability proposal id")
        return self.proposal_root / proposal_id

    def _load(self, proposal_id: str) -> dict:
        return read_json(self._proposal_path(proposal_id) / "proposal.json")

    @staticmethod
    def _files(root: Path, *, suffix: str | None = None) -> list[Path]:
        files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
        if suffix:
            files = [path for path in files if path.suffix == suffix]
        return sorted(files)

    @staticmethod
    def _manifest(root: Path, files: list[Path]) -> list[dict[str, str]]:
        return [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            for path in files
        ]


def _require_identifier(value: str, label: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", value):
        raise PlanFlowError("INVALID_CAPABILITY", f"Invalid {label}: {value}")


def _require_module(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", value):
        raise PlanFlowError("INVALID_CAPABILITY", f"Invalid import module: {value}")

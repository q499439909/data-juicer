"""Resolve the operators used by one plan into one immutable runtime image."""

from __future__ import annotations

import re
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .capability_schema import ArtifactRef, CapabilityCatalog, OperatorArtifact, OperatorArtifactCatalog
from .common import FileLock, PlanFlowError, canonical_json, read_json, sha256_bytes, write_json_atomic
from .runtime_manifest import RuntimeCatalog, RuntimeManifest

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PACKAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True, order=True)
class DependencyPin:
    package: str
    version: str
    wheel_sha256: str

    def __post_init__(self) -> None:
        if not _PACKAGE.fullmatch(str(self.package or "")) or not str(self.version or "").strip():
            raise PlanFlowError("INVALID_RUNTIME_DEPENDENCY", "Dependency package and version must be pinned")
        if not _SHA256.fullmatch(str(self.wheel_sha256 or "")):
            raise PlanFlowError("INVALID_RUNTIME_DEPENDENCY", "Dependency wheel must have an immutable SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {"package": self.package, "version": self.version, "wheel_sha256": self.wheel_sha256}


@dataclass(frozen=True)
class RuntimeBuildSpec:
    composition_hash: str
    base_image_id: str
    data_juicer_identity: str
    operator_artifact_ids: tuple[str, ...]
    dependencies: tuple[DependencyPin, ...]
    model_refs: tuple[ArtifactRef, ...]
    bootstrap_version: str
    profile_family: str


ImageBuilder = Callable[[RuntimeBuildSpec], str]


class RuntimeResolver:
    """Compose every non-built-in operator in a plan into a single cached runtime."""

    def __init__(
        self,
        worker_root: str | Path,
        *,
        base_image_id: str,
        data_juicer_identity: str,
        image_builder: ImageBuilder,
        dependency_pins: dict[str, tuple[DependencyPin, ...]] | None = None,
        bootstrap_version: str = "1",
    ):
        if not _SHA256.fullmatch(str(base_image_id or "")):
            raise PlanFlowError("INVALID_RUNTIME_MANIFEST", "base_image_id must be immutable")
        self.worker_root = Path(worker_root).resolve()
        self.base_image_id = base_image_id
        self.data_juicer_identity = str(data_juicer_identity)
        self.image_builder = image_builder
        self.dependency_pins = dependency_pins or {}
        self.bootstrap_version = str(bootstrap_version)
        self.capabilities = CapabilityCatalog(self.worker_root)
        self.artifacts = OperatorArtifactCatalog(self.worker_root)
        self.runtimes = RuntimeCatalog(self.worker_root)
        self.index_root = self.worker_root / "runtime-catalog" / "composition-index"
        self.index_root.mkdir(parents=True, exist_ok=True)

    def resolve(
        self,
        *,
        capability_ids: tuple[str, ...],
        operator_names: tuple[str, ...],
        profile_family: str,
    ) -> RuntimeManifest:
        if not capability_ids or not operator_names:
            raise PlanFlowError("RUNTIME_RESOLUTION_FAILED", "A custom runtime needs capabilities and operators")
        artifact_ids: list[str] = []
        models: list[ArtifactRef] = []
        for capability_id in capability_ids:
            capability = self.capabilities.resolve(capability_id)
            if profile_family not in {item.removeprefix("local-") for item in capability.resource_profiles}:
                raise PlanFlowError("RUNTIME_PROFILE_UNSUPPORTED", f"Capability does not support {profile_family}")
            for artifact_id in capability.operator_artifact_ids:
                if artifact_id not in artifact_ids:
                    artifact_ids.append(artifact_id)
            models.extend(capability.model_refs)

        artifacts = tuple(self.artifacts.resolve(item) for item in artifact_ids)
        exposed = {definition.name for artifact in artifacts for definition in artifact.operators}
        missing = sorted(set(operator_names) - exposed)
        if missing:
            raise PlanFlowError(
                "RUNTIME_OPERATOR_MISSING",
                "Resolved capabilities do not provide all plan operators",
                details={"missing": missing},
            )
        models.extend(model for artifact in artifacts for model in artifact.model_refs)
        model_refs = self._merge_models(models)
        dependencies = self._merge_dependencies(artifacts)
        dependency_lock_hash = sha256_bytes(
            canonical_json({
                "artifact_locks": [
                    {"artifact_id": item.artifact_id, "dependency_lock_hash": item.dependency_lock_hash}
                    for item in artifacts
                ],
                "pins": [item.to_dict() for item in dependencies],
            })
        )
        operator_refs = tuple(ArtifactRef(item.artifact_id, item.content_hash) for item in artifacts)
        composition = {
            "base_image_id": self.base_image_id,
            "data_juicer_identity": self.data_juicer_identity,
            "operator_artifacts": [item.to_dict() for item in operator_refs],
            "dependency_lock_hash": dependency_lock_hash,
            "model_refs": [item.to_dict() for item in model_refs],
            "bootstrap_version": self.bootstrap_version,
            "profile_family": profile_family,
        }
        composition_hash = sha256_bytes(canonical_json(composition))
        index = self.index_root / f"{composition_hash.removeprefix('sha256:')}.json"
        with FileLock(self.index_root / f".{composition_hash.removeprefix('sha256:')}.lock"):
            if index.is_file():
                try:
                    return self.runtimes.resolve(str(read_json(index)["runtime_id"]))
                except (KeyError, TypeError, json.JSONDecodeError, OSError, UnicodeError, PlanFlowError) as exc:
                    raise PlanFlowError(
                        "RUNTIME_CACHE_CORRUPT",
                        "Runtime composition cache is corrupt; it must be repaired before rebuilding",
                    ) from exc
            build_spec = RuntimeBuildSpec(
                composition_hash,
                self.base_image_id,
                self.data_juicer_identity,
                tuple(artifact_ids),
                dependencies,
                model_refs,
                self.bootstrap_version,
                profile_family,
            )
            image_id = self.image_builder(build_spec)
            if not _SHA256.fullmatch(str(image_id or "")):
                raise PlanFlowError("RUNTIME_BUILD_FAILED", "Runtime builder did not return an immutable image id")
            manifest = RuntimeManifest.create(
                base_image_id=self.base_image_id,
                data_juicer_identity=self.data_juicer_identity,
                operator_artifacts=operator_refs,
                dependency_lock_hash=dependency_lock_hash,
                model_refs=model_refs,
                image_id=image_id,
                bootstrap_version=self.bootstrap_version,
                profile_family=profile_family,
            )
            self.runtimes.publish(manifest)
            write_json_atomic(index, {"schema_version": 1, "composition_hash": composition_hash, "runtime_id": manifest.runtime_id})
            return manifest

    def _merge_dependencies(self, artifacts: tuple[OperatorArtifact, ...]) -> tuple[DependencyPin, ...]:
        merged: dict[str, DependencyPin] = {}
        for artifact in artifacts:
            for pin in self.dependency_pins.get(artifact.artifact_id, ()):
                key = re.sub(r"[-_.]+", "-", pin.package).casefold()
                previous = merged.get(key)
                if previous is not None and previous != pin:
                    raise PlanFlowError(
                        "RUNTIME_DEPENDENCY_CONFLICT",
                        f"Conflicting pinned dependency: {pin.package}",
                        details={"first": previous.to_dict(), "second": pin.to_dict()},
                    )
                merged[key] = pin
        return tuple(sorted(merged.values(), key=lambda item: item.package.casefold()))

    @staticmethod
    def _merge_models(models: list[ArtifactRef]) -> tuple[ArtifactRef, ...]:
        merged: dict[str, ArtifactRef] = {}
        for model in models:
            previous = merged.get(model.artifact_id.casefold())
            if previous is not None and previous.sha256 != model.sha256:
                raise PlanFlowError("RUNTIME_MODEL_CONFLICT", f"Conflicting model artifact: {model.artifact_id}")
            merged[model.artifact_id.casefold()] = model
        return tuple(sorted(merged.values()))

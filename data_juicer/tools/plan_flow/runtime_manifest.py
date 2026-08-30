"""Immutable runtime composition values; runtime images live here, not in capabilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .capability_schema import ArtifactRef
from .common import FileLock, PlanFlowError, canonical_json, now_iso, read_json, sha256_bytes, write_json_atomic

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROFILE = re.compile(r"[a-z][a-z0-9._-]{1,63}\Z")


def _hash(value: str, field: str) -> str:
    result = str(value or "")
    if not _SHA256.fullmatch(result):
        raise PlanFlowError("INVALID_RUNTIME_MANIFEST", f"{field} must be a lowercase SHA-256 value")
    return result


def _normalize_refs(values: tuple[ArtifactRef, ...], field: str) -> tuple[ArtifactRef, ...]:
    by_id: dict[str, ArtifactRef] = {}
    for item in values:
        key = item.artifact_id.casefold()
        previous = by_id.get(key)
        if previous is not None and previous.sha256 != item.sha256:
            raise PlanFlowError(
                "INVALID_RUNTIME_MANIFEST",
                f"{field} contains duplicate artifact_id with conflicting content: {item.artifact_id}",
            )
        by_id[key] = item
    return tuple(sorted(by_id.values()))


@dataclass(frozen=True)
class RuntimeManifest:
    runtime_id: str
    content_hash: str
    base_image_id: str
    data_juicer_identity: str
    operator_artifacts: tuple[ArtifactRef, ...]
    dependency_lock_hash: str
    model_refs: tuple[ArtifactRef, ...]
    image_id: str
    bootstrap_version: str
    profile_family: str
    created_at: str
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        base_image_id: str,
        data_juicer_identity: str,
        operator_artifacts: tuple[ArtifactRef, ...],
        dependency_lock_hash: str,
        model_refs: tuple[ArtifactRef, ...],
        image_id: str,
        bootstrap_version: str,
        profile_family: str,
    ) -> "RuntimeManifest":
        base_image_id = _hash(base_image_id, "base_image_id")
        image_id = _hash(image_id, "image_id")
        dependency_lock_hash = _hash(dependency_lock_hash, "dependency_lock_hash")
        data_juicer_identity = str(data_juicer_identity or "").strip()
        bootstrap_version = str(bootstrap_version or "").strip()
        if not data_juicer_identity or not bootstrap_version:
            raise PlanFlowError("INVALID_RUNTIME_MANIFEST", "runtime identities must not be empty")
        if not _PROFILE.fullmatch(str(profile_family or "")):
            raise PlanFlowError("INVALID_RUNTIME_MANIFEST", "profile_family contains unsupported characters")
        operators = _normalize_refs(operator_artifacts, "operator_artifacts")
        models = _normalize_refs(model_refs, "model_refs")
        payload = {
            "schema_version": 1,
            "base_image_id": base_image_id,
            "data_juicer_identity": data_juicer_identity,
            "operator_artifacts": [item.to_dict() for item in operators],
            "dependency_lock_hash": dependency_lock_hash,
            "model_refs": [item.to_dict() for item in models],
            "image_id": image_id,
            "bootstrap_version": bootstrap_version,
            "profile_family": profile_family,
        }
        content_hash = sha256_bytes(canonical_json(payload))
        return cls(
            "runtime-" + content_hash.removeprefix("sha256:")[:24],
            content_hash,
            base_image_id,
            data_juicer_identity,
            operators,
            dependency_lock_hash,
            models,
            image_id,
            bootstrap_version,
            profile_family,
            now_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_id": self.runtime_id,
            "content_hash": self.content_hash,
            "base_image_id": self.base_image_id,
            "data_juicer_identity": self.data_juicer_identity,
            "operator_artifacts": [item.to_dict() for item in self.operator_artifacts],
            "dependency_lock_hash": self.dependency_lock_hash,
            "model_refs": [item.to_dict() for item in self.model_refs],
            "image_id": self.image_id,
            "bootstrap_version": self.bootstrap_version,
            "profile_family": self.profile_family,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuntimeManifest":
        expected = {
            "schema_version", "runtime_id", "content_hash", "base_image_id", "data_juicer_identity",
            "operator_artifacts", "dependency_lock_hash", "model_refs", "image_id", "bootstrap_version",
            "profile_family", "created_at",
        }
        if not isinstance(value, dict) or set(value) != expected or value.get("schema_version") != 1:
            raise PlanFlowError("INVALID_RUNTIME_MANIFEST", "RuntimeManifest fields do not match schema")
        runtime_id = str(value["runtime_id"])
        if not re.fullmatch(r"runtime-[0-9a-f]{24}", runtime_id):
            raise PlanFlowError("INVALID_RUNTIME_MANIFEST", "runtime_id is invalid")
        result = cls(
            runtime_id,
            _hash(str(value["content_hash"]), "content_hash"),
            _hash(str(value["base_image_id"]), "base_image_id"),
            str(value["data_juicer_identity"]),
            _normalize_refs(tuple(ArtifactRef.from_dict(item) for item in value["operator_artifacts"]), "operator_artifacts"),
            _hash(str(value["dependency_lock_hash"]), "dependency_lock_hash"),
            _normalize_refs(tuple(ArtifactRef.from_dict(item) for item in value["model_refs"]), "model_refs"),
            _hash(str(value["image_id"]), "image_id"),
            str(value["bootstrap_version"]),
            str(value["profile_family"]),
            str(value["created_at"]),
        )
        return result


class RuntimeCatalog:
    """Immutable catalog keyed by a runtime composition identity."""

    def __init__(self, worker_root: str | Path):
        self.root = Path(worker_root).resolve() / "runtime-catalog"
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, manifest: RuntimeManifest) -> RuntimeManifest:
        path = self.root / f"{manifest.runtime_id}.json"
        with FileLock(self.root / f".{manifest.runtime_id}.lock"):
            if path.is_file():
                existing = self.resolve(manifest.runtime_id)
                if existing.content_hash != manifest.content_hash:
                    raise PlanFlowError("RUNTIME_CONFLICT", "runtime_id already has different content")
                return existing
            write_json_atomic(path, manifest.to_dict())
        return manifest

    def resolve(self, runtime_id: str) -> RuntimeManifest:
        if not re.fullmatch(r"runtime-[0-9a-f]{24}", str(runtime_id or "")):
            raise PlanFlowError("INVALID_RUNTIME_MANIFEST", "runtime_id is invalid")
        path = self.root / f"{runtime_id}.json"
        if not path.is_file():
            raise PlanFlowError("RUNTIME_MISSING", f"Runtime is not published: {runtime_id}")
        return RuntimeManifest.from_dict(read_json(path))

"""Versioned capability and operator-artifact values independent of runtimes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import FileLock, PlanFlowError, canonical_json, now_iso, read_json, sha256_bytes, write_json_atomic

_IDENTIFIER = re.compile(r"[a-z][a-z0-9._-]{2,127}\Z")
_OPERATOR_NAME = re.compile(r"[a-z][a-z0-9_]{2,127}\Z")
_MODULE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTRACT = re.compile(r"[a-z][a-z0-9_]*@[1-9][0-9]*\Z")
_KINDS = {"mapper", "filter", "selector", "deduplicator", "aggregator", "grouper", "pipeline"}


def _identifier(value: str, field: str) -> str:
    result = str(value or "")
    if not _IDENTIFIER.fullmatch(result):
        raise PlanFlowError("INVALID_CAPABILITY_SCHEMA", f"{field} contains unsupported characters")
    return result


def _sha256(value: str, field: str) -> str:
    result = str(value or "")
    if not _SHA256.fullmatch(result):
        raise PlanFlowError("INVALID_CAPABILITY_SCHEMA", f"{field} must be a lowercase SHA-256 value")
    return result


def _unique(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    if not values or len({item.casefold() for item in values}) != len(values):
        raise PlanFlowError("INVALID_CAPABILITY_SCHEMA", f"{field} must be non-empty and unique")
    return values


@dataclass(frozen=True, order=True)
class ArtifactRef:
    artifact_id: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _identifier(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))

    def to_dict(self) -> dict[str, str]:
        return {"artifact_id": self.artifact_id, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArtifactRef":
        if not isinstance(value, dict) or set(value) != {"artifact_id", "sha256"}:
            raise PlanFlowError("INVALID_CAPABILITY_SCHEMA", "ArtifactRef fields do not match schema")
        return cls(str(value["artifact_id"]), str(value["sha256"]))


@dataclass(frozen=True)
class OperatorDefinition:
    name: str
    import_module: str
    kind: str
    schema_hash: str

    def __post_init__(self) -> None:
        if not _OPERATOR_NAME.fullmatch(str(self.name or "")):
            raise PlanFlowError("INVALID_OPERATOR_ARTIFACT", "operator name contains unsupported characters")
        if not _MODULE.fullmatch(str(self.import_module or "")):
            raise PlanFlowError("INVALID_OPERATOR_ARTIFACT", "import_module is invalid")
        if self.kind not in _KINDS:
            raise PlanFlowError("INVALID_OPERATOR_ARTIFACT", f"unsupported operator kind: {self.kind}")
        object.__setattr__(self, "schema_hash", _sha256(self.schema_hash, "schema_hash"))

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "import_module": self.import_module,
            "kind": self.kind,
            "schema_hash": self.schema_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OperatorDefinition":
        expected = {"name", "import_module", "kind", "schema_hash"}
        if not isinstance(value, dict) or set(value) != expected:
            raise PlanFlowError("INVALID_OPERATOR_ARTIFACT", "OperatorDefinition fields do not match schema")
        return cls(**{key: str(value[key]) for key in expected})


@dataclass(frozen=True)
class OperatorArtifact:
    artifact_id: str
    content_hash: str
    source_hash: str
    dependency_lock_hash: str
    operators: tuple[OperatorDefinition, ...]
    model_refs: tuple[ArtifactRef, ...]
    created_at: str
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        source_hash: str,
        dependency_lock_hash: str,
        operators: tuple[OperatorDefinition, ...],
        model_refs: tuple[ArtifactRef, ...],
    ) -> "OperatorArtifact":
        artifact_id = _identifier(artifact_id, "artifact_id")
        source_hash = _sha256(source_hash, "source_hash")
        dependency_lock_hash = _sha256(dependency_lock_hash, "dependency_lock_hash")
        if not operators or len({item.name.casefold() for item in operators}) != len(operators):
            raise PlanFlowError("INVALID_OPERATOR_ARTIFACT", "operators must be non-empty and unique")
        models = tuple(sorted(model_refs))
        payload = {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "source_hash": source_hash,
            "dependency_lock_hash": dependency_lock_hash,
            "operators": [item.to_dict() for item in operators],
            "model_refs": [item.to_dict() for item in models],
        }
        return cls(
            artifact_id,
            sha256_bytes(canonical_json(payload)),
            source_hash,
            dependency_lock_hash,
            tuple(operators),
            models,
            now_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "source_hash": self.source_hash,
            "dependency_lock_hash": self.dependency_lock_hash,
            "operators": [item.to_dict() for item in self.operators],
            "model_refs": [item.to_dict() for item in self.model_refs],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OperatorArtifact":
        expected = {
            "schema_version", "artifact_id", "content_hash", "source_hash", "dependency_lock_hash",
            "operators", "model_refs", "created_at",
        }
        if not isinstance(value, dict) or set(value) != expected or value.get("schema_version") != 1:
            raise PlanFlowError("INVALID_OPERATOR_ARTIFACT", "OperatorArtifact fields do not match schema")
        result = cls(
            _identifier(str(value["artifact_id"]), "artifact_id"),
            _sha256(str(value["content_hash"]), "content_hash"),
            _sha256(str(value["source_hash"]), "source_hash"),
            _sha256(str(value["dependency_lock_hash"]), "dependency_lock_hash"),
            tuple(OperatorDefinition.from_dict(item) for item in value["operators"]),
            tuple(ArtifactRef.from_dict(item) for item in value["model_refs"]),
            str(value["created_at"]),
        )
        if not result.operators or len({item.name.casefold() for item in result.operators}) != len(result.operators):
            raise PlanFlowError("INVALID_OPERATOR_ARTIFACT", "operators must be non-empty and unique")
        return result


@dataclass(frozen=True)
class CapabilityDescriptor:
    capability_id: str
    content_hash: str
    implements: tuple[str, ...]
    operator_artifact_ids: tuple[str, ...]
    model_refs: tuple[ArtifactRef, ...]
    run_network: str
    resource_profiles: tuple[str, ...]
    approval_scope: str
    created_at: str
    schema_version: int = 2

    @classmethod
    def create(
        cls,
        *,
        capability_id: str,
        implements: tuple[str, ...],
        operator_artifact_ids: tuple[str, ...],
        model_refs: tuple[ArtifactRef, ...],
        run_network: str,
        resource_profiles: tuple[str, ...],
        approval_scope: str,
    ) -> "CapabilityDescriptor":
        capability_id = _identifier(capability_id, "capability_id")
        implements = _unique(tuple(implements), "implements")
        if any(not _CONTRACT.fullmatch(item) for item in implements):
            raise PlanFlowError("INVALID_CAPABILITY_SCHEMA", "implements contains an invalid contract")
        artifact_ids = _unique(
            tuple(_identifier(item, "operator_artifact_id") for item in operator_artifact_ids),
            "operator_artifact_ids",
        )
        profiles = _unique(tuple(_identifier(item, "resource_profile") for item in resource_profiles), "resource_profiles")
        if run_network != "none":
            raise PlanFlowError("INVALID_CAPABILITY_SCHEMA", "only run_network=none is supported")
        approval_scope = _identifier(approval_scope, "approval_scope")
        models = tuple(sorted(model_refs))
        payload = {
            "schema_version": 2,
            "capability_id": capability_id,
            "implements": list(implements),
            "operator_artifact_ids": list(artifact_ids),
            "model_refs": [item.to_dict() for item in models],
            "run_network": run_network,
            "resource_profiles": list(profiles),
            "approval_scope": approval_scope,
        }
        return cls(
            capability_id,
            sha256_bytes(canonical_json(payload)),
            implements,
            artifact_ids,
            models,
            run_network,
            profiles,
            approval_scope,
            now_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capability_id": self.capability_id,
            "content_hash": self.content_hash,
            "implements": list(self.implements),
            "operator_artifact_ids": list(self.operator_artifact_ids),
            "model_refs": [item.to_dict() for item in self.model_refs],
            "run_network": self.run_network,
            "resource_profiles": list(self.resource_profiles),
            "approval_scope": self.approval_scope,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CapabilityDescriptor":
        expected = {
            "schema_version", "capability_id", "content_hash", "implements", "operator_artifact_ids",
            "model_refs", "run_network", "resource_profiles", "approval_scope", "created_at",
        }
        if not isinstance(value, dict) or set(value) != expected or value.get("schema_version") != 2:
            raise PlanFlowError("INVALID_CAPABILITY_SCHEMA", "CapabilityDescriptor fields do not match schema")
        result = cls(
            _identifier(str(value["capability_id"]), "capability_id"),
            _sha256(str(value["content_hash"]), "content_hash"),
            tuple(str(item) for item in value["implements"]),
            tuple(_identifier(str(item), "operator_artifact_id") for item in value["operator_artifact_ids"]),
            tuple(ArtifactRef.from_dict(item) for item in value["model_refs"]),
            str(value["run_network"]),
            tuple(str(item) for item in value["resource_profiles"]),
            _identifier(str(value["approval_scope"]), "approval_scope"),
            str(value["created_at"]),
        )
        if result.run_network != "none":
            raise PlanFlowError("INVALID_CAPABILITY_SCHEMA", "only run_network=none is supported")
        _unique(result.implements, "implements")
        _unique(result.operator_artifact_ids, "operator_artifact_ids")
        _unique(result.resource_profiles, "resource_profiles")
        return result


class OperatorArtifactCatalog:
    """Immutable local catalog for independently reusable operator artifacts."""

    def __init__(self, worker_root: str | Path):
        self.root = Path(worker_root).resolve() / "operator-artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, artifact: OperatorArtifact) -> OperatorArtifact:
        path = self.root / artifact.artifact_id / "artifact.json"
        lock = self.root / f".{artifact.artifact_id}.lock"
        with FileLock(lock):
            if path.is_file():
                existing = self.resolve(artifact.artifact_id)
                if existing.content_hash != artifact.content_hash:
                    raise PlanFlowError("OPERATOR_ARTIFACT_CONFLICT", "artifact_id already has different content")
                return existing
            path.parent.mkdir(parents=False, exist_ok=False)
            write_json_atomic(path, artifact.to_dict())
        return artifact

    def resolve(self, artifact_id: str) -> OperatorArtifact:
        artifact_id = _identifier(artifact_id, "artifact_id")
        path = self.root / artifact_id / "artifact.json"
        if not path.is_file():
            raise PlanFlowError("OPERATOR_ARTIFACT_MISSING", f"Operator artifact is not published: {artifact_id}")
        return OperatorArtifact.from_dict(read_json(path))


class CapabilityCatalog:
    """Immutable semantic capability catalog, deliberately independent of runtime images."""

    def __init__(self, worker_root: str | Path):
        self.root = Path(worker_root).resolve() / "broker-state" / "capabilities-v2"
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, descriptor: CapabilityDescriptor) -> CapabilityDescriptor:
        path = self.root / f"{descriptor.capability_id}.json"
        with FileLock(self.root / f".{descriptor.capability_id}.lock"):
            if path.is_file():
                existing = self.resolve(descriptor.capability_id)
                if existing.content_hash != descriptor.content_hash:
                    raise PlanFlowError("CAPABILITY_CONFLICT", "capability_id already has different content")
                return existing
            write_json_atomic(path, descriptor.to_dict())
        return descriptor

    def resolve(self, capability_id: str) -> CapabilityDescriptor:
        capability_id = _identifier(capability_id, "capability_id")
        path = self.root / f"{capability_id}.json"
        if not path.is_file():
            raise PlanFlowError("CAPABILITY_MISSING", f"Capability is not published: {capability_id}")
        return CapabilityDescriptor.from_dict(read_json(path))

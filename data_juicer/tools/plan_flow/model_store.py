"""Local, content-verified model staging and atomic publication."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import FileLock, PlanFlowError, is_within, now_iso, read_yaml, sha256_file, write_json_atomic

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MANIFEST_NAME = "model-manifest.yaml"
_STORE_METADATA = {"_publication.json"}
_STAGE_METADATA = {"_stage.json"}


def _tree_hash(root: Path, files: tuple["ModelFile", ...]) -> str:
    """Hash ordered path names and bytes, making a multi-file artifact unambiguous."""
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.path.encode("utf-8"))
        digest.update(b"\0")
        with (root / item.path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class ModelFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class ModelManifest:
    artifact_id: str
    source: str
    revision: str
    sha256: str
    size_bytes: int
    license_status: str
    files: tuple[ModelFile, ...]

    @classmethod
    def load(cls, path: Path) -> "ModelManifest":
        raw = read_yaml(path)
        expected = {"artifact_id", "source", "revision", "sha256", "size_bytes", "license", "files"}
        if set(raw) != expected:
            raise PlanFlowError(
                "INVALID_MODEL_MANIFEST",
                "Model manifest fields do not match schema",
                details={"missing": sorted(expected - set(raw)), "unknown": sorted(set(raw) - expected)},
            )
        artifact_id = str(raw.get("artifact_id") or "")
        if not _IDENTIFIER.fullmatch(artifact_id):
            raise PlanFlowError("INVALID_MODEL_MANIFEST", "artifact_id contains unsupported characters")
        source = str(raw.get("source") or "").strip()
        revision = str(raw.get("revision") or "").strip()
        if not source or not revision:
            raise PlanFlowError("INVALID_MODEL_MANIFEST", "source and revision must not be empty")
        artifact_hash = str(raw.get("sha256") or "")
        if not _SHA256.fullmatch(artifact_hash):
            raise PlanFlowError("INVALID_MODEL_MANIFEST", "sha256 must be a lowercase SHA-256 value")
        size = raw.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise PlanFlowError("INVALID_MODEL_MANIFEST", "size_bytes must be a non-negative integer")
        license_value = raw.get("license")
        if not isinstance(license_value, dict) or set(license_value) != {"status"}:
            raise PlanFlowError("INVALID_MODEL_MANIFEST", "license must contain exactly status")
        license_status = str(license_value.get("status") or "")
        if license_status not in {"approved", "approved-for-test"}:
            raise PlanFlowError("MODEL_LICENSE_NOT_APPROVED", "Model license status is not approved")
        raw_files = raw.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise PlanFlowError("INVALID_MODEL_MANIFEST", "files must be a non-empty array")
        files: list[ModelFile] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_files):
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise PlanFlowError("INVALID_MODEL_MANIFEST", f"files[{index}] has an invalid field set")
            raw_path = str(item.get("path") or "")
            relative = Path(raw_path)
            normalized = relative.as_posix()
            if (
                not normalized
                or relative.is_absolute()
                or ".." in relative.parts
                or ":" in raw_path
                or "\\" in raw_path
                or normalized.startswith("/")
                or normalized.casefold() in seen
                or normalized in {_MANIFEST_NAME, *_STORE_METADATA, *_STAGE_METADATA}
            ):
                raise PlanFlowError("INVALID_MODEL_MANIFEST", f"files[{index}].path is unsafe or duplicated")
            file_hash = str(item.get("sha256") or "")
            if not _SHA256.fullmatch(file_hash):
                raise PlanFlowError("INVALID_MODEL_MANIFEST", f"files[{index}].sha256 is invalid")
            seen.add(normalized.casefold())
            files.append(ModelFile(normalized, file_hash))
        files.sort(key=lambda item: item.path)
        return cls(artifact_id, source, revision, artifact_hash, size, license_status, tuple(files))

    def to_provenance(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "source": self.source,
            "revision": self.revision,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "license_status": self.license_status,
        }


@dataclass(frozen=True)
class StagedModel:
    request_id: str
    path: Path


@dataclass(frozen=True)
class ModelArtifact:
    manifest: ModelManifest
    path: Path


class LocalModelInstaller:
    """Copy a local fixture into model-downloads without access to run inputs."""

    def __init__(self, worker_root: str | Path, allowed_source_root: str | Path):
        self.worker_root = Path(worker_root).resolve()
        self.allowed_source_root = Path(allowed_source_root).resolve()
        if not self.worker_root.is_dir() or not self.allowed_source_root.is_dir():
            raise PlanFlowError("MODEL_INSTALLER_ROOT_NOT_FOUND", "Installer roots must already exist")
        fixtures_root = (self.worker_root / "fixtures").resolve()
        if not is_within(self.allowed_source_root, fixtures_root):
            raise PlanFlowError("MODEL_SOURCE_NOT_ALLOWED", "Installer source root must be under worker fixtures")
        self.downloads_root = self.worker_root / "model-downloads"
        self.downloads_root.mkdir(parents=True, exist_ok=True)

    def stage_local(self, request_id: str, source_dir: str | Path) -> StagedModel:
        if not _IDENTIFIER.fullmatch(str(request_id or "")):
            raise PlanFlowError("INVALID_MODEL_REQUEST", "Model request_id contains unsupported characters")
        source = Path(source_dir)
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise PlanFlowError("MODEL_SOURCE_NOT_FOUND", f"Model source is unavailable: {source}") from exc
        if (
            not is_within(resolved, self.allowed_source_root)
            or source.absolute().is_symlink()
            or not resolved.is_dir()
        ):
            raise PlanFlowError("MODEL_SOURCE_NOT_ALLOWED", "Installer source must be a real directory under fixtures")
        for child in resolved.rglob("*"):
            if child.is_symlink():
                raise PlanFlowError("MODEL_SOURCE_NOT_ALLOWED", f"Model source contains a symbolic link: {child}")
        destination = (self.downloads_root / request_id).resolve()
        if destination.parent != self.downloads_root.resolve():
            raise PlanFlowError("MODEL_SOURCE_NOT_ALLOWED", "Model request escaped downloads root")
        with FileLock(self.downloads_root / f".{request_id}.lock"):
            if destination.exists():
                raise PlanFlowError("MODEL_REQUEST_EXISTS", f"Model request already exists: {request_id}")
            shutil.copytree(resolved, destination)
            write_json_atomic(
                destination / "_stage.json",
                {"schema_version": 1, "request_id": request_id, "status": "staged", "staged_at": now_iso()},
            )
        return StagedModel(request_id, destination)


class LocalModelStore:
    """Verify and atomically publish immutable local model artifacts."""

    def __init__(self, worker_root: str | Path):
        self.worker_root = Path(worker_root).resolve()
        if not self.worker_root.is_dir():
            raise PlanFlowError("MODEL_STORE_ROOT_NOT_FOUND", "ModelStore worker root must already exist")
        self.downloads_root = (self.worker_root / "model-downloads").resolve()
        self.models_root = (self.worker_root / "models").resolve()
        self.downloads_root.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)

    def publish(self, request_id: str) -> ModelArtifact:
        if not _IDENTIFIER.fullmatch(str(request_id or "")):
            raise PlanFlowError("INVALID_MODEL_REQUEST", "Model request_id contains unsupported characters")
        staging = (self.downloads_root / request_id).resolve()
        if staging.parent != self.downloads_root or not staging.is_dir() or staging.is_symlink():
            raise PlanFlowError("MODEL_REQUEST_NOT_FOUND", f"Unknown staged model request: {request_id}")
        manifest = self._verify_directory(staging, allow_stage_metadata=True)
        destination = (self.models_root / manifest.artifact_id).resolve()
        with FileLock(self.models_root / f".{manifest.artifact_id}.lock"):
            if destination.exists():
                existing = self.resolve(manifest.artifact_id)
                if existing.manifest != manifest:
                    raise PlanFlowError("MODEL_ARTIFACT_CONFLICT", "Published artifact_id has different content")
                return existing
            publishing = self.models_root / f".publishing-{manifest.artifact_id}-{uuid.uuid4().hex}"
            try:
                publishing.mkdir(parents=False, exist_ok=False)
                shutil.copy2(staging / _MANIFEST_NAME, publishing / _MANIFEST_NAME)
                for item in manifest.files:
                    source = staging / item.path
                    target = publishing / item.path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                write_json_atomic(
                    publishing / "_publication.json",
                    {
                        "schema_version": 1,
                        "artifact_id": manifest.artifact_id,
                        "request_id": request_id,
                        "status": "published",
                        "published_at": now_iso(),
                    },
                )
                self._verify_directory(publishing, allow_store_metadata=True)
                os.replace(publishing, destination)
            except Exception:
                if publishing.is_dir():
                    shutil.rmtree(publishing)
                raise
        return ModelArtifact(manifest, destination)

    def resolve(self, artifact_id: str) -> ModelArtifact:
        if not _IDENTIFIER.fullmatch(str(artifact_id or "")):
            raise PlanFlowError("INVALID_MODEL_ARTIFACT", "Model artifact_id contains unsupported characters")
        path = (self.models_root / artifact_id).resolve()
        if path.parent != self.models_root or not path.is_dir() or path.is_symlink():
            raise PlanFlowError("MODEL_NOT_FOUND", f"Model artifact is not published: {artifact_id}")
        manifest = self._verify_directory(path, allow_store_metadata=True)
        if manifest.artifact_id != artifact_id:
            raise PlanFlowError("MODEL_INTEGRITY_FAILED", "Model directory and manifest identities differ")
        return ModelArtifact(manifest, path)

    def verify(self, artifact_id: str) -> ModelManifest:
        return self.resolve(artifact_id).manifest

    @staticmethod
    def _verify_directory(
        root: Path, *, allow_stage_metadata: bool = False, allow_store_metadata: bool = False
    ) -> ModelManifest:
        manifest = ModelManifest.load(root / _MANIFEST_NAME)
        permitted = {_MANIFEST_NAME, *(item.path for item in manifest.files)}
        if allow_stage_metadata:
            permitted.update(_STAGE_METADATA)
        if allow_store_metadata:
            permitted.update(_STORE_METADATA)
        actual_files: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise PlanFlowError("MODEL_INTEGRITY_FAILED", f"Model artifact contains a symbolic link: {path}")
            if path.is_file():
                actual_files.add(path.relative_to(root).as_posix())
        unexpected = actual_files - permitted
        missing = {item.path for item in manifest.files} - actual_files
        if unexpected or missing:
            raise PlanFlowError(
                "MODEL_INTEGRITY_FAILED",
                "Model artifact file inventory does not match manifest",
                details={"missing": sorted(missing), "unexpected": sorted(unexpected)},
            )
        total = 0
        for item in manifest.files:
            path = root / item.path
            if sha256_file(path) != item.sha256:
                raise PlanFlowError("MODEL_HASH_MISMATCH", f"Model file hash mismatch: {item.path}")
            total += path.stat().st_size
        if total != manifest.size_bytes:
            raise PlanFlowError("MODEL_SIZE_MISMATCH", "Model artifact size does not match manifest")
        if _tree_hash(root, manifest.files) != manifest.sha256:
            raise PlanFlowError("MODEL_HASH_MISMATCH", "Model aggregate hash does not match manifest")
        return manifest

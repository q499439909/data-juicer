"""Policy-constrained artifact acquisition into an isolated quarantine."""

from __future__ import annotations

import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import quote, urlparse

from .common import FileLock, PlanFlowError, is_within, now_iso, sha256_file, write_json_atomic

_IDENTIFIER = re.compile(r"[a-z][a-z0-9._-]{2,127}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FLOATING_REVISIONS = {"head", "latest", "main", "master", "stable", "dev", "develop"}
_REMOTE_HOSTS = {
    "github.com",
    "raw.githubusercontent.com",
    "huggingface.co",
    "modelscope.cn",
    "www.modelscope.cn",
    "pypi.org",
    "files.pythonhosted.org",
}
_SAFE_MODEL_SUFFIXES = {
    ".json", ".md", ".onnx", ".safetensors", ".txt", ".yaml", ".yml",
    ".model", ".tflite", ".xml", ".data",
}
_SAFE_OPERATOR_SUFFIXES = {".py", ".pyi", ".toml", ".txt", ".whl", ".json", ".md"}


@dataclass(frozen=True)
class FetchRequest:
    """Closed request schema accepted from the control plane; never a command."""

    request_id: str
    artifact_id: str
    kind: str
    source: str
    revision: str
    files: tuple[str, ...]
    expected_sha256: dict[str, str]
    max_bytes: int
    license_status: str

    def validate(self) -> None:
        for field, value in (("request_id", self.request_id), ("artifact_id", self.artifact_id)):
            if not _IDENTIFIER.fullmatch(str(value or "")):
                raise PlanFlowError("INVALID_FETCH_REQUEST", f"{field} contains unsupported characters")
        if self.kind not in {"model", "operator-source", "dependency"}:
            raise PlanFlowError("INVALID_FETCH_REQUEST", f"unsupported artifact kind: {self.kind}")
        revision = str(self.revision or "").strip()
        if not revision or revision.casefold() in _FLOATING_REVISIONS:
            raise PlanFlowError("FLOATING_REVISION", "Artifact revision must be immutable, not a floating alias")
        if self.license_status not in {"approved", "approved-for-test"}:
            raise PlanFlowError("ARTIFACT_LICENSE_NOT_APPROVED", "Artifact license is not approved")
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int) or self.max_bytes <= 0:
            raise PlanFlowError("INVALID_FETCH_REQUEST", "max_bytes must be a positive integer")
        if not self.files or len(set(self.files)) != len(self.files):
            raise PlanFlowError("INVALID_FETCH_REQUEST", "files must be non-empty and unique")
        if set(self.expected_sha256) != set(self.files):
            raise PlanFlowError("INVALID_FETCH_REQUEST", "expected_sha256 must cover exactly the requested files")
        suffixes = _SAFE_MODEL_SUFFIXES if self.kind == "model" else _SAFE_OPERATOR_SUFFIXES
        for raw_path in self.files:
            path = PurePosixPath(raw_path)
            if path.is_absolute() or ".." in path.parts or "\\" in raw_path or ":" in raw_path:
                raise PlanFlowError("INVALID_FETCH_REQUEST", f"unsafe artifact path: {raw_path}")
            if path.suffix.casefold() not in suffixes:
                raise PlanFlowError("UNSAFE_ARTIFACT_FORMAT", f"Artifact format is not allowed: {raw_path}")
            if self.kind == "dependency" and path.suffix.casefold() == ".whl" and "manylinux" not in path.name:
                raise PlanFlowError(
                    "WHEEL_PLATFORM_MISMATCH",
                    f"Docker runtime dependency must be a manylinux wheel: {raw_path}",
                )
            if not _SHA256.fullmatch(str(self.expected_sha256[raw_path] or "")):
                raise PlanFlowError("INVALID_FETCH_REQUEST", f"Invalid SHA-256 for {raw_path}")


@dataclass(frozen=True)
class StagedArtifact:
    request: FetchRequest
    path: Path
    manifest: dict


Downloader = Callable[[FetchRequest, str, Path], None]


class ArtifactFetcher:
    """Fetch exact files with no shell, credentials, or access to run inputs."""

    def __init__(
        self,
        worker_root: str | Path,
        *,
        allowed_local_roots: tuple[str | Path, ...] = (),
        downloader: Downloader | None = None,
    ):
        self.worker_root = Path(worker_root).resolve()
        if not self.worker_root.is_dir():
            raise PlanFlowError("FETCHER_ROOT_NOT_FOUND", "ArtifactFetcher worker root must already exist")
        self.allowed_local_roots = tuple(Path(item).resolve() for item in allowed_local_roots)
        self.downloader = downloader or self._download_remote
        self.quarantine_root = self.worker_root / "quarantine"
        self.quarantine_root.mkdir(parents=True, exist_ok=True)

    def fetch(self, request: FetchRequest) -> StagedArtifact:
        request.validate()
        local_source = self._validate_source(request.source)
        destination = self.quarantine_root / request.request_id
        with FileLock(self.quarantine_root / f".{request.request_id}.lock"):
            if destination.exists():
                raise PlanFlowError("FETCH_REQUEST_EXISTS", f"Fetch request already exists: {request.request_id}")
            destination.mkdir(parents=False)
            try:
                total = 0
                inventory = []
                for relative in request.files:
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if local_source is not None:
                        source = (local_source / relative).resolve()
                        if not is_within(source, local_source) or not source.is_file() or source.is_symlink():
                            raise PlanFlowError("ARTIFACT_SOURCE_MISSING", f"Requested artifact file is unavailable: {relative}")
                        shutil.copy2(source, target)
                    else:
                        self.downloader(request, relative, target)
                    size = target.stat().st_size
                    total += size
                    if total > request.max_bytes:
                        raise PlanFlowError("ARTIFACT_TOO_LARGE", "Artifact exceeds the approved byte limit")
                    digest = sha256_file(target)
                    if digest != request.expected_sha256[relative]:
                        raise PlanFlowError("ARTIFACT_HASH_MISMATCH", f"Artifact hash mismatch: {relative}")
                    inventory.append({"path": relative, "sha256": digest, "size_bytes": size})
                manifest = {
                    "schema_version": 1,
                    "request_id": request.request_id,
                    "artifact_id": request.artifact_id,
                    "kind": request.kind,
                    "source": request.source,
                    "revision": request.revision,
                    "license_status": request.license_status,
                    "status": "verified",
                    "total_bytes": total,
                    "files": inventory,
                    "verified_at": now_iso(),
                }
                write_json_atomic(destination / "_fetch.json", manifest)
                return StagedArtifact(request, destination, manifest)
            except Exception:
                shutil.rmtree(destination, ignore_errors=True)
                raise

    def _validate_source(self, source: str) -> Path | None:
        parsed = urlparse(str(source or ""))
        if "://" in str(source):
            if parsed.scheme != "https" or parsed.hostname not in _REMOTE_HOSTS:
                raise PlanFlowError("ARTIFACT_SOURCE_NOT_ALLOWED", "Remote artifact source is not approved")
            return None
        try:
            resolved = Path(source).resolve(strict=True)
        except OSError as exc:
            raise PlanFlowError("ARTIFACT_SOURCE_MISSING", "Local artifact source does not exist") from exc
        if not resolved.is_dir() or resolved.is_symlink():
            raise PlanFlowError("ARTIFACT_SOURCE_NOT_ALLOWED", "Local artifact source must be a real directory")
        if not any(is_within(resolved, root) for root in self.allowed_local_roots):
            raise PlanFlowError("ARTIFACT_SOURCE_NOT_ALLOWED", "Local artifact source is outside approved roots")
        return resolved

    @staticmethod
    def _download_remote(request: FetchRequest, relative: str, target: Path) -> None:
        source = request.source.rstrip("/")
        parsed = urlparse(source)
        if parsed.hostname == "huggingface.co":
            url = f"{source}/resolve/{quote(request.revision, safe='')}/{quote(relative)}"
        elif parsed.hostname in {"github.com", "raw.githubusercontent.com"}:
            url = f"{source}/raw/{quote(request.revision, safe='')}/{quote(relative)}"
        else:
            url = f"{source}/{quote(relative)}?revision={quote(request.revision, safe='')}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        except Exception as exc:
            raise PlanFlowError("ARTIFACT_DOWNLOAD_FAILED", f"Could not fetch approved artifact file: {relative}") from exc

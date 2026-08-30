"""One-copy input snapshots and bounded recursive output artifact collection."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import PlanFlowError, canonical_json, is_within, sha256_bytes, sha256_file, write_json_atomic

_MEDIA_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif",
    ".mp3", ".wav", ".flac", ".mp4", ".mov", ".avi", ".mkv",
}
_OUTPUT_MANIFEST = "_output-manifest.json"


@dataclass(frozen=True)
class DatasetSnapshot:
    path: Path
    dataset_path: Path
    manifest: dict[str, Any]


class DatasetSnapshotter:
    """Snapshot raw media once and rewrite every reference for the container mount."""

    def __init__(
        self,
        *,
        allowed_input_roots: tuple[str | Path, ...],
        max_files: int = 500,
        max_total_bytes: int = 10 * 1024 * 1024 * 1024,
    ):
        self.allowed_roots = tuple(Path(item).resolve() for item in allowed_input_roots)
        if not self.allowed_roots or any(not item.is_dir() for item in self.allowed_roots):
            raise PlanFlowError("INPUT_ROOT_NOT_FOUND", "Every allowed input root must be an existing directory")
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes

    def create(
        self,
        source: str | Path,
        staging_root: str | Path,
        *,
        container_root: str = "/workspace/input",
    ) -> DatasetSnapshot:
        source_path = Path(source).resolve()
        self._require_allowed(source_path)
        if source_path.is_symlink():
            raise PlanFlowError("INPUT_SYMLINK_REJECTED", "Input source must not be a symbolic link")
        staging = Path(staging_root).resolve()
        if staging.exists() and any(staging.iterdir()):
            raise PlanFlowError("INPUT_STAGING_EXISTS", "Input staging directory is not empty")
        staging.mkdir(parents=True, exist_ok=True)
        media_root = staging / "media"
        media_root.mkdir()
        copied: dict[Path, str] = {}
        inventory: list[dict[str, Any]] = []
        total_bytes = 0

        def snapshot_file(raw: str | Path, *, relative_to: Path) -> str:
            nonlocal total_bytes
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = relative_to / candidate
            resolved = candidate.resolve()
            self._require_allowed(resolved)
            if not resolved.is_file() or resolved.is_symlink():
                raise PlanFlowError("INPUT_REFERENCE_MISSING", f"Media reference is unavailable: {raw}")
            if resolved in copied:
                return copied[resolved]
            if len(copied) >= self.max_files:
                raise PlanFlowError("INPUT_TOO_MANY_FILES", "Input snapshot exceeds the approved file count")
            size = resolved.stat().st_size
            total_bytes += size
            if total_bytes > self.max_total_bytes:
                raise PlanFlowError("INPUT_TOO_LARGE", "Input snapshot exceeds the approved byte limit")
            digest = sha256_file(resolved)
            target_name = digest.removeprefix("sha256:")[:24] + resolved.suffix.casefold()
            target = media_root / target_name
            if not target.exists():
                shutil.copy2(resolved, target)
            container_path = f"{container_root.rstrip('/')}/media/{target_name}"
            copied[resolved] = container_path
            inventory.append({
                "source_name": resolved.name,
                "container_path": container_path,
                "sha256": digest,
                "size_bytes": size,
            })
            return container_path

        if source_path.is_dir():
            files = sorted(
                item for item in source_path.rglob("*")
                if item.is_file() and item.suffix.casefold() in _MEDIA_SUFFIXES
            )
            if not files:
                raise PlanFlowError("INPUT_EMPTY", "Input directory contains no supported media")
            records = [
                {"source_id": item.relative_to(source_path).as_posix(), "images": [snapshot_file(item, relative_to=source_path)]}
                for item in files
            ]
        elif source_path.is_file() and source_path.suffix.casefold() == ".jsonl":
            records = []
            for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PlanFlowError("INVALID_INPUT_MANIFEST", f"Invalid JSONL at line {line_number}") from exc
                records.append(self._rewrite_record(record, source_path.parent, snapshot_file))
            if not records:
                raise PlanFlowError("INPUT_EMPTY", "Input JSONL contains no records")
        else:
            raise PlanFlowError("INPUT_FORMAT_UNSUPPORTED", "Input must be a media directory or JSONL manifest")

        dataset_path = staging / "dataset.jsonl"
        dataset_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        snapshot_hash = sha256_bytes(canonical_json({"files": inventory, "dataset_sha256": sha256_file(dataset_path)}))
        manifest = {
            "schema_version": 1,
            "snapshot_hash": snapshot_hash,
            "record_count": len(records),
            "copied_file_count": len(copied),
            "total_bytes": total_bytes,
            "dataset_path": f"{container_root.rstrip('/')}/dataset.jsonl",
            "files": inventory,
        }
        write_json_atomic(staging / "snapshot-manifest.json", manifest)
        return DatasetSnapshot(staging, dataset_path, manifest)

    def _require_allowed(self, path: Path) -> None:
        if not any(is_within(path, root) for root in self.allowed_roots):
            raise PlanFlowError("INPUT_PATH_NOT_ALLOWED", f"Input path is outside explicitly allowed roots: {path}")

    @staticmethod
    def _rewrite_record(value: Any, base: Path, snapshot_file) -> Any:
        if isinstance(value, dict):
            return {key: DatasetSnapshotter._rewrite_record(item, base, snapshot_file) for key, item in value.items()}
        if isinstance(value, list):
            return [DatasetSnapshotter._rewrite_record(item, base, snapshot_file) for item in value]
        if isinstance(value, str):
            candidate = Path(value)
            resolved = candidate if candidate.is_absolute() else base / candidate
            if candidate.suffix.casefold() in _MEDIA_SUFFIXES:
                return snapshot_file(value, relative_to=base)
        return value


class OutputArtifactCollector:
    """Inventory output recursively while rejecting links and unbounded artifacts."""

    def __init__(self, *, max_total_bytes: int, max_files: int = 10_000):
        self.max_total_bytes = max_total_bytes
        self.max_files = max_files

    def collect(self, output_root: str | Path) -> dict[str, Any]:
        root = Path(output_root).resolve()
        if not root.is_dir() or root.is_symlink():
            raise PlanFlowError("OUTPUT_NOT_FOUND", "Output root must be a real directory")
        files = []
        total = 0
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise PlanFlowError("OUTPUT_SYMLINK_REJECTED", f"Output contains a symbolic link: {path}")
            if not path.is_file() or path.name == _OUTPUT_MANIFEST:
                continue
            if len(files) >= self.max_files:
                raise PlanFlowError("OUTPUT_TOO_MANY_FILES", "Output exceeds the approved file count")
            size = path.stat().st_size
            total += size
            if total > self.max_total_bytes:
                raise PlanFlowError("OUTPUT_TOO_LARGE", "Output exceeds the approved byte limit")
            files.append({
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": size,
            })
        manifest = {
            "schema_version": 1,
            "total_bytes": total,
            "file_count": len(files),
            "files": files,
            "content_hash": sha256_bytes(canonical_json(files)),
        }
        write_json_atomic(root / _OUTPUT_MANIFEST, manifest)
        return manifest

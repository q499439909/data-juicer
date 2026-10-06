"""Build self-contained output datasets from manifests that reference source media."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .common import PlanFlowError, is_within, sha256_file


DEFAULT_MEDIA_KEYS = {
    "images",
    "image",
    "image_path",
    "videos",
    "video",
    "video_path",
    "audios",
    "audio",
    "audio_path",
}


def _safe_output_path(root: Path, value: str, label: str) -> Path:
    path = (root / value).resolve()
    if not is_within(path, root):
        raise PlanFlowError("PATH_NOT_ALLOWED", f"{label} must stay inside the Run output directory")
    return path


def package_datasets(
    output_root: str | Path, workspace: str | Path, manifests: list[dict[str, Any]]
) -> dict[str, Any]:
    """Copy manifest-referenced media into outputs and rewrite references to relative paths."""
    output_root = Path(output_root).resolve()
    workspace = Path(workspace).resolve()
    results = []
    for spec in manifests:
        manifest = _safe_output_path(output_root, str(spec.get("path", "")), "Package manifest")
        media_root = _safe_output_path(output_root, str(spec.get("media_dir", "")), "Package media_dir")
        if manifest.suffix.casefold() != ".jsonl" or not manifest.is_file() or manifest.is_symlink():
            raise PlanFlowError("PACKAGE_MANIFEST_INVALID", f"Package manifest is unavailable: {manifest}")
        if media_root.exists():
            if not media_root.is_dir() or media_root.is_symlink() or any(media_root.iterdir()):
                raise PlanFlowError("PACKAGE_OUTPUT_EXISTS", f"Package media directory must be new or empty: {media_root}")
        else:
            media_root.mkdir(parents=True)
        media_keys = set(DEFAULT_MEDIA_KEYS)
        media_keys.update(str(key) for key in spec.get("media_keys", []) if str(key))
        copied: dict[Path, str] = {}
        targets: dict[str, str] = {}
        copied_bytes = 0

        def package_value(value, base: Path, in_media=False):
            nonlocal copied_bytes
            if isinstance(value, dict):
                return {key: package_value(item, base, in_media or key in media_keys) for key, item in value.items()}
            if isinstance(value, list):
                return [package_value(item, base, in_media) for item in value]
            if not in_media or not isinstance(value, str):
                return value
            if "://" in value:
                raise PlanFlowError("PACKAGE_REMOTE_MEDIA_UNSUPPORTED", "Materialize remote media before packaging")
            source = (base / value).resolve()
            if not is_within(source, workspace) or not source.is_file() or source.is_symlink():
                raise PlanFlowError("PACKAGE_MEDIA_MISSING", f"Package media is unavailable: {source}")
            if is_within(source, output_root):
                return Path(os.path.relpath(source, manifest.parent)).as_posix()
            if source not in copied:
                before = source.stat()
                digest = sha256_file(source)
                name = source.name
                target = media_root / name
                relative_target = Path(os.path.relpath(target, manifest.parent)).as_posix()
                if relative_target in targets and targets[relative_target] != digest:
                    target = media_root / f"{source.stem}-{digest.removeprefix('sha256:')[:12]}{source.suffix}"
                    relative_target = Path(os.path.relpath(target, manifest.parent)).as_posix()
                created = not target.exists()
                if not created:
                    if target.is_symlink() or sha256_file(target) != digest:
                        raise PlanFlowError(
                            "PACKAGE_OUTPUT_CONFLICT", f"Package output conflicts with source media: {target}"
                        )
                else:
                    shutil.copyfile(source, target)
                    after = source.stat()
                    if (
                        (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                        or sha256_file(target) != digest
                    ):
                        target.unlink(missing_ok=True)
                        raise PlanFlowError("PACKAGE_MEDIA_CHANGED", f"Package media changed while copying: {source}")
                copied[source] = relative_target
                targets[relative_target] = digest
                if created:
                    copied_bytes += source.stat().st_size
            return copied[source]

        temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
        records = 0
        try:
            with manifest.open(encoding="utf-8-sig") as reader, temporary.open("w", encoding="utf-8") as writer:
                for line in reader:
                    if not line.strip():
                        continue
                    records += 1
                    writer.write(
                        json.dumps(package_value(json.loads(line), manifest.parent), ensure_ascii=False) + "\n"
                    )
            os.replace(temporary, manifest)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        results.append(
            {
                "manifest": str(manifest),
                "media_dir": str(media_root),
                "records": records,
                "media_files": len(targets),
                "media_bytes": copied_bytes,
            }
        )
    return {"packages": results}

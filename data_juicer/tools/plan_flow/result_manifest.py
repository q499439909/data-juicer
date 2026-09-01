"""Create the verified inventory consumed by the result-center gateway."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import PlanFlowError, sha256_file, write_json_atomic


def write_result_manifest(
    output_root: str | Path,
    *,
    run_id: str,
    started_at: str,
    finished_at: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze every current output file into one path-safe integrity manifest."""
    root = Path(output_root).resolve()
    if not root.is_dir():
        raise PlanFlowError("RESULT_OUTPUT_MISSING", f"Run output directory does not exist: {root}")
    outputs: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.name == "result-manifest.json" or relative.parts[:1] == (".dataset-archives",):
            continue
        if path.is_symlink():
            raise PlanFlowError("OUTPUT_SYMLINK_NOT_ALLOWED", f"Output contains a symbolic link: {relative}")
        if path.is_file():
            outputs.append({
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    manifest = {
        **(metadata or {}),
        "schema_version": 1,
        "run_id": str(run_id),
        "status": "succeeded",
        "started_at": str(started_at),
        "finished_at": str(finished_at),
        "output_count": len(outputs),
        "output_size_bytes": sum(item["size_bytes"] for item in outputs),
        "outputs": outputs,
    }
    write_json_atomic(root / "result-manifest.json", manifest)
    return manifest

"""Resolve immutable user-supplied model files by digest."""

from __future__ import annotations

import os
from pathlib import Path

from ..common import PlanFlowError, sha256_file


class LocalFileModelBackend:
    name = "local-file"

    def prepare(self, binding: dict, *, offline: bool) -> Path:
        candidates = []
        configured = os.environ.get("DSH_PRELOADED_MODEL_ROOT")
        if configured:
            root = Path(configured).expanduser().resolve()
            candidates.extend(
                (
                    root / binding["lock_id"] / binding.get("filename", ""),
                    root / binding.get("filename", ""),
                )
            )
        source = binding.get("source_path")
        if source:
            candidates.append(Path(source).expanduser().resolve())
        for path in candidates:
            if path.is_file() and sha256_file(path).removeprefix("sha256:") == binding["sha256"]:
                return path
        raise PlanFlowError("MODEL_NOT_PREPARED", f"Locked local model is not available: {binding['lock_id']}")

    def verify(self, binding: dict, path: Path) -> None:
        if not path.is_file():
            raise PlanFlowError("MODEL_FILE_MISSING", f"Model file is missing: {path}")
        if sha256_file(path).removeprefix("sha256:") != binding["sha256"]:
            raise PlanFlowError("MODEL_HASH_MISMATCH", f"Locked local model changed: {binding['lock_id']}")

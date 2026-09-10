"""Resolve resources shipped inside a uv-locked Python distribution."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path

from ..common import PlanFlowError, sha256_file


class PythonDistributionModelBackend:
    name = "python-distribution"

    def prepare(self, binding: dict, *, offline: bool) -> Path:
        try:
            actual = importlib.metadata.version(binding["distribution"])
        except importlib.metadata.PackageNotFoundError as exc:
            raise PlanFlowError(
                "RUNTIME_PACKAGE_MISSING", f"Required locked package is absent: {binding['distribution']}"
            ) from exc
        if actual != binding["version"]:
            raise PlanFlowError(
                "RUNTIME_PACKAGE_MISMATCH",
                f"Installed {binding['distribution']}=={actual}, expected {binding['version']}",
            )
        module = importlib.import_module(binding["module"])
        root = Path(module.__file__).resolve().parent
        path = (root / binding["resource"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PlanFlowError("MODEL_PATH_FORBIDDEN", "Package resource escaped its distribution") from exc
        return path

    def verify(self, binding: dict, path: Path) -> None:
        if not path.is_file() or path.stat().st_size != int(binding["size"]):
            raise PlanFlowError("MODEL_FILE_MISSING", f"Locked package resource is missing: {path}")
        if sha256_file(path).removeprefix("sha256:") != binding["sha256"]:
            raise PlanFlowError("MODEL_HASH_MISMATCH", f"Locked package resource changed: {binding['lock_id']}")

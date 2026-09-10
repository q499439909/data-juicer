"""Freeze and verify the Python runtime independently from model artifacts."""

from __future__ import annotations

import importlib.metadata
import sys
import tomllib
from pathlib import Path

from .common import PlanFlowError, sha256_file


_AUDITED_PACKAGES = {
    "data-juicer",
    "huggingface-hub",
    "modelscope",
    "numpy",
    "opencv-contrib-python",
    "torch",
    "transformers",
    "ultralytics",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def locked_package_versions(lock_path: str | Path | None = None) -> dict[str, str]:
    lock_path = Path(lock_path).resolve() if lock_path else project_root() / "uv.lock"
    try:
        raw = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PlanFlowError("RUNTIME_LOCK_MISSING", f"Cannot read Python lock: {lock_path}") from exc
    return {
        str(item["name"]).casefold(): str(item["version"])
        for item in raw.get("package", [])
        if isinstance(item, dict) and item.get("name") and item.get("version")
    }


def freeze_runtime_lock(lock_path: str | Path | None = None) -> dict:
    """Return a path-free identity for the Python dependency graph."""
    path = Path(lock_path).resolve() if lock_path else project_root() / "uv.lock"
    versions = locked_package_versions(path)
    return {
        "schema_version": 1,
        "manager": "uv",
        "lockfile": "uv.lock",
        "sha256": sha256_file(path).removeprefix("sha256:"),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "audited_packages": {name: versions[name] for name in sorted(_AUDITED_PACKAGES & versions.keys())},
    }


def verify_runtime_lock(
    expected: dict, lock_path: str | Path | None = None, *, required_packages: set[str] | None = None
) -> dict:
    """Fail closed if source lock or an installed audited package has drifted."""
    if not isinstance(expected, dict) or expected.get("schema_version") != 1:
        raise PlanFlowError("RUNTIME_LOCK_MISSING", "Plan has no supported Python runtime lock")
    actual = freeze_runtime_lock(lock_path)
    for field in ("manager", "lockfile", "sha256", "python"):
        if expected.get(field) != actual[field]:
            raise PlanFlowError(
                "RUNTIME_LOCK_MISMATCH",
                f"Python runtime lock differs at {field}: expected {expected.get(field)!r}, got {actual[field]!r}",
            )
    required = {name.casefold() for name in (required_packages or set())}
    for name, version in expected.get("audited_packages", {}).items():
        if name.casefold() not in required:
            continue
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # Optional packages are checked by the model/operator binding that needs them.
            continue
        if installed != version:
            raise PlanFlowError(
                "RUNTIME_PACKAGE_MISMATCH",
                f"Installed {name}=={installed} differs from uv.lock ({version})",
            )
    return actual

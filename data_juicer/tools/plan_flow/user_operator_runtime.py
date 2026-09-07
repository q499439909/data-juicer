"""Account-local dependency environments; the shared DJ installation is read-only."""

import importlib.metadata
import os
import sys

from packaging.requirements import Requirement, InvalidRequirement
from packaging.utils import canonicalize_name

from .common import FileLock, PlanFlowError, read_json, write_json_atomic, write_text_atomic
from .user_operator_store import digest


def dependency_lock(requirements):
    if not isinstance(requirements, list) or len(requirements) > 100:
        raise PlanFlowError(
            "DEPENDENCY_LOCK_REQUIRED",
            "dependencies must be a list of at most 100 exact pins",
        )
    pinned = {}
    for value in requirements:
        try:
            req = Requirement(value)
        except (InvalidRequirement, TypeError) as exc:
            raise PlanFlowError("DEPENDENCY_LOCK_REQUIRED", "Invalid dependency requirement") from exc
        specs = list(req.specifier)
        if req.url or len(specs) != 1 or specs[0].operator != "==" or "*" in specs[0].version or req.extras:
            raise PlanFlowError(
                "DEPENDENCY_LOCK_REQUIRED",
                "Supply all additional packages as exact name==version pins without URLs or extras",
            )
        if req.marker and not req.marker.evaluate():
            continue
        name = canonicalize_name(req.name)
        if name in pinned and pinned[name] != specs[0].version:
            raise PlanFlowError("DEPENDENCY_CONFLICT", f"Conflicting versions for {name}")
        pinned[name] = specs[0].version
    return pinned


def runtime_python(store, requirements, run_command=None):
    pinned = dependency_lock(requirements)
    missing = []
    for name, version in pinned.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        # Do not shadow core DJ packages with an incompatible version.
        if installed and installed != version:
            raise PlanFlowError(
                "DEPENDENCY_CONFLICT",
                f"{name} requires {version}; DJ runtime has {installed}",
            )
        if installed is None:
            missing.append(f"{name}=={version}")
    base_packages = [path for path in sys.path if path.rstrip("/\\").endswith("site-packages")]
    runtime_id = digest(
        {
            "python": sys.executable,
            "version": sys.version,
            "dependencies": pinned,
            "base_packages": base_packages,
            "layout": 2,
        }
    )[:24]
    target = store.path(store.home / "operator_runtimes" / runtime_id)
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    executable = str(python).removeprefix("\\\\?\\")
    marker = target / "ready.json"
    if marker.exists() and python.exists() and read_json(marker).get("dependencies") == pinned:
        return executable
    if run_command is None:
        raise PlanFlowError(
            "OPERATOR_RUNTIME_BLOCKED",
            "Validate this dependency combination before running it",
            details=missing,
        )
    with FileLock(target.with_suffix(".lock")):
        if marker.exists() and python.exists():
            return executable
        run_command([sys.executable, "-m", "venv", str(target).removeprefix("\\\\?\\")])
        # A nested venv's system-site-packages points at base Python, not the DJ
        # venv. Explicitly expose the actual DJ packages after this overlay.
        site = target / (
            "Lib/site-packages"
            if os.name == "nt"
            else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
        write_text_atomic(site / "dsh_dj_base.pth", "\n".join(base_packages) + "\n")
        if missing:
            run_command(
                [
                    executable,
                    "-m",
                    "pip",
                    "--disable-pip-version-check",
                    "install",
                    "--no-input",
                    "--no-deps",
                    "--require-virtualenv",
                    *missing,
                ]
            )
        # All transitive requirements must be explicitly present in the lock or DJ base.
        if missing:
            run_command([executable, "-m", "pip", "check"])
        write_json_atomic(marker, {"dependencies": pinned, "python": sys.version})
    return executable

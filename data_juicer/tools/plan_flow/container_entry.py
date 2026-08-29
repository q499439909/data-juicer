"""Validated entry point for an isolated plan-flow run container."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import posixpath
import re
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path, PurePosixPath
from typing import Any

from .common import now_iso, read_yaml, write_json_atomic

DEFAULT_RUN_SPEC = Path("/run/bundle/run-spec.json")
_RECIPE_PATH = PurePosixPath("/run/bundle/materialized-recipe.yaml")
_FIXED_MOUNTS = {
    "input": PurePosixPath("/workspace/input"),
    "bundle": PurePosixPath("/run/bundle"),
    "output": PurePosixPath("/workspace/output"),
    "work": PurePosixPath("/run/work"),
    "temp": PurePosixPath("/tmp"),
}
_RUN_ID = re.compile(r"run[_-][A-Za-z0-9][A-Za-z0-9._-]{0,126}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_WINDOWS_ABSOLUTE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)")
_PATH_KEYS = {
    "custom_operator_paths",
    "dataset_path",
    "export_path",
    "model_path",
    "stats_export_path",
    "temp_dir",
    "work_dir",
}


class ExitCode(IntEnum):
    """Stable process exit codes consumed by an execution backend."""

    SUCCESS = 0
    INVALID_SPEC = 10
    RECIPE_HASH_MISMATCH = 11
    PATH_NOT_ALLOWED = 12
    MOUNT_POLICY_VIOLATION = 13
    EXECUTION_FAILED = 20
    RESULT_MANIFEST_FAILED = 21


class ContainerEntryError(ValueError):
    """A safe, structured run failure."""

    def __init__(self, code: str, message: str, exit_code: ExitCode, *, details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            error["details"] = self.details
        return {"ok": False, "error": error}


@dataclass(frozen=True)
class ModelMount:
    artifact_id: str
    path: PurePosixPath


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    tenant_id: str
    recipe_path: PurePosixPath
    recipe_sha256: str
    mounts: dict[str, PurePosixPath]
    models: tuple[ModelMount, ...]


@dataclass(frozen=True)
class PreparedRun:
    spec: RunSpec
    recipe: dict[str, Any]
    recipe_path: Path
    output_root: Path


def _invalid_spec(message: str, *, details: Any = None) -> ContainerEntryError:
    return ContainerEntryError("INVALID_RUN_SPEC", message, ExitCode.INVALID_SPEC, details=details)


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid_spec(f"{location} must be an object")
    return value


def _require_exact_keys(value: dict[str, Any], location: str, expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        raise _invalid_spec(
            f"{location} has an invalid field set",
            details={"missing": sorted(expected - actual), "unexpected": sorted(actual - expected)},
        )


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid_spec(f"{location} must be a non-empty string")
    return value


def _parse_run_spec(value: Any) -> RunSpec:
    raw = _require_object(value, "run-spec")
    _require_exact_keys(raw, "run-spec", {"schema_version", "run_id", "tenant_id", "recipe", "mounts", "models"})
    if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
        raise _invalid_spec("schema_version must be 1")

    run_id = _require_string(raw["run_id"], "run_id")
    tenant_id = _require_string(raw["tenant_id"], "tenant_id")
    if not _RUN_ID.fullmatch(run_id):
        raise _invalid_spec("run_id must start with run_ or run- and contain only safe identifier characters")
    if not _IDENTIFIER.fullmatch(tenant_id):
        raise _invalid_spec("tenant_id contains unsupported characters")

    raw_recipe = _require_object(raw["recipe"], "recipe")
    _require_exact_keys(raw_recipe, "recipe", {"path", "sha256"})
    recipe_path = PurePosixPath(_require_string(raw_recipe["path"], "recipe.path"))
    recipe_sha256 = _require_string(raw_recipe["sha256"], "recipe.sha256")
    if recipe_path != _RECIPE_PATH:
        raise _invalid_spec(f"recipe.path must be {_RECIPE_PATH}")
    if not _SHA256.fullmatch(recipe_sha256):
        raise _invalid_spec("recipe.sha256 must be a lowercase sha256 digest")

    raw_mounts = _require_object(raw["mounts"], "mounts")
    _require_exact_keys(raw_mounts, "mounts", set(_FIXED_MOUNTS))
    mounts: dict[str, PurePosixPath] = {}
    for name, expected in _FIXED_MOUNTS.items():
        path = PurePosixPath(_require_string(raw_mounts[name], f"mounts.{name}"))
        if path != expected:
            raise _invalid_spec(f"mounts.{name} must be {expected}")
        mounts[name] = path

    if not isinstance(raw["models"], list):
        raise _invalid_spec("models must be an array")
    models: list[ModelMount] = []
    seen_artifacts: set[str] = set()
    for index, item in enumerate(raw["models"]):
        raw_model = _require_object(item, f"models[{index}]")
        _require_exact_keys(raw_model, f"models[{index}]", {"artifact_id", "path"})
        artifact_id = _require_string(raw_model["artifact_id"], f"models[{index}].artifact_id")
        if not _IDENTIFIER.fullmatch(artifact_id):
            raise _invalid_spec(f"models[{index}].artifact_id contains unsupported characters")
        if artifact_id in seen_artifacts:
            raise _invalid_spec(f"Duplicate model artifact: {artifact_id}")
        seen_artifacts.add(artifact_id)
        path = PurePosixPath(_require_string(raw_model["path"], f"models[{index}].path"))
        if path != PurePosixPath("/models") / artifact_id:
            raise _invalid_spec(f"models[{index}].path must be /models/{artifact_id}")
        models.append(ModelMount(artifact_id=artifact_id, path=path))

    return RunSpec(
        run_id=run_id,
        tenant_id=tenant_id,
        recipe_path=recipe_path,
        recipe_sha256=recipe_sha256,
        mounts=mounts,
        models=tuple(models),
    )


def _load_run_spec(path: Path) -> RunSpec:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _invalid_spec(f"Run spec does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _invalid_spec(f"Run spec is not valid UTF-8 JSON: {exc}") from exc
    return _parse_run_spec(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _normal_posix_path(value: str, location: str) -> PurePosixPath:
    if _WINDOWS_ABSOLUTE.match(value):
        raise ContainerEntryError(
            "HOST_PATH_NOT_ALLOWED",
            f"{location} contains a Windows host path",
            ExitCode.PATH_NOT_ALLOWED,
        )
    path = PurePosixPath(value)
    if not path.is_absolute():
        raise ContainerEntryError(
            "RELATIVE_PATH_NOT_ALLOWED",
            f"{location} must be an absolute materialized container path",
            ExitCode.PATH_NOT_ALLOWED,
        )
    if ".." in path.parts or value.startswith("//"):
        raise ContainerEntryError(
            "PATH_TRAVERSAL",
            f"{location} contains a disallowed path traversal",
            ExitCode.PATH_NOT_ALLOWED,
        )
    return PurePosixPath(posixpath.normpath(value))


def _is_within_posix(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path == root or root in path.parents


def _require_under(value: str, root: PurePosixPath, location: str) -> PurePosixPath:
    path = _normal_posix_path(value, location)
    if not _is_within_posix(path, root):
        raise ContainerEntryError(
            "PATH_NOT_ALLOWED",
            f"{location} must be inside {root}",
            ExitCode.PATH_NOT_ALLOWED,
        )
    return path


def _walk_path_values(value: Any, location: str = "recipe"):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{location}.{key}"
            key_is_path = key in _PATH_KEYS or key.endswith("_path") or key.endswith("_dir")
            if key_is_path:
                values = item if isinstance(item, list) else [item]
                for index, candidate in enumerate(values):
                    if isinstance(candidate, str) and candidate:
                        suffix = f"[{index}]" if isinstance(item, list) else ""
                        yield child + suffix, candidate
            yield from _walk_path_values(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_path_values(item, f"{location}[{index}]")


def _validate_recipe_paths(recipe: dict[str, Any], spec: RunSpec) -> None:
    allowed_roots = [
        spec.mounts["input"],
        spec.mounts["output"],
        spec.mounts["work"],
        spec.mounts["temp"],
        spec.mounts["bundle"],
        *(model.path for model in spec.models),
    ]
    for location, value in _walk_path_values(recipe):
        path = _normal_posix_path(value, location)
        if not any(_is_within_posix(path, root) for root in allowed_roots):
            raise ContainerEntryError(
                "PATH_NOT_ALLOWED",
                f"{location} is outside the declared container roots",
                ExitCode.PATH_NOT_ALLOWED,
            )

    export_path = recipe.get("export_path")
    if not isinstance(export_path, str) or not export_path:
        raise ContainerEntryError("EXPORT_PATH_REQUIRED", "recipe.export_path is required", ExitCode.PATH_NOT_ALLOWED)
    _require_under(export_path, spec.mounts["output"], "recipe.export_path")

    if recipe.get("work_dir") != str(spec.mounts["work"]):
        raise ContainerEntryError(
            "WORK_DIR_NOT_ALLOWED",
            f"recipe.work_dir must be {spec.mounts['work']}",
            ExitCode.PATH_NOT_ALLOWED,
        )
    if recipe.get("temp_dir") != str(spec.mounts["temp"]):
        raise ContainerEntryError(
            "TEMP_DIR_NOT_ALLOWED",
            f"recipe.temp_dir must be {spec.mounts['temp']}",
            ExitCode.PATH_NOT_ALLOWED,
        )

    if recipe.get("dataset_path"):
        _require_under(str(recipe["dataset_path"]), spec.mounts["input"], "recipe.dataset_path")
    dataset = recipe.get("dataset")
    if isinstance(dataset, dict):
        configs = dataset.get("configs")
        if not isinstance(configs, list) or not configs:
            raise ContainerEntryError(
                "INVALID_DATASET",
                "recipe.dataset.configs must be a non-empty array",
                ExitCode.PATH_NOT_ALLOWED,
            )
        for index, config in enumerate(configs):
            if not isinstance(config, dict) or config.get("type") != "local" or not isinstance(config.get("path"), str):
                raise ContainerEntryError(
                    "NON_LOCAL_DATASET",
                    f"recipe.dataset.configs[{index}] must declare a local path",
                    ExitCode.PATH_NOT_ALLOWED,
                )
            _require_under(config["path"], spec.mounts["input"], f"recipe.dataset.configs[{index}].path")


def _decode_mount_field(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _mount_is_read_only(path: Path, mountinfo: str | None = None) -> bool:
    supplied_mountinfo = mountinfo is not None
    if mountinfo is None:
        try:
            mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
        except OSError as exc:
            raise ContainerEntryError(
                "MOUNTINFO_UNAVAILABLE",
                f"Could not read container mount policy: {exc}",
                ExitCode.MOUNT_POLICY_VIOLATION,
            ) from exc
    target = path.as_posix() if supplied_mountinfo else str(path.resolve())
    matches: list[tuple[int, set[str]]] = []
    for line in mountinfo.splitlines():
        before_separator = line.split(" - ", 1)[0].split()
        if len(before_separator) < 6:
            continue
        mount_point = _decode_mount_field(before_separator[4])
        options = set(before_separator[5].split(","))
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            matches.append((len(mount_point), options))
    if not matches:
        return False
    return "ro" in max(matches, key=lambda item: item[0])[1]


def _require_directory(path: Path, location: str) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ContainerEntryError(
            "MOUNT_NOT_FOUND",
            f"{location} must be a mounted directory: {path}",
            ExitCode.MOUNT_POLICY_VIOLATION,
        )


def _require_read_only_mount(path: Path, location: str) -> None:
    _require_directory(path, location)
    if not _mount_is_read_only(path):
        raise ContainerEntryError(
            "READ_ONLY_MOUNT_REQUIRED",
            f"{location} must be mounted read-only: {path}",
            ExitCode.MOUNT_POLICY_VIOLATION,
        )


def _require_writable_mount(path: Path, location: str) -> None:
    _require_directory(path, location)
    if _mount_is_read_only(path) or not os.access(path, os.W_OK):
        raise ContainerEntryError(
            "WRITABLE_MOUNT_REQUIRED",
            f"{location} must be mounted writable: {path}",
            ExitCode.MOUNT_POLICY_VIOLATION,
        )


def _prepare_run(run_spec_path: Path) -> PreparedRun:
    spec = _load_run_spec(run_spec_path)
    recipe_path = Path(str(spec.recipe_path))
    try:
        actual_hash = _sha256_file(recipe_path)
    except OSError as exc:
        raise ContainerEntryError(
            "RECIPE_NOT_FOUND",
            f"Could not read materialized recipe: {recipe_path}",
            ExitCode.RECIPE_HASH_MISMATCH,
        ) from exc
    if not hmac.compare_digest(actual_hash, spec.recipe_sha256):
        raise ContainerEntryError(
            "RECIPE_HASH_MISMATCH",
            "Materialized recipe does not match run-spec",
            ExitCode.RECIPE_HASH_MISMATCH,
            details={"expected": spec.recipe_sha256, "actual": actual_hash},
        )

    recipe = read_yaml(recipe_path)
    _validate_recipe_paths(recipe, spec)
    _require_read_only_mount(Path(str(spec.mounts["input"])), "input")
    _require_read_only_mount(Path(str(spec.mounts["bundle"])), "bundle")
    for model in spec.models:
        _require_read_only_mount(Path(str(model.path)), f"model {model.artifact_id}")
    _require_writable_mount(Path(str(spec.mounts["output"])), "output")
    _require_writable_mount(Path(str(spec.mounts["work"])), "work")
    _require_writable_mount(Path(str(spec.mounts["temp"])), "temp")
    return PreparedRun(spec=spec, recipe=recipe, recipe_path=recipe_path, output_root=Path(str(spec.mounts["output"])))


def _execute_recipe(recipe_path: Path) -> None:
    from data_juicer.config import init_configs
    from data_juicer.core.executor import ExecutorFactory

    cfg = init_configs(["--config", str(recipe_path)], load_configs_only=False)
    ExecutorFactory.create_executor(cfg.executor_type)(cfg).run()


def _output_inventory(output_root: Path) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.name == "result-manifest.json":
            continue
        if path.is_symlink():
            raise ContainerEntryError(
                "OUTPUT_SYMLINK_NOT_ALLOWED",
                f"Output contains a symbolic link: {path.relative_to(output_root)}",
                ExitCode.RESULT_MANIFEST_FAILED,
            )
        if path.is_file():
            relative = path.relative_to(output_root).as_posix()
            outputs.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return outputs


def run_from_spec(run_spec_path: str | Path = DEFAULT_RUN_SPEC) -> dict[str, Any]:
    """Validate one run spec, execute its recipe, and return its result manifest."""
    started_at = now_iso()
    prepared = _prepare_run(Path(run_spec_path))
    _execute_recipe(prepared.recipe_path)
    outputs = _output_inventory(prepared.output_root)
    manifest = {
        "schema_version": 1,
        "run_id": prepared.spec.run_id,
        "tenant_id": prepared.spec.tenant_id,
        "status": "succeeded",
        "started_at": started_at,
        "finished_at": now_iso(),
        "recipe_sha256": prepared.spec.recipe_sha256,
        "output_count": len(outputs),
        "output_size_bytes": sum(item["size_bytes"] for item in outputs),
        "outputs": outputs,
    }
    try:
        write_json_atomic(prepared.output_root / "result-manifest.json", manifest)
    except OSError as exc:
        raise ContainerEntryError(
            "RESULT_MANIFEST_WRITE_FAILED",
            f"Could not write result manifest: {exc}",
            ExitCode.RESULT_MANIFEST_FAILED,
        ) from exc
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an approved Data-Juicer bundle inside an isolated container")
    parser.add_argument("--run-spec", default=str(DEFAULT_RUN_SPEC))
    args = parser.parse_args(argv)
    try:
        manifest = run_from_spec(args.run_spec)
    except ContainerEntryError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return int(exc.exit_code)
    except Exception as exc:
        error = ContainerEntryError(
            "EXECUTION_FAILED",
            f"Data-Juicer execution failed: {exc}",
            ExitCode.EXECUTION_FAILED,
        )
        print(json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return int(error.exit_code)
    print(json.dumps({"ok": True, "result": manifest}, ensure_ascii=False, sort_keys=True))
    return int(ExitCode.SUCCESS)


if __name__ == "__main__":
    raise SystemExit(main())

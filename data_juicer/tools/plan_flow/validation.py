"""Strict plan, recipe, and postprocess validation."""

from __future__ import annotations

import ast
import copy
import re
from pathlib import Path
from typing import Any

from data_juicer.config.config import build_base_parser

from ._runtime import bind_api_operator
from .common import is_within, require_workspace, resolve_workspace_path
from .discovery import operator_schema

_COMMON_OPERATOR_PARAMS = {
    "text_key",
    "image_key",
    "audio_key",
    "video_key",
    "image_bytes_key",
    "query_key",
    "response_key",
    "history_key",
    "system_key",
    "instruction_key",
    "prompt_key",
    "index_key",
    "batch_size",
    "work_dir",
    "skip_op_error",
    "auto_op_parallelism",
    "batch_mode",
    "accelerator",
    "num_cpus",
    "num_gpus",
    "memory",
    "runtime_env",
    "ray_execution_mode",
    "stats_export_path",
    "min_closed_interval",
    "max_closed_interval",
    "reversed_range",
}
_SECRET_MARKERS = ("api_key", "apikey", "password", "secret", "credential", "access_token")
_MODEL_ARTIFACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MODEL_URI = re.compile(r"model-store://([A-Za-z0-9][A-Za-z0-9._-]{0,127})(?:/(.+))?\Z")


def _config_fields() -> set[str]:
    parser = build_base_parser()
    return {action.dest for action in parser._actions if action.dest and action.dest not in {"help", "config"}}


def _validate_secrets(value: Any, path: str, errors: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if (
                any(marker in str(key).lower() for marker in _SECRET_MARKERS)
                and str(key).lower() != "secret_ref"
                and item not in (None, "", False)
            ):
                errors.append(
                    {
                        "code": "PLAINTEXT_SECRET",
                        "path": child,
                        "message": "Secrets must be supplied through the server environment",
                    }
                )
            _validate_secrets(item, child, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_secrets(item, f"{path}[{index}]", errors)


def _walk_strings(value: Any, path: str):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def _validate_models(plan: dict[str, Any], errors: list[dict[str, str]]) -> None:
    raw_models = plan.get("models", [])
    if not isinstance(raw_models, list):
        errors.append({"code": "INVALID_MODELS", "path": "models", "message": "models must be an array"})
        plan["models"] = []
        raw_models = []
    declared: set[str] = set()
    for index, item in enumerate(raw_models):
        location = f"models[{index}]"
        if not isinstance(item, dict) or set(item) != {"artifact_id"}:
            errors.append(
                {"code": "INVALID_MODEL_REF", "path": location, "message": "Model ref must contain exactly artifact_id"}
            )
            continue
        artifact_id = str(item.get("artifact_id") or "")
        if not _MODEL_ARTIFACT_ID.fullmatch(artifact_id) or artifact_id.casefold() in declared:
            errors.append(
                {
                    "code": "INVALID_MODEL_REF",
                    "path": f"{location}.artifact_id",
                    "message": "Model artifact_id is invalid or duplicated",
                }
            )
            continue
        declared.add(artifact_id.casefold())
    for location, value in _walk_strings(plan.get("recipe", {}), "recipe"):
        if not value.startswith("model-store://"):
            continue
        match = _MODEL_URI.fullmatch(value)
        relative_text = match.group(2) if match else None
        if (
            not match
            or (relative_text is not None and Path(relative_text).is_absolute())
            or (relative_text is not None and ".." in Path(relative_text).parts)
            or (relative_text is not None and ":" in relative_text)
            or (relative_text is not None and "\\" in relative_text)
        ):
            errors.append({"code": "INVALID_MODEL_URI", "path": location, "message": "Model URI is invalid"})
        elif match.group(1).casefold() not in declared:
            errors.append(
                {
                    "code": "MODEL_NOT_DECLARED",
                    "path": location,
                    "message": f"Model artifact is not declared: {match.group(1)}",
                }
            )


def normalize_and_validate(
    workspace_root: str, raw_plan: dict[str, Any], *, external_operator_names: frozenset[str] = frozenset(), operator_schemas=None
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    workspace = require_workspace(workspace_root)
    plan = copy.deepcopy(raw_plan)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not isinstance(plan, dict):
        return (
            {},
            {"ok": False, "errors": [{"code": "INVALID_PLAN", "message": "plan must be an object"}], "warnings": []},
            [],
        )
    if not str(plan.get("user_intent", "")).strip():
        errors.append({"code": "USER_INTENT_REQUIRED", "path": "user_intent", "message": "user_intent is required"})
    plan.setdefault("modality", "unknown")
    plan.setdefault("risk_notes", [])
    plan.setdefault("acceptance_criteria", [])
    plan.setdefault("approval_required", True)
    plan.setdefault("postprocess", [])
    plan.setdefault("models", [])

    recipe = plan.get("recipe")
    if not isinstance(recipe, dict):
        errors.append({"code": "RECIPE_REQUIRED", "path": "recipe", "message": "plan.recipe must be an object"})
        recipe = {}
        plan["recipe"] = recipe

    sources = [name for name in ("dataset_path", "dataset", "generated_dataset_config") if recipe.get(name)]
    if len(sources) != 1:
        errors.append(
            {
                "code": "INPUT_SOURCE_COUNT",
                "path": "recipe",
                "message": "Exactly one of dataset_path, dataset, or generated_dataset_config is required",
            }
        )
    if recipe.get("dataset_path"):
        path = resolve_workspace_path(recipe["dataset_path"], workspace)
        if not is_within(path, workspace):
            errors.append(
                {
                    "code": "PATH_NOT_ALLOWED",
                    "path": "recipe.dataset_path",
                    "message": f"Input must be inside workspace: {path}",
                }
            )
        elif not path.exists():
            errors.append(
                {"code": "INPUT_NOT_FOUND", "path": "recipe.dataset_path", "message": f"Input does not exist: {path}"}
            )
        recipe["dataset_path"] = str(path)
    if isinstance(recipe.get("dataset"), dict):
        configs = recipe["dataset"].get("configs")
        if not isinstance(configs, list) or not configs:
            errors.append(
                {
                    "code": "INVALID_DATASET",
                    "path": "recipe.dataset.configs",
                    "message": "dataset.configs must be a non-empty list",
                }
            )
        else:
            types = set()
            for index, config in enumerate(configs):
                if not isinstance(config, dict):
                    errors.append(
                        {
                            "code": "INVALID_DATASET",
                            "path": f"recipe.dataset.configs[{index}]",
                            "message": "Each dataset config must be an object",
                        }
                    )
                    continue
                types.add(config.get("type"))
                if config.get("type") == "local" and config.get("path"):
                    path = resolve_workspace_path(config["path"], workspace)
                    if not is_within(path, workspace):
                        errors.append(
                            {
                                "code": "PATH_NOT_ALLOWED",
                                "path": f"recipe.dataset.configs[{index}].path",
                                "message": f"Local input must be inside workspace: {path}",
                            }
                        )
                    elif not path.exists():
                        errors.append(
                            {
                                "code": "INPUT_NOT_FOUND",
                                "path": f"recipe.dataset.configs[{index}].path",
                                "message": f"Input does not exist: {path}",
                            }
                        )
                    config["path"] = str(path)
            if len(types) > 1:
                errors.append(
                    {
                        "code": "MIXED_DATASET_TYPES",
                        "path": "recipe.dataset.configs",
                        "message": "Data-Juicer does not support mixed dataset types in one config",
                    }
                )

    process = recipe.get("process")
    if not isinstance(process, list) or not process:
        errors.append(
            {"code": "PROCESS_REQUIRED", "path": "recipe.process", "message": "recipe.process must be a non-empty list"}
        )
    else:
        for index, step in enumerate(process):
            location = f"recipe.process[{index}]"
            if not isinstance(step, dict) or len(step) != 1:
                errors.append(
                    {
                        "code": "INVALID_PROCESS_STEP",
                        "path": location,
                        "message": "Each process step must contain exactly one operator",
                    }
                )
                continue
            name, params = next(iter(step.items()))
            schema = (operator_schemas or {}).get(name) or operator_schema(name)
            if schema is None:
                if name not in external_operator_names:
                    errors.append({"code": "OP_NOT_FOUND", "path": location, "message": f"Unknown operator: {name}"})
                    continue
                if params is None:
                    params = {}
                    step[name] = params
                if not isinstance(params, dict):
                    errors.append(
                        {
                            "code": "INVALID_OPERATOR_PARAMS",
                            "path": location,
                            "message": "External operator parameters must be an object",
                        }
                    )
                continue
            if params is None:
                params = {}
                step[name] = params
            if not isinstance(params, dict):
                errors.append(
                    {
                        "code": "INVALID_OPERATOR_PARAMS",
                        "path": location,
                        "message": "Operator parameters must be an object",
                    }
                )
                continue
            tags = set(schema.get("tags", []))
            has_api_mode_switch = "is_api_model" in schema["parameters"]
            api_selected = "api" in tags and (not has_api_mode_switch or params.get("is_api_model") is True)
            is_api_vlm = api_selected and "multimodal" in tags and "api_or_hf_model" in schema["parameters"]
            if api_selected:
                operator_path = f"{location}.{name}"
                resolved_model, binding_errors, binding_warnings = bind_api_operator(
                    operator=name,
                    path=operator_path,
                    is_vlm=is_api_vlm,
                    explicit_model=params.get("api_or_hf_model"),
                )
                errors.extend(binding_errors)
                warnings.extend(binding_warnings)
                if is_api_vlm and resolved_model:
                    params["api_or_hf_model"] = resolved_model
            allowed = set(schema["parameters"]) | _COMMON_OPERATOR_PARAMS
            for unknown in sorted(set(params) - allowed):
                errors.append(
                    {
                        "code": "UNKNOWN_OPERATOR_PARAM",
                        "path": f"{location}.{name}.{unknown}",
                        "message": f"Unknown parameter for {name}: {unknown}",
                    }
                )
            for param_name, details in schema["parameters"].items():
                if details["required"] and param_name not in params:
                    errors.append(
                        {
                            "code": "MISSING_OPERATOR_PARAM",
                            "path": f"{location}.{name}.{param_name}",
                            "message": f"Required parameter is missing: {param_name}",
                        }
                    )
    executor = recipe.get("executor_type", "default")
    if executor not in {"default", "ray", "ray_partitioned"}:
        errors.append(
            {
                "code": "INVALID_EXECUTOR",
                "path": "recipe.executor_type",
                "message": f"Unsupported executor_type: {executor}",
            }
        )
    recipe["executor_type"] = executor
    try:
        np_value = int(recipe.get("np", 1))
        if np_value < 1:
            raise ValueError
        recipe["np"] = np_value
    except (TypeError, ValueError):
        errors.append({"code": "INVALID_NP", "path": "recipe.np", "message": "np must be a positive integer"})

    allowed_config = _config_fields()
    for key in sorted(set(recipe) - allowed_config - {"process"}):
        errors.append(
            {
                "code": "UNKNOWN_RECIPE_FIELD",
                "path": f"recipe.{key}",
                "message": f"Unknown Data-Juicer config field: {key}",
            }
        )

    requested_export = str(recipe.get("export_path") or "processed_data.jsonl")
    output_name = Path(requested_export).name or "processed_data.jsonl"
    recipe["export_path"] = f"${{RUN_OUTPUT}}/{output_name}"
    recipe.pop("work_dir", None)
    recipe.pop("temp_dir", None)

    _validate_models(plan, errors)

    artifact_paths: list[str] = []
    for item in plan.get("artifacts", []) or []:
        raw = item.get("path") if isinstance(item, dict) else item
        if raw:
            artifact_paths.append(str(raw))
    for index, step in enumerate(plan.get("postprocess", []) or []):
        location = f"postprocess[{index}]"
        if not isinstance(step, dict) or step.get("kind") != "python":
            errors.append(
                {
                    "code": "INVALID_POSTPROCESS",
                    "path": location,
                    "message": "Only kind=python postprocess steps are supported",
                }
            )
            continue
        script = step.get("script")
        if not script:
            errors.append(
                {"code": "POSTPROCESS_SCRIPT_REQUIRED", "path": location, "message": "postprocess script is required"}
            )
            continue
        source = resolve_workspace_path(script, workspace)
        if not is_within(source, workspace) or not source.is_file():
            errors.append(
                {
                    "code": "ARTIFACT_NOT_FOUND",
                    "path": f"{location}.script",
                    "message": f"Script must be an existing workspace file: {source}",
                }
            )
            continue
        if str(source) not in artifact_paths:
            artifact_paths.append(str(source))
        try:
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError) as exc:
            errors.append({"code": "POSTPROCESS_SYNTAX", "path": f"{location}.script", "message": str(exc)})

    _validate_secrets(plan, "", errors)
    return plan, {"ok": not errors, "errors": errors, "warnings": warnings}, artifact_paths

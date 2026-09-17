"""Freeze logical model identities and materialize them for one machine at run time."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .common import PlanFlowError, canonical_json, now_iso, sha256_file
from .model_backends import (
    HttpFileModelBackend,
    HuggingFaceModelBackend,
    LocalFileModelBackend,
    ModelScopeModelBackend,
    PythonDistributionModelBackend,
    TorchHubModelBackend,
)

_REVISION = re.compile(r"[a-fA-F0-9]{40,64}")
_SHA256 = re.compile(r"[a-f0-9]{64}")


def _valid_file(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and set(item) == {"path", "size", "sha256"}
        and not Path(str(item["path"])).is_absolute()
        and ".." not in Path(str(item["path"])).parts
        and type(item["size"]) is int
        and item["size"] >= 0
        and bool(_SHA256.fullmatch(str(item["sha256"])))
    )


def load_builtin_model_catalog(path: str | Path | None = None) -> dict[str, Any]:
    catalog_path = Path(path) if path else Path(__file__).with_name("builtin_model_catalog.json")
    value = json.loads(catalog_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("models"), list):
        raise PlanFlowError("MODEL_LOCK_MISSING", f"Invalid built-in model catalog: {catalog_path}")
    if path is None:
        from data_juicer import __version__ as dj_version

        if value.get("dj_version") != dj_version:
            raise PlanFlowError(
                "MODEL_LOCK_MISSING",
                f"Built-in model catalog targets DJ {value.get('dj_version')}, running {dj_version}",
            )
        from .runtime_environment_lock import locked_package_versions

        runtime_versions = locked_package_versions()
    else:
        runtime_versions = None
    consumers: dict[tuple, list[set]] = {}
    for requirement in value.get("blocked_requirements", []):
        if not isinstance(requirement, dict) or not requirement.get("operator") or not requirement.get("reason"):
            raise PlanFlowError("MODEL_LOCK_MISSING", f"Invalid blocked model requirement in {catalog_path}")
        if not isinstance(requirement.get("when", {}), dict):
            raise PlanFlowError("MODEL_LOCK_MISSING", "Invalid blocked model requirement condition")
        defaults = requirement.get("defaults")
        if defaults is not None and (
            not requirement.get("parameter") or not isinstance(defaults, list) or not defaults
        ):
            raise PlanFlowError("MODEL_LOCK_MISSING", "Invalid blocked model requirement defaults")
    for requirement in value.get("exempt_requirements", []):
        model_types = requirement.get("model_types")
        if (
            not isinstance(requirement, dict)
            or not requirement.get("reason")
            or ("model_type" not in requirement and not isinstance(model_types, list))
            or not isinstance(requirement.get("when", {}), dict)
        ):
            raise PlanFlowError("MODEL_LOCK_MISSING", f"Invalid exempt model requirement in {catalog_path}")
    for model in value["models"]:
        backend = model.get("backend")
        if backend not in {"huggingface", "modelscope", "http-file", "torch-hub", "python-distribution"}:
            raise PlanFlowError("MODEL_LOCK_MISSING", f"Invalid model backend in {catalog_path}: {backend}")
        if not model.get("lock_id"):
            raise PlanFlowError("MODEL_LOCK_MISSING", f"Invalid model lock in {catalog_path}")
        if backend in {"huggingface", "modelscope"}:
            valid_identity = model.get("model_id") and _REVISION.fullmatch(str(model.get("revision", "")))
        elif backend == "torch-hub":
            valid_identity = model.get("repository_url") and _REVISION.fullmatch(str(model.get("revision", "")))
        elif backend == "http-file":
            valid_identity = (
                str(model.get("url", "")).startswith(("https://", "http://127.0.0.1:"))
                and model.get("filename")
                and type(model.get("size")) is int
                and bool(_SHA256.fullmatch(str(model.get("sha256", ""))))
            )
        else:
            valid_identity = all(model.get(key) for key in ("distribution", "version", "module", "resource")) and (
                type(model.get("size")) is int and bool(_SHA256.fullmatch(str(model.get("sha256", ""))))
            )
        if not valid_identity:
            raise PlanFlowError("MODEL_LOCK_MISSING", f"Invalid {backend} identity in {model.get('lock_id')}")
        if backend == "python-distribution" and runtime_versions is not None:
            locked_version = runtime_versions.get(str(model["distribution"]).casefold())
            if locked_version != model["version"]:
                raise PlanFlowError(
                    "RUNTIME_LOCK_MISMATCH",
                    f"{model['lock_id']} expects {model['distribution']}=={model['version']}, uv.lock has {locked_version}",
                )
        files = model.get("files", [])
        if backend in {"huggingface", "modelscope", "torch-hub"} and (not files or not isinstance(files, list)):
            raise PlanFlowError("MODEL_LOCK_MISSING", f"Model lock has no file identities: {model.get('lock_id')}")
        for file in files:
            if not _valid_file(file):
                raise PlanFlowError("MODEL_LOCK_MISSING", f"Invalid file identity in {model.get('lock_id')}")
        for consumer in model.get("consumers", []):
            key = (consumer.get("operator"), consumer.get("parameter"))
            defaults = consumer.get("defaults")
            if not key[0] or not isinstance(defaults, list) or not defaults:
                raise PlanFlowError("MODEL_BINDING_CONFLICT", f"Invalid model consumer: {key}")
            if not isinstance(consumer.get("when", {}), dict):
                raise PlanFlowError("MODEL_BINDING_CONFLICT", f"Invalid consumer condition: {key}")
            subpath = consumer.get("subpath")
            if subpath and (Path(str(subpath)).is_absolute() or ".." in Path(str(subpath)).parts):
                raise PlanFlowError("MODEL_PATH_FORBIDDEN", f"Invalid consumer subpath: {subpath}")
            values = set(json.dumps(item, sort_keys=True) for item in defaults)
            if any(values & existing for existing in consumers.get(key, [])):
                raise PlanFlowError("MODEL_BINDING_CONFLICT", f"Overlapping model consumer defaults: {key}")
            consumers.setdefault(key, []).append(values)
    return value


class ModelLockResolver:
    """Single seam between Plan model identities and environment-specific files."""

    def __init__(self, *, catalog_path=None, backends=None):
        self.catalog = load_builtin_model_catalog(catalog_path)
        self.backends = backends or {
            "huggingface": HuggingFaceModelBackend(),
            "http-file": HttpFileModelBackend(),
            "local-file": LocalFileModelBackend(),
            "modelscope": ModelScopeModelBackend(),
            "python-distribution": PythonDistributionModelBackend(),
            "torch-hub": TorchHubModelBackend(),
        }
        self._consumers: dict[str, list[tuple[dict, dict]]] = {}
        self._blocked: dict[str, list[dict]] = {}
        for item in self.catalog.get("blocked_requirements", []):
            self._blocked.setdefault(item["operator"], []).append(item)
        for model in self.catalog["models"]:
            for item in model.get("consumers", []):
                self._consumers.setdefault(item["operator"], []).append((model, item))

    def freeze(self, plan: dict[str, Any], *, personal: dict[str, dict] | None = None) -> list[dict[str, Any]]:
        """Return deterministic, path-free bindings for every declared model consumer."""
        personal = personal or {}
        by_identity: dict[tuple, dict[str, Any]] = {}
        steps = plan.get("recipe", {}).get("process", [])
        for index, step in enumerate(steps):
            if not isinstance(step, dict) or len(step) != 1:
                continue
            operator, params = next(iter(step.items()))
            params = params or {}
            for requirement in self._blocked.get(operator, []):
                if not self._condition_matches(operator, params, requirement.get("when", {})):
                    continue
                parameter = requirement.get("parameter")
                defaults = requirement.get("defaults")
                if parameter and defaults:
                    from .discovery import operator_schema

                    schema = operator_schema(operator) or {}
                    actual = params.get(parameter, schema.get("parameters", {}).get(parameter, {}).get("default"))
                    if actual not in defaults:
                        continue
                raise PlanFlowError(
                    "MODEL_DOWNLOAD_BLOCKED",
                    f"{operator} has an unvalidated built-in model requirement: {requirement['reason']}",
                )
            catalog_matches = [
                item
                for item in self._consumers.get(operator, [])
                if self._condition_matches(operator, params, item[1].get("when", {}))
            ]
            grouped: dict[str | None, list[tuple[dict, dict]]] = {}
            for match in catalog_matches:
                grouped.setdefault(match[1].get("parameter"), []).append(match)
            for parameter, alternatives in grouped.items():
                explicit = (
                    params.get(parameter, alternatives[0][1]["defaults"][0])
                    if parameter
                    else alternatives[0][1]["defaults"][0]
                )
                selected = [(model, consumer) for model, consumer in alternatives if explicit in consumer["defaults"]]
                if len(selected) != 1:
                    raise PlanFlowError(
                        "MODEL_LOCK_MISSING",
                        f"No curated model lock uniquely matches {operator}.{parameter}={explicit!r}",
                    )
                model, consumer = selected[0]
                binding = self._builtin_binding(model)
                self._add_consumer(binding, index, operator, parameter, consumer.get("subpath"))
                self._merge(by_identity, binding)
            candidate = personal.get(operator)
            if candidate:
                for ref in candidate.get("_manifest", {}).get("model_refs", []):
                    parameter = str(ref.get("parameter") or "").strip()
                    if not parameter:
                        raise PlanFlowError(
                            "MODEL_REQUIREMENT_UNDECLARED",
                            f"Custom operator {operator} model_refs must name the constructor parameter",
                        )
                    binding = self._custom_binding(ref)
                    self._add_consumer(binding, index, operator, parameter)
                    self._merge(by_identity, binding)
            elif not candidate:
                undeclared = self._undeclared_hf_parameters(operator, params, catalog_matches)
                if undeclared:
                    raise PlanFlowError(
                        "MODEL_REQUIREMENT_UNDECLARED",
                        f"{operator} has uncurated Hugging Face model parameter(s): {', '.join(undeclared)}",
                    )
        result = sorted(by_identity.values(), key=lambda item: item["binding_id"])
        for item in result:
            item["consumers"].sort(key=lambda value: (value["step_index"], value["parameter"]))
        return result

    def has_catalog_consumer(self, plan: dict[str, Any]) -> bool:
        return any(
            isinstance(step, dict)
            and len(step) == 1
            and (next(iter(step)) in self._consumers or next(iter(step)) in self._blocked)
            for step in plan.get("recipe", {}).get("process", [])
        )

    @staticmethod
    def _condition_matches(operator, params, conditions):
        if not conditions:
            return True
        from .discovery import operator_schema

        schema = operator_schema(operator) or {}
        parameters = schema.get("parameters", {})
        for name, expected in conditions.items():
            actual = params.get(name, parameters.get(name, {}).get("default"))
            if actual != expected:
                return False
        return True

    @staticmethod
    def _undeclared_hf_parameters(operator, params, catalog_matches):
        from .discovery import operator_schema

        schema = operator_schema(operator) or {}
        parameters = schema.get("parameters", {})
        tags = set(schema.get("tags", []))

        def selected(name, fallback=None):
            details = parameters.get(name, {})
            return params.get(name, details.get("default", fallback))

        hf_selected = "hf" in tags
        if "is_hf_model" in parameters:
            hf_selected = bool(selected("is_hf_model", False))
        if "is_api_model" in parameters:
            hf_selected = not bool(selected("is_api_model", False))
        if not hf_selected:
            return []
        declared_parameters = {item[1].get("parameter") for item in catalog_matches}
        result = []
        for name, details in parameters.items():
            lowered = name.casefold()
            is_identity = (
                lowered.startswith("hf_")
                or lowered.endswith("_hf_model")
                or lowered in {"hf_model", "api_or_hf_model", "pretrained_model_name_or_path"}
            )
            if not is_identity or name in declared_parameters or lowered.endswith("_params"):
                continue
            value = params.get(name, details.get("default"))
            if value not in (None, "", False):
                result.append(name)
        return sorted(result)

    def prepare(self, bindings: list[dict[str, Any]], *, offline: bool | None = None) -> dict[str, Path]:
        if offline is None:
            offline = os.environ.get("DSH_MODEL_NETWORK_POLICY", "online").casefold() == "offline"
        resolved = {}
        for binding in bindings:
            backend = self.backends.get(binding.get("backend"))
            if backend is None:
                raise PlanFlowError("MODEL_BINDING_CONFLICT", f"Unknown model backend: {binding.get('backend')}")
            path = backend.prepare(binding, offline=offline)
            backend.verify(binding, path)
            resolved[binding["binding_id"]] = path
        return resolved

    def materialize(self, recipe: dict[str, Any], bindings: list[dict[str, Any]], resolved: dict[str, Path]):
        result = copy.deepcopy(recipe)
        records = []
        for binding in bindings:
            path = resolved.get(binding["binding_id"])
            if path is None:
                raise PlanFlowError("MODEL_NOT_PREPARED", f"Model was not resolved: {binding['binding_id']}")
            for consumer in binding["consumers"]:
                if consumer["parameter"] is None:
                    continue
                step = result["process"][consumer["step_index"]]
                params = step[consumer["operator"]]
                if params is None:
                    params = {}
                    step[consumer["operator"]] = params
                materialized_path = path
                if consumer.get("subpath"):
                    materialized_path = (path / consumer["subpath"]).resolve()
                    try:
                        materialized_path.relative_to(path.resolve())
                    except ValueError as exc:
                        raise PlanFlowError(
                            "MODEL_PATH_FORBIDDEN", "Consumer model subpath escaped its snapshot"
                        ) from exc
                    if not materialized_path.exists():
                        raise PlanFlowError(
                            "MODEL_FILE_MISSING", f"Consumer model path is missing: {materialized_path}"
                        )
                params[consumer["parameter"]] = str(materialized_path)
            records.append(
                {
                    "binding_id": binding["binding_id"],
                    "lock_id": binding["lock_id"],
                    "backend": binding["backend"],
                    "revision": binding.get("revision"),
                    "resolved_path": str(path),
                    "verified_at": now_iso(),
                }
            )
        return result, {"schema_version": 1, "models": records}

    @staticmethod
    def _builtin_binding(model: dict[str, Any]) -> dict[str, Any]:
        stable = {key: copy.deepcopy(value) for key, value in model.items() if key != "consumers"}
        stable["binding_id"] = "model_" + hashlib.sha256(canonical_json(stable)).hexdigest()[:20]
        stable["provider"] = "dj"
        stable["consumers"] = []
        return stable

    @staticmethod
    def _custom_binding(ref: dict[str, Any]) -> dict[str, Any]:
        if ref.get("path"):
            path = Path(ref["path"]).expanduser().resolve()
            digest = str(ref.get("sha256") or sha256_file(path)).removeprefix("sha256:")
            stable = {
                "lock_id": "local-" + digest[:20],
                "backend": "local-file",
                "sha256": digest,
                "filename": path.name,
            }
            runtime = {}
        elif ref.get("backend", "huggingface") in {"huggingface", "modelscope"}:
            backend = ref.get("backend", "huggingface")
            revision = str(ref.get("revision") or "")
            if not re.fullmatch(r"[a-fA-F0-9]{40,64}", revision):
                raise PlanFlowError("MODEL_REVISION_REQUIRED", "Custom model revision must be an immutable commit")
            stable = {
                "lock_id": backend[:2]
                + "-"
                + hashlib.sha256(f"{ref['model_id']}@{revision}".encode()).hexdigest()[:20],
                "backend": backend,
                "model_id": ref["model_id"],
                "revision": revision,
                "files": copy.deepcopy(ref.get("files", [])),
                "runtime_packages": (
                    ["huggingface-hub", "torch", "transformers"] if backend == "huggingface" else ["modelscope"]
                ),
            }
            runtime = {}
        elif ref.get("backend") == "http-file":
            stable = {key: copy.deepcopy(ref[key]) for key in ("backend", "url", "filename", "size", "sha256")}
            stable["sha256"] = str(stable["sha256"]).removeprefix("sha256:")
            stable["lock_id"] = "http-" + stable["sha256"][:20]
            runtime = {}
        elif ref.get("backend") == "torch-hub":
            revision = str(ref.get("revision") or "")
            stable = {
                "lock_id": "th-" + hashlib.sha256(f"{ref['repository_url']}@{revision}".encode()).hexdigest()[:20],
                "backend": "torch-hub",
                "repository_url": ref["repository_url"],
                "revision": revision,
                "entrypoint": ref.get("entrypoint"),
                "files": copy.deepcopy(ref.get("files", [])),
                "runtime_packages": ["torch"],
            }
            runtime = {}
        else:
            raise PlanFlowError("INVALID_MODEL_REFS", f"Unsupported custom model backend: {ref.get('backend')}")
        binding = {**stable, **runtime, "provider": "user", "consumers": []}
        identity = {key: value for key, value in stable.items()}
        binding["binding_id"] = "model_" + hashlib.sha256(canonical_json(identity)).hexdigest()[:20]
        return binding

    @staticmethod
    def attach_local_sources(bindings: list[dict[str, Any]], personal: dict[str, dict]) -> list[dict[str, Any]]:
        """Add host-only source paths in memory; they are never written into the Plan."""
        result = copy.deepcopy(bindings)
        paths = {}
        for candidate in personal.values():
            for ref in candidate.get("_manifest", {}).get("model_refs", []):
                if ref.get("path") and ref.get("sha256"):
                    paths[str(ref["sha256"]).removeprefix("sha256:")] = str(Path(ref["path"]).resolve())
        for binding in result:
            if binding.get("backend") == "local-file":
                source = paths.get(binding.get("sha256"))
                if source:
                    binding["source_path"] = source
        return result

    @staticmethod
    def _add_consumer(binding, index, operator, parameter, subpath=None):
        item = {"step_index": index, "operator": operator, "parameter": parameter}
        if subpath:
            item["subpath"] = subpath
        binding["consumers"].append(item)

    @staticmethod
    def _merge(by_identity, binding):
        identity = {key: value for key, value in binding.items() if key not in {"binding_id", "provider", "consumers"}}
        key = canonical_json(identity)
        existing = by_identity.get(key)
        if existing:
            existing["consumers"].extend(binding["consumers"])
        else:
            by_identity[key] = binding

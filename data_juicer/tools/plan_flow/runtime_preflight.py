"""Side-effect-free runtime assessment for immutable plans."""

from __future__ import annotations

import platform
from collections.abc import Callable
from typing import Any

from .discovery import operator_schema


class RuntimePreflight:
    """Resolve a plan against the configured execution seam without starting work."""

    def __init__(
        self,
        *,
        cuda_available: Callable[[], bool] | None = None,
        platform_name: Callable[[], str] | None = None,
    ):
        self._cuda_available = cuda_available or self._detect_cuda
        self._platform_name = platform_name or platform.platform

    @staticmethod
    def _detect_cuda() -> bool:
        from data_juicer.utils.resource_utils import is_cuda_available

        return bool(is_cuda_available())

    def assess(
        self,
        plan: dict[str, Any],
        *,
        execution_mode: str,
        runtime_resolver_configured: bool,
        broker_configured: bool,
    ) -> dict[str, Any]:
        recipe = plan.get("recipe", {}) if isinstance(plan, dict) else {}
        profile = str(plan.get("execution_profile") or "local-cpu")
        local_cuda = self._safe_cuda_probe()
        profile_requests_cuda = "gpu" in profile.casefold() or "cuda" in profile.casefold()
        target_cuda = local_cuda if execution_mode == "native" else profile_requests_cuda
        available_accelerators = ["cpu", *(["cuda"] if target_cuda else [])]
        blockers: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        operators: list[dict[str, Any]] = []
        model_inputs: list[dict[str, Any]] = []

        if execution_mode == "broker":
            backend = "broker"
            if not broker_configured:
                blockers.append(
                    {
                        "code": "BROKER_REQUIRED",
                        "message": "The execution broker is not configured.",
                    }
                )
        else:
            backend = "local-process"

        bindings = plan.get("capability_bindings", []) if isinstance(plan, dict) else []
        has_external_operators = bool(bindings)
        if execution_mode == "broker" and has_external_operators and not runtime_resolver_configured:
            blockers.append(
                {
                    "code": "RUNTIME_RESOLVER_REQUIRED",
                    "message": "Custom operators require a configured Runtime Resolver.",
                }
            )

        if execution_mode == "native" and profile_requests_cuda and not local_cuda:
            blockers.append(
                {
                    "code": "GPU_REQUIRED",
                    "path": "execution_profile",
                    "message": f"Execution profile {profile!r} requires CUDA, but CUDA is unavailable.",
                }
            )

        process = recipe.get("process", []) if isinstance(recipe, dict) else []
        personal_schemas = {}
        if plan.get("operator_bindings"):
            from .user_operator_store import resolve_bindings
            personal_schemas = resolve_bindings(plan)
        for index, step in enumerate(process if isinstance(process, list) else []):
            if not isinstance(step, dict) or len(step) != 1:
                continue
            name, raw_params = next(iter(step.items()))
            params = raw_params if isinstance(raw_params, dict) else {}
            schema = personal_schemas.get(str(name)) or operator_schema(str(name))
            tags = set((schema or {}).get("tags", []))
            preferred = "cuda" if "gpu" in tags else "cpu"
            explicit_accelerator = str(params.get("accelerator") or "").strip().casefold()
            try:
                requested_gpus = float(params.get("num_gpus", 0) or 0)
            except (TypeError, ValueError):
                requested_gpus = 0
            explicitly_requires_cuda = explicit_accelerator == "cuda" or requested_gpus > 0

            if preferred == "cuda" and target_cuda:
                device = "cuda"
                resolution = "available"
            elif preferred == "cuda":
                device = "cpu"
                resolution = "cpu_fallback"
                warnings.append(
                    {
                        "code": "CPU_FALLBACK",
                        "path": f"recipe.process[{index}].{name}",
                        "operator": str(name),
                        "message": "CUDA is unavailable; Data-Juicer will attempt this GPU-tagged operator on CPU.",
                    }
                )
            else:
                device = "cpu"
                resolution = "available"

            if explicitly_requires_cuda and not target_cuda:
                blockers.append(
                    {
                        "code": "GPU_REQUIRED",
                        "path": f"recipe.process[{index}].{name}",
                        "operator": str(name),
                        "message": "The operator explicitly requests CUDA resources, but the target runtime has none.",
                    }
                )

            operators.append(
                {
                    "process_index": index,
                    "operator": str(name),
                    "preferred_accelerator": preferred,
                    "resolved_device": device,
                    "resolution": resolution,
                }
            )
            if schema:
                for parameter_name, details in schema.get("parameters", {}).items():
                    if "model" not in parameter_name.casefold():
                        continue
                    value = params.get(parameter_name, details.get("default"))
                    if value in (None, ""):
                        continue
                    model_inputs.append(
                        {
                            "operator": str(name),
                            "parameter": parameter_name,
                            "value": value,
                            "source": "plan" if parameter_name in params else "operator_default",
                        }
                    )

        if model_inputs and any("hf" in set((operator_schema(item["operator"]) or {}).get("tags", [])) for item in model_inputs):
            warnings.append(
                {
                    "code": "MODEL_DOWNLOAD_MAY_BE_REQUIRED",
                    "message": "One or more Hugging Face model artifacts may be downloaded on first use.",
                }
            )

        return {
            "ok": not blockers,
            "execution_mode": execution_mode,
            "execution_backend": backend,
            "execution_profile": profile,
            "host": {
                "platform": self._platform_name() if execution_mode == "native" else "broker-managed",
                "probe_source": "local_host" if execution_mode == "native" else "execution_profile",
                "available_accelerators": available_accelerators,
                "cuda_available": target_cuda,
            },
            "runtime_resolution": {
                "required": has_external_operators,
                "configured": runtime_resolver_configured,
                "status": (
                    "ready"
                    if has_external_operators and runtime_resolver_configured
                    else "not_configured" if has_external_operators else "not_required"
                ),
            },
            "operators": operators,
            "model_inputs": model_inputs,
            "blocking_issues": blockers,
            "warnings": warnings,
        }

    def _safe_cuda_probe(self) -> bool:
        try:
            return bool(self._cuda_available())
        except Exception:
            return False

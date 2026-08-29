"""Private runtime binding for operators selected by a plan."""

from __future__ import annotations

import os
from typing import Any


def bind_api_operator(
    *, operator: str, path: str, is_vlm: bool, explicit_model: str | None
) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve server configuration only after an API operator is selected."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    config_file = str(os.environ.get("DJ_PLAN_FLOW_CONFIG_FILE") or "the plan-flow MCP environment file")
    config_template = f"{config_file}.example" if config_file != "the plan-flow MCP environment file" else None
    restart_hint = "then restart the plan-flow MCP; reopening the chat alone does not reload it"

    credentials_configured = bool(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("SK")
    )
    if not credentials_configured:
        template_hint = f" (copy {config_template} first if needed)" if config_template else ""
        errors.append(
            {
                "code": "RUNTIME_API_CREDENTIAL_MISSING",
                "path": path,
                "operator": operator,
                "missing": "api_credentials",
                "message": (
                    f"Operator {operator} requires API credentials. Configure OPENAI_API_KEY or "
                    f"DASHSCOPE_API_KEY in {config_file}{template_hint}, {restart_hint}."
                ),
            }
        )

    base_url_configured = bool(
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
    )
    if not base_url_configured:
        warnings.append(
            {
                "code": "RUNTIME_API_BASE_URL_NOT_CONFIGURED",
                "path": path,
                "operator": operator,
                "missing": "api_base_url",
                "message": (
                    f"Operator {operator} has no custom API base URL configured and will use its provider default. "
                    f"For DashScope or another compatible endpoint, configure OPENAI_BASE_URL, OPENAI_API_URL, "
                    f"or DASHSCOPE_BASE_URL in {config_file}, {restart_hint}."
                ),
            }
        )

    resolved_model = str(explicit_model or "").strip() or None
    if is_vlm and resolved_model is None:
        resolved_model = str(os.environ.get("DJ_VLM_MODEL") or "").strip() or None
        if resolved_model is None:
            template_hint = f" (copy {config_template} first if needed)" if config_template else ""
            errors.append(
                {
                    "code": "RUNTIME_VLM_MODEL_MISSING",
                    "path": f"{path}.api_or_hf_model",
                    "operator": operator,
                    "missing": "vlm_model",
                    "message": (
                        f"API VLM operator {operator} requires an explicit model or DJ_VLM_MODEL. Configure "
                        f"DJ_VLM_MODEL in {config_file}{template_hint}, {restart_hint}."
                    ),
                }
            )

    return resolved_model, errors, warnings

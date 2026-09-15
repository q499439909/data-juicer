# -*- coding: utf-8 -*-
"""MCP server exposing built-in tool specifications to generic agents."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Sequence

from pydantic import ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from data_juicer.agent_tools.core.tool import (
    ToolContext,
    ToolResult,
    ToolSpec,
    get_active_tool_profile,
    list_tool_specs,
)

DEFAULT_SERVER_NAME = "Data-Juicer Agent Tools"
SERVER_INSTRUCTIONS = (
    "Atomic Data-Juicer tools for dataset inspection, operator retrieval, "
    "recipe planning, operator development, and recipe execution. "
    "Tool results are JSON objects with an 'ok' flag; inspect 'message' and "
    "'error_type' when 'ok' is false."
)


def _parameter_annotation(field: FieldInfo) -> Any:
    """Re-attach field constraints to the annotation without its default."""
    annotation = field.annotation if field.annotation is not None else Any
    metadata = copy.copy(field)
    metadata.default = PydanticUndefined
    metadata.default_factory = None
    return Annotated[annotation, metadata]


def build_tool_signature(spec: ToolSpec) -> inspect.Signature:
    """Expand the pydantic input model into a flat keyword signature."""
    required: List[inspect.Parameter] = []
    optional: List[inspect.Parameter] = []
    for name, field in spec.input_model.model_fields.items():
        parameter = inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=_parameter_annotation(field),
            default=(
                inspect.Parameter.empty
                if field.is_required()
                else (field.default_factory() if field.default_factory else field.default)
            ),
        )
        (required if field.is_required() else optional).append(parameter)
    return inspect.Signature(parameters=required + optional, return_annotation=Dict[str, Any])


def invoke_tool_spec(
    spec: ToolSpec,
    *,
    ctx: ToolContext,
    raw_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        result = spec.execute(ctx, raw_kwargs)
    except ValidationError as exc:
        return {
            "ok": False,
            "action": spec.name,
            "error_type": "invalid_arguments",
            "message": f"invalid arguments for {spec.name}: {exc}",
            "validation_errors": exc.errors(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "action": spec.name,
            "error_type": "tool_exception",
            "message": f"{spec.name} failed: {exc}",
        }

    if isinstance(result, ToolResult):
        return result.to_payload(action=spec.name)
    if isinstance(result, dict):
        return result
    return {
        "ok": False,
        "action": spec.name,
        "error_type": "invalid_tool_result",
        "message": f"{spec.name} returned unsupported result type: {type(result)}",
    }


def build_mcp_tool_function(
    spec: ToolSpec,
    *,
    ctx_factory: Callable[[], ToolContext],
) -> Callable[..., Dict[str, Any]]:
    signature = build_tool_signature(spec)

    def _wrapped(**kwargs: Any) -> Dict[str, Any]:
        return invoke_tool_spec(spec, ctx=ctx_factory(), raw_kwargs=kwargs)

    _wrapped.__signature__ = signature  # type: ignore[attr-defined]
    _wrapped.__name__ = spec.name
    _wrapped.__doc__ = _tool_description(spec)
    return _wrapped


def _tool_description(spec: ToolSpec) -> str:
    description = str(spec.description or "").strip()
    notes = [f"effects={spec.effects}"]
    if spec.tags:
        notes.append("tags=" + ",".join(spec.tags))
    if spec.confirmation != "none":
        notes.append(f"confirmation={spec.confirmation} (ask the user before calling this tool)")
    return f"{description}\n\n[{'; '.join(notes)}]"


def _tool_annotations(spec: ToolSpec):
    from mcp.types import ToolAnnotations

    return ToolAnnotations(
        title=spec.name,
        readOnlyHint=spec.effects == "read",
        destructiveHint=spec.effects in {"write", "execute"},
        openWorldHint=spec.effects == "external",
    )


def build_tool_context(working_dir: str | None) -> ToolContext:
    raw = str(working_dir or "").strip() or "./.djx"
    resolved = str(Path(raw).expanduser())
    return ToolContext(working_dir=resolved, artifacts_dir=resolved)


def select_tool_specs(
    *,
    profile: str | None = None,
    tags: Sequence[str] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> List[ToolSpec]:
    specs = list_tool_specs(tags=list(tags or []) or None, profile=profile)
    if include:
        wanted = {str(name).strip() for name in include if str(name).strip()}
        unknown = wanted.difference({spec.name for spec in specs})
        if unknown:
            raise KeyError(f"unknown tool(s): {', '.join(sorted(unknown))}")
        specs = [spec for spec in specs if spec.name in wanted]
    if exclude:
        unwanted = {str(name).strip() for name in exclude if str(name).strip()}
        specs = [spec for spec in specs if spec.name not in unwanted]
    return specs


def build_mcp_server(
    *,
    name: str = DEFAULT_SERVER_NAME,
    working_dir: str | None = None,
    profile: str | None = None,
    tags: Sequence[str] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """Create a FastMCP server exposing the built-in tool registry."""
    from mcp.server.fastmcp import FastMCP

    resolved_profile = profile if profile is not None else get_active_tool_profile()
    specs = select_tool_specs(
        profile=resolved_profile,
        tags=tags,
        include=include,
        exclude=exclude,
    )

    mcp = FastMCP(name, instructions=SERVER_INSTRUCTIONS, host=host, port=port)
    ctx_factory = lambda: build_tool_context(working_dir)  # noqa: E731
    for spec in specs:
        mcp.add_tool(
            build_mcp_tool_function(spec, ctx_factory=ctx_factory),
            name=spec.name,
            description=_tool_description(spec),
            annotations=_tool_annotations(spec),
            structured_output=False,
        )
    return mcp


__all__ = [
    "DEFAULT_SERVER_NAME",
    "build_mcp_server",
    "build_mcp_tool_function",
    "build_tool_context",
    "build_tool_signature",
    "invoke_tool_spec",
    "select_tool_specs",
]

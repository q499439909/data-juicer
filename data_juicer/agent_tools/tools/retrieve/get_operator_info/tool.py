# -*- coding: utf-8 -*-
"""Tool spec for get_operator_info."""

from __future__ import annotations

from data_juicer.agent_tools.core.tool import ToolContext, ToolResult, ToolSpec

from .input import GenericOutput, GetOperatorInfoInput
from .logic import get_operator_info


def _get_operator_info(_ctx: ToolContext, args: GetOperatorInfoInput) -> ToolResult:
    if not args.operator_name.strip():
        return ToolResult.failure(
            summary="operator_name is required for get_operator_info",
            error_type="missing_required",
            data={
                "ok": False,
                "requires": ["operator_name"],
                "message": "operator_name is required for get_operator_info",
            },
        )

    payload = get_operator_info(args.operator_name.strip())
    if payload.get("ok"):
        return ToolResult.success(summary="retrieved operator info", data=payload)
    return ToolResult.failure(
        summary=str(payload.get("message", "operator lookup failed")),
        error_type=str(payload.get("error_type", "operator_lookup_failed")),
        data=payload,
    )


GET_OPERATOR_INFO = ToolSpec(
    name="get_operator_info",
    description="Return structured Data-Juicer operator metadata and parameter schema.",
    input_model=GetOperatorInfoInput,
    output_model=GenericOutput,
    executor=_get_operator_info,
    tags=("retrieve", "operators"),
    effects="read",
    confirmation="none",
)


__all__ = ["GET_OPERATOR_INFO"]

"""Tool spec for list_operator_catalog."""

from __future__ import annotations

from pydantic import BaseModel

from data_juicer.agent_tools.core.tool import ToolContext, ToolResult, ToolSpec

from .input import ListOperatorCatalogInput
from .logic import list_operator_catalog


class ListOperatorCatalogOutput(BaseModel):
    ok: bool = True
    message: str = ""
    operators: list = []
    total_count: int = 0
    returned_count: int = 0
    op_type_filter: str | None = None
    requested_tags: list = []
    include_parameters: bool = False
    limit: int = 0


def _list_operator_catalog(_ctx: ToolContext, args: ListOperatorCatalogInput) -> ToolResult:
    result = list_operator_catalog(
        op_type=args.op_type,
        tags=args.tags,
        include_parameters=args.include_parameters,
        limit=args.limit,
    )

    if result.get("ok"):
        return ToolResult.success(
            summary=f"listed {result['returned_count']} operators",
            data=result,
        )

    return ToolResult.failure(
        summary=result.get("message", "Failed to list operator catalog"),
        error_type=str(result.get("error_type", "operator_catalog_list_failed")),
        data=result,
    )


LIST_OPERATOR_CATALOG = ToolSpec(
    name="list_operator_catalog",
    description=(
        "List the local Data-Juicer built-in operator catalog with descriptions, types, tags, "
        "and optional parameter schemas. Use this as a fallback when targeted retrieval is "
        "insufficient and the agent needs broader catalog context for reasoning."
    ),
    input_model=ListOperatorCatalogInput,
    output_model=ListOperatorCatalogOutput,
    executor=_list_operator_catalog,
    tags=("retrieve", "operators"),
    effects="read",
    confirmation="none",
)

__all__ = ["LIST_OPERATOR_CATALOG"]

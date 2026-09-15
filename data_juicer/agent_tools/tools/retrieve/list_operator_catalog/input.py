"""Input models for list_operator_catalog."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ListOperatorCatalogInput(BaseModel):
    op_type: str = Field(
        default="",
        description=(
            "Optional operator type filter (e.g. 'filter', 'mapper', 'deduplicator', "
            "'selector', 'grouper', 'aggregator', 'pipeline')."
        ),
    )
    tags: List[str] = Field(
        default_factory=list,
        description=(
            "Optional tag filter (e.g. 'text', 'image', 'multimodal'). "
            "Only operators containing ALL requested tags are returned."
        ),
    )
    include_parameters: bool = Field(
        default=False,
        description=(
            "Whether to include full structured parameter schemas for every returned operator. "
            "Enable this only when targeted retrieval is insufficient and the agent needs broad "
            "catalog context for reasoning."
        ),
    )
    limit: int = Field(
        default=0,
        ge=0,
        description=("Maximum number of operators to return. Use 0 to return the full filtered catalog."),
    )


class GenericOutput(BaseModel):
    ok: bool = True

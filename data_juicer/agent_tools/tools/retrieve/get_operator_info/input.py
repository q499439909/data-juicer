# -*- coding: utf-8 -*-
"""Input models for get_operator_info."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GetOperatorInfoInput(BaseModel):
    operator_name: str = Field(description="Canonical or approximate Data-Juicer operator name to inspect.")


class GenericOutput(BaseModel):
    ok: bool = True

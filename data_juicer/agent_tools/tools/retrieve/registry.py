# -*- coding: utf-8 -*-
"""Registry for retrieve tool specs."""

from __future__ import annotations

from typing import List

from data_juicer.agent_tools.core.tool import ToolSpec

from .get_operator_info.tool import GET_OPERATOR_INFO
from .list_operator_catalog.tool import LIST_OPERATOR_CATALOG
from .retrieve_operators.tool import RETRIEVE_OPERATORS
from .retrieve_operators_api.tool import RETRIEVE_OPERATORS_API

TOOL_SPECS: List[ToolSpec] = [
    RETRIEVE_OPERATORS,
    RETRIEVE_OPERATORS_API,
    GET_OPERATOR_INFO,
    LIST_OPERATOR_CATALOG,
]

__all__ = [
    "GET_OPERATOR_INFO",
    "LIST_OPERATOR_CATALOG",
    "RETRIEVE_OPERATORS",
    "RETRIEVE_OPERATORS_API",
    "TOOL_SPECS",
]

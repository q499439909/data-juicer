"""MCP server exposing Data-Juicer agent tools.

Unlike ``DJ_mcp_granular_ops`` (one MCP tool per operator) and
``DJ_mcp_recipe_flow`` (recipe-level execution), this server exposes the
workflow-level agent tools in :mod:`data_juicer.agent_tools`: dataset
inspection, operator retrieval, recipe planning/validation, operator
development, and recipe execution.
"""

import os
from typing import List, Optional

from data_juicer.agent_tools.mcp.server import build_mcp_server


def _env_list(name: str) -> Optional[List[str]]:
    raw = os.getenv(name, "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def create_mcp_server(port: str = "8000"):
    """Create the agent-tools MCP server, configured from the environment.

    Environment variables:
    - ``DJ_AGENT_TOOLS_WORKING_DIR``: working/artifact dir for tool runs
    - ``DJ_AGENT_TOOLS_PROFILE``: tool profile name (e.g. ``default``, ``harness``)
    - ``DJ_AGENT_TOOLS_TAGS``: comma-separated tag filter
    - ``DJ_AGENT_TOOLS_INCLUDE`` / ``DJ_AGENT_TOOLS_EXCLUDE``: comma-separated tool names
    - ``SERVER_HOST``: bind host for HTTP transports
    """
    return build_mcp_server(
        working_dir=os.getenv("DJ_AGENT_TOOLS_WORKING_DIR"),
        profile=os.getenv("DJ_AGENT_TOOLS_PROFILE"),
        tags=_env_list("DJ_AGENT_TOOLS_TAGS"),
        include=_env_list("DJ_AGENT_TOOLS_INCLUDE"),
        exclude=_env_list("DJ_AGENT_TOOLS_EXCLUDE"),
        host=os.getenv("SERVER_HOST", "127.0.0.1"),
        port=int(port),
    )


if __name__ == "__main__":
    mcp = create_mcp_server()
    mcp.run(transport=os.getenv("SERVER_TRANSPORT", "stdio"))

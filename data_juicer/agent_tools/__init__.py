"""Agent-facing atomic tools for dataset inspection, planning, and execution.

The tools are runtime agnostic: every tool is declared as a ``ToolSpec`` with a
pydantic input model, and adapters expose them to a concrete runtime (currently
MCP, via ``data_juicer.tools.DJ_mcp_agent_tools``).
"""

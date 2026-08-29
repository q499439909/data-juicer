"""Persistent plan-first workflow support for the Data-Juicer MCP server."""

__all__ = ["PlanFlowService"]


def __getattr__(name):
    """Keep the MCP module lazy so the container entry can validate first."""
    if name == "PlanFlowService":
        from .service import PlanFlowService

        return PlanFlowService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

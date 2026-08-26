"""FastMCP adapter for the independent plan-first workflow."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from data_juicer.utils.lazy_loader import LazyLoader

from .common import PlanFlowError
from .service import PlanFlowService

fastmcp = LazyLoader("mcp.server.fastmcp", "mcp[cli]")
service = PlanFlowService()

WorkspaceRoot = Annotated[
    str,
    Field(
        description=(
            "Absolute root of the workspace selected in the current DSH session. "
            "This is never the MCP server source directory or process cwd. Relative data paths are resolved from here."
        )
    ),
]


def _call(method, *args, **kwargs) -> dict[str, Any]:
    try:
        return method(*args, **kwargs)
    except PlanFlowError as exc:
        return exc.to_dict()


def inspect_input(workspace_root: WorkspaceRoot, input: dict[str, Any], sample_size: int = 20) -> dict[str, Any]:
    """Inspect input in the DSH-selected workspace; raw media folders become reproducible DJ JSONL manifests."""
    return _call(service.inspect_input, workspace_root, input, sample_size)


def search_capabilities(
    requirements: list[str], modality: str | None = None, executor_type: str = "default", top_k: int = 5
) -> dict[str, Any]:
    """Search Data-Juicer operators; each requirement returns at most five candidates with full schemas."""
    return _call(service.search_capabilities, requirements, modality, executor_type, top_k)


def prepare_plan(
    workspace_root: WorkspaceRoot,
    plan: dict[str, Any],
    task_id: str | None = None,
    base_plan_version: str | None = None,
) -> dict[str, Any]:
    """Validate and save a new immutable plan_vNNN. Invalid drafts are saved for audit but cannot be approved."""
    return _call(service.prepare_plan, workspace_root, plan, task_id, base_plan_version)


def get_plan(
    workspace_root: WorkspaceRoot,
    task_id: str,
    plan_version: str | None = None,
    include_versions: bool = False,
) -> dict[str, Any]:
    """Read a plan, validation findings, diff, approval, and optionally all versions."""
    return _call(service.get_plan, workspace_root, task_id, plan_version, include_versions)


def preview_plan(workspace_root: WorkspaceRoot, task_id: str, plan_version: str) -> dict[str, Any]:
    """Show exactly what DJ and generic postprocessing would run, without executing it."""
    return _call(service.preview_plan, workspace_root, task_id, plan_version)


def approve_plan(
    workspace_root: WorkspaceRoot,
    task_id: str,
    plan_version: str,
    content_hash: str,
    note: str = "",
) -> dict[str, Any]:
    """Approve the exact validated plan bundle identified by its single content hash."""
    return _call(service.approve_plan, workspace_root, task_id, plan_version, content_hash, note)


def run_plan(workspace_root: WorkspaceRoot, task_id: str, plan_version: str) -> dict[str, Any]:
    """Start an approved plan asynchronously in a fresh versioned output directory."""
    return _call(service.run_plan, workspace_root, task_id, plan_version)


def get_run(workspace_root: WorkspaceRoot, task_id: str, run_id: str | None = None) -> dict[str, Any]:
    """Get run status and paths to logs, output, and the final report."""
    return _call(service.get_run, workspace_root, task_id, run_id)


def cancel_run(workspace_root: WorkspaceRoot, task_id: str, run_id: str) -> dict[str, Any]:
    """Stop a running worker and mark the run cancelled."""
    return _call(service.cancel_run, workspace_root, task_id, run_id)


def create_mcp_server(port: str = "8000"):
    mcp = fastmcp.FastMCP(
        "Data-Juicer Plan Flow",
        instructions=(
            "This server has no independent data workspace. For every workspace_root argument, pass the absolute "
            "workspace selected in the current DSH session. Never use the MCP source directory or process cwd. "
            "Never put API keys in plans; API operators inherit credentials from the MCP runtime environment."
        ),
        port=port,
    )
    for tool in (
        inspect_input,
        search_capabilities,
        prepare_plan,
        get_plan,
        preview_plan,
        approve_plan,
        run_plan,
        get_run,
        cancel_run,
    ):
        mcp.tool()(tool)
    return mcp

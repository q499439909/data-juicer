"""FastMCP adapter for the independent plan-first workflow."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

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
    requirements: list[str], modality: str | None = None, executor_type: str = "default", top_k: int = 3
) -> dict[str, Any]:
    """BM25-search all atomic requirements; return compact, cross-requirement-deduplicated operator definitions."""
    return _call(service.search_capabilities, requirements, modality, executor_type, top_k)


def get_capability_schemas(operator_names: list[str]) -> dict[str, Any]:
    """Load full schemas by exact name for operators already found by search_capabilities; this is not a new search."""
    return _call(service.get_capability_schemas, operator_names)


def resolve_capabilities(requirements: list[str]) -> dict[str, Any]:
    """Resolve already approved reusable capabilities before proposing any new operator."""
    return _call(service.resolve_capabilities, requirements)


def prepare_capability(capability: dict[str, Any], operator_artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze, sandbox-build, and validate one capability proposal containing one or more operator artifacts."""
    return _call(service.prepare_capability, capability, operator_artifacts)


def get_capability(capability_id: str) -> dict[str, Any]:
    """Read an approved immutable capability descriptor."""
    return _call(service.get_capability, capability_id)


def approve_capability(proposal_id: str, content_hash: str, note: str = "") -> dict[str, Any]:
    """Approve the exact post-validation capability proposal hash and publish all covered artifacts."""
    return _call(service.approve_capability, proposal_id, content_hash, note)


def operator_catalog() -> dict[str, Any]:
    """List every operator visible in the live Data-Juicer registry."""
    return _call(service.operator_catalog)


def operator_detail(name: str) -> dict[str, Any]:
    """Load parameter details for one exact operator name."""
    return _call(service.operator_detail, name)


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
        get_capability_schemas,
        resolve_capabilities,
        prepare_capability,
        get_capability,
        approve_capability,
        prepare_plan,
        get_plan,
        preview_plan,
        approve_plan,
        run_plan,
        get_run,
        cancel_run,
    ):
        mcp.tool()(tool)

    @mcp.custom_route("/operator-catalog", methods=["GET"], include_in_schema=False)
    async def get_operator_catalog(_request: Request) -> JSONResponse:
        payload = operator_catalog()
        return JSONResponse(
            payload,
            status_code=200 if payload.get("ok") else 503,
            headers={"Cache-Control": "no-store"},
        )

    @mcp.custom_route("/operator-detail", methods=["GET"], include_in_schema=False)
    async def get_operator_detail(request: Request) -> JSONResponse:
        payload = operator_detail(request.query_params.get("name", ""))
        return JSONResponse(
            payload,
            status_code=200 if payload.get("ok") else 404,
            headers={"Cache-Control": "no-store"},
        )

    return mcp

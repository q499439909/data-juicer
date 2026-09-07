"""FastMCP adapter for the independent plan-first workflow."""

from __future__ import annotations

import hmac
import inspect
import os
from contextlib import contextmanager
from typing import Annotated, Any

from pydantic import Field
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from data_juicer.utils.lazy_loader import LazyLoader

from .common import PlanFlowError
from .run_output_gateway import RunOutputGateway
from .service import PlanFlowService
from .user_operator_store import current_user, safe_id
from .user_operator_validation import validation_jobs

fastmcp = LazyLoader("mcp.server.fastmcp", "mcp[cli]")


def _service_from_environment() -> PlanFlowService:
    mode = os.environ.get("DJ_PLAN_FLOW_EXECUTION_MODE", "broker")
    try:
        return PlanFlowService(execution_mode=mode)
    except ValueError as exc:
        raise RuntimeError("DJ_PLAN_FLOW_EXECUTION_MODE must be either 'broker' or 'native'") from exc


service = _service_from_environment()
run_outputs = RunOutputGateway()

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


def _internal_authorized(request: Request) -> bool:
    expected = os.environ.get("DSH_DJ_INTERNAL_TOKEN", "")
    supplied = request.headers.get("x-dsh-internal-token", "")
    return bool(expected and supplied and hmac.compare_digest(expected, supplied))


@contextmanager
def _request_user(request):
    user = request.headers.get("x-dsh-user-id") if _internal_authorized(request) else None
    token = current_user.set(safe_id(user) if user else None)
    try:
        yield
    finally:
        current_user.reset(token)


def get_custom_operator_authoring_spec(
    category: str,
    modality: str = "generic",
) -> dict[str, Any]:
    """Return the running DJ version's authoring contract for a category.

    Call this once after confirming a built-in capability gap and before
    generating Python. Do not search the filesystem or web for DJ base classes.
    Select one strategy from spec.model_policy, replace scaffold placeholders,
    declare its exact dependencies/model_refs, then call develop_custom_operator.
    """
    from .operator_authoring import operator_authoring

    result = _call(operator_authoring.get_spec, category, modality)
    return {"ok": True, "spec": result} if "ok" not in result else result


def develop_custom_operator(proposal: dict[str, Any], samples: list[dict[str, Any]], parameters: dict[str, Any] | None = None,
                            timeout_seconds: int = 180) -> dict[str, Any]:
    """Submit generated Python to the current account only. proposal: name, category, source, validation_contract
    (purpose, row_count, equals=[{row,field,value}], limitations), optional exact dependencies/model_refs/replaces.
    Samples are temporary JSON records. Runs real DJ validation asynchronously, cleans test files, and publishes
    validated (declared assertions passed) or experimental (smoke only). Never edits the DJ built-in library.
    """
    return _call(validation_jobs.develop, proposal, samples, parameters, timeout_seconds)


def get_custom_operator_job(job_id: str, cancel: bool = False) -> dict[str, Any]:
    """Read this account's validation job and cleanup report, or request cancellation and process cleanup."""
    return _call(validation_jobs.get, job_id, cancel)


def validate_custom_operator(candidate_id: str, samples: list[dict[str, Any]], parameters: dict[str, Any] | None = None,
                             timeout_seconds: int = 180) -> dict[str, Any]:
    """Re-run an existing personal version's frozen contract with temporary samples; does not weaken its contract."""
    from pathlib import Path
    from .common import read_json
    from .user_operator_store import UserOperatorStore
    try:
        item = UserOperatorStore().resolve(candidate_id)
        path = Path(item["_path"])
        contract = read_json(path.parent / "validation-contract.json")
        proposal = {"name": item["name"], "category": item["type"], "source": path.read_text(encoding="utf-8"),
                    "validation_contract": {key: value for key, value in contract.items() if not key.startswith("_")},
                    **item["_manifest"]}
        return _call(validation_jobs.develop, proposal, samples, parameters, timeout_seconds)
    except PlanFlowError as exc:
        return exc.to_dict()


def _result_arguments(request: Request) -> tuple[str, str, str, str]:
    return (
        request.query_params.get("workspace_root", ""),
        request.query_params.get("task_id", ""),
        request.query_params.get("plan_version", ""),
        request.query_params.get("result_ref", ""),
    )


def inspect_input(workspace_root: WorkspaceRoot, input: dict[str, Any], sample_size: int = 20) -> dict[str, Any]:
    """Inspect input in the DSH-selected workspace; raw media folders become reproducible DJ JSONL manifests."""
    return _call(service.inspect_input, workspace_root, input, sample_size)


def search_capabilities(
    requirements: list[str], modality: str | None = None, executor_type: str = "default", top_k: int = 3
) -> dict[str, Any]:
    """Return at most three ranked candidates per requirement with full descriptions but no parameter schemas."""
    return _call(service.search_capabilities, requirements, modality, executor_type, top_k)


def get_capability_schemas(operator_names: list[str]) -> dict[str, Any]:
    """Load executable parameter contracts by exact candidate ID or DJ name for an already selected shortlist."""
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
    view_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and save a new immutable plan_vNNN. Invalid drafts are saved for audit but cannot be approved."""
    return _call(service.prepare_plan, workspace_root, plan, task_id, base_plan_version, view_spec)


def get_plan(
    workspace_root: WorkspaceRoot,
    task_id: str,
    plan_version: str | None = None,
    include_versions: bool = False,
) -> dict[str, Any]:
    """Read a plan, execution preview, runtime preflight, validation, diff, approval, and versions."""
    return _call(service.get_plan, workspace_root, task_id, plan_version, include_versions)


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
    tool_functions = (
        inspect_input,
        search_capabilities,
        get_capability_schemas,
        resolve_capabilities,
        prepare_plan,
        get_plan,
        approve_plan,
        run_plan,
        get_run,
        cancel_run,
        get_custom_operator_authoring_spec,
        develop_custom_operator,
        get_custom_operator_job,
        validate_custom_operator,
    )
    for tool in tool_functions:
        mcp.tool()(tool)

    @mcp.custom_route("/internal/operator-tools", methods=["GET", "POST"], include_in_schema=False)
    async def account_tool_gateway(request: Request) -> JSONResponse:
        if not _internal_authorized(request):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=403)
        if request.method == "GET":
            return JSONResponse({"tools": [item.model_dump(mode="json") for item in await mcp.list_tools()]})
        if not request.headers.get("x-dsh-user-id"):
            return JSONResponse({"ok": False, "error": "account_required"}, status_code=403)
        body = await request.body()
        if len(body) > 2_000_000:
            return JSONResponse({"ok": False, "error": "request_too_large"}, status_code=413)
        import json
        import anyio
        try:
            payload = json.loads(body)
            function = {fn.__name__: fn for fn in tool_functions}.get(payload.get("name"))
            if function is None:
                raise ValueError("Unknown tool")
            arguments = payload.get("arguments", {})
            inspect.signature(function).bind(**arguments)
            with _request_user(request):
                result = await anyio.to_thread.run_sync(lambda: function(**arguments))
            return JSONResponse(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @mcp.custom_route("/operator-catalog", methods=["GET"], include_in_schema=False)
    async def get_operator_catalog(_request: Request) -> JSONResponse:
        with _request_user(_request):
            payload = operator_catalog()
        return JSONResponse(
            payload,
            status_code=200 if payload.get("ok") else 503,
            headers={"Cache-Control": "no-store"},
        )

    @mcp.custom_route("/operator-detail", methods=["GET"], include_in_schema=False)
    async def get_operator_detail(request: Request) -> JSONResponse:
        with _request_user(request):
            payload = operator_detail(request.query_params.get("name", ""))
        return JSONResponse(
            payload,
            status_code=200 if payload.get("ok") else 404,
            headers={"Cache-Control": "no-store"},
        )

    @mcp.custom_route("/plan-view", methods=["GET"], include_in_schema=False)
    async def get_plan_view(request: Request) -> JSONResponse:
        with _request_user(request):
            payload = get_plan(
                request.query_params.get("workspace_root", ""),
                request.query_params.get("task_id", ""),
                request.query_params.get("plan_version") or None,
                request.query_params.get("include_versions") == "true",
            )
        return JSONResponse(payload, status_code=200 if payload.get("ok") else 404, headers={"Cache-Control": "no-store"})

    @mcp.custom_route("/run-steps", methods=["GET"], include_in_schema=False)
    async def get_run_steps(request: Request) -> JSONResponse:
        with _request_user(request):
            payload = get_run(
                request.query_params.get("workspace_root", ""),
                request.query_params.get("task_id", ""),
                request.query_params.get("run_id") or None,
            )
        return JSONResponse(payload, status_code=200 if payload.get("ok") else 404, headers={"Cache-Control": "no-store"})

    @mcp.custom_route("/internal/run-output", methods=["GET", "DELETE"], include_in_schema=False)
    async def internal_run_output(request: Request) -> JSONResponse:
        if not _internal_authorized(request):
            return JSONResponse({"ok": False, "error": {"code": "INTERNAL_UNAUTHORIZED", "message": "Unauthorized"}}, status_code=403)
        args = _result_arguments(request)
        if request.method == "DELETE":
            payload = _call(run_outputs.delete_outputs, *args, request.headers.get("if-match", ""))
        else:
            payload = _call(run_outputs.inspect_run, *args)
            if payload.get("eligible") is not None:
                payload = {"ok": True, **payload}
        return JSONResponse(
            payload,
            status_code=200 if payload.get("ok") or payload.get("deleted") else 404,
            headers={"Cache-Control": "no-store"},
        )

    @mcp.custom_route("/internal/run-asset", methods=["GET"], include_in_schema=False)
    async def internal_run_asset(request: Request):
        if not _internal_authorized(request):
            return JSONResponse({"ok": False, "error": {"code": "INTERNAL_UNAUTHORIZED", "message": "Unauthorized"}}, status_code=403)
        try:
            opened = run_outputs.open_asset(*_result_arguments(request), request.query_params.get("asset_id", ""))
        except PlanFlowError as exc:
            return JSONResponse(exc.to_dict(), status_code=404, headers={"Cache-Control": "no-store"})
        return FileResponse(
            opened.path,
            media_type=opened.media_type,
            filename=opened.display_name if request.query_params.get("download") == "1" else None,
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
                "ETag": opened.sha256,
            },
        )

    @mcp.custom_route("/internal/run-archive", methods=["GET"], include_in_schema=False)
    async def internal_run_archive(request: Request):
        if not _internal_authorized(request):
            return JSONResponse({"ok": False, "error": {"code": "INTERNAL_UNAUTHORIZED", "message": "Unauthorized"}}, status_code=403)
        try:
            selected = None if request.query_params.get("all") == "1" else request.query_params.getlist("asset_id")
            archive = run_outputs.create_archive(*_result_arguments(request), selected)
        except PlanFlowError as exc:
            return JSONResponse(exc.to_dict(), status_code=404, headers={"Cache-Control": "no-store"})
        return FileResponse(
            archive.path,
            media_type="application/zip",
            filename=archive.display_name,
            headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff", "ETag": archive.sha256},
        )

    return mcp

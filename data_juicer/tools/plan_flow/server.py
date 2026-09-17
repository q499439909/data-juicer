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
from .plan_contract import PlanDraft
from .plan_contract import contract as plan_contract
from .run_output_gateway import RunOutputGateway
from .service import PlanFlowService
from .task_control import authorize_task, trusted_decision
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
        if (
            os.environ.get("DSH_DJ_INTERNAL_TOKEN")
            and not current_user.get()
            and method.__name__ not in {"operator_catalog", "operator_detail"}
        ):
            raise PlanFlowError("ACCOUNT_REQUIRED", "Use the authenticated DJ gateway")
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


def develop_custom_operator(
    proposal: dict[str, Any],
    samples: list[dict[str, Any]],
    parameters: dict[str, Any] | None = None,
    timeout_seconds: int = 180,
    request_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Submit generated Python to the current account only. proposal: name, category, source, validation_contract
    (purpose, row_count, equals=[{row,field,value}], limitations), optional exact dependencies/model_refs/replaces.
    Samples are temporary JSON records. Runs real DJ validation asynchronously, cleans test files, and publishes
    validated (declared assertions passed) or experimental (smoke only). Never edits the DJ built-in library.
    """
    return _call(validation_jobs.develop, proposal, samples, parameters, timeout_seconds, request_id, task_id)


def get_custom_operator_job(job_id: str, cancel: bool = False) -> dict[str, Any]:
    """Read this account's validation job and cleanup report, or request cancellation and process cleanup."""
    return _call(validation_jobs.get, job_id, cancel)


def validate_custom_operator(
    candidate_id: str,
    samples: list[dict[str, Any]],
    parameters: dict[str, Any] | None = None,
    timeout_seconds: int = 180,
    request_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Re-run an existing personal version's frozen contract with temporary samples; does not weaken its contract."""
    from pathlib import Path

    from .common import read_json
    from .user_operator_store import UserOperatorStore

    try:
        item = UserOperatorStore().resolve(candidate_id)
        path = Path(item["_path"])
        contract = read_json(path.parent / "validation-contract.json")
        proposal = {
            "name": item["name"],
            "category": item["type"],
            "source": path.read_text(encoding="utf-8"),
            "validation_contract": {key: value for key, value in contract.items() if not key.startswith("_")},
            **item["_manifest"],
        }
        return _call(validation_jobs.develop, proposal, samples, parameters, timeout_seconds, request_id, task_id)
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
    requirements: list[str],
    modality: str | None = None,
    executor_type: str = "default",
    top_k: int = 3,
    include_details: bool = False,
) -> dict[str, Any]:
    """Return up to three candidates per requirement. image includes image/multimodal operators; multimodal includes component media operators. Fetch schemas after selecting candidates; include_details expands retrieval evidence."""
    result = _call(service.search_capabilities, requirements, modality, executor_type, top_k)
    if not include_details and result.get("ok"):
        for row in result.get("results", []):
            row.pop("retrieval", None)
            row.pop("ranking", None)
        for item in result.get("operators", []):
            item["description"] = item.get("description", "").split("\n\n")[0][:360]
            item["schema_ref"] = item.get("candidate_id", item["name"])
            item.pop("ranking", None)
            item.pop("match_score", None)
            item["match_semantics"] = (
                "Candidate rank only; verify the requested behavior using its capability contract."
            )
    return result


def get_plan_contract() -> dict[str, Any]:
    """Get versioned Plan schema, path semantics, native image audit/report configuration and approval behavior."""
    return plan_contract()


def inspect_runtime(workspace_root: WorkspaceRoot, task_id: str, plan_version: str | None = None) -> dict[str, Any]:
    """Read the authorized Plan's platform-specific dependency closure, runtime identity and recovery actions; never install packages."""
    result = _call(service.get_plan, workspace_root, task_id, plan_version)
    if not result.get("ok"):
        return result
    return {
        "ok": True,
        "task_id": task_id,
        "plan_version": result["plan_version"],
        "runtime_assessment": result["runtime_assessment"],
    }


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
    plan: PlanDraft,
    task_id: str | None = None,
    base_plan_version: str | None = None,
    view_spec: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Validate and save a new immutable plan_vNNN. Invalid drafts are saved for audit but cannot be approved."""
    raw = plan.model_dump(exclude_unset=True) if isinstance(plan, PlanDraft) else dict(plan)
    raw.pop("submission_ref", None)
    return _call(service.prepare_plan, workspace_root, raw, task_id, base_plan_version, view_spec, request_id)


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
    accepted_gaps: list[str] | None = None,
) -> dict[str, Any]:
    """Approve the exact validated plan bundle identified by its single content hash."""
    return _call(service.approve_plan, workspace_root, task_id, plan_version, content_hash, note, accepted_gaps)


def run_plan(
    workspace_root: WorkspaceRoot,
    task_id: str,
    plan_version: str,
    request_id: str | None = None,
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    """Start an approved plan asynchronously in a fresh versioned output directory."""
    return _call(service.run_plan, workspace_root, task_id, plan_version, request_id, timeout_seconds)


def get_run(workspace_root: WorkspaceRoot, task_id: str, run_id: str | None = None) -> dict[str, Any]:
    """Get run status and paths to logs, output, and the final report."""
    return _call(service.get_run, workspace_root, task_id, run_id)


def cancel_run(workspace_root: WorkspaceRoot, task_id: str, run_id: str) -> dict[str, Any]:
    """Stop a running worker and mark the run cancelled."""
    return _call(service.cancel_run, workspace_root, task_id, run_id)


def create_mcp_server(port: str = "8000"):
    from .deployment import planning_health, service_contract

    contract = service_contract(service.execution_mode)
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
        get_plan_contract,
        inspect_runtime,
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
    if service.capability_catalog is None:
        tool_functions = tuple(tool for tool in tool_functions if tool is not resolve_capabilities)
    for tool in tool_functions:
        mcp.tool()(tool)

    @mcp.custom_route("/internal/health", methods=["GET"], include_in_schema=False)
    async def internal_health(request: Request):
        if not _internal_authorized(request):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=403)
        payload = planning_health(contract)
        return JSONResponse(payload, status_code=200 if payload["ok"] else 503, headers={"Cache-Control": "no-store"})

    @mcp.custom_route("/internal/operator-tools", methods=["GET", "POST"], include_in_schema=False)
    async def account_tool_gateway(request: Request) -> JSONResponse:
        if not _internal_authorized(request):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=403)
        if request.method == "GET":
            return JSONResponse(
                {**contract, "tools": [item.model_dump(mode="json") for item in await mcp.list_tools()]},
                headers={"Cache-Control": "no-store"},
            )
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
            if payload.get("name") == "approve_plan":
                return JSONResponse(
                    {
                        "ok": False,
                        "error": {"code": "USER_DECISION_REQUIRED", "message": "Approve the exact plan through the UI"},
                    },
                    status_code=403,
                )
            if request.headers.get("x-dsh-dj-protocol") != str(contract["protocol"]["version"]):
                return JSONResponse(
                    {
                        "ok": False,
                        "error": {
                            "code": "PROTOCOL_MISMATCH",
                            "message": "Negotiate the DJ gateway protocol before calling tools",
                        },
                    },
                    status_code=409,
                )
            if request.headers.get("x-dsh-dj-instance") != contract["instance_id"]:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": {
                            "code": "SERVICE_RESTARTED",
                            "message": "DJ service changed; reconnect the bridge before retrying",
                        },
                    },
                    status_code=409,
                )
            arguments = payload.get("arguments", {})
            inspect.signature(function).bind(**arguments)
            metadata = mcp._tool_manager.get_tool(function.__name__).fn_metadata
            arguments = metadata.arg_model.model_validate(metadata.pre_parse_json(arguments)).model_dump_one_level()
            with _request_user(request):
                result = await anyio.to_thread.run_sync(lambda: function(**arguments))
            return JSONResponse(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @mcp.custom_route("/internal/plan-decision", methods=["POST"], include_in_schema=False)
    async def plan_decision(request: Request):
        if not _internal_authorized(request) or not request.headers.get("x-dsh-user-id"):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=403)
        import uuid

        try:
            raw = await request.body()
            if len(raw) > 16384:
                raise ValueError("Decision payload too large")
            import json

            body = json.loads(raw)
            decision_id = str(uuid.UUID(body["decision_id"]))
            with _request_user(request):
                token = trusted_decision.set("ui:" + decision_id)
                try:
                    payload = approve_plan(
                        body["workspace_root"],
                        body["task_id"],
                        body["plan_version"],
                        body["content_hash"],
                        accepted_gaps=body.get("accepted_gaps"),
                    )
                finally:
                    trusted_decision.reset(token)
            return JSONResponse(payload, status_code=200 if payload.get("ok") else 403)
        except (ValueError, KeyError, TypeError):
            return JSONResponse({"ok": False, "error": "invalid_decision"}, status_code=400)

    @mcp.custom_route("/internal/workspace-access", methods=["GET"], include_in_schema=False)
    async def workspace_access(request: Request):
        if not _internal_authorized(request):
            return JSONResponse({"ok": False}, status_code=403)
        from .task_control import authorize_workspace

        try:
            with _request_user(request):
                authorize_workspace(request.query_params.get("workspace_root", ""))
            return JSONResponse({"ok": True})
        except PlanFlowError as exc:
            return JSONResponse(exc.to_dict(), status_code=403)

    @mcp.custom_route("/internal/task-runs", methods=["GET"], include_in_schema=False)
    async def task_runs(request: Request):
        if not _internal_authorized(request):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=403)
        from .task_control import list_owned_runs

        with _request_user(request):
            result = _call(list_owned_runs, request.query_params.get("cursor", "0"))
        return JSONResponse(result, status_code=200 if result.get("ok") else 403)

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
        return JSONResponse(
            payload, status_code=200 if payload.get("ok") else 404, headers={"Cache-Control": "no-store"}
        )

    @mcp.custom_route("/run-steps", methods=["GET"], include_in_schema=False)
    async def get_run_steps(request: Request) -> JSONResponse:
        with _request_user(request):
            payload = get_run(
                request.query_params.get("workspace_root", ""),
                request.query_params.get("task_id", ""),
                request.query_params.get("run_id") or None,
            )
        return JSONResponse(
            payload, status_code=200 if payload.get("ok") else 404, headers={"Cache-Control": "no-store"}
        )

    @mcp.custom_route("/internal/run-output", methods=["GET", "DELETE"], include_in_schema=False)
    async def internal_run_output(request: Request) -> JSONResponse:
        if not _internal_authorized(request):
            return JSONResponse(
                {"ok": False, "error": {"code": "INTERNAL_UNAUTHORIZED", "message": "Unauthorized"}}, status_code=403
            )
        args = _result_arguments(request)
        from .store import PlanStore

        try:
            with _request_user(request):
                authorize_task(PlanStore(args[0]), args[1])
        except PlanFlowError as exc:
            return JSONResponse(exc.to_dict(), status_code=403)
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
            return JSONResponse(
                {"ok": False, "error": {"code": "INTERNAL_UNAUTHORIZED", "message": "Unauthorized"}}, status_code=403
            )
        try:
            from .store import PlanStore

            with _request_user(request):
                args = _result_arguments(request)
                authorize_task(PlanStore(args[0]), args[1])
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
            return JSONResponse(
                {"ok": False, "error": {"code": "INTERNAL_UNAUTHORIZED", "message": "Unauthorized"}}, status_code=403
            )
        try:
            from .store import PlanStore

            with _request_user(request):
                args = _result_arguments(request)
                authorize_task(PlanStore(args[0]), args[1])
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

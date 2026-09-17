"""Actionable recovery hints, not permission to bypass an execution boundary."""


def recovery(code):
    if code == "CAPABILITY_CONTROL_NOT_CONFIGURED":
        return "planning", [
            {
                "action": "search_capabilities",
                "tool": "mcp__dj__search_capabilities",
                "message": "This deployment uses the native operator catalog. Search it directly; the optional broker catalog is not configured.",
            }
        ]
    if code.startswith("MODEL_"):
        return "blocked", [
            {
                "action": "inspect_runtime",
                "tool": "mcp__dj__inspect_runtime",
                "message": "Inspect exact model revision, missing files and download bytes; preserve valid cached weights.",
            },
            {
                "action": "retry_run",
                "requires_user_action": True,
                "message": "After restoring cache permissions/network, explicitly retry this approved version; the host prepares and verifies missing files.",
            },
        ]
    if code in {"APPROVAL_REQUIRED", "USER_DECISION_REQUIRED"}:
        return "awaiting_approval", [
            {
                "action": "confirm_plan",
                "tool": "mcp__dj__confirm_plan",
                "message": "Show and confirm the exact saved Plan; do not retry run_plan or recreate a version.",
            }
        ]
    if code in {"TASK_FORBIDDEN", "WORKSPACE_FORBIDDEN"}:
        return "blocked", [
            {
                "action": "request_resource_access",
                "message": "Use an authorized task/workspace or have its owner grant/migrate access. Do not inspect denied internal files through shell.",
            }
        ]
    if code.startswith("RUNTIME_"):
        return "blocked", [
            {
                "action": "inspect_runtime",
                "tool": "mcp__dj__inspect_runtime",
                "message": "Inspect the locked dependency closure. Restore the isolated runtime through authorized maintenance; never weaken the lock.",
            }
        ]
    if code in {"CONTENT_CHANGED", "PLAN_SUPERSEDED", "INPUT_SNAPSHOT_CHANGED"}:
        return "blocked", [
            {
                "action": "get_plan",
                "tool": "mcp__dj__get_plan",
                "message": "Review the latest content and input identity; obtain a fresh approval only if execution identity changes.",
            }
        ]
    return "blocked", [
        {
            "action": "diagnose",
            "message": "Use the error code and authorized task logs for bounded diagnosis; preserve the current Plan.",
        }
    ]

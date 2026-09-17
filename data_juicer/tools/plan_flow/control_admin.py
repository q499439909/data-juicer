"""Explicit local administration; deliberately not exposed as an Agent/MCP tool."""

import argparse
import os
from pathlib import Path

from .common import FileLock, PlanFlowError, read_yaml, write_yaml_atomic
from .task_control import register_workspace
from .user_operator_store import safe_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-data-root", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--migrate-task", action="append", default=[])
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve(strict=True)
    if not workspace.is_dir():
        parser.error("workspace must be an existing directory")
    owner = safe_id(args.owner)
    os.environ["DSH_USER_DATA_ROOT"] = str(Path(args.user_data_root).resolve())
    os.environ["DSH_DJ_ALLOWED_WORKSPACES"] = str(workspace)
    # An explicit named grant, never first-visit adoption or chat-history inference.
    register_workspace(workspace, owner)
    from .store import PlanStore

    store = PlanStore(workspace)
    for task_id in args.migrate_task:
        task_path = store.task_path(task_id)
        with FileLock(task_path / ".lock"):
            path = task_path / "task.yaml"
            task = read_yaml(path)
            if task.get("owner") not in {None, owner}:
                raise PlanFlowError("OWNER_CONFLICT", "Refusing to reassign an existing task owner")
            backup = task_path / "task.before-owner-migration.yaml"
            if not backup.exists():
                write_yaml_atomic(backup, task)
            task["owner"] = owner
            write_yaml_atomic(path, task)
    print(
        f"Workspace granted to {owner}; {len(args.migrate_task)} named tasks migrated. Plan/approval bundles unchanged."
    )


if __name__ == "__main__":
    main()

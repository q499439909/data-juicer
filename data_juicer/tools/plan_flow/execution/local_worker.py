"""Internal worker process launched by LocalProcessBackend."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from ..common import FileLock, PlanFlowError, now_iso, read_json, read_yaml, write_json_atomic, write_text_atomic
from ..store import PlanStore


def _replace(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("${" + key + "}", replacement)
        return value
    if isinstance(value, list):
        return [_replace(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, variables) for key, item in value.items()}
    return value


def _postprocess_arguments(arguments: Any) -> list[str]:
    if isinstance(arguments, list):
        return [str(item) for item in arguments]
    result: list[str] = []
    for key, value in (arguments or {}).items():
        option = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(option)
        elif isinstance(value, list):
            for item in value:
                result.extend([option, str(item)])
        elif value is not None:
            result.extend([option, str(value)])
    return result


def execute_worker(workspace: str, task_id: str, plan_version: str, run_id: str) -> int:
    store = PlanStore(workspace)
    plan_path = store.plan_path(task_id, plan_version)
    run_path = store.task_path(task_id) / "runs" / run_id
    for _ in range(100):
        if (run_path / ".started").is_file():
            break
        time.sleep(0.05)
    state = read_json(run_path / "run.json")
    exit_code = 0
    try:
        store.verify_bundle(task_id, plan_version)
        plan = read_yaml(plan_path / "plan.yaml")
        recipe = read_yaml(run_path / "materialized-recipe.yaml")
        from data_juicer.config import init_configs
        from data_juicer.core.executor import ExecutorFactory

        cfg = init_configs(["--config", str(run_path / "materialized-recipe.yaml")], load_configs_only=False)
        ExecutorFactory.create_executor(cfg.executor_type)(cfg).run()
        variables = {
            "RUN_OUTPUT": state["output_dir"],
            "RUN_DIR": str(run_path),
            "recipe.output": str(recipe["export_path"]),
        }
        post_results = []
        for index, step in enumerate(plan.get("postprocess", []) or []):
            script = (plan_path / step["script"]).resolve()
            if plan_path.resolve() not in script.parents:
                raise PlanFlowError("PATH_NOT_ALLOWED", f"Postprocess artifact escaped plan bundle: {script}")
            arguments = _replace(step.get("arguments", {}), variables)
            log_path = run_path / "logs" / f"postprocess-{index + 1}.log"
            command = [sys.executable, "-X", "utf8", str(script), *_postprocess_arguments(arguments)]
            with log_path.open("wb") as log:
                completed = subprocess.run(
                    command, cwd=state["output_dir"], stdout=log, stderr=subprocess.STDOUT, check=False
                )
            if completed.returncode:
                raise RuntimeError(f"Postprocess step {index + 1} failed; see {log_path}")
            post_results.append({"step": index + 1, "script": step["script"], "log": str(log_path)})
        report = "\n".join(
            [
                "# Data-Juicer run report",
                "",
                f"- Task: `{task_id}`",
                f"- Plan: `{plan_version}`",
                f"- Run: `{run_id}`",
                f"- Output: `{state['output_dir']}`",
                f"- Recipe output: `{recipe['export_path']}`",
                f"- DJ steps: {len(recipe.get('process', []))}",
                f"- Postprocess steps: {len(post_results)}",
                "",
            ]
        )
        write_text_atomic(run_path / "report.md", report)
        state.update(
            {
                "status": "succeeded",
                "updated_at": now_iso(),
                "recipe_output": str(recipe["export_path"]),
                "postprocess_results": post_results,
                "report_path": str(run_path / "report.md"),
            }
        )
    except Exception as exc:
        traceback.print_exc()
        exit_code = 20
        error_code = exc.code if isinstance(exc, PlanFlowError) else "EXECUTION_FAILED"
        state.update({"status": "failed", "updated_at": now_iso(), "error_code": error_code, "error": str(exc)})
    write_json_atomic(run_path / "run.json", state)
    return exit_code


def _finish_backend_record(path: Path, exit_code: int) -> None:
    try:
        with FileLock(path.with_suffix(".lock")):
            record = read_json(path)
            if record.get("status") != "cancelled":
                record.update({"status": "exited", "exit_code": exit_code, "finished_at": now_iso()})
                write_json_atomic(path, record)
    except Exception:
        traceback.print_exc()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-record", required=True)
    parser.add_argument("workspace")
    parser.add_argument("task_id")
    parser.add_argument("plan_version")
    parser.add_argument("run_id")
    args = parser.parse_args(argv)
    exit_code = execute_worker(args.workspace, args.task_id, args.plan_version, args.run_id)
    _finish_backend_record(Path(args.backend_record), exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

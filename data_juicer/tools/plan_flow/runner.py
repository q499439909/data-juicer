"""Approved-plan execution and run lifecycle management."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .common import (
    FileLock,
    PlanFlowError,
    now_iso,
    read_json,
    read_yaml,
    write_json_atomic,
    write_text_atomic,
    write_yaml_atomic,
)
from .store import PlanStore


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


class PlanRunner:
    def __init__(self, workspace_root: str | Path):
        self.store = PlanStore(workspace_root)

    def start(self, task_id: str, plan_version: str) -> dict[str, Any]:
        content_hash = self.store.verify_bundle(task_id, plan_version)
        plan_info = self.store.get_plan(task_id, plan_version)
        approval = plan_info.get("approval")
        if not approval or approval.get("content_hash") != content_hash:
            raise PlanFlowError("APPROVAL_REQUIRED", "Approve this exact plan version before running it")
        task_path = self.store.task_path(task_id)
        with FileLock(task_path / ".lock"):
            runs_root = task_path / "runs"
            run_id = f"run_r{self.store._next_number(runs_root, 'run_r'):03d}"
            run_path = runs_root / run_id
            run_path.mkdir(parents=True)
            task = read_yaml(task_path / "task.yaml")
            output = self.store.outputs_root / task["task_slug"] / plan_version / run_id
            output.mkdir(parents=True, exist_ok=False)
            (run_path / "logs").mkdir()
            recipe = self._materialize(plan_info["plan"]["recipe"], run_path, output)
            write_yaml_atomic(run_path / "materialized-recipe.yaml", recipe)
            state = {
                "task_id": task_id,
                "plan_version": plan_version,
                "run_id": run_id,
                "status": "starting",
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "output_dir": str(output),
                "content_hash": content_hash,
            }
            write_json_atomic(run_path / "run.json", state)
            stdout = (run_path / "logs" / "stdout.log").open("ab")
            stderr = (run_path / "logs" / "stderr.log").open("ab")
            command = [
                sys.executable,
                "-m",
                "data_juicer.tools.plan_flow.runner",
                "--worker",
                str(self.store.workspace),
                task_id,
                plan_version,
                run_id,
            ]
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            environment = os.environ.copy()
            source_root = str(Path(__file__).resolve().parents[3])
            environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
            process = subprocess.Popen(
                command,
                cwd=str(self.store.workspace),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=flags,
            )
            stdout.close()
            stderr.close()
            state.update({"status": "running", "pid": process.pid, "updated_at": now_iso()})
            try:
                import psutil

                state["pid_create_time"] = psutil.Process(process.pid).create_time()
            except Exception:
                pass
            write_json_atomic(run_path / "run.json", state)
            write_text_atomic(run_path / ".started", "ready\n")
            current = read_json(task_path / "current.json")
            current["latest_run"] = run_id
            write_json_atomic(task_path / "current.json", current)
        return state

    @staticmethod
    def _materialize(recipe: dict[str, Any], run_path: Path, output: Path) -> dict[str, Any]:
        result = _replace(copy.deepcopy(recipe), {"RUN_OUTPUT": str(output), "RUN_DIR": str(run_path)})
        # Avoid Data-Juicer's CLI-style dataset_path parser treating Windows
        # backslashes as shell escapes. The persisted plan stays human-friendly;
        # only the executor-specific recipe uses the structured local form.
        if result.get("dataset_path"):
            result["dataset"] = {"configs": [{"type": "local", "path": result.pop("dataset_path")}]}
        result["work_dir"] = str(run_path / "work")
        result["temp_dir"] = str(run_path / "tmp")
        return result

    def get(self, task_id: str, run_id: str | None = None) -> dict[str, Any]:
        task_path = self.store.task_path(task_id)
        run_id = run_id or read_json(task_path / "current.json").get("latest_run")
        run_path = task_path / "runs" / str(run_id)
        if not (run_path / "run.json").is_file():
            raise PlanFlowError("RUN_NOT_FOUND", f"Unknown run: {run_id}")
        state = read_json(run_path / "run.json")
        if state.get("status") in {"starting", "running"} and not _same_process(state):
            state.update(
                {"status": "failed", "updated_at": now_iso(), "error": "Worker process exited without a final result"}
            )
            write_json_atomic(run_path / "run.json", state)
        state["stdout_log"] = str(run_path / "logs" / "stdout.log")
        state["stderr_log"] = str(run_path / "logs" / "stderr.log")
        report = run_path / "report.md"
        if report.is_file():
            state["report_path"] = str(report)
        return state

    def cancel(self, task_id: str, run_id: str) -> dict[str, Any]:
        state = self.get(task_id, run_id)
        if state["status"] not in {"starting", "running"}:
            return state
        try:
            import psutil

            process = psutil.Process(int(state["pid"]))
            for child in process.children(recursive=True):
                child.terminate()
            process.terminate()
        except Exception as exc:
            raise PlanFlowError("CANCEL_FAILED", f"Could not stop run {run_id}: {exc}") from exc
        run_path = self.store.task_path(task_id) / "runs" / run_id
        state.update({"status": "cancelled", "updated_at": now_iso()})
        write_json_atomic(run_path / "run.json", state)
        return state


def _same_process(state: dict[str, Any]) -> bool:
    try:
        import psutil

        process = psutil.Process(int(state["pid"]))
        expected = state.get("pid_create_time")
        return process.is_running() and (expected is None or abs(process.create_time() - float(expected)) < 0.01)
    except Exception:
        return False


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


def execute_worker(workspace: str, task_id: str, plan_version: str, run_id: str) -> None:
    store = PlanStore(workspace)
    plan_path = store.plan_path(task_id, plan_version)
    run_path = store.task_path(task_id) / "runs" / run_id
    for _ in range(100):
        if (run_path / ".started").is_file():
            break
        time.sleep(0.05)
    state = read_json(run_path / "run.json")
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
            command = [sys.executable, str(script), *_postprocess_arguments(arguments)]
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
        state.update({"status": "failed", "updated_at": now_iso(), "error": str(exc)})
    write_json_atomic(run_path / "run.json", state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("workspace", nargs="?")
    parser.add_argument("task_id", nargs="?")
    parser.add_argument("plan_version", nargs="?")
    parser.add_argument("run_id", nargs="?")
    args = parser.parse_args()
    if not args.worker:
        parser.error("runner is an internal worker")
    execute_worker(args.workspace, args.task_id, args.plan_version, args.run_id)


if __name__ == "__main__":
    main()

"""Internal worker process launched by LocalProcessBackend."""

from __future__ import annotations
from filelock import FileLock

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from ..common import PlanFlowError, now_iso, read_json, read_yaml, sha256_file, write_json_atomic, write_text_atomic
from ..result_manifest import write_result_manifest
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


def _write_worker_state(path, state):
    with FileLock(path.with_suffix('.lock')):
        current = read_json(path)
        if current.get('status') in {'cancelling', 'cancelled', 'failed', 'succeeded'}:
            return
        write_json_atomic(path, state)


def execute_worker(workspace: str, task_id: str, plan_version: str, run_id: str) -> int:
    store = PlanStore(workspace)
    plan_path = store.plan_path(task_id, plan_version)
    run_path = store.task_path(task_id) / "runs" / run_id
    for _ in range(100):
        if (run_path / ".started").is_file():
            break
        time.sleep(0.05)
    else:
        state = read_json(run_path / "run.json")
        state.update(status="failed", error_code="START_INTERRUPTED", error="Host did not commit the run start")
        _write_worker_state(run_path / "run.json", state)
        return 1
    state = read_json(run_path / "run.json")
    exit_code = 0
    try:
        store.verify_bundle(task_id, plan_version)
        plan = read_yaml(plan_path / "plan.yaml")
        from ..runtime_environment_lock import verify_runtime_lock

        bindings = plan.get("model_bindings", [])
        required_packages = {
            str(package).casefold()
            for binding in bindings
            for package in binding.get("runtime_packages", [])
        }
        required_packages.update(
            str(binding["distribution"]).casefold()
            for binding in bindings
            if binding.get("backend") == "python-distribution"
        )
        verify_runtime_lock(plan.get("runtime_lock"), required_packages=required_packages)
        from ..runtime_environment_lock import inspect_plan_runtime
        environment=inspect_plan_runtime(plan)
        if not environment['ok']:
            issue=environment['blocking_issues'][0]
            raise PlanFlowError(issue['code'],issue['message'],details=environment)
        # No production execution path may mutate the shared DJ environment.
        os.environ["DATA_JUICER_DISABLE_AUTO_INSTALL"] = "1"
        recipe = read_yaml(run_path / "materialized-recipe.yaml")
        selected = {}
        if plan.get("operator_bindings"):
            from ..user_operator_store import UserOperatorStore, resolve_bindings
            selected = resolve_bindings(plan, UserOperatorStore(user_id=plan["operator_owner"]))
            expected_paths = [str(run_path / "custom_operators" / f"{name}.py") for name in selected]
            if recipe.get("custom_operator_paths") != expected_paths:
                raise PlanFlowError("OPERATOR_HASH_MISMATCH", "Materialized operator paths changed")
            for name, item in selected.items():
                if sha256_file(Path(item["_path"])) != sha256_file(run_path / "custom_operators" / f"{name}.py"):
                    raise PlanFlowError("OPERATOR_HASH_MISMATCH", "Materialized operator source changed")
                for filename, content in item["_manifest"].get("assets", {}).items():
                    if (run_path / "custom_operators" / "assets" / filename).read_text(encoding="utf-8") != content:
                        raise PlanFlowError("OPERATOR_HASH_MISMATCH", "Materialized operator asset changed")
                import importlib.metadata
                from ..user_operator_runtime import dependency_lock
                for package, expected in dependency_lock(item["_manifest"].get("dependencies", [])).items():
                    if importlib.metadata.version(package) != expected:
                        raise PlanFlowError("OPERATOR_RUNTIME_BLOCKED", "Installed dependency differs from the validated lock")
        if bindings:
            from ..model_lock_resolver import ModelLockResolver

            state.update({"status": "preparing_models", "updated_at": now_iso()})
            _write_worker_state(run_path / "run.json", state)
            resolver = ModelLockResolver()
            runtime_bindings = resolver.attach_local_sources(bindings, selected)
            paths = resolver.prepare(runtime_bindings)
            recipe, provenance = resolver.materialize(recipe, bindings, paths)
            write_json_atomic(run_path / "resolved-models.json", provenance)
            from ..common import write_yaml_atomic

            write_yaml_atomic(run_path / "materialized-recipe.yaml", recipe)
        # The preparation phase is the only phase allowed to contact a model
        # registry. DJ/Transformers must consume verified local snapshots.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        state.update({"status": "running", "updated_at": now_iso()})
        _write_worker_state(run_path / "run.json", state)
        from data_juicer.config import init_configs
        from data_juicer.core.executor import ExecutorFactory

        cfg = init_configs(["--config", str(run_path / "materialized-recipe.yaml")], load_configs_only=False)
        if any(step.get('kind')=='image_audit' for step in plan.get('postprocess',[])):
            from ..image_audit import score_audit_input
            score_audit_input(cfg, lambda steps: (state.update(steps=steps), _write_worker_state(run_path/'run.json',state)), plan['recipe']['dataset_path'])
        else:
            ExecutorFactory.create_executor(cfg.executor_type)(cfg).run()
        variables = {
            "RUN_OUTPUT": state["output_dir"],
            "RUN_DIR": str(run_path),
            "recipe.output": str(recipe["export_path"]),
        }
        post_results = []
        for index, step in enumerate(plan.get("postprocess", []) or []):
            state['active_postprocess_index']=index
            state['postprocess_results']=post_results
            _write_worker_state(run_path/'run.json',state)
            if step.get('kind')=='image_audit':
                from ..image_audit import run_image_audit
                post_results.append({'step':index+1, **run_image_audit(step,recipe['export_path'],state['output_dir'],plan['recipe']['dataset_path'],coverage=plan.get('coverage',[]))})
                state['postprocess_results']=post_results
                _write_worker_state(run_path/'run.json',state)
                continue
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
            _write_worker_state(run_path/'run.json',state)
        from ..delivery import verify_delivery
        verification = verify_delivery(plan, state['output_dir'])
        write_json_atomic(run_path / 'acceptance.json', verification)
        state.update({k: verification[k] for k in ('delivery_status', 'acceptance_status', 'task_status')})
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
                f"- Delivery: {verification['delivery_status']}",
                f"- Acceptance: {verification['acceptance_status']}",
                f"- Evidence: {run_path / 'acceptance.json'}",
                f"- Postprocess steps: {len(post_results)}",
                "",
            ]
        )
        write_text_atomic(run_path / "report.md", report)
        finished_at = now_iso()
        manifest = write_result_manifest(
            state["output_dir"],
            run_id=run_id,
            started_at=state["created_at"],
            finished_at=finished_at,
            metadata={"recipe_sha256": sha256_file(run_path / "materialized-recipe.yaml")},
        )
        state.update(
            {
                "status": "succeeded",
                "execution_status": "succeeded",
                "updated_at": finished_at,
                "recipe_output": str(recipe["export_path"]),
                "postprocess_results": post_results,
                "report_path": str(run_path / "report.md"),
                "result_manifest_path": str(Path(state["output_dir"]) / "result-manifest.json"),
                "result_output_count": manifest["output_count"],
            }
        )
    except Exception as exc:
        traceback.print_exc()
        state.update(execution_status='failed', delivery_status='failed', acceptance_status='unverified', task_status='failed')
        write_text_atomic(run_path / 'report.md', f'# Failed run\n\n{type(exc).__name__}: {exc}\n')
        state['report_path'] = str(run_path / 'report.md')
        exit_code = 20
        error_code = exc.code if isinstance(exc, PlanFlowError) else "EXECUTION_FAILED"
        state.update({"status": "failed", "updated_at": now_iso(), "error_code": error_code, "error": str(exc)})
        from ..recovery import recovery
        state['recovery_actions']=recovery(error_code)[1]
        state['error_details']=getattr(exc,'details',None)
        state['stdout_log']=str(run_path/'logs'/'stdout.log')
        state['stderr_log']=str(run_path/'logs'/'stderr.log')
        diagnostic_path = run_path / 'failure-manifest.json'
        write_json_atomic(diagnostic_path, {'task_id': task_id, 'plan_version': plan_version, 'run_id': run_id,
            'execution_status': 'failed', 'error_code': error_code, 'error': str(exc),
            'report_path': state['report_path'], 'stdout_log': state.get('stdout_log'), 'stderr_log': state.get('stderr_log'),
            'error_details':state['error_details'],'recovery_actions':state['recovery_actions']})
        state['failure_manifest_path'] = str(diagnostic_path)
    _write_worker_state(run_path / "run.json", state)
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

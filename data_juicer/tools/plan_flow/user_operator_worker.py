"""Disposable validation worker. User code is imported only here, never in MCP."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

from .common import read_json, write_json_atomic


def run(root):
    from data_juicer.config.config import load_custom_operators
    from data_juicer.ops import OPERATORS
    from data_juicer.tools.op_search import analyze_tag_from_cls
    from .discovery import _json_safe_value

    request = read_json(root / "request.json")
    name, category = request["name"], request["category"]
    before = dict(OPERATORS.modules)
    if name in before:
        raise ValueError("A personal operator cannot use a built-in registration name")
    load_custom_operators([str(root / f"{name}.py")])
    after = dict(OPERATORS.modules)
    if set(after) - set(before) != {name} or any(after[key] is not value for key, value in before.items()):
        raise ValueError("Source must register exactly one new operator and must not change built-ins")
    cls = after[name]
    if category not in {base.__name__.lower() for base in cls.__mro__}:
        raise ValueError("Operator must inherit the declared DJ base class")
    parameters = {}
    for param in inspect.signature(cls.__init__).parameters.values():
        if param.name == "self" or param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        parameters[param.name] = {
            "type": "Any"
            if param.annotation is inspect.Parameter.empty
            else inspect.formatannotation(param.annotation),
            "required": param.default is inspect.Parameter.empty,
            "default": None if param.default is inspect.Parameter.empty else _json_safe_value(param.default),
            "description": "",
        }
    schema = {
        "name": name,
        "type": category,
        "description": inspect.getdoc(cls) or request.get("description", ""),
        "parameters": parameters,
        "tags": analyze_tag_from_cls(cls, name),
        "signature": str(inspect.signature(cls.__init__)),
    }
    write_json_atomic(root / "schema.json", schema)
    # Run in another fresh process so init_configs loads the module exactly once.
    from .common import write_yaml_atomic

    test_parameters = dict(request.get("parameters", {}))
    for key in ("save_dir", "output_dir", "work_dir"):
        if key in parameters or key in test_parameters:
            test_parameters[key] = str(root / "generated")
    for key in ("output_path", "export_path", "stats_export_path"):
        if test_parameters.get(key):
            test_parameters[key] = str(root / "generated" / Path(str(test_parameters[key])).name)
    (root / "generated").mkdir(exist_ok=True)
    recipe = {
        "job_id": "validation",
        "work_dir": str(root / "validation"),
        "np": 1,
        "dataset": {"configs": [{"type": "local", "path": str(root / "input.jsonl")}]},
        "export_path": str(root / "validation" / "output.jsonl"),
        "use_cache": False,
        "custom_operator_paths": [str(root / f"{name}.py")],
        "process": [{name: test_parameters}],
    }
    write_yaml_atomic(root / "recipe.yaml", recipe)


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    try:
        if sys.argv[2] == "schema":
            run(root)
        else:
            import shutil
            import time
            from .common import read_yaml
            from .runner import PlanRunner

            # The service constructed these fixture-only paths and froze the contract.
            # Internal approval is for the disposable test, never a business Plan.
            runner = PlanRunner(root)
            task_id, _ = runner.store.create_task("validation")
            recipe = read_yaml(root / "recipe.yaml")
            recipe["export_path"] = "${RUN_OUTPUT}/output.jsonl"
            saved = runner.store.save_plan(
                task_id=task_id,
                plan={"user_intent": "validation", "recipe": recipe},
                validation={"ok": True, "errors": [], "warnings": []},
                artifact_paths=[],
                base_plan_version=None,
            )
            runner.store.approve(
                task_id,
                saved["plan_version"],
                saved["content_hash"],
                "Disposable operator test",
            )
            state = runner.start(task_id, saved["plan_version"])
            while state["status"] in {"starting", "running"}:
                time.sleep(0.1)
                state = runner.get(task_id, state["run_id"])
            if state["status"] != "succeeded":
                raise RuntimeError(state.get("error") or "Native validation run failed")
            (root / "validation").mkdir(exist_ok=True)
            shutil.copyfile(
                Path(state["output_dir"]) / "output.jsonl",
                root / "validation" / "output.jsonl",
            )
            from .user_operator_resources import cache_snapshot

            write_json_atomic(root / "cache-snapshot.json", cache_snapshot())
    except Exception as exc:
        messages = []
        current = exc
        while current is not None and len(messages) < 4:
            messages.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
        write_json_atomic(root / "error.json", {"error": "; ".join(messages)})
        raise

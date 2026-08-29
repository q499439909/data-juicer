import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from data_juicer.tools.plan_flow import container_entry
from data_juicer.tools.plan_flow.container_entry import ContainerEntryError, ExitCode, ModelMount, RunSpec


def _spec(**overrides):
    value = {
        "schema_version": 1,
        "run_id": "run_r001",
        "tenant_id": "local-test",
        "recipe": {
            "path": "/run/bundle/materialized-recipe.yaml",
            "sha256": "sha256:" + "a" * 64,
        },
        "mounts": {
            "input": "/workspace/input",
            "bundle": "/run/bundle",
            "output": "/workspace/output",
            "work": "/run/work",
            "temp": "/tmp",
        },
        "models": [],
    }
    value.update(overrides)
    return value


def _parsed_spec(models=()):
    return RunSpec(
        run_id="run_r001",
        tenant_id="local-test",
        recipe_path=PurePosixPath("/run/bundle/materialized-recipe.yaml"),
        recipe_sha256="sha256:" + "a" * 64,
        mounts={
            "input": PurePosixPath("/workspace/input"),
            "bundle": PurePosixPath("/run/bundle"),
            "output": PurePosixPath("/workspace/output"),
            "work": PurePosixPath("/run/work"),
            "temp": PurePosixPath("/tmp"),
        },
        models=tuple(models),
    )


def _recipe(**overrides):
    value = {
        "dataset": {"configs": [{"type": "local", "path": "/workspace/input/input.jsonl"}]},
        "export_path": "/workspace/output/result.jsonl",
        "work_dir": "/run/work",
        "temp_dir": "/tmp",
        "process": [],
    }
    value.update(overrides)
    return value


def test_run_spec_has_a_strict_small_schema():
    parsed = container_entry._parse_run_spec(_spec())

    assert parsed.run_id == "run_r001"
    assert parsed.mounts["output"] == PurePosixPath("/workspace/output")

    invalid = _spec(unexpected=True)
    with pytest.raises(ContainerEntryError) as exc_info:
        container_entry._parse_run_spec(invalid)
    assert exc_info.value.exit_code == ExitCode.INVALID_SPEC
    assert exc_info.value.details["unexpected"] == ["unexpected"]


def test_container_entry_does_not_eagerly_import_the_mcp_module():
    script = """
import sys
import data_juicer.tools.plan_flow.container_entry

assert "data_juicer.tools.plan_flow.service" not in sys.modules
assert "data_juicer.tools.plan_flow.discovery" not in sys.modules
"""

    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("run_id", ["r001", "run_r001/escape", "run_r001\\escape", "run_r001 shell"])
def test_run_id_rejects_unsafe_values(run_id):
    with pytest.raises(ContainerEntryError, match="run_id"):
        container_entry._parse_run_spec(_spec(run_id=run_id))


def test_model_mount_must_match_its_declared_artifact():
    value = _spec(models=[{"artifact_id": "fixture-v1", "path": "/models/other"}])

    with pytest.raises(ContainerEntryError, match="must be /models/fixture-v1"):
        container_entry._parse_run_spec(value)


def test_recipe_accepts_only_materialized_declared_paths():
    spec = _parsed_spec(models=[ModelMount("fixture-v1", PurePosixPath("/models/fixture-v1"))])
    recipe = _recipe(model_path="/models/fixture-v1/weights.bin")

    container_entry._validate_recipe_paths(recipe, spec)

    with pytest.raises(ContainerEntryError) as windows_path:
        container_entry._validate_recipe_paths(_recipe(dataset_path=r"D:\private\input.jsonl"), spec)
    assert windows_path.value.code == "HOST_PATH_NOT_ALLOWED"

    with pytest.raises(ContainerEntryError) as undeclared_model:
        container_entry._validate_recipe_paths(_recipe(model_path="/models/other/weights.bin"), spec)
    assert undeclared_model.value.code == "PATH_NOT_ALLOWED"

    with pytest.raises(ContainerEntryError) as traversal:
        container_entry._validate_recipe_paths(_recipe(export_path="/workspace/output/../input/stolen.jsonl"), spec)
    assert traversal.value.code == "PATH_TRAVERSAL"


def test_recipe_requires_fixed_work_and_temp_directories():
    spec = _parsed_spec()

    with pytest.raises(ContainerEntryError) as work_error:
        container_entry._validate_recipe_paths(_recipe(work_dir="/workspace/output/work"), spec)
    assert work_error.value.code == "WORK_DIR_NOT_ALLOWED"

    with pytest.raises(ContainerEntryError) as temp_error:
        container_entry._validate_recipe_paths(_recipe(temp_dir="/workspace/output/tmp"), spec)
    assert temp_error.value.code == "TEMP_DIR_NOT_ALLOWED"


def test_mountinfo_uses_the_most_specific_mount():
    mountinfo = "\n".join(
        [
            "1 0 0:1 / / rw,relatime - overlay overlay rw",
            "2 1 0:2 / /run/bundle ro,relatime - 9p host ro",
            "3 1 0:3 / /workspace/output rw,relatime - 9p host rw",
        ]
    )

    assert container_entry._mount_is_read_only(Path("/run/bundle/recipe.yaml"), mountinfo) is True
    assert container_entry._mount_is_read_only(Path("/workspace/output/result.jsonl"), mountinfo) is False


def test_recipe_hash_mismatch_has_a_stable_exit_code(tmp_path, monkeypatch):
    recipe = tmp_path / "materialized-recipe.yaml"
    recipe.write_text("process: []\n", encoding="utf-8")
    expected = "sha256:" + "0" * 64
    parsed = _parsed_spec()
    parsed = RunSpec(
        run_id=parsed.run_id,
        tenant_id=parsed.tenant_id,
        recipe_path=PurePosixPath(str(recipe).replace("\\", "/")),
        recipe_sha256=expected,
        mounts=parsed.mounts,
        models=parsed.models,
    )
    monkeypatch.setattr(container_entry, "_load_run_spec", lambda path: parsed)

    with pytest.raises(ContainerEntryError) as exc_info:
        container_entry._prepare_run(tmp_path / "run-spec.json")
    assert exc_info.value.code == "RECIPE_HASH_MISMATCH"
    assert exc_info.value.exit_code == ExitCode.RECIPE_HASH_MISMATCH


def test_run_from_spec_executes_once_and_writes_manifest(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text("process: []\n", encoding="utf-8")
    recipe_hash = "sha256:" + hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    spec = _parsed_spec()
    spec = RunSpec(
        run_id=spec.run_id,
        tenant_id=spec.tenant_id,
        recipe_path=spec.recipe_path,
        recipe_sha256=recipe_hash,
        mounts=spec.mounts,
        models=spec.models,
    )
    prepared = container_entry.PreparedRun(spec=spec, recipe=_recipe(), recipe_path=recipe_path, output_root=output)
    calls = []

    monkeypatch.setattr(container_entry, "_prepare_run", lambda path: prepared)

    def execute(path):
        calls.append(path)
        (output / "result.jsonl").write_text('{"text":"ok"}\n', encoding="utf-8")

    monkeypatch.setattr(container_entry, "_execute_recipe", execute)

    manifest = container_entry.run_from_spec(tmp_path / "run-spec.json")

    assert calls == [recipe_path]
    assert manifest["status"] == "succeeded"
    assert manifest["output_count"] == 1
    assert manifest["outputs"][0]["path"] == "result.jsonl"
    assert json.loads((output / "result-manifest.json").read_text(encoding="utf-8")) == manifest


def test_main_returns_structured_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        container_entry,
        "run_from_spec",
        lambda path: (_ for _ in ()).throw(
            ContainerEntryError("BAD_MOUNT", "mount rejected", ExitCode.MOUNT_POLICY_VIOLATION)
        ),
    )

    exit_code = container_entry.main(["--run-spec", "/run/bundle/run-spec.json"])

    assert exit_code == ExitCode.MOUNT_POLICY_VIOLATION
    payload = json.loads(capsys.readouterr().err)
    assert payload == {"ok": False, "error": {"code": "BAD_MOUNT", "message": "mount rejected"}}

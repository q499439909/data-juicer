import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.execution import DockerBackend, RunHandle, RuntimeSpec
from data_juicer.tools.plan_flow.model_store import LocalModelInstaller, LocalModelStore
from data_juicer.tools.plan_flow.runner import PlanRunner
from data_juicer.tools.plan_flow.store import PlanStore

IMAGE_ID = "sha256:" + "a" * 64
CONTAINER_ID = "b" * 64


def _approved_plan(
    workspace: Path,
    dataset: Path | None = None,
    *,
    models: list[dict] | None = None,
    recipe_extra: dict | None = None,
) -> tuple[str, str]:
    dataset = dataset or workspace / "input.jsonl"
    if not dataset.exists():
        dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    store = PlanStore(workspace)
    task_id, _ = store.create_task("Docker backend", "task_docker")
    recipe = {
        "dataset_path": str(dataset),
        "export_path": "${RUN_OUTPUT}/result.jsonl",
        "process": [],
        "executor_type": "default",
        "np": 1,
    }
    recipe.update(recipe_extra or {})
    saved = store.save_plan(
        task_id=task_id,
        plan={
            "user_intent": "Docker backend",
            "recipe": recipe,
            "postprocess": [],
            "models": models or [],
        },
        validation={"ok": True, "errors": [], "warnings": []},
        artifact_paths=[],
        base_plan_version=None,
    )
    store.approve(task_id, saved["plan_version"], saved["content_hash"], "test")
    return task_id, saved["plan_version"]


class FakeDocker:
    def __init__(self, state="running", exit_code=0, oom=False):
        self.calls = []
        self.state = state
        self.exit_code = exit_code
        self.oom = oom

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        args = command[1:]
        stdout, stderr, returncode = "", "", 0
        if args[:2] == ["image", "inspect"]:
            stdout = json.dumps([{"Id": IMAGE_ID}])
        elif args[0] == "create":
            stdout = CONTAINER_ID + "\n"
        elif args[0] == "inspect":
            stdout = json.dumps([{"State": {"Status": self.state, "ExitCode": self.exit_code, "OOMKilled": self.oom}}])
        elif args[0] == "logs":
            stdout, stderr = "container stdout\n", "container stderr\n"
        elif args[0] == "stop":
            self.state, self.exit_code = "exited", 137
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _backend(
    workspace: Path, worker: Path, fake: FakeDocker, model_store: LocalModelStore | None = None
) -> DockerBackend:
    return DockerBackend(workspace, worker, "test-image:tag", model_store=model_store, command_runner=fake)


def _published_model(worker: Path) -> LocalModelStore:
    fixture = worker / "fixtures" / "model-fixtures" / "fixture-model-v1"
    fixture.mkdir(parents=True)
    content = b"model-bytes\n"
    (fixture / "weights.bin").write_bytes(content)
    file_hash = "sha256:" + hashlib.sha256(content).hexdigest()
    aggregate = "sha256:" + hashlib.sha256(b"weights.bin\0" + content + b"\0").hexdigest()
    (fixture / "model-manifest.yaml").write_text(
        "\n".join(
            [
                "artifact_id: fixture-model-v1",
                "source: local-fixture",
                "revision: v1",
                f"sha256: {aggregate}",
                f"size_bytes: {len(content)}",
                "license:",
                "  status: approved-for-test",
                "files:",
                "  - path: weights.bin",
                f"    sha256: {file_hash}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    installer = LocalModelInstaller(worker, fixture.parent)
    installer.stage_local("request-model", fixture)
    store = LocalModelStore(worker)
    store.publish("request-model")
    return store


def _write_result(worker_run: Path, run_id: str) -> None:
    output = worker_run / "output"
    result = output / "result.jsonl"
    result.write_text('{"text":"hello"}\n', encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(result.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "tenant_id": "local-test",
        "status": "succeeded",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "recipe_sha256": "sha256:" + "0" * 64,
        "output_count": 1,
        "output_size_bytes": result.stat().st_size,
        "outputs": [{"path": "result.jsonl", "size_bytes": result.stat().st_size, "sha256": digest}],
    }
    (output / "result-manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def test_docker_backend_uses_safe_locked_down_create_and_private_identity(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    task_id, version = _approved_plan(workspace)
    fake = FakeDocker()
    runner = PlanRunner(workspace, backend=_backend(workspace, worker, fake))

    started = runner.start(task_id, version)

    handle = started["handle"]
    assert set(handle) == {"schema_version", "backend", "run_id", "created_at", "deadline", "backend_ref"}
    assert "container_id" not in json.dumps(started)
    record = json.loads((workspace / ".dj" / "execution" / "docker" / f"{handle['backend_ref']}.json").read_text())
    assert record["container_id"] == CONTAINER_ID
    assert record["image_id"] == IMAGE_ID
    create, kwargs = next(call for call in fake.calls if call[0][1] == "create")
    joined = " ".join(create)
    for expected in ("--read-only", "--network none", "--cap-drop ALL", "no-new-privileges:true", "--memory-swap"):
        assert expected in joined
    assert IMAGE_ID in create
    assert kwargs["check"] is False
    assert all(isinstance(item, str) for item in create)
    mounts = [create[index + 1] for index, value in enumerate(create) if value == "--mount"]
    assert len(mounts) == 4
    assert all(str(worker.resolve()) in mount for mount in mounts)
    assert "readonly" in mounts[0] and "readonly" in mounts[1]
    assert "readonly" not in mounts[2] and "readonly" not in mounts[3]


def test_docker_backend_collects_verified_output_and_survives_adapter_restart(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    task_id, version = _approved_plan(workspace)
    fake = FakeDocker()
    runner = PlanRunner(workspace, backend=_backend(workspace, worker, fake))
    started = runner.start(task_id, version)
    handle = RunHandle.from_dict(started["handle"])
    record_path = workspace / ".dj" / "execution" / "docker" / f"{handle.backend_ref}.json"
    record = json.loads(record_path.read_text())
    _write_result(Path(record["worker_run_root"]), handle.run_id)
    fake.state = "exited"

    restarted = _backend(workspace, worker, fake)
    result = restarted.collect(handle)

    assert result.status == "succeeded"
    assert result.provenance["image_id"] == IMAGE_ID
    spec = RuntimeSpec.from_dict(record["runtime_spec"])
    assert (Path(spec.output_dir) / "result.jsonl").is_file()
    assert (Path(spec.run_dir) / "runtime-provenance.json").is_file()
    restarted.cleanup(handle)
    assert record_path.exists() is False
    assert (Path(record["worker_run_root"]) / "work").exists() is False
    assert (Path(record["worker_run_root"]) / "bundle").is_dir()


def test_docker_backend_rejects_dataset_outside_workspace_before_create(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    task_id, version = _approved_plan(workspace, outside)
    fake = FakeDocker()

    with pytest.raises(PlanFlowError) as error:
        PlanRunner(workspace, backend=_backend(workspace, worker, fake)).start(task_id, version)

    assert error.value.code == "PATH_NOT_ALLOWED"
    assert not any(call[0][1] == "create" for call in fake.calls)


@pytest.mark.parametrize(
    ("exit_code", "oom", "error_code"),
    [(20, False, "EXECUTION_FAILED"), (137, True, "RUN_OOM")],
)
def test_docker_backend_maps_failure_and_retains_work(tmp_path, exit_code, oom, error_code):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    task_id, version = _approved_plan(workspace)
    fake = FakeDocker(state="running")
    backend = _backend(workspace, worker, fake)
    started = PlanRunner(workspace, backend=backend).start(task_id, version)
    handle = RunHandle.from_dict(started["handle"])
    record_path = workspace / ".dj" / "execution" / "docker" / f"{handle.backend_ref}.json"
    run_root = Path(json.loads(record_path.read_text())["worker_run_root"])
    fake.state, fake.exit_code, fake.oom = "exited", exit_code, oom

    result = backend.collect(handle)
    backend.cleanup(handle)

    assert result.status == "failed"
    assert result.error_code == error_code
    assert (run_root / "work").is_dir()


def test_docker_backend_cancel_has_stable_terminal_semantics(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    task_id, version = _approved_plan(workspace)
    fake = FakeDocker()
    backend = _backend(workspace, worker, fake)
    started = PlanRunner(workspace, backend=backend).start(task_id, version)
    handle = RunHandle.from_dict(started["handle"])

    backend.cancel(handle)
    result = backend.collect(handle)

    assert result.status == "cancelled"
    assert any(call[0][1] == "stop" for call in fake.calls)
    forged_deadline = RunHandle(
        "docker", handle.run_id, handle.created_at - timedelta(minutes=2),
        datetime.now(timezone.utc) - timedelta(minutes=1), handle.backend_ref,
    )
    with pytest.raises(PlanFlowError) as forged:
        backend.inspect(forged_deadline)
    assert forged.value.code == "INVALID_BACKEND_STATE"


def test_docker_backend_enforces_deadline_and_reports_timeout(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    task_id, version = _approved_plan(workspace)
    fake = FakeDocker()
    backend = _backend(workspace, worker, fake)
    started = PlanRunner(workspace, backend=backend).start(task_id, version)
    original = RunHandle.from_dict(started["handle"])
    record_path = workspace / ".dj" / "execution" / "docker" / f"{original.backend_ref}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    deadline = original.created_at + timedelta(microseconds=1)
    record["runtime_spec"]["deadline"] = deadline.isoformat()
    record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    handle = RunHandle("docker", original.run_id, original.created_at, deadline, original.backend_ref)

    assert backend.inspect(handle).status == "failed"
    result = backend.collect(handle)

    assert result.error_code == "RUN_TIMED_OUT"
    assert any(call[0][1] == "stop" for call in fake.calls)


def test_docker_backend_rejects_cross_backend_and_forged_refs(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    backend = _backend(workspace, worker, FakeDocker())
    now = datetime.now(timezone.utc)
    with pytest.raises(PlanFlowError) as cross:
        backend.inspect(RunHandle("local-process", "run_r001", now, None, "opaque"))
    assert cross.value.code == "BACKEND_MISMATCH"
    with pytest.raises(PlanFlowError) as forged:
        backend.inspect(RunHandle("docker", "run_r001", now, None, "../../state"))
    assert forged.value.code == "INVALID_BACKEND_REF"


def test_docker_backend_materializes_and_mounts_verified_models_readonly(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    model_store = _published_model(worker)
    task_id, version = _approved_plan(
        workspace,
        models=[{"artifact_id": "fixture-model-v1"}],
        recipe_extra={"model_path": "model-store://fixture-model-v1"},
    )
    fake = FakeDocker()
    backend = _backend(workspace, worker, fake, model_store)

    started = PlanRunner(workspace, backend=backend).start(task_id, version)
    handle = RunHandle.from_dict(started["handle"])
    record_path = workspace / ".dj" / "execution" / "docker" / f"{handle.backend_ref}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    recipe = (Path(record["worker_run_root"]) / "bundle" / "materialized-recipe.yaml").read_text(
        encoding="utf-8"
    )
    create = next(call[0] for call in fake.calls if call[0][1] == "create")

    assert "model_path: /models/fixture-model-v1" in recipe
    assert any(
        value.endswith("dst=/models/fixture-model-v1,readonly")
        for index, value in enumerate(create)
        if index > 0 and create[index - 1] == "--mount"
    )
    assert record["models"][0]["manifest"]["sha256"].startswith("sha256:")
    _write_result(Path(record["worker_run_root"]), handle.run_id)
    fake.state = "exited"
    result = backend.collect(handle)
    assert result.provenance["models"][0]["artifact_id"] == "fixture-model-v1"


def test_docker_backend_rejects_model_file_absent_from_manifest_before_create(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    model_store = _published_model(worker)
    task_id, version = _approved_plan(
        workspace,
        models=[{"artifact_id": "fixture-model-v1"}],
        recipe_extra={"model_path": "model-store://fixture-model-v1/missing.bin"},
    )
    fake = FakeDocker()

    with pytest.raises(PlanFlowError) as error:
        PlanRunner(workspace, backend=_backend(workspace, worker, fake, model_store)).start(task_id, version)

    assert error.value.code == "MODEL_FILE_NOT_DECLARED"
    assert not any(call[0][1] == "create" for call in fake.calls)


@pytest.mark.skipif(os.environ.get("DJ_RUN_DOCKER_INTEGRATION") != "1", reason="explicit Docker integration opt-in")
def test_real_docker_backend_end_to_end_and_cleanup(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_id, version = _approved_plan(workspace)
    backend = DockerBackend(workspace, Path("D:/dsh-worker"), "dj-plan-flow-cpu:local-v1")
    runner = PlanRunner(workspace, backend=backend)

    started = runner.start(task_id, version)
    handle = RunHandle.from_dict(started["handle"])
    record_path = workspace / ".dj" / "execution" / "docker" / f"{handle.backend_ref}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    inspected = json.loads(
        subprocess.run(
            ["docker", "inspect", record["container_id"]],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )[0]
    host_config = inspected["HostConfig"]
    mount_access = {mount["Destination"]: mount["RW"] for mount in inspected["Mounts"]}
    assert inspected["Config"]["User"] == "10001:10001"
    assert host_config["ReadonlyRootfs"] is True
    assert host_config["NetworkMode"] == "none"
    assert host_config["CapDrop"] == ["ALL"]
    assert "no-new-privileges:true" in host_config["SecurityOpt"]
    assert host_config["PidsLimit"] == 256
    assert host_config["NanoCpus"] == 2_000_000_000
    assert host_config["Memory"] == host_config["MemorySwap"] == 8 * 1024**3
    assert mount_access == {
        "/workspace/input": False,
        "/run/bundle": False,
        "/workspace/output": True,
        "/run/work": True,
    }
    for _ in range(120):
        state = runner.get(task_id, handle.run_id)
        if state["status"] not in {"starting", "running"}:
            break
        time.sleep(0.5)
    else:
        backend.cancel(handle)
        pytest.fail("real Docker run did not finish within 60 seconds")

    assert state["status"] == "succeeded", state
    assert state["runtime_provenance"]["image_id"].startswith("sha256:")
    assert not {"container_id", "image_id"} & set(state["handle"])
    assert (Path(state["output_dir"]) / "result.jsonl").is_file()
    restarted = DockerBackend(workspace, Path("D:/dsh-worker"), "dj-plan-flow-cpu:local-v1")
    assert restarted.collect(handle).status == "succeeded"
    restarted.cleanup(handle)
    run_root = Path(record["worker_run_root"])
    assert (run_root / "work").exists() is False
    assert (run_root / "bundle" / "run-spec.json").is_file()
    remaining = subprocess.run(
        ["docker", "inspect", record["container_id"]], capture_output=True, text=True, check=False
    )
    assert remaining.returncode != 0


@pytest.mark.skipif(os.environ.get("DJ_RUN_DOCKER_INTEGRATION") != "1", reason="explicit Docker integration opt-in")
def test_real_docker_backend_mounts_published_model_and_records_provenance(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    worker = Path("D:/dsh-worker")
    store = LocalModelStore(worker)
    artifact = store.resolve("fixture-tiny-model-v1")
    task_id, version = _approved_plan(workspace, models=[{"artifact_id": artifact.manifest.artifact_id}])
    backend = DockerBackend(
        workspace, worker, "dj-plan-flow-cpu:local-v1", model_store=store
    )
    runner = PlanRunner(workspace, backend=backend)

    started = runner.start(task_id, version)
    handle = RunHandle.from_dict(started["handle"])
    for _ in range(120):
        state = runner.get(task_id, handle.run_id)
        if state["status"] not in {"starting", "running"}:
            break
        time.sleep(0.5)
    else:
        backend.cancel(handle)
        pytest.fail("real Docker model run did not finish within 60 seconds")

    assert state["status"] == "succeeded", state
    assert state["runtime_provenance"]["models"] == [artifact.manifest.to_provenance()]
    backend.cleanup(handle)

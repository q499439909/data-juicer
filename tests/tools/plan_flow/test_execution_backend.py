import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.execution import (
    ExecutionBackend,
    LocalProcessBackend,
    RunHandle,
    RunResult,
    RunStatus,
    RuntimeSpec,
)
from data_juicer.tools.plan_flow.runner import PlanRunner
from data_juicer.tools.plan_flow.store import PlanStore


def _runtime_spec(tmp_path: Path) -> RuntimeSpec:
    run_dir = tmp_path / ".dj" / "tasks" / "task_test" / "runs" / "run_r001"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    output = tmp_path / "outputs" / "run_r001"
    output.mkdir(parents=True)
    recipe = run_dir / "materialized-recipe.yaml"
    recipe.write_text("process: []\n", encoding="utf-8")
    return RuntimeSpec(
        task_id="task_test",
        plan_version="plan_v001",
        run_id="run_r001",
        workspace_root=str(tmp_path.resolve()),
        run_dir=str(run_dir),
        output_dir=str(output),
        recipe_path=str(recipe),
        stdout_log=str(logs / "stdout.log"),
        stderr_log=str(logs / "stderr.log"),
        content_hash="sha256:" + "0" * 64,
        created_at=datetime.now(timezone.utc),
    )


def _approved_plan(tmp_path: Path) -> tuple[str, str]:
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    store = PlanStore(tmp_path)
    task_id, _ = store.create_task("Backend contract", "task_backend")
    saved = store.save_plan(
        task_id=task_id,
        plan={
            "user_intent": "Backend contract",
            "recipe": {
                "dataset_path": str(dataset),
                "export_path": "${RUN_OUTPUT}/result.jsonl",
                "process": [{"text_length_filter": {"min_len": 2}}],
                "executor_type": "default",
                "np": 1,
            },
            "postprocess": [],
        },
        validation={"ok": True, "errors": [], "warnings": []},
        artifact_paths=[],
        base_plan_version=None,
    )
    store.approve(task_id, saved["plan_version"], saved["content_hash"], "test")
    return task_id, saved["plan_version"]


class RecordingBackend:
    name = "fake"

    def __init__(self, observed_status="running"):
        self.started = []
        self.inspected = []
        self.cancelled = []
        self.cleaned = []
        self.observed_status = observed_status

    def start(self, spec):
        self.started.append(spec)
        return RunHandle(self.name, spec.run_id, spec.created_at, spec.deadline, "opaque-ref")

    def inspect(self, handle):
        self.inspected.append(handle)
        message = "backend record missing" if self.observed_status == "lost" else None
        return RunStatus(self.observed_status, datetime.now(timezone.utc), message)

    def cancel(self, handle):
        self.cancelled.append(handle)

    def collect(self, handle):
        status = "failed" if self.observed_status == "lost" else self.observed_status
        return RunResult(status, datetime.now(timezone.utc), exit_code=0)

    def cleanup(self, handle):
        self.cleaned.append(handle)


def test_runtime_spec_and_handle_round_trip_without_backend_specific_fields(tmp_path):
    spec = _runtime_spec(tmp_path)
    restored_spec = RuntimeSpec.from_dict(spec.to_dict())
    handle = RunHandle(
        backend="local-process",
        run_id=spec.run_id,
        created_at=spec.created_at,
        deadline=spec.created_at + timedelta(minutes=5),
        backend_ref="opaque-reference",
    )

    assert restored_spec == spec
    assert RunHandle.from_dict(handle.to_dict()) == handle
    assert set(handle.to_dict()) == {
        "schema_version",
        "backend",
        "run_id",
        "created_at",
        "deadline",
        "backend_ref",
    }
    assert not {"pid", "pid_create_time", "container_id", "image_id", "pod_uid"} & set(handle.to_dict())

    leaked = handle.to_dict() | {"container_id": "not-allowed"}
    with pytest.raises(PlanFlowError) as leaked_error:
        RunHandle.from_dict(leaked)
    assert leaked_error.value.code == "INVALID_RUN_HANDLE"


def test_plan_runner_uses_injected_backend_and_persists_only_opaque_handle(tmp_path):
    task_id, plan_version = _approved_plan(tmp_path)
    backend = RecordingBackend()
    assert isinstance(backend, ExecutionBackend)
    runner = PlanRunner(tmp_path, backend=backend)

    started = runner.start(task_id, plan_version)

    assert len(backend.started) == 1
    assert backend.started[0].run_id == started["run_id"]
    assert started["handle"]["backend"] == "fake"
    assert started["handle"]["backend_ref"] == "opaque-ref"
    serialized = json.dumps(started)
    assert "pid" not in serialized
    assert "container_id" not in serialized
    assert "image_id" not in serialized

    observed = runner.get(task_id, started["run_id"])
    assert observed["status"] == "running"
    assert len(backend.inspected) == 1

    cancelled = runner.cancel(task_id, started["run_id"])
    assert cancelled["status"] == "cancelled"
    assert backend.cancelled == [backend.inspected[0]]


def test_runner_module_does_not_own_process_implementation():
    from data_juicer.tools.plan_flow import runner

    assert not hasattr(runner, "subprocess")
    assert not hasattr(runner, "psutil")


def test_plan_runner_maps_backend_state_loss_to_stable_failure(tmp_path):
    task_id, plan_version = _approved_plan(tmp_path)
    backend = RecordingBackend(observed_status="lost")
    runner = PlanRunner(tmp_path, backend=backend)
    started = runner.start(task_id, plan_version)

    state = runner.get(task_id, started["run_id"])

    assert state["status"] == "failed"
    assert state["error_code"] == "RUNNER_LOST"
    assert state["error"] == "backend record missing"


def test_plan_runner_collects_terminal_backend_status(tmp_path):
    task_id, plan_version = _approved_plan(tmp_path)
    backend = RecordingBackend(observed_status="succeeded")
    runner = PlanRunner(tmp_path, backend=backend)
    started = runner.start(task_id, plan_version)

    state = runner.get(task_id, started["run_id"])

    assert state["status"] == "succeeded"


def test_local_backend_keeps_pid_in_private_state(tmp_path):
    spec = _runtime_spec(tmp_path)
    backend = LocalProcessBackend(tmp_path)
    process = MagicMock(pid=4242)

    with patch(
        "data_juicer.tools.plan_flow.execution.local_process.subprocess.Popen", return_value=process
    ) as popen:
        handle = backend.start(spec)

    assert handle.backend == "local-process"
    assert not hasattr(handle, "pid")
    record_path = tmp_path / ".dj" / "execution" / "local-process" / f"{handle.backend_ref}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["pid"] == 4242
    assert record["run_id"] == spec.run_id
    assert record["runtime_spec"]["run_id"] == spec.run_id
    command = popen.call_args.args[0]
    assert "data_juicer.tools.plan_flow.execution.local_worker" in command


def test_local_backend_rejects_cross_backend_and_forged_refs(tmp_path):
    backend = LocalProcessBackend(tmp_path)
    created = datetime.now(timezone.utc)
    cross_backend = RunHandle("docker", "run_r001", created, None, "opaque")
    with pytest.raises(PlanFlowError) as cross_error:
        backend.inspect(cross_backend)
    assert cross_error.value.code == "BACKEND_MISMATCH"

    forged = RunHandle("local-process", "run_r001", created, None, "../../state")
    with pytest.raises(PlanFlowError) as forged_error:
        backend.inspect(forged)
    assert forged_error.value.code == "INVALID_BACKEND_REF"


def test_local_backend_reports_missing_private_state_as_runner_lost(tmp_path):
    backend = LocalProcessBackend(tmp_path)
    handle = RunHandle("local-process", "run_r001", datetime.now(timezone.utc), None, "0" * 32)

    status = backend.inspect(handle)
    result = backend.collect(handle)
    backend.cleanup(handle)

    assert status.status == "lost"
    assert result.status == "lost"
    assert result.error_code == "RUNNER_LOST"


def test_local_backend_collects_through_private_state_after_adapter_restart(tmp_path):
    spec = _runtime_spec(tmp_path)
    run_state = Path(spec.run_dir) / "run.json"
    run_state.write_text(
        json.dumps({"run_id": spec.run_id, "status": "succeeded"}) + "\n",
        encoding="utf-8",
    )
    backend = LocalProcessBackend(tmp_path)
    process = MagicMock(pid=4242)
    with patch("data_juicer.tools.plan_flow.execution.local_process.subprocess.Popen", return_value=process):
        handle = backend.start(spec)
    record_path = tmp_path / ".dj" / "execution" / "local-process" / f"{handle.backend_ref}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.update({"status": "exited", "exit_code": 0})
    record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    restarted_backend = LocalProcessBackend(tmp_path)
    result = restarted_backend.collect(handle)
    restarted_backend.cleanup(handle)

    assert result.status == "succeeded"
    assert record_path.exists() is False


def test_local_backend_rejects_runtime_paths_outside_its_workspace(tmp_path):
    spec = _runtime_spec(tmp_path)
    payload = spec.to_dict()
    payload["output_dir"] = str(tmp_path.parent / "escaped-output")
    escaped = RuntimeSpec.from_dict(payload)

    with pytest.raises(PlanFlowError) as error:
        LocalProcessBackend(tmp_path).start(escaped)

    assert error.value.code == "PATH_NOT_ALLOWED"


def test_local_backend_cancel_uses_only_its_private_process_identity(tmp_path):
    spec = _runtime_spec(tmp_path)
    backend = LocalProcessBackend(tmp_path)
    launched = MagicMock(pid=4242)
    with patch("data_juicer.tools.plan_flow.execution.local_process.subprocess.Popen", return_value=launched):
        handle = backend.start(spec)

    child = MagicMock()
    process = MagicMock()
    process.children.return_value = [child]
    with patch.object(backend, "_same_process", return_value=True), patch("psutil.Process", return_value=process):
        backend.cancel(handle)

    child.terminate.assert_called_once_with()
    process.terminate.assert_called_once_with()
    record_path = tmp_path / ".dj" / "execution" / "local-process" / f"{handle.backend_ref}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "cancelled"
    backend.cleanup(handle)
    assert record_path.exists() is False

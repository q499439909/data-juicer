import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from data_juicer.tools.plan_flow.broker import (
    ExecutionBroker,
    PROFILES,
    create_broker_app,
    serve_broker,
)
from data_juicer.tools.plan_flow.capability import (
    CapabilityDescriptor,
    LocalCapabilityCatalog,
)
from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.execution import (
    DockerBackend,
    RunHandle,
    RunResult,
    RunStatus,
)
from data_juicer.tools.plan_flow.runner import PlanRunner
from data_juicer.tools.plan_flow.store import PlanStore

IMAGE_ID = "sha256:" + "a" * 64


class CompletedBackend:
    name = "fake"

    def start(self, spec):
        return RunHandle(self.name, spec.run_id, spec.created_at, spec.deadline, "opaque-ref")

    def inspect(self, handle):
        return RunStatus("succeeded", datetime.now(timezone.utc))

    def collect(self, handle):
        return RunResult("succeeded", datetime.now(timezone.utc), exit_code=0)

    def cancel(self, handle):
        return None

    def cleanup(self, handle):
        return None


class LifecycleBackend(CompletedBackend):
    def __init__(self):
        self.status = "running"
        self.cleanup_count = 0

    def inspect(self, handle):
        return RunStatus(self.status, datetime.now(timezone.utc))

    def collect(self, handle):
        return RunResult(self.status, datetime.now(timezone.utc), exit_code=137)

    def cancel(self, handle):
        self.status = "cancelled"

    def cleanup(self, handle):
        self.cleanup_count += 1


class DiscoverableDocker:
    def __init__(self):
        self.container_id = "b" * 64
        self.labels = {}
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        args = command[1:]
        stdout = ""
        if args[:2] == ["image", "inspect"]:
            stdout = json.dumps([{"Id": args[2]}])
        elif args[0] == "create":
            for index, value in enumerate(args):
                if value == "--label":
                    key, label_value = args[index + 1].split("=", 1)
                    self.labels[key] = label_value
            stdout = self.container_id + "\n"
        elif args[:2] == ["ps", "-a"]:
            stdout = (self.container_id if "--no-trunc" in args else self.container_id[:12]) + "\n"
        elif args[0] == "inspect":
            stdout = json.dumps(
                [{"Config": {"Labels": self.labels}, "State": {"Status": "running", "ExitCode": 0}}]
            )
        return subprocess.CompletedProcess(command, 0, stdout, "")


def _approved_plan(workspace: Path) -> tuple[str, str]:
    dataset = workspace / "input.jsonl"
    dataset.write_text('{"text":"broker"}\n', encoding="utf-8")
    store = PlanStore(workspace)
    task_id, _ = store.create_task("Broker contract", "task_broker")
    saved = store.save_plan(
        task_id=task_id,
        plan={
            "user_intent": "Broker contract",
            "models": [],
            "recipe": {
                "dataset_path": str(dataset),
                "export_path": "${RUN_OUTPUT}/result.jsonl",
                "process": [],
                "executor_type": "default",
                "np": 1,
            },
            "postprocess": [],
        },
        validation={"ok": True, "errors": [], "warnings": []},
        artifact_paths=[],
        base_plan_version=None,
    )
    store.approve(task_id, saved["plan_version"], saved["content_hash"], "broker test")
    return task_id, saved["plan_version"]


def _broker(workspace: Path, worker: Path, backend=None) -> ExecutionBroker:
    LocalCapabilityCatalog(worker).register(
        CapabilityDescriptor(
            capability_id="broker-demo-v1",
            operator_name="demo_mapper",
            content_hash="sha256:" + "1" * 64,
            backend="docker",
            backend_ref={"image_id": IMAGE_ID},
            base_image_id="sha256:" + "b" * 64,
            created_at="2026-08-30T00:00:00+00:00",
        )
    )
    return ExecutionBroker(
        workspace,
        worker,
        allowed_capabilities=("broker-demo-v1",),
        backend_factory=lambda descriptor, profile: backend or CompletedBackend(),
    )


def test_http_client_can_start_and_query_an_approved_allowlisted_run(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    worker.mkdir()
    task_id, plan_version = _approved_plan(workspace)
    client = TestClient(create_broker_app(_broker(workspace, worker)))

    started = client.post(
        "/v1/runs",
        json={
            "task_id": task_id,
            "plan_version": plan_version,
            "capability_id": "broker-demo-v1",
            "profile": "local-tiny",
        },
    )
    observed = client.get(f"/v1/runs/{started.json()['run_id']}")

    assert started.status_code == 201
    assert observed.status_code == 200
    assert observed.json()["status"] == "succeeded"
    assert observed.json()["capability_id"] == "broker-demo-v1"
    assert not {"handle", "backend_ref", "container_id", "image_id"} & set(observed.json())


def test_restarted_broker_recovers_a_run_by_its_public_id(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    worker.mkdir()
    task_id, plan_version = _approved_plan(workspace)
    started = TestClient(create_broker_app(_broker(workspace, worker))).post(
        "/v1/runs",
        json={
            "task_id": task_id,
            "plan_version": plan_version,
            "capability_id": "broker-demo-v1",
            "profile": "local-tiny",
        },
    ).json()

    restarted = TestClient(create_broker_app(_broker(workspace, worker)))
    recovered = restarted.get(f"/v1/runs/{started['run_id']}")

    assert recovered.status_code == 200
    assert recovered.json()["run_id"] == started["run_id"]
    assert recovered.json()["status"] == "succeeded"


def test_app_startup_reconciles_an_allowlisted_same_tenant_docker_orphan(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    worker.mkdir()
    task_id, plan_version = _approved_plan(workspace)
    docker = DiscoverableDocker()
    backend = DockerBackend(
        workspace,
        worker,
        IMAGE_ID,
        tenant_id="local-test",
        limits=PROFILES["local-tiny"].limits,
        command_runner=docker,
    )
    PlanRunner(workspace, backend=backend).start(task_id, plan_version, timeout_seconds=300)
    _broker(workspace, worker)
    broker = ExecutionBroker(
        workspace,
        worker,
        allowed_capabilities=("broker-demo-v1",),
        backend_factory=lambda descriptor, profile: DockerBackend(
            workspace,
            worker,
            IMAGE_ID,
            tenant_id="local-test",
            limits=profile.limits,
            command_runner=docker,
        ),
    )

    client = TestClient(create_broker_app(broker))
    records = list(broker.state_root.glob("run_*.json"))

    assert len(records) == 1
    recovered = client.get(f"/v1/runs/{records[0].stem}").json()
    assert recovered["task_id"] == task_id
    assert recovered["plan_version"] == plan_version
    assert recovered["status"] == "running"
    assert broker.reconcile() == ()
    assert any(call[1:4] == ["ps", "-a", "--no-trunc"] for call in docker.calls)


@pytest.mark.parametrize(
    ("orphan_tenant", "orphan_image"),
    [("another-tenant", IMAGE_ID), ("local-test", "sha256:" + "c" * 64)],
)
def test_reconcile_ignores_foreign_tenant_and_unknown_image(tmp_path, orphan_tenant, orphan_image):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    worker.mkdir()
    task_id, plan_version = _approved_plan(workspace)
    docker = DiscoverableDocker()
    PlanRunner(
        workspace,
        backend=DockerBackend(
            workspace,
            worker,
            orphan_image,
            tenant_id=orphan_tenant,
            limits=PROFILES["local-tiny"].limits,
            command_runner=docker,
        ),
    ).start(task_id, plan_version, timeout_seconds=300)
    _broker(workspace, worker)
    broker = ExecutionBroker(
        workspace,
        worker,
        allowed_capabilities=("broker-demo-v1",),
        backend_factory=lambda descriptor, profile: DockerBackend(
            workspace,
            worker,
            IMAGE_ID,
            tenant_id="local-test",
            limits=profile.limits,
            command_runner=docker,
        ),
    )

    assert broker.reconcile() == ()


def test_cancel_is_idempotent_and_cleanup_runs_once(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    worker.mkdir()
    task_id, plan_version = _approved_plan(workspace)
    backend = LifecycleBackend()
    client = TestClient(create_broker_app(_broker(workspace, worker, backend)))
    started = client.post(
        "/v1/runs",
        json={
            "task_id": task_id,
            "plan_version": plan_version,
            "capability_id": "broker-demo-v1",
            "profile": "local-tiny",
        },
    ).json()

    first_cancel = client.post(f"/v1/runs/{started['run_id']}:cancel")
    second_cancel = client.post(f"/v1/runs/{started['run_id']}:cancel")
    first_cleanup = client.post(f"/v1/runs/{started['run_id']}:cleanup")
    second_cleanup = client.post(f"/v1/runs/{started['run_id']}:cleanup")

    assert first_cancel.json()["status"] == second_cancel.json()["status"] == "cancelled"
    assert first_cleanup.json()["cleaned"] is True
    assert second_cleanup.json()["cleaned"] is True
    assert backend.cleanup_count == 1


def test_local_cpu_profile_rejects_a_second_active_run(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    worker.mkdir()
    task_id, plan_version = _approved_plan(workspace)
    backend = LifecycleBackend()
    request = {
        "task_id": task_id,
        "plan_version": plan_version,
        "capability_id": "broker-demo-v1",
        "profile": "local-cpu",
    }
    first_client = TestClient(create_broker_app(_broker(workspace, worker, backend)))

    first = first_client.post("/v1/runs", json=request)
    restarted_client = TestClient(create_broker_app(_broker(workspace, worker, backend)))
    second = restarted_client.post("/v1/runs", json=request)

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "PROFILE_CAPACITY_EXCEEDED"


def test_server_refuses_to_bind_outside_loopback(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    worker.mkdir()
    broker = _broker(workspace, worker)

    with pytest.raises(PlanFlowError) as error:
        serve_broker(broker, host="0.0.0.0", port=8765)

    assert error.value.code == "BROKER_LOOPBACK_REQUIRED"


def test_plan_models_must_match_the_allowlisted_capability_contract(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    worker.mkdir()
    task_id, plan_version = _approved_plan(workspace)
    LocalCapabilityCatalog(worker).register(
        CapabilityDescriptor(
            capability_id="broker-model-v1",
            operator_name="demo_model_mapper",
            content_hash="sha256:" + "2" * 64,
            backend="docker",
            backend_ref={"image_id": IMAGE_ID},
            base_image_id="sha256:" + "b" * 64,
            created_at="2026-08-30T00:00:00+00:00",
            model_refs=(
                {"artifact_id": "required-model-v1", "sha256": "sha256:" + "3" * 64},
            ),
        )
    )
    broker = ExecutionBroker(
        workspace,
        worker,
        allowed_capabilities=("broker-model-v1",),
        backend_factory=lambda descriptor, profile: CompletedBackend(),
    )
    client = TestClient(create_broker_app(broker))

    response = client.post(
        "/v1/runs",
        json={
            "task_id": task_id,
            "plan_version": plan_version,
            "capability_id": "broker-model-v1",
            "profile": "local-tiny",
        },
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CAPABILITY_MODEL_MISMATCH"


def test_http_request_cannot_supply_docker_arguments(tmp_path):
    workspace, worker = tmp_path / "workspace", tmp_path / "worker"
    workspace.mkdir()
    worker.mkdir()
    task_id, plan_version = _approved_plan(workspace)
    client = TestClient(create_broker_app(_broker(workspace, worker)))

    response = client.post(
        "/v1/runs",
        json={
            "task_id": task_id,
            "plan_version": plan_version,
            "capability_id": "broker-demo-v1",
            "profile": "local-tiny",
            "docker_args": ["--privileged"],
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"

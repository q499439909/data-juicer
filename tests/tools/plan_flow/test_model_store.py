import hashlib
import os
import subprocess
import uuid
from pathlib import Path

import pytest
import yaml

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.model_store import LocalModelInstaller, LocalModelStore
from data_juicer.tools.plan_flow.validation import normalize_and_validate

IMAGE_ID = "sha256:c8815bf653a3e4fe7946ce1bf1c5501a37949b44401dfe06914c77a30db76490"


def _fixture(worker: Path, *, artifact_id="fixture-model-v1", content=b"tiny-model-weights\n") -> Path:
    source = worker / "fixtures" / "model-fixtures" / artifact_id
    source.mkdir(parents=True)
    weights = source / "weights.bin"
    weights.write_bytes(content)
    file_hash = "sha256:" + hashlib.sha256(content).hexdigest()
    aggregate = hashlib.sha256(b"weights.bin\0" + content + b"\0").hexdigest()
    manifest = {
        "artifact_id": artifact_id,
        "source": "local-fixture",
        "revision": "v1",
        "sha256": "sha256:" + aggregate,
        "size_bytes": len(content),
        "license": {"status": "approved-for-test"},
        "files": [{"path": "weights.bin", "sha256": file_hash}],
    }
    (source / "model-manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return source


def test_installer_stages_and_store_atomically_publishes_verified_artifact(tmp_path):
    worker = tmp_path / "worker"
    source = _fixture(worker)
    installer = LocalModelInstaller(worker, worker / "fixtures" / "model-fixtures")
    store = LocalModelStore(worker)

    staged = installer.stage_local("request-001", source)
    artifact = store.publish(staged.request_id)
    resolved = store.resolve("fixture-model-v1")

    assert staged.path == worker / "model-downloads" / "request-001"
    assert artifact == resolved
    assert resolved.manifest.sha256.startswith("sha256:")
    assert (resolved.path / "weights.bin").read_bytes() == b"tiny-model-weights\n"
    assert (resolved.path / "_publication.json").is_file()
    assert not list((worker / "models").glob(".publishing-*"))


def test_hash_mismatch_never_exposes_partial_published_artifact(tmp_path):
    worker = tmp_path / "worker"
    source = _fixture(worker)
    installer = LocalModelInstaller(worker, worker / "fixtures" / "model-fixtures")
    store = LocalModelStore(worker)
    staged = installer.stage_local("request-bad", source)
    (staged.path / "weights.bin").write_bytes(b"tampered")

    with pytest.raises(PlanFlowError) as error:
        store.publish("request-bad")

    assert error.value.code == "MODEL_HASH_MISMATCH"
    assert (worker / "models" / "fixture-model-v1").exists() is False
    assert not list((worker / "models").glob(".publishing-*"))


def test_installer_cannot_stage_from_outside_controlled_fixture_root(tmp_path):
    worker = tmp_path / "worker"
    _fixture(worker)
    outside = tmp_path / "workspace" / "run-input"
    outside.mkdir(parents=True)
    (outside / "secret.bin").write_bytes(b"business-input")
    installer = LocalModelInstaller(worker, worker / "fixtures" / "model-fixtures")

    with pytest.raises(PlanFlowError) as error:
        installer.stage_local("request-escape", outside)

    assert error.value.code == "MODEL_SOURCE_NOT_ALLOWED"
    assert (worker / "model-downloads" / "request-escape").exists() is False


def test_published_artifact_is_reverified_on_every_resolve(tmp_path):
    worker = tmp_path / "worker"
    source = _fixture(worker)
    installer = LocalModelInstaller(worker, worker / "fixtures" / "model-fixtures")
    store = LocalModelStore(worker)
    artifact = store.publish(installer.stage_local("request-verify", source).request_id)
    (artifact.path / "weights.bin").write_bytes(b"changed-after-publication")

    with pytest.raises(PlanFlowError) as error:
        store.resolve("fixture-model-v1")

    assert error.value.code == "MODEL_HASH_MISMATCH"


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ("license", "MODEL_LICENSE_NOT_APPROVED"),
        ("unsafe-path", "INVALID_MODEL_MANIFEST"),
        ("extra-file", "MODEL_INTEGRITY_FAILED"),
    ],
)
def test_publish_rejects_unapproved_unsafe_or_unlisted_content(tmp_path, mutation, error_code):
    worker = tmp_path / "worker"
    source = _fixture(worker)
    manifest_path = source / "model-manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if mutation == "license":
        manifest["license"]["status"] = "unknown"
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    elif mutation == "unsafe-path":
        manifest["files"][0]["path"] = "../weights.bin"
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    else:
        (source / "unlisted.bin").write_bytes(b"not-in-manifest")
    installer = LocalModelInstaller(worker, worker / "fixtures" / "model-fixtures")
    installer.stage_local("request-invalid", source)

    with pytest.raises(PlanFlowError) as error:
        LocalModelStore(worker).publish("request-invalid")

    assert error.value.code == error_code
    assert (worker / "models" / "fixture-model-v1").exists() is False


def test_plan_validation_accepts_declared_model_uri_and_rejects_undeclared_one(tmp_path, monkeypatch):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "data_juicer.tools.plan_flow.validation.operator_schema",
        lambda name: {"parameters": {"model_path": {"required": True}}, "tags": []},
    )
    plan = {
        "user_intent": "Use a published model",
        "models": [{"artifact_id": "fixture-model-v1"}],
        "recipe": {
            "dataset_path": str(dataset),
            "process": [
                {"fixture_mapper": {"model_path": "model-store://fixture-model-v1/weights.bin"}}
            ],
        },
    }

    normalized, validation, _ = normalize_and_validate(str(tmp_path), plan)
    plan["recipe"]["process"][0]["fixture_mapper"]["model_path"] = "model-store://fixture-model-v1"
    _, root_validation, _ = normalize_and_validate(str(tmp_path), plan)
    plan["models"] = []
    _, invalid, _ = normalize_and_validate(str(tmp_path), plan)

    assert validation["ok"] is True
    assert root_validation["ok"] is True
    assert normalized["models"] == [{"artifact_id": "fixture-model-v1"}]
    assert {item["code"] for item in invalid["errors"]} == {"MODEL_NOT_DECLARED"}


@pytest.mark.skipif(os.environ.get("DJ_RUN_DOCKER_INTEGRATION") != "1", reason="explicit Docker integration opt-in")
def test_real_two_containers_read_same_model_but_cannot_modify_it():
    worker = Path("D:/dsh-worker")
    fixture = worker / "fixtures" / "model-fixtures" / "fixture-tiny-model-v1"
    installer = LocalModelInstaller(worker, fixture.parent)
    request_id = "request-g-" + uuid.uuid4().hex
    installer.stage_local(request_id, fixture)
    artifact = LocalModelStore(worker).publish(request_id)
    mount = f"type=bind,src={artifact.path},dst=/models/{artifact.manifest.artifact_id},readonly"
    model_path = f"/models/{artifact.manifest.artifact_id}/weights.bin"
    names = ["dj-model-g-" + uuid.uuid4().hex[:16] for _ in range(2)]
    container_ids = []
    try:
        for name in names:
            created = subprocess.run(
                [
                    "docker", "create", "--name", name, "--label", "dj.model-test=true", "--read-only",
                    "--network", "none", "--user", "10001:10001", "--mount", mount,
                    "--entrypoint", "python", IMAGE_ID, "-c",
                    f"from pathlib import Path; import time; assert Path('{model_path}').read_bytes(); time.sleep(2)",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            container_ids.append(created.stdout.strip())
        for container_id in container_ids:
            subprocess.run(["docker", "start", container_id], capture_output=True, text=True, check=True)
        running = [
            subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", container_id],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            for container_id in container_ids
        ]
        assert running == ["true", "true"]
        for container_id in container_ids:
            assert subprocess.run(["docker", "wait", container_id], capture_output=True, text=True, check=True).stdout.strip() == "0"
        write_attempt = subprocess.run(
            [
                "docker", "run", "--rm", "--read-only", "--network", "none", "--user", "10001:10001",
                "--mount", mount, "--entrypoint", "python", IMAGE_ID, "-c",
                (
                    f"from pathlib import Path; import sys; p=Path('{model_path}'); "
                    "\ntry: p.unlink()\nexcept OSError: sys.exit(0)\nelse: sys.exit(3)"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert write_attempt.returncode == 0, write_attempt.stderr
    finally:
        for container_id in container_ids:
            subprocess.run(["docker", "rm", "--force", container_id], capture_output=True, text=True, check=False)
    assert LocalModelStore(worker).verify(artifact.manifest.artifact_id) == artifact.manifest

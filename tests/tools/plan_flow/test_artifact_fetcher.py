from pathlib import Path

import pytest

from data_juicer.tools.plan_flow.artifact_fetcher import ArtifactFetcher, FetchRequest
from data_juicer.tools.plan_flow.common import PlanFlowError, sha256_file


def _request(source: str, revision: str, digest: str, *, size: int) -> FetchRequest:
    return FetchRequest(
        request_id="face-model-request",
        artifact_id="face-model-v1",
        kind="model",
        source=source,
        revision=revision,
        files=("model.onnx",),
        expected_sha256={"model.onnx": digest},
        max_bytes=size,
        license_status="approved-for-test",
    )


def test_fetch_local_artifact_into_quarantine_with_verified_inventory(tmp_path):
    source_root = tmp_path / "allowed"
    source = source_root / "fixture"
    source.mkdir(parents=True)
    payload = source / "model.onnx"
    payload.write_bytes(b"fixed-model")
    worker = tmp_path / "worker"
    worker.mkdir()
    request = _request(str(source), "local-fixture-v1", sha256_file(payload), size=64)

    staged = ArtifactFetcher(worker, allowed_local_roots=(source_root,)).fetch(request)

    assert staged.path == worker / "quarantine" / request.request_id
    assert (staged.path / "model.onnx").read_bytes() == b"fixed-model"
    assert staged.manifest["status"] == "verified"
    assert staged.manifest["total_bytes"] == len(b"fixed-model")


def test_fetch_rejects_floating_revision_before_downloader_is_called(tmp_path):
    calls = []
    worker = tmp_path / "worker"
    worker.mkdir()
    request = _request(
        "https://huggingface.co/department/face-model",
        "main",
        "sha256:" + "1" * 64,
        size=64,
    )

    with pytest.raises(PlanFlowError, match="immutable"):
        ArtifactFetcher(worker, downloader=lambda *_: calls.append(True)).fetch(request)

    assert calls == []


def test_fetch_rejects_unapproved_host_and_unsafe_model_format(tmp_path):
    worker = tmp_path / "worker"
    worker.mkdir()
    request = _request(
        "https://example.invalid/model",
        "a" * 40,
        "sha256:" + "1" * 64,
        size=64,
    )
    with pytest.raises(PlanFlowError, match="source"):
        ArtifactFetcher(worker).fetch(request)

    unsafe = FetchRequest(
        request_id="unsafe-model-request",
        artifact_id="unsafe-model-v1",
        kind="model",
        source=str(tmp_path),
        revision="local-v1",
        files=("weights.pkl",),
        expected_sha256={"weights.pkl": "sha256:" + "2" * 64},
        max_bytes=64,
        license_status="approved-for-test",
    )
    with pytest.raises(PlanFlowError, match="format"):
        ArtifactFetcher(worker, allowed_local_roots=(tmp_path,)).fetch(unsafe)


def test_fetch_removes_partial_quarantine_after_hash_failure(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"tampered")
    worker = tmp_path / "worker"
    worker.mkdir()
    request = _request(str(source), "local-v1", "sha256:" + "0" * 64, size=64)

    with pytest.raises(PlanFlowError, match="hash"):
        ArtifactFetcher(worker, allowed_local_roots=(tmp_path,)).fetch(request)

    assert not (worker / "quarantine" / request.request_id).exists()


def test_fetcher_exposes_only_structured_requests_not_shell_commands(tmp_path):
    worker = tmp_path / "worker"
    worker.mkdir()
    fields = FetchRequest.__dataclass_fields__

    assert "command" not in fields
    assert "args" not in fields
    assert not hasattr(ArtifactFetcher(worker), "run")


def test_fetch_rejects_windows_wheel_for_linux_runtime(tmp_path):
    request = FetchRequest(
        request_id="wrong-wheel-request",
        artifact_id="wrong-wheel-v1",
        kind="dependency",
        source=str(tmp_path),
        revision="4.10.0.84",
        files=("opencv_python-4.10.0-cp312-win_amd64.whl",),
        expected_sha256={"opencv_python-4.10.0-cp312-win_amd64.whl": "sha256:" + "1" * 64},
        max_bytes=100,
        license_status="approved-for-test",
    )

    with pytest.raises(PlanFlowError) as mismatch:
        ArtifactFetcher(tmp_path, allowed_local_roots=(tmp_path,)).fetch(request)
    assert mismatch.value.code == "WHEEL_PLATFORM_MISMATCH"

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.run_output_gateway import RunOutputGateway


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _run(tmp_path: Path, *, status: str = "succeeded", error: str | None = None):
    workspace = tmp_path / "workspace"
    task_id, plan_version, run_id = "task_result", "plan_v001", "run_r001"
    run_root = workspace / ".dj" / "tasks" / task_id / "runs" / run_id
    output = workspace / "outputs" / "result" / task_id / plan_version / run_id
    run_root.mkdir(parents=True)
    output.mkdir(parents=True)
    (workspace / ".dj" / "tasks" / task_id / "task.yaml").write_text(
        "schema_version: 1\ntask_id: task_result\ntitle: Result task\ntask_slug: result\n",
        encoding="utf-8",
    )
    state = {
        "task_id": task_id,
        "plan_version": plan_version,
        "run_id": run_id,
        "status": status,
        "created_at": "2026-08-31T00:00:00+00:00",
        "updated_at": "2026-08-31T00:01:00+00:00",
        "output_dir": str(output),
    }
    if error:
        state.update({"error_code": "EXECUTION_FAILED", "error": error})
    (run_root / "run.json").write_text(json.dumps(state), encoding="utf-8")
    return workspace, task_id, plan_version, run_id, output


def _manifest(output: Path, run_id: str, files: dict[str, bytes]):
    entries = []
    total = 0
    for relative, content in files.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        total += len(content)
        entries.append({"path": relative, "size_bytes": len(content), "sha256": _sha(target)})
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "succeeded",
        "started_at": "2026-08-31T00:00:00+00:00",
        "finished_at": "2026-08-31T00:01:00+00:00",
        "output_count": len(entries),
        "output_size_bytes": total,
        "outputs": entries,
    }
    (output / "result-manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_inspects_verified_outputs_and_applies_dataset_view_without_exposing_paths(tmp_path):
    workspace, task_id, version, run_id, output = _run(tmp_path)
    _manifest(output, run_id, {"images/a.png": b"png", "result.jsonl": b'{"score":0.9}\n'})
    (output / "dataset-view.json").write_text(
        json.dumps({
            "schema_version": 1,
            "title": "图片质量结果",
            "summary": {"record_count": 1, "labels": {"保留": 1}, "metrics": {"quality_score": {"mean": 0.9}}},
            "items": [{
                "item_id": "item_1",
                "sample_id": "sample_1",
                "variant": "overlay",
                "variant_label": "叠加预览",
                "asset_path": "images/a.png",
                "display_name": "a.png",
                "media_type": "image/png",
                "labels": ["保留"],
                "metrics": {"quality_score": 0.9},
            }],
            "documents": [{"asset_path": "result.jsonl", "kind": "jsonl"}],
        }),
        encoding="utf-8",
    )

    gateway = RunOutputGateway()
    result = gateway.inspect_run(workspace, task_id, version, run_id)

    assert result["eligible"] is True
    assert result["status"] == "available"
    assert result["title"] == "图片质量结果"
    assert result["labels"] == {"保留": 1}
    assert result["metrics"]["quality_score"]["mean"] == 0.9
    assert {item["name"] for item in result["assets"]} == {"a.png", "result.jsonl"}
    assert all("path" not in item and "output" not in item for item in result["assets"])
    image = next(item for item in result["assets"] if item["name"] == "a.png")
    assert image["sampleId"] == "sample_1"
    assert image["variant"] == "overlay"
    assert image["variantLabel"] == "叠加预览"
    opened = gateway.open_asset(workspace, task_id, version, run_id, image["assetId"])
    assert opened.path.read_bytes() == b"png"
    assert opened.media_type == "image/png"


@pytest.mark.parametrize(
    ("status", "error", "expected_reason"),
    [
        ("failed", "boom", "run_failed"),
        ("succeeded", "boom", "run_failed"),
        ("succeeded", None, "no_verified_output"),
    ],
)
def test_excludes_failed_errored_and_output_free_runs(tmp_path, status, error, expected_reason):
    workspace, task_id, version, run_id, output = _run(tmp_path, status=status, error=error)
    if expected_reason != "no_verified_output":
        _manifest(output, run_id, {"result.jsonl": b"{}\n"})

    result = RunOutputGateway().inspect_run(workspace, task_id, version, run_id)

    assert result == {"eligible": False, "reason": expected_reason}


def test_marks_cancelled_run_with_verified_outputs_as_partial(tmp_path):
    workspace, task_id, version, run_id, output = _run(tmp_path, status="cancelled")
    _manifest(output, run_id, {"partial.jsonl": b"{}\n"})

    result = RunOutputGateway().inspect_run(workspace, task_id, version, run_id)

    assert result["eligible"] is True
    assert result["status"] == "partial"


def test_rejects_manifest_path_traversal_before_opening_an_asset(tmp_path):
    workspace, task_id, version, run_id, output = _run(tmp_path)
    outside = workspace / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "succeeded",
        "output_count": 1,
        "output_size_bytes": outside.stat().st_size,
        "outputs": [{"path": "../../secret.txt", "size_bytes": outside.stat().st_size, "sha256": _sha(outside)}],
    }
    (output / "result-manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PlanFlowError) as raised:
        RunOutputGateway().inspect_run(workspace, task_id, version, run_id)

    assert raised.value.code == "INVALID_RESULT_MANIFEST"


def test_deletes_only_the_exact_output_when_manifest_hash_still_matches(tmp_path):
    workspace, task_id, version, run_id, output = _run(tmp_path)
    _manifest(output, run_id, {"result.jsonl": b"{}\n"})
    gateway = RunOutputGateway()
    inspected = gateway.inspect_run(workspace, task_id, version, run_id)

    with pytest.raises(PlanFlowError) as raised:
        gateway.delete_outputs(workspace, task_id, version, run_id, "sha256:stale")
    assert raised.value.code == "RESULT_CHANGED"
    assert output.is_dir()

    deleted = gateway.delete_outputs(workspace, task_id, version, run_id, inspected["manifestHash"])

    assert deleted == {"deleted": True, "resultRef": run_id}
    assert not output.exists()
    assert (workspace / ".dj" / "tasks" / task_id / "runs" / run_id / "run.json").is_file()


def test_creates_reusable_safe_zip_for_all_or_selected_assets(tmp_path):
    workspace, task_id, version, run_id, output = _run(tmp_path)
    _manifest(output, run_id, {"images/a.png": b"a", "images/b.png": b"b"})
    gateway = RunOutputGateway()
    inspected = gateway.inspect_run(workspace, task_id, version, run_id)
    selected_id = next(item["assetId"] for item in inspected["assets"] if item["name"] == "a.png")

    selected = gateway.create_archive(workspace, task_id, version, run_id, [selected_id])
    repeated = gateway.create_archive(workspace, task_id, version, run_id, [selected_id])
    complete = gateway.create_archive(workspace, task_id, version, run_id, None)

    assert repeated.path == selected.path
    with zipfile.ZipFile(selected.path) as archive:
        assert archive.namelist() == ["images/a.png"]
    with zipfile.ZipFile(complete.path) as archive:
        assert archive.namelist() == ["images/a.png", "images/b.png"]
    assert selected.media_type == "application/zip"

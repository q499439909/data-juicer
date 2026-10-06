import json

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.output_packaging import package_datasets
from data_juicer.tools.plan_flow.validation import normalize_and_validate


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_package_datasets_copies_media_and_rewrites_manifest_to_relative_paths(tmp_path):
    workspace = tmp_path
    source = workspace / "data"
    source.mkdir()
    image = source / "cat.jpg"
    image.write_bytes(b"cat")
    output = workspace / "outputs" / "run"
    output.mkdir(parents=True)
    manifest = output / "cats.jsonl"
    manifest.write_text(json.dumps({"images": [str(image)], "label": "cat"}) + "\n", encoding="utf-8")

    result = package_datasets(output, workspace, [{"path": "cats.jsonl", "media_dir": "cats"}])

    assert (output / "cats" / "cat.jpg").read_bytes() == b"cat"
    assert _read_jsonl(manifest) == [{"images": ["cats/cat.jpg"], "label": "cat"}]
    assert result["packages"][0]["media_files"] == 1


def test_package_datasets_handles_name_collisions_by_content_hash(tmp_path):
    workspace = tmp_path
    first = workspace / "a"
    second = workspace / "b"
    first.mkdir()
    second.mkdir()
    (first / "same.jpg").write_bytes(b"first")
    (second / "same.jpg").write_bytes(b"second")
    output = workspace / "outputs" / "run"
    output.mkdir(parents=True)
    manifest = output / "selected.jsonl"
    manifest.write_text(
        "\n".join(json.dumps({"images": [str(path / "same.jpg")]}) for path in (first, second)) + "\n",
        encoding="utf-8",
    )

    package_datasets(output, workspace, [{"path": "selected.jsonl", "media_dir": "media"}])

    records = _read_jsonl(manifest)
    assert records[0]["images"] == ["media/same.jpg"]
    assert records[1]["images"][0].startswith("media/same-")
    assert len(list((output / "media").iterdir())) == 2


def test_package_datasets_rejects_media_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    output = workspace / "outputs" / "run"
    output.mkdir(parents=True)
    manifest = output / "selected.jsonl"
    manifest.write_text(json.dumps({"images": [str(outside)]}) + "\n", encoding="utf-8")

    with pytest.raises(PlanFlowError) as denied:
        package_datasets(output, workspace, [{"path": "selected.jsonl", "media_dir": "media"}])
    assert denied.value.code == "PACKAGE_MEDIA_MISSING"


def test_plan_validation_accepts_safe_dataset_package_and_rejects_escape(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    plan = {
        "user_intent": "Package the selected media",
        "recipe": {
            "dataset_path": str(dataset),
            "export_path": "selected.jsonl",
            "process": [{"text_length_filter": {"min_len": 1}}],
        },
        "postprocess": [
            {"kind": "dataset_package", "manifests": [{"path": "selected.jsonl", "media_dir": "media"}]}
        ],
    }

    _, valid, _ = normalize_and_validate(str(tmp_path), plan)
    assert valid["ok"] is True

    plan["postprocess"][0]["manifests"][0]["media_dir"] = "../outside"
    _, invalid, _ = normalize_and_validate(str(tmp_path), plan)
    assert invalid["ok"] is False
    assert any(error["code"] == "INVALID_DATASET_PACKAGE" for error in invalid["errors"])

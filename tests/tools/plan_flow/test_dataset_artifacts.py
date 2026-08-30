import json
from pathlib import Path

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.dataset_artifacts import DatasetSnapshotter, OutputArtifactCollector


def test_image_directory_snapshot_copies_each_file_once_and_uses_container_paths(tmp_path):
    source = tmp_path / "face" / "images"
    source.mkdir(parents=True)
    (source / "a.jpg").write_bytes(b"a")
    (source / "b.jpg").write_bytes(b"b")
    staging = tmp_path / "run" / "input"

    snapshot = DatasetSnapshotter(allowed_input_roots=(tmp_path / "face",)).create(source, staging)

    records = [json.loads(line) for line in snapshot.dataset_path.read_text(encoding="utf-8").splitlines()]
    copied = list((staging / "media").iterdir())
    assert len(records) == len(copied) == 2
    assert all(record["images"][0].startswith("/workspace/input/media/") for record in records)
    assert "C:\\" not in snapshot.dataset_path.read_text(encoding="utf-8")
    assert snapshot.manifest["copied_file_count"] == 2
    assert snapshot.manifest["snapshot_hash"].startswith("sha256:")


def test_jsonl_snapshot_deduplicates_repeated_media_references(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    image = source / "same.jpg"
    image.write_bytes(b"same")
    dataset = source / "data.jsonl"
    dataset.write_text(
        "\n".join((
            json.dumps({"id": 1, "images": [str(image)]}),
            json.dumps({"id": 2, "images": [str(image)]}),
        )) + "\n",
        encoding="utf-8",
    )

    snapshot = DatasetSnapshotter(allowed_input_roots=(source,)).create(dataset, tmp_path / "stage")

    records = [json.loads(line) for line in snapshot.dataset_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["images"] == records[1]["images"]
    assert snapshot.manifest["copied_file_count"] == 1


def test_snapshot_rejects_input_outside_explicit_roots(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.jpg").write_bytes(b"a")
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    with pytest.raises(PlanFlowError) as denied:
        DatasetSnapshotter(allowed_input_roots=(allowed,)).create(outside, tmp_path / "stage")
    assert denied.value.code == "INPUT_PATH_NOT_ALLOWED"


def test_output_collector_recursively_hashes_directory_artifacts(tmp_path):
    output = tmp_path / "output"
    masks = output / "masks"
    masks.mkdir(parents=True)
    (masks / "a.png").write_bytes(b"mask-a")
    (output / "result.jsonl").write_text("{}\n", encoding="utf-8")

    result = OutputArtifactCollector(max_total_bytes=1024).collect(output)

    assert [item["path"] for item in result["files"]] == ["masks/a.png", "result.jsonl"]
    assert result["total_bytes"] == (masks / "a.png").stat().st_size + (output / "result.jsonl").stat().st_size


def test_output_collector_rejects_symlinks_and_size_limit(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    target = output / "large.bin"
    target.write_bytes(b"12345")
    with pytest.raises(PlanFlowError) as too_large:
        OutputArtifactCollector(max_total_bytes=4).collect(output)
    assert too_large.value.code == "OUTPUT_TOO_LARGE"

    target.unlink()
    external = tmp_path / "external.txt"
    external.write_text("secret", encoding="utf-8")
    try:
        (output / "link.txt").symlink_to(external)
    except OSError:
        pytest.skip("Creating symlinks requires Windows developer mode")
    with pytest.raises(PlanFlowError) as linked:
        OutputArtifactCollector(max_total_bytes=1024).collect(output)
    assert linked.value.code == "OUTPUT_SYMLINK_REJECTED"

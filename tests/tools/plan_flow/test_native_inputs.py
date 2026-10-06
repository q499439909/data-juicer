import json

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError, sha256_file
from data_juicer.tools.plan_flow.native_inputs import freeze_inputs, verify_inputs


def _plan(dataset):
    return {"recipe": {"dataset_path": str(dataset), "image_key": "images"}}


def test_native_input_freeze_references_source_media_without_copying(tmp_path):
    workspace = tmp_path
    source = workspace / "data"
    source.mkdir()
    image = source / "cat.jpg"
    image.write_bytes(b"cat")
    dataset = source / "input.jsonl"
    dataset.write_text(json.dumps({"text": "<__dj__image>", "images": [str(image)]}) + "\n")
    version = workspace / ".dj" / "tasks" / "task_1" / "plans" / "plan_v001"
    version.mkdir(parents=True)
    plan = _plan(dataset)

    freeze_inputs(plan, version, workspace)

    frozen_dataset = version / "input" / "dataset-0.jsonl"
    frozen_record = json.loads(frozen_dataset.read_text(encoding="utf-8"))
    assert frozen_record["images"] == [str(image.resolve())]
    assert not (version / "input" / "media").exists()
    inventory = [
        json.loads(line)
        for line in (version / "input" / "media-inventory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert inventory == [{"path": str(image.resolve()), "size_bytes": 3, "sha256": sha256_file(image)}]
    assert plan["input_snapshot"]["schema_version"] == 2
    assert plan["input_snapshot"]["source_media"]["count"] == 1
    verify_inputs(plan, version, workspace)


def test_native_input_verification_rejects_changed_or_missing_source_media(tmp_path):
    workspace = tmp_path
    source = workspace / "data"
    source.mkdir()
    image = source / "cat.jpg"
    image.write_bytes(b"cat")
    dataset = source / "input.jsonl"
    dataset.write_text(json.dumps({"images": [str(image)]}) + "\n")
    version = workspace / ".dj" / "tasks" / "task_1" / "plans" / "plan_v001"
    version.mkdir(parents=True)
    plan = _plan(dataset)
    freeze_inputs(plan, version, workspace)

    image.write_bytes(b"dog")
    with pytest.raises(PlanFlowError, match="Referenced input changed") as changed:
        verify_inputs(plan, version, workspace)
    assert changed.value.code == "INPUT_SNAPSHOT_CHANGED"

    image.unlink()
    with pytest.raises(PlanFlowError, match="Referenced input changed") as missing:
        verify_inputs(plan, version, workspace)
    assert missing.value.code == "INPUT_SNAPSHOT_CHANGED"

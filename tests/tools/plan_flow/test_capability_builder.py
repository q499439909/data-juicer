import subprocess
from pathlib import Path

import pytest

from data_juicer.tools.plan_flow.capability import (
    CapabilityBuilder,
    CapabilityModelRef,
    CapabilitySpec,
    LocalCapabilityCatalog,
)
from data_juicer.tools.plan_flow.common import PlanFlowError

BASE_IMAGE_ID = "sha256:c8815bf653a3e4fe7946ce1bf1c5501a37949b44401dfe06914c77a30db76490"


def _fixture(worker: Path) -> tuple[Path, Path]:
    source = worker / "fixtures" / "capabilities" / "demo" / "source"
    source.mkdir(parents=True)
    (source / "operator.py").write_text("VALUE = 1\n", encoding="utf-8")
    wheelhouse = worker / "fixtures" / "capabilities" / "demo" / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "dsh_demo_token-1.0.0-py3-none-any.whl").write_bytes(b"fixture-wheel")
    return source, wheelhouse


def test_publish_requires_approval_for_the_exact_proposal_content(tmp_path):
    worker = tmp_path / "worker"
    source, wheelhouse = _fixture(worker)
    builder = CapabilityBuilder(worker, base_image_ref="base:test", base_image_id=BASE_IMAGE_ID)
    proposal = builder.prepare(
        CapabilitySpec(
            capability_id="demo-text-signature-v1",
            operator_name="demo_text_signature_mapper",
            import_module="operator",
            source_dir=source,
            wheelhouse_dir=wheelhouse,
        )
    )

    assert proposal.status == "pending-approval"
    assert proposal.content_hash.startswith("sha256:")
    with pytest.raises(PlanFlowError) as missing:
        builder.publish(proposal.proposal_id)
    assert missing.value.code == "CAPABILITY_APPROVAL_REQUIRED"

    with pytest.raises(PlanFlowError) as changed:
        builder.approve(proposal.proposal_id, "sha256:" + "0" * 64, note="reviewed")
    assert changed.value.code == "CAPABILITY_CONTENT_CHANGED"


def test_published_capability_is_registered_and_reused_without_rebuilding(tmp_path):
    worker = tmp_path / "worker"
    source, wheelhouse = _fixture(worker)
    calls = []
    derived_image_id = "sha256:" + "d" * 64

    def docker_boundary(argv, **kwargs):
        calls.append(argv)
        stdout = ""
        if argv[1:3] == ["image", "inspect"]:
            stdout = (BASE_IMAGE_ID if argv[-1] == "base:test" else derived_image_id) + "\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    builder = CapabilityBuilder(
        worker,
        base_image_ref="base:test",
        base_image_id=BASE_IMAGE_ID,
        command_runner=docker_boundary,
    )
    proposal = builder.prepare(
        CapabilitySpec(
            capability_id="demo-text-signature-v1",
            operator_name="demo_text_signature_mapper",
            import_module="operator",
            source_dir=source,
            wheelhouse_dir=wheelhouse,
        )
    )
    builder.approve(proposal.proposal_id, proposal.content_hash, note="reviewed fixture capability")

    first = builder.publish(proposal.proposal_id)
    second = builder.publish(proposal.proposal_id)
    resolved = LocalCapabilityCatalog(worker).resolve("demo-text-signature-v1")

    assert first == second == resolved
    assert resolved.backend == "docker"
    assert resolved.backend_ref == {"image_id": derived_image_id}
    assert resolved.content_hash == proposal.content_hash
    assert resolved.source_hash.startswith("sha256:")
    assert resolved.dependency_lock_hash.startswith("sha256:")
    assert resolved.model_refs == ()
    assert sum(call[1] == "build" for call in calls) == 1


def test_model_contract_is_part_of_the_approved_capability_hash(tmp_path):
    worker = tmp_path / "worker"
    source, wheelhouse = _fixture(worker)
    builder = CapabilityBuilder(worker, base_image_ref="base:test", base_image_id=BASE_IMAGE_ID)
    common = {
        "capability_id": "demo-model-capability-v1",
        "operator_name": "demo_model_mapper",
        "import_module": "operator",
        "source_dir": source,
        "wheelhouse_dir": wheelhouse,
    }

    first = builder.prepare(
        CapabilitySpec(
            **common,
            model_refs=(CapabilityModelRef("model-v1", "sha256:" + "1" * 64),),
        )
    )
    second = builder.prepare(
        CapabilitySpec(
            **common,
            model_refs=(CapabilityModelRef("model-v1", "sha256:" + "2" * 64),),
        )
    )

    assert first.content_hash != second.content_hash

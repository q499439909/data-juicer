import pytest

from data_juicer.tools.plan_flow.capability_lifecycle import CapabilityLifecycle
from data_juicer.tools.plan_flow.capability_schema import (
    CapabilityCatalog,
    CapabilityDescriptor,
    OperatorArtifact,
    OperatorArtifactCatalog,
    OperatorDefinition,
)
from data_juicer.tools.plan_flow.common import PlanFlowError


def _artifact(artifact_id: str, operator_name: str, marker: str) -> OperatorArtifact:
    return OperatorArtifact.create(
        artifact_id=artifact_id,
        source_hash="sha256:" + marker * 64,
        dependency_lock_hash="sha256:" + "d" * 64,
        operators=(
            OperatorDefinition(
                name=operator_name,
                import_module=f"department_ops.{operator_name}",
                kind="mapper",
                schema_hash="sha256:" + "e" * 64,
            ),
        ),
        model_refs=(),
    )


def _capability(artifacts):
    return CapabilityDescriptor.create(
        capability_id="face-region-analysis-v1",
        implements=("face_mask@1", "masked_region_statistics@1"),
        operator_artifact_ids=tuple(item.artifact_id for item in artifacts),
        model_refs=(),
        run_network="none",
        resource_profiles=("local-cpu",),
        approval_scope="local-test",
    )


def test_new_capability_is_validated_before_one_aggregated_approval(tmp_path):
    artifacts = (
        _artifact("op-face-mask-v1", "face_mask_mapper", "1"),
        _artifact("op-region-stats-v1", "masked_region_statistics_mapper", "2"),
    )
    capability = _capability(artifacts)
    reports = []
    lifecycle = CapabilityLifecycle(tmp_path, validator=lambda proposal: reports.append(proposal.proposal_id) or {"passed": True})

    prepared = lifecycle.prepare(capability, artifacts)
    assert prepared.status == "staging"
    with pytest.raises(PlanFlowError) as early:
        lifecycle.approve(prepared.proposal_id, prepared.content_hash, note="too early")
    assert early.value.code == "CAPABILITY_NOT_VALIDATED"

    validated = lifecycle.build_and_validate(prepared.proposal_id)
    assert validated.status == "pending_approval"
    assert reports == [prepared.proposal_id]
    lifecycle.approve(prepared.proposal_id, prepared.content_hash, note="approve both operators")
    published = lifecycle.publish(prepared.proposal_id)

    assert published.status == "available"
    assert CapabilityCatalog(tmp_path).resolve(capability.capability_id) == capability
    assert [OperatorArtifactCatalog(tmp_path).resolve(item.artifact_id) for item in artifacts] == list(artifacts)


def test_approval_is_bound_to_exact_immutable_content(tmp_path):
    artifacts = (_artifact("op-face-mask-v1", "face_mask_mapper", "1"),)
    capability = CapabilityDescriptor.create(
        capability_id="face-mask-v1",
        implements=("face_mask@1",),
        operator_artifact_ids=(artifacts[0].artifact_id,),
        model_refs=(),
        run_network="none",
        resource_profiles=("local-cpu",),
        approval_scope="local-test",
    )
    lifecycle = CapabilityLifecycle(tmp_path, validator=lambda _: {"passed": True})
    proposal = lifecycle.prepare(capability, artifacts)
    lifecycle.build_and_validate(proposal.proposal_id)

    with pytest.raises(PlanFlowError) as changed:
        lifecycle.approve(proposal.proposal_id, "sha256:" + "0" * 64, note="wrong hash")
    assert changed.value.code == "CAPABILITY_CONTENT_CHANGED"


def test_available_capability_is_reused_without_validation_or_approval(tmp_path):
    artifact = _artifact("op-face-mask-v1", "face_mask_mapper", "1")
    capability = CapabilityDescriptor.create(
        capability_id="face-mask-v1",
        implements=("face_mask@1",),
        operator_artifact_ids=(artifact.artifact_id,),
        model_refs=(),
        run_network="none",
        resource_profiles=("local-cpu",),
        approval_scope="local-test",
    )
    first = CapabilityLifecycle(tmp_path, validator=lambda _: {"passed": True})
    proposal = first.prepare(capability, (artifact,))
    first.build_and_validate(proposal.proposal_id)
    first.approve(proposal.proposal_id, proposal.content_hash, note="approved")
    first.publish(proposal.proposal_id)

    reused = CapabilityLifecycle(
        tmp_path,
        validator=lambda _: (_ for _ in ()).throw(AssertionError("must not validate reused capability")),
    ).prepare(capability, (artifact,))

    assert reused.status == "available"
    assert reused.reused is True

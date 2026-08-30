import pytest

from data_juicer.tools.plan_flow.capability_schema import (
    CapabilityCatalog,
    CapabilityDescriptor,
    OperatorArtifact,
    OperatorArtifactCatalog,
    OperatorDefinition,
)
from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.runtime_resolver import DependencyPin, RuntimeResolver


def _publish(worker, artifact_id, operator_name, contract, marker):
    artifact = OperatorArtifact.create(
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
    capability = CapabilityDescriptor.create(
        capability_id=f"{artifact_id}-capability",
        implements=(contract,),
        operator_artifact_ids=(artifact_id,),
        model_refs=(),
        run_network="none",
        resource_profiles=("local-cpu",),
        approval_scope="local-test",
    )
    OperatorArtifactCatalog(worker).publish(artifact)
    CapabilityCatalog(worker).publish(capability)
    return artifact, capability


def test_resolver_composes_two_operator_artifacts_into_one_cached_runtime(tmp_path):
    face, face_cap = _publish(tmp_path, "op-face-mask-v1", "face_mask_mapper", "face_mask@1", "1")
    stats, stats_cap = _publish(
        tmp_path,
        "op-region-stats-v1",
        "masked_region_statistics_mapper",
        "masked_region_statistics@1",
        "2",
    )
    builds = []

    def build(spec):
        builds.append(spec)
        return "sha256:" + "f" * 64

    resolver = RuntimeResolver(
        tmp_path,
        base_image_id="sha256:" + "a" * 64,
        data_juicer_identity="git:plan-flow-mcp@0123456",
        image_builder=build,
        dependency_pins={
            face.artifact_id: (DependencyPin("opencv-python-headless", "4.10.0.84", "sha256:" + "3" * 64),),
            stats.artifact_id: (),
        },
    )

    first = resolver.resolve(
        capability_ids=(face_cap.capability_id, stats_cap.capability_id),
        operator_names=("face_mask_mapper", "masked_region_statistics_mapper"),
        profile_family="cpu",
    )
    second = resolver.resolve(
        capability_ids=(face_cap.capability_id, stats_cap.capability_id),
        operator_names=("face_mask_mapper", "masked_region_statistics_mapper"),
        profile_family="cpu",
    )

    assert first == second
    assert len(builds) == 1
    assert builds[0].operator_artifact_ids == (face.artifact_id, stats.artifact_id)
    assert [item.artifact_id for item in first.operator_artifacts] == [face.artifact_id, stats.artifact_id]


def test_resolver_rejects_dependency_version_conflicts_before_build(tmp_path):
    first, first_cap = _publish(tmp_path, "op-first-v1", "first_mapper", "first@1", "1")
    second, second_cap = _publish(tmp_path, "op-second-v1", "second_mapper", "second@1", "2")
    resolver = RuntimeResolver(
        tmp_path,
        base_image_id="sha256:" + "a" * 64,
        data_juicer_identity="git:test@1",
        image_builder=lambda _: (_ for _ in ()).throw(AssertionError("must not build")),
        dependency_pins={
            first.artifact_id: (DependencyPin("numpy", "1.26.4", "sha256:" + "3" * 64),),
            second.artifact_id: (DependencyPin("numpy", "2.0.0", "sha256:" + "4" * 64),),
        },
    )

    with pytest.raises(PlanFlowError) as conflict:
        resolver.resolve(
            capability_ids=(first_cap.capability_id, second_cap.capability_id),
            operator_names=("first_mapper", "second_mapper"),
            profile_family="cpu",
        )
    assert conflict.value.code == "RUNTIME_DEPENDENCY_CONFLICT"

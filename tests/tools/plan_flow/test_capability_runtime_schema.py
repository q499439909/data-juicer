import json

import pytest

from data_juicer.tools.plan_flow.capability_schema import (
    ArtifactRef,
    CapabilityCatalog,
    CapabilityDescriptor,
    OperatorArtifactCatalog,
    OperatorArtifact,
    OperatorDefinition,
)
from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.runtime_manifest import RuntimeCatalog, RuntimeManifest


def test_capability_describes_semantics_and_artifacts_without_an_execution_image():
    capability = CapabilityDescriptor.create(
        capability_id="masked-region-stats-v1",
        implements=("masked_region_statistics@1",),
        operator_artifact_ids=("op-masked-region-stats-v1",),
        model_refs=(),
        run_network="none",
        resource_profiles=("local-cpu",),
        approval_scope="local-test",
    )

    payload = capability.to_dict()

    assert payload["schema_version"] == 2
    assert payload["implements"] == ["masked_region_statistics@1"]
    assert "image_id" not in json.dumps(payload)
    assert "backend_ref" not in payload
    assert CapabilityDescriptor.from_dict(payload) == capability


def test_operator_artifact_can_publish_multiple_registered_operators():
    artifact = OperatorArtifact.create(
        artifact_id="op-region-toolkit-v1",
        source_hash="sha256:" + "1" * 64,
        dependency_lock_hash="sha256:" + "2" * 64,
        operators=(
            OperatorDefinition(
                name="masked_region_statistics_mapper",
                import_module="department_ops.region_stats",
                kind="mapper",
                schema_hash="sha256:" + "3" * 64,
            ),
            OperatorDefinition(
                name="embedding_cluster_deduplicator",
                import_module="department_ops.embedding_cluster",
                kind="deduplicator",
                schema_hash="sha256:" + "4" * 64,
            ),
        ),
        model_refs=(),
    )

    assert [item.name for item in artifact.operators] == [
        "masked_region_statistics_mapper",
        "embedding_cluster_deduplicator",
    ]
    assert OperatorArtifact.from_dict(artifact.to_dict()) == artifact


def test_runtime_manifest_owns_the_composed_image_identity():
    runtime = RuntimeManifest.create(
        base_image_id="sha256:" + "a" * 64,
        data_juicer_identity="git:plan-flow-mcp@0123456",
        operator_artifacts=(
            ArtifactRef("op-face-mask-v1", "sha256:" + "b" * 64),
            ArtifactRef("op-region-toolkit-v1", "sha256:" + "c" * 64),
        ),
        dependency_lock_hash="sha256:" + "d" * 64,
        model_refs=(ArtifactRef("face-parser-v1", "sha256:" + "e" * 64),),
        image_id="sha256:" + "f" * 64,
        bootstrap_version="1",
        profile_family="cpu",
    )

    assert runtime.runtime_id.startswith("runtime-")
    assert [item.artifact_id for item in runtime.operator_artifacts] == [
        "op-face-mask-v1",
        "op-region-toolkit-v1",
    ]
    assert RuntimeManifest.from_dict(runtime.to_dict()) == runtime


def test_runtime_manifest_rejects_duplicate_artifact_id_with_different_content():
    with pytest.raises(PlanFlowError, match="conflicting content"):
        RuntimeManifest.create(
            base_image_id="sha256:" + "a" * 64,
            data_juicer_identity="git:plan-flow-mcp@0123456",
            operator_artifacts=(
                ArtifactRef("op-face-mask-v1", "sha256:" + "b" * 64),
                ArtifactRef("op-face-mask-v1", "sha256:" + "c" * 64),
            ),
            dependency_lock_hash="sha256:" + "d" * 64,
            model_refs=(),
            image_id="sha256:" + "f" * 64,
            bootstrap_version="1",
            profile_family="cpu",
        )


def test_artifact_capability_and_runtime_catalogs_publish_independent_identities(tmp_path):
    artifact = OperatorArtifact.create(
        artifact_id="op-region-toolkit-v1",
        source_hash="sha256:" + "1" * 64,
        dependency_lock_hash="sha256:" + "2" * 64,
        operators=(
            OperatorDefinition(
                name="masked_region_statistics_mapper",
                import_module="department_ops.region_stats",
                kind="mapper",
                schema_hash="sha256:" + "3" * 64,
            ),
        ),
        model_refs=(),
    )
    capability = CapabilityDescriptor.create(
        capability_id="masked-region-stats-v1",
        implements=("masked_region_statistics@1",),
        operator_artifact_ids=(artifact.artifact_id,),
        model_refs=(),
        run_network="none",
        resource_profiles=("local-cpu",),
        approval_scope="local-test",
    )
    runtime = RuntimeManifest.create(
        base_image_id="sha256:" + "a" * 64,
        data_juicer_identity="git:plan-flow-mcp@0123456",
        operator_artifacts=(ArtifactRef(artifact.artifact_id, artifact.content_hash),),
        dependency_lock_hash=artifact.dependency_lock_hash,
        model_refs=(),
        image_id="sha256:" + "f" * 64,
        bootstrap_version="1",
        profile_family="cpu",
    )

    OperatorArtifactCatalog(tmp_path).publish(artifact)
    CapabilityCatalog(tmp_path).publish(capability)
    RuntimeCatalog(tmp_path).publish(runtime)

    assert OperatorArtifactCatalog(tmp_path).resolve(artifact.artifact_id) == artifact
    assert CapabilityCatalog(tmp_path).resolve(capability.capability_id) == capability
    assert RuntimeCatalog(tmp_path).resolve(runtime.runtime_id) == runtime

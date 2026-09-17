import json
from pathlib import Path

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.model_backends.huggingface import HuggingFaceModelBackend
from data_juicer.tools.plan_flow.model_backends.http_file import HttpFileModelBackend
from data_juicer.tools.plan_flow.model_lock_resolver import ModelLockResolver
from data_juicer.tools.plan_flow.model_requirement_scanner import scan
from data_juicer.tools.plan_flow.model_bundle import import_bundle
from data_juicer.tools.plan_flow.runtime_environment_lock import freeze_runtime_lock, verify_runtime_lock
from data_juicer.tools.plan_flow.service import PlanFlowService
from data_juicer.tools.plan_flow import operator_catalog_service


def _catalog(tmp_path: Path):
    model_file = tmp_path / "seed.bin"
    model_file.write_bytes(b"locked")
    import hashlib

    digest = hashlib.sha256(b"locked").hexdigest()
    catalog = {
        "schema_version": 1,
        "models": [
            {
                "lock_id": "fixture-v1",
                "backend": "huggingface",
                "model_id": "org/model",
                "revision": "a" * 40,
                "files": [{"path": "model.bin", "size": 6, "sha256": digest}],
                "consumers": [
                    {"operator": "image_nsfw_filter", "parameter": "hf_nsfw_model", "defaults": ["org/model"]}
                ],
            }
        ],
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


def _plan():
    return {"recipe": {"process": [{"image_nsfw_filter": {"hf_nsfw_model": "org/model"}}]}}


def test_freeze_is_deterministic_and_contains_no_host_path(tmp_path):
    resolver = ModelLockResolver(catalog_path=_catalog(tmp_path))

    first = resolver.freeze(_plan())
    second = resolver.freeze(_plan())

    assert first == second
    assert first[0]["revision"] == "a" * 40
    assert first[0]["consumers"] == [
        {"step_index": 0, "operator": "image_nsfw_filter", "parameter": "hf_nsfw_model"}
    ]
    assert "source_path" not in first[0]
    assert "\\" not in json.dumps(first)


def test_unknown_override_is_not_allowed_to_follow_main(tmp_path):
    resolver = ModelLockResolver(catalog_path=_catalog(tmp_path))
    plan = _plan()
    plan["recipe"]["process"][0]["image_nsfw_filter"]["hf_nsfw_model"] = "org/other"

    with pytest.raises(PlanFlowError, match="No curated model lock") as caught:
        resolver.freeze(plan)

    assert caught.value.code == "MODEL_LOCK_MISSING"


def test_uncurated_huggingface_operator_is_blocked():
    resolver = ModelLockResolver()
    plan = {
        "recipe": {
            "process": [
                {
                    "llm_condition_filter": {
                        "is_hf_model": True,
                        "api_or_hf_model": "org/not-curated",
                    }
                }
            ]
        }
    }

    with pytest.raises(PlanFlowError) as caught:
        resolver.freeze(plan)

    assert caught.value.code == "MODEL_REQUIREMENT_UNDECLARED"


def test_explicitly_unvalidated_builtin_requirement_is_blocked():
    resolver = ModelLockResolver()

    with pytest.raises(PlanFlowError) as caught:
        resolver.freeze({"recipe": {"process": [{"sentence_split_mapper": {}}]}})

    assert caught.value.code == "MODEL_DOWNLOAD_BLOCKED"


def test_builtin_model_requirement_scan_has_no_unresolved_call_sites():
    result = scan()

    assert result["complete"] is True
    assert result["status_counts"]["unresolved"] == 0


def test_offline_preloaded_root_is_portable_and_materializes_local_path(tmp_path, monkeypatch):
    catalog_path = _catalog(tmp_path)
    root = tmp_path / "linux-preload"
    snapshot = root / "models--org--model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"locked")
    monkeypatch.setenv("DSH_PRELOADED_MODEL_ROOT", str(root))
    resolver = ModelLockResolver(catalog_path=catalog_path)
    bindings = resolver.freeze(_plan())

    paths = resolver.prepare(bindings, offline=True)
    recipe, provenance = resolver.materialize(_plan()["recipe"], bindings, paths)

    assert recipe["process"][0]["image_nsfw_filter"]["hf_nsfw_model"] == str(snapshot)
    assert provenance["models"][0]["resolved_path"] == str(snapshot)


def test_hash_mismatch_blocks_execution(tmp_path):
    catalog_path = _catalog(tmp_path)
    cache = tmp_path / "cache"
    snapshot = cache / "models--org--model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"broken")
    backend = HuggingFaceModelBackend(cache_root=cache)
    resolver = ModelLockResolver(catalog_path=catalog_path, backends={"huggingface": backend})

    with pytest.raises(PlanFlowError) as caught:
        resolver.prepare(resolver.freeze(_plan()), offline=True)

    assert caught.value.code in {"MODEL_FILE_MISSING", "MODEL_HASH_MISMATCH"}


def test_same_model_used_by_multiple_steps_has_one_binding(tmp_path):
    resolver = ModelLockResolver(catalog_path=_catalog(tmp_path))
    plan = _plan()
    plan["recipe"]["process"].append({"image_nsfw_filter": {"hf_nsfw_model": "org/model"}})

    bindings = resolver.freeze(plan)

    assert len(bindings) == 1
    assert [item["step_index"] for item in bindings[0]["consumers"]] == [0, 1]


def test_prepare_plan_freezes_builtin_model_without_downloading(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"images":["image.jpg"]}\n', encoding="utf-8")
    (tmp_path / "image.jpg").write_bytes(b"fixture media")
    plan = {
        "user_intent": "Filter unsafe images",
        "modality": "image",
        "recipe": {
            "dataset_path": str(dataset),
            "process": [{"image_nsfw_filter": {"max_score": 0.5}}],
        },
    }

    prepared = PlanFlowService.native().prepare_plan(str(tmp_path), plan)

    assert prepared["valid"] is True
    binding = prepared["plan"]["model_bindings"][0]
    assert binding["revision"] == "96cb0d0342c7afb80cab76ecc58b265fa44da256"
    assert binding["provider"] == "dj"
    assert "resolved_path" not in json.dumps(binding)
    assert prepared["plan"]["runtime_lock"]["lockfile"] == "uv.lock"
    assert len(prepared["plan"]["runtime_lock"]["sha256"]) == 64


def test_import_bundle_uses_standard_huggingface_snapshot_layout(tmp_path):
    catalog_path = _catalog(tmp_path)
    bundle = tmp_path / "bundle"
    source = bundle / "fixture-v1"
    source.mkdir(parents=True)
    (source / "model.bin").write_bytes(b"locked")
    binding = ModelLockResolver(catalog_path=catalog_path).freeze(_plan())[0]
    (bundle / "model-bundle.json").write_text(
        json.dumps({"schema_version": 1, "models": [{k: v for k, v in binding.items() if k != "consumers"}]}),
        encoding="utf-8",
    )
    destination = tmp_path / "preloaded"
    resolver = ModelLockResolver(catalog_path=catalog_path)

    imported = import_bundle(bundle, destination, resolver=resolver)

    expected = destination / "models--org--model" / "snapshots" / ("a" * 40)
    assert imported["models"] == [{"lock_id": "fixture-v1", "status": "imported"}]
    assert (expected / "model.bin").read_bytes() == b"locked"
    offline_resolver = ModelLockResolver(
        catalog_path=catalog_path,
        backends={"huggingface": HuggingFaceModelBackend(preloaded_root=destination)},
    )
    assert offline_resolver.prepare([binding], offline=True)[binding["binding_id"]] == expected


def test_builtin_catalog_detail_exposes_logical_lock_without_host_path():
    detail = operator_catalog_service.detail("image_nsfw_filter")["operator"]

    assert detail["model_locks"][0]["revision"] == "96cb0d0342c7afb80cab76ecc58b265fa44da256"
    assert "path" not in detail["model_locks"][0]


def test_opencv_package_resource_is_locked_and_injected():
    resolver = ModelLockResolver()
    plan = {"recipe": {"process": [{"image_face_count_filter": {}}]}}

    bindings = resolver.freeze(plan)
    paths = resolver.prepare(bindings, offline=True)
    recipe, provenance = resolver.materialize(plan["recipe"], bindings, paths)

    assert bindings[0]["backend"] == "python-distribution"
    assert bindings[0]["version"] == "4.11.0.86"
    assert recipe["process"][0]["image_face_count_filter"]["cv_classifier"].endswith(
        "haarcascade_frontalface_alt.xml"
    )
    assert provenance["models"][0]["backend"] == "python-distribution"


def test_http_file_backend_reuses_only_hash_verified_artifact(tmp_path):
    payload = b"fixed artifact"
    import hashlib

    binding = {
        "lock_id": "fixture-http-v1",
        "filename": "weights.bin",
        "url": "https://example.invalid/weights.bin",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    target = tmp_path / "http-file" / binding["lock_id"] / binding["filename"]
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    backend = HttpFileModelBackend(cache_root=tmp_path)

    assert backend.prepare(binding, offline=True) == target.resolve()
    target.write_bytes(b"wrong")
    with pytest.raises(PlanFlowError) as caught:
        backend.prepare(binding, offline=True)
    assert caught.value.code == "MODEL_NOT_PREPARED"


def test_python_runtime_lock_detects_lockfile_drift(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text('version = 1\n[[package]]\nname = "fixture-package"\nversion = "1.2.3"\n', encoding="utf-8")
    frozen = freeze_runtime_lock(lock)

    assert verify_runtime_lock(frozen, lock)["sha256"] == frozen["sha256"]
    lock.write_text(lock.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    with pytest.raises(PlanFlowError) as caught:
        verify_runtime_lock(frozen, lock)
    assert caught.value.code == "RUNTIME_LOCK_MISMATCH"

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.discovery import (
    capability_schemas,
    inspect_input,
    operator_catalog,
    operator_detail,
    search_capabilities,
)
from data_juicer.tools.plan_flow.execution import LocalProcessBackend, RunHandle
from data_juicer.tools.plan_flow.run_output_gateway import RunOutputGateway
from data_juicer.tools.plan_flow.service import PlanFlowService
from data_juicer.tools.plan_flow.validation import normalize_and_validate


def _plan(dataset: Path, min_len: int = 2):
    return {
        "user_intent": "Keep useful text records",
        "modality": "text",
        "risk_notes": [],
        "approval_required": True,
        "recipe": {
            "dataset_path": str(dataset),
            "export_path": "result.jsonl",
            "process": [{"text_length_filter": {"min_len": min_len}}],
            "executor_type": "default",
            "np": 1,
        },
        "postprocess": [],
    }


def test_relative_input_path_is_resolved_from_workspace(tmp_path):
    input_dir = tmp_path / "data" / "input"
    input_dir.mkdir(parents=True)

    result = inspect_input(str(tmp_path), {"path": "data/input"})

    assert result["ok"] is True
    assert result["workspace_root"] == str(tmp_path.resolve())
    assert result["dataset_path"] == str(input_dir.resolve())


def test_operator_catalog_projects_the_live_registry_without_internal_paths():
    result = operator_catalog()

    assert result["ok"] is True
    assert result["total"] == len(result["operators"])
    assert result["total"] > 0
    assert result["facets"]["categories"]
    assert result["facets"]["modalities"]
    assert result["facets"]["devices"] == ["cpu", "gpu"]
    assert [item["name"] for item in result["operators"]] == sorted(item["name"] for item in result["operators"])
    first = result["operators"][0]
    assert set(first) == {
        "name",
        "description",
        "display_name_zh",
        "description_zh",
        "translation_status",
        "category",
        "modalities",
        "devices",
    }
    assert first["name"]
    assert first["category"]
    assert first["modalities"]
    assert first["devices"]
    assert result["translation"] == {"locale": "zh-CN", "translated": result["total"], "pending": 0}
    assert all(operator["translation_status"] == "translated" for operator in result["operators"])
    assert all(operator["display_name_zh"] != operator["name"] for operator in result["operators"])
    assert all(operator["description_zh"] for operator in result["operators"])


def test_operator_catalog_uses_general_for_operators_without_a_modality_tag():
    result = operator_catalog()

    item = next(operator for operator in result["operators"] if operator["name"] == "general_field_filter")
    assert item["modalities"] == ["general"]
    assert item["devices"] == ["cpu"]


def test_operator_detail_includes_presentation_safe_parameter_metadata():
    result = operator_detail("text_length_filter")

    assert result["ok"] is True
    operator = result["operator"]
    assert operator["name"] == "text_length_filter"
    assert operator["category"] == "filter"
    assert operator["modalities"] == ["text"]
    assert operator["devices"] == ["cpu"]
    parameter = next(item for item in operator["parameters"] if item["name"] == "min_len")
    assert set(parameter) == {
        "name",
        "display_name_zh",
        "type",
        "required",
        "default",
        "description",
        "description_zh",
    }
    assert parameter["required"] is False
    assert operator["display_name_zh"] == "文本长度过滤器"
    assert operator["description_zh"]
    assert operator["summary_zh"]
    assert operator["translation_status"] == "translated"
    assert parameter["display_name_zh"] == "最小长度"
    assert parameter["description_zh"]
    json.dumps(result)


def test_operator_detail_reports_an_unknown_exact_name():
    result = operator_detail("not_a_registered_operator")

    assert result == {
        "ok": False,
        "error": "operator_not_found",
        "message": "Operator was not found.",
    }


def test_zh_cn_asset_covers_every_live_operator_and_parameter():
    from data_juicer.tools.plan_flow.localization import contains_han, load_zh_cn

    catalog = operator_catalog()
    localized = load_zh_cn()["operators"]

    assert set(localized) == {operator["name"] for operator in catalog["operators"]}
    for item in catalog["operators"]:
        entry = localized[item["name"]]
        assert contains_han(entry["display_name"])
        assert contains_han(entry["summary"])
        assert contains_han(entry["description"])
        detail = operator_detail(item["name"])["operator"]
        assert set(entry["parameters"]) == {parameter["name"] for parameter in detail["parameters"]}
        assert all(contains_han(value["display_name"]) for value in entry["parameters"].values())
        assert all(contains_han(value["description"]) for value in entry["parameters"].values())


def test_image_filter_localization_preserves_operator_specific_methods_and_rules():
    aesthetics = operator_detail("image_aesthetics_filter")["operator"]
    aspect_ratio = operator_detail("image_aspect_ratio_filter")["operator"]
    face_count = operator_detail("image_face_count_filter")["operator"]
    face_ratio = operator_detail("image_face_ratio_filter")["operator"]

    assert "Hugging Face" in aesthetics["summary_zh"]
    assert "美学得分" in aesthetics["description_zh"]
    assert "any" in aesthetics["description_zh"] and "all" in aesthetics["description_zh"]
    assert "宽度除以高度" in aspect_ratio["description_zh"]
    assert "OpenCV" in face_count["summary_zh"]
    assert "人脸数量" in face_count["description_zh"]
    assert "最大人脸面积" in face_ratio["description_zh"]
    assert len({
        aesthetics["summary_zh"],
        aspect_ratio["summary_zh"],
        face_count["summary_zh"],
        face_ratio["summary_zh"],
    }) == 4


def test_method_detection_prefers_registered_algorithm_over_incidental_platform_name():
    operator = operator_detail("document_minhash_deduplicator")["operator"]
    captioning = operator_detail("image_captioning_mapper")["operator"]
    alphanumeric = operator_detail("alphanumeric_filter")["operator"]

    assert "MinHash" in operator["summary_zh"]
    assert "Hugging Face 模型" not in operator["summary_zh"]
    assert "Hugging Face" in captioning["summary_zh"]
    assert "SimHash" not in captioning["summary_zh"]
    assert "Hugging Face" not in alphanumeric["summary_zh"]


def test_untranslated_future_operator_remains_visible_with_pending_status():
    from data_juicer.tools.plan_flow.localization import localize_catalog_item

    item = {
        "name": "future_custom_mapper",
        "description": "A future custom operator.",
        "category": "mapper",
        "modalities": ["general"],
        "devices": ["cpu"],
    }

    localized = localize_catalog_item(item)

    assert localized["display_name_zh"] == item["name"]
    assert localized["description_zh"] == item["description"]
    assert localized["translation_status"] == "pending"


def test_search_does_not_expose_runtime_configuration(monkeypatch):
    secret = "must-not-be-returned"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("DJ_VLM_MODEL", "qwen3.7-plus")

    result = search_capabilities(["filter text"], modality="text", top_k=1)

    serialized = json.dumps(result)
    assert "runtime" not in result
    assert secret not in serialized
    assert "qwen3.7-plus" not in serialized
    assert "example.invalid" not in serialized


def test_runtime_vlm_model_is_materialized_for_api_vlm_operator(tmp_path, monkeypatch):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"<__dj__image>","images":["image.jpg"]}\n', encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-secret")
    monkeypatch.setenv("DJ_VLM_MODEL", "qwen3.7-plus")
    plan = {
        "user_intent": "Tag images",
        "modality": "image",
        "recipe": {
            "dataset_path": str(dataset),
            "export_path": "result.jsonl",
            "process": [{"image_tagging_vlm_mapper": {"is_api_model": True}}],
        },
    }

    normalized, validation, _ = normalize_and_validate(str(tmp_path), plan)

    assert validation["ok"] is True
    params = normalized["recipe"]["process"][0]["image_tagging_vlm_mapper"]
    assert params["api_or_hf_model"] == "qwen3.7-plus"


def test_prepare_reports_operator_specific_missing_api_credentials(tmp_path, monkeypatch):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"<__dj__image>","images":["image.jpg"]}\n', encoding="utf-8")
    for name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "SK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DJ_VLM_MODEL", "qwen3.7-plus")
    monkeypatch.setenv("DJ_PLAN_FLOW_CONFIG_FILE", r"D:\dsh-app\dj-plan-flow.env")
    plan = {
        "user_intent": "Tag images",
        "modality": "image",
        "recipe": {
            "dataset_path": str(dataset),
            "export_path": "result.jsonl",
            "process": [{"image_tagging_vlm_mapper": {"is_api_model": True}}],
        },
    }

    prepared = PlanFlowService().prepare_plan(str(tmp_path), plan)

    assert prepared["valid"] is False
    error = next(item for item in prepared["validation"]["errors"] if item["code"] == "RUNTIME_API_CREDENTIAL_MISSING")
    assert error["operator"] == "image_tagging_vlm_mapper"
    assert error["path"] == "recipe.process[0].image_tagging_vlm_mapper"
    assert "OPENAI_API_KEY or DASHSCOPE_API_KEY" in error["message"]
    assert r"D:\dsh-app\dj-plan-flow.env" in error["message"]
    assert "restart" in error["message"].lower()


def test_prepare_reports_operator_specific_missing_vlm_model(tmp_path, monkeypatch):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"<__dj__image>","images":["image.jpg"]}\n', encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-secret")
    monkeypatch.delenv("DJ_VLM_MODEL", raising=False)
    monkeypatch.setenv("DJ_PLAN_FLOW_CONFIG_FILE", r"D:\dsh-app\dj-plan-flow.env")
    plan = {
        "user_intent": "Tag images",
        "modality": "image",
        "recipe": {
            "dataset_path": str(dataset),
            "export_path": "result.jsonl",
            "process": [{"image_tagging_vlm_mapper": {"is_api_model": True}}],
        },
    }

    prepared = PlanFlowService().prepare_plan(str(tmp_path), plan)

    assert prepared["valid"] is False
    error = next(item for item in prepared["validation"]["errors"] if item["code"] == "RUNTIME_VLM_MODEL_MISSING")
    assert error["operator"] == "image_tagging_vlm_mapper"
    assert "DJ_VLM_MODEL" in error["message"]
    assert r"D:\dsh-app\dj-plan-flow.env" in error["message"]


def test_prepare_response_does_not_expose_global_runtime_inventory(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")

    prepared = PlanFlowService().prepare_plan(str(tmp_path), _plan(dataset))

    assert "runtime" not in prepared


def test_exact_operator_name_bypasses_modality_filter():
    result = search_capabilities(["image_tagging_vlm_mapper"], modality="image", top_k=1)

    assert result["results"][0]["operator_names"] == ["image_tagging_vlm_mapper"]
    assert result["operators"][0]["name"] == "image_tagging_vlm_mapper"
    assert result["operators"][0]["description"]
    assert result["operators"][0]["match_score"] == 1.0
    assert "parameters" not in result["operators"][0]
    assert "signature" not in result["operators"][0]
    assert "parameter_descriptions" not in result["operators"][0]


def test_image_search_includes_multimodal_operators():
    result = search_capabilities(["tag images with a vision language model"], modality="image", top_k=30)

    names = result["results"][0]["operator_names"]
    assert result["top_k"] == 3
    assert len(names) <= 3
    assert "image_tagging_vlm_mapper" in names


def test_search_capabilities_defaults_to_three_compact_candidates():
    result = search_capabilities(["image"], modality="image")

    assert result["top_k"] == 3
    assert len(result["results"][0]["operator_names"]) == 3


@pytest.mark.parametrize(
    "requirement",
    ["image aesthetic quality scoring", "image aesthetics quality scoring"],
)
def test_search_normalizes_english_inflections(requirement):
    result = search_capabilities([requirement], modality="image", top_k=5)

    assert result["results"][0]["operator_names"][0] == "image_aesthetics_filter"


def test_service_separates_discovery_metadata_from_executable_schema():
    service = PlanFlowService()
    result = service.search_capabilities(["image aesthetic quality scoring"], modality="image", top_k=5)

    assert result["top_k"] == 3
    candidate = result["operators"][0]
    assert set(candidate) == {
        "candidate_id",
        "name",
        "type",
        "tags",
        "description",
        "match_score",
        "matched_requirements",
        "provider",
        "status",
        "version",
    }
    assert candidate["name"] == "image_aesthetics_filter"
    assert 0.0 <= candidate["match_score"] <= 1.0

    schema = service.get_capability_schemas([candidate["candidate_id"]])["operators"][0]
    assert "parameters" in schema
    assert "description" not in schema
    assert "tags" not in schema
    assert "match_score" not in schema


def test_search_deduplicates_compact_definitions_across_requirements():
    result = search_capabilities(["filter text", "filter text"], modality="text", top_k=3)

    first = result["results"][0]["operator_names"]
    second = result["results"][1]["operator_names"]
    assert first == second
    assert len(result["operators"]) == len(set(first))
    assert all(operator["matched_requirements"] == ["filter text", "filter text"] for operator in result["operators"])


def test_search_uses_bm25_over_registered_definition_fields(monkeypatch):
    from data_juicer.tools.plan_flow import discovery

    searcher = discovery._searcher()
    original = searcher.search_by_bm25
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(searcher, "search_by_bm25", capture)
    monkeypatch.setattr(
        searcher,
        "search_by_regex",
        lambda *args, **kwargs: pytest.fail("plan-flow discovery must not use regex"),
    )

    search_capabilities(["filter text"], modality="text", top_k=1)

    assert calls[0]["fields"] == ["name", "desc", "param_desc", "sig"]


def test_full_capability_schemas_are_loaded_by_exact_name():
    result = capability_schemas(["text_length_filter", "text_length_filter", "missing_operator"])

    assert [operator["name"] for operator in result["operators"]] == ["text_length_filter"]
    assert "parameters" in result["operators"][0]
    assert "description" not in result["operators"][0]
    assert "tags" not in result["operators"][0]
    assert "signature" not in result["operators"][0]
    assert result["missing"] == ["missing_operator"]
    assert result["ok"] is False


def test_prepare_versions_are_immutable_and_diffed(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    service = PlanFlowService()
    first = service.prepare_plan(str(tmp_path), _plan(dataset))
    second = service.prepare_plan(str(tmp_path), _plan(dataset, 3), first["task_id"], first["plan_version"])
    assert first["plan_version"] == "plan_v001"
    assert second["plan_version"] == "plan_v002"
    assert first["plan"]["plan_id"] == f"{first['task_id']}/plan_v001"
    assert first["content_hash"] != second["content_hash"]
    versions = service.get_plan(str(tmp_path), first["task_id"], include_versions=True)["versions"]
    assert [item["plan_version"] for item in versions] == ["plan_v001", "plan_v002"]
    assert any(change["path"].endswith("min_len") for change in second["changes"])


def test_materialized_recipe_enables_real_operation_events(tmp_path):
    from data_juicer.tools.plan_flow.runner import PlanRunner

    recipe = PlanRunner._materialize({"process": []}, tmp_path / "run", tmp_path / "output")

    assert recipe["use_dag"] is True


def test_invalid_plan_cannot_be_approved(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    plan = _plan(dataset)
    plan["recipe"]["process"][0]["text_length_filter"]["not_a_parameter"] = 1
    service = PlanFlowService()
    prepared = service.prepare_plan(str(tmp_path), plan)
    assert prepared["valid"] is False
    with pytest.raises(PlanFlowError, match="valid plan"):
        service.approve_plan(str(tmp_path), prepared["task_id"], prepared["plan_version"], prepared["content_hash"])


def test_run_requires_approval_and_detects_tampering(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    service = PlanFlowService()
    prepared = service.prepare_plan(str(tmp_path), _plan(dataset))
    with pytest.raises(PlanFlowError, match="Approve"):
        service.run_plan(str(tmp_path), prepared["task_id"], prepared["plan_version"])
    plan_path = Path(prepared["plan_path"])
    plan_path.write_text(plan_path.read_text(encoding="utf-8").replace("min_len: 2", "min_len: 99"), encoding="utf-8")
    with pytest.raises(PlanFlowError, match="changed"):
        service.approve_plan(str(tmp_path), prepared["task_id"], prepared["plan_version"], prepared["content_hash"])


def test_postprocess_is_snapshotted_outside_recipe(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    script = tmp_path / "finish.py"
    script.write_text("print('done')\n", encoding="utf-8")
    plan = _plan(dataset)
    plan["postprocess"] = [{"kind": "python", "script": str(script), "arguments": []}]
    prepared = PlanFlowService().prepare_plan(str(tmp_path), plan)
    assert prepared["valid"] is True
    assert "postprocess" not in prepared["plan"]["recipe"]
    assert prepared["plan"]["postprocess"][0]["script"] == "artifacts/finish.py"


def test_mcp_exposes_small_plan_first_surface():
    from data_juicer.tools.DJ_mcp_plan_flow import create_mcp_server

    tools = create_mcp_server()._tool_manager.list_tools()
    assert {tool.name for tool in tools} == {
        "inspect_input",
        "search_capabilities",
        "get_capability_schemas",
        "resolve_capabilities",
        "prepare_plan",
        "get_plan",
        "approve_plan",
        "run_plan",
        "get_run",
        "cancel_run",
        "get_custom_operator_authoring_spec",
        "develop_custom_operator",
        "get_custom_operator_job",
        "validate_custom_operator",
    }
    assert len(tools) == 14


def test_production_service_refuses_to_fall_back_to_shared_local_process(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    service = PlanFlowService()
    prepared = service.prepare_plan(str(tmp_path), _plan(dataset))
    service.approve_plan(str(tmp_path), prepared["task_id"], prepared["plan_version"], prepared["content_hash"])

    with pytest.raises(PlanFlowError) as missing:
        service.run_plan(str(tmp_path), prepared["task_id"], prepared["plan_version"])
    assert missing.value.code == "BROKER_REQUIRED"


def test_native_execution_mode_must_be_selected_explicitly():
    service = PlanFlowService.native()

    assert service.execution_mode == "native"

    with pytest.raises(ValueError, match="Unsupported plan-flow execution mode"):
        PlanFlowService(execution_mode="anything")


def test_mcp_service_selects_native_execution_from_environment(monkeypatch):
    from data_juicer.tools.plan_flow.server import _service_from_environment

    monkeypatch.setenv("DJ_PLAN_FLOW_EXECUTION_MODE", "native")

    assert _service_from_environment().execution_mode == "native"


def test_production_service_resolves_runtime_then_submits_public_broker_run(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    resolution_calls = []
    broker_calls = []

    class Resolver:
        def resolve(self, **kwargs):
            resolution_calls.append(kwargs)
            return SimpleNamespace(runtime_id="runtime-" + "1" * 24)

    class Broker:
        def start(self, **kwargs):
            broker_calls.append(kwargs)
            return {"run_id": "run_" + "2" * 32, "task_id": kwargs["task_id"], "status": "running"}

    plan = _plan(dataset)
    plan["capability_bindings"] = [
        {"capability_id": "op-region-stats-v1-capability", "operators": ["masked_region_statistics_mapper"]}
    ]
    service = PlanFlowService(runtime_resolver=Resolver(), broker_client=Broker())
    prepared = service.prepare_plan(str(tmp_path), plan)
    service.approve_plan(str(tmp_path), prepared["task_id"], prepared["plan_version"], prepared["content_hash"])

    run = service.run_plan(str(tmp_path), prepared["task_id"], prepared["plan_version"])["run"]

    assert run["run_id"].startswith("run_")
    assert resolution_calls[0]["capability_ids"] == ("op-region-stats-v1-capability",)
    assert broker_calls[0]["runtime_id"] == "runtime-" + "1" * 24


def test_approved_plan_runs_and_writes_report(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text("\n".join(json.dumps({"text": value}) for value in ["a", "hello"]) + "\n", encoding="utf-8")
    service = PlanFlowService.local_for_tests()
    prepared = service.prepare_plan(str(tmp_path), _plan(dataset))
    service.approve_plan(str(tmp_path), prepared["task_id"], prepared["plan_version"], prepared["content_hash"])
    started = service.run_plan(str(tmp_path), prepared["task_id"], prepared["plan_version"])["run"]
    assert set(started["handle"]) == {
        "schema_version",
        "backend",
        "run_id",
        "created_at",
        "deadline",
        "backend_ref",
    }
    assert started["handle"]["backend"] == "local-process"
    assert "pid" not in started
    deadline = time.time() + 60
    while time.time() < deadline:
        state = service.get_run(str(tmp_path), prepared["task_id"], started["run_id"])["run"]
        if state["status"] not in {"starting", "running"}:
            break
        time.sleep(0.2)
    assert state["status"] == "succeeded", state
    assert [step["status"] for step in state["steps"]] == ["succeeded"]
    assert state["step_telemetry"]["mapping_complete"] is True
    assert Path(state["report_path"]).is_file()
    assert Path(state["recipe_output"]).is_file()
    assert Path(state["result_manifest_path"]).is_file()
    inspected = RunOutputGateway().inspect_run(tmp_path, prepared["task_id"], prepared["plan_version"], started["run_id"])
    assert inspected["eligible"] is True
    assert inspected["status"] == "partial"
    assert inspected["fileCount"] >= 1
    backend = LocalProcessBackend(tmp_path)
    handle = RunHandle.from_dict(started["handle"])
    result = backend.collect(handle)
    assert result.status == "succeeded"
    assert result.exit_code == 0
    backend.cleanup(handle)

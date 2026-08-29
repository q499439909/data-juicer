import json
import time
from pathlib import Path

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
    assert set(first) == {"name", "description", "category", "modalities", "devices"}
    assert first["name"]
    assert first["category"]
    assert first["modalities"]
    assert first["devices"]


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
    assert set(parameter) == {"name", "type", "required", "default", "description"}
    assert parameter["required"] is False
    json.dumps(result)


def test_operator_detail_reports_an_unknown_exact_name():
    result = operator_detail("not_a_registered_operator")

    assert result == {
        "ok": False,
        "error": "operator_not_found",
        "message": "Operator was not found.",
    }


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
    assert result["operators"][0]["modality_compatible"] is True
    assert "parameters" not in result["operators"][0]
    assert "signature" in result["operators"][0]


def test_image_search_includes_multimodal_operators():
    result = search_capabilities(["tag images with a vision language model"], modality="image", top_k=30)

    names = result["results"][0]["operator_names"]
    assert result["top_k"] == 5
    assert len(names) <= 5
    assert "image_tagging_vlm_mapper" in names


def test_search_capabilities_defaults_to_three_compact_candidates():
    result = search_capabilities(["image"], modality="image")

    assert result["top_k"] == 3
    assert len(result["results"][0]["operator_names"]) == 3


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
        "prepare_plan",
        "get_plan",
        "preview_plan",
        "approve_plan",
        "run_plan",
        "get_run",
        "cancel_run",
    }


def test_approved_plan_runs_and_writes_report(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text("\n".join(json.dumps({"text": value}) for value in ["a", "hello"]) + "\n", encoding="utf-8")
    service = PlanFlowService()
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
    assert "pid" not in started
    deadline = time.time() + 60
    while time.time() < deadline:
        state = service.get_run(str(tmp_path), prepared["task_id"], started["run_id"])["run"]
        if state["status"] not in {"starting", "running"}:
            break
        time.sleep(0.2)
    assert state["status"] == "succeeded", state
    assert Path(state["report_path"]).is_file()
    assert Path(state["recipe_output"]).is_file()
    backend = LocalProcessBackend(tmp_path)
    handle = RunHandle.from_dict(started["handle"])
    result = backend.collect(handle)
    assert result.status == "succeeded"
    assert result.exit_code == 0
    backend.cleanup(handle)

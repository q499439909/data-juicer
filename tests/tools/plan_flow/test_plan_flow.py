import json
import time
from pathlib import Path

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.discovery import inspect_input, search_capabilities
from data_juicer.tools.plan_flow.service import PlanFlowService


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


def test_runtime_capabilities_report_presence_without_secret(monkeypatch):
    secret = "must-not-be-returned"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")

    result = search_capabilities(["filter text"], modality="text", top_k=1)

    assert result["runtime"]["api_credentials_configured"] is True
    assert result["runtime"]["api_base_url_configured"] is True
    assert secret not in json.dumps(result)


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
    deadline = time.time() + 60
    while time.time() < deadline:
        state = service.get_run(str(tmp_path), prepared["task_id"], started["run_id"])["run"]
        if state["status"] not in {"starting", "running"}:
            break
        time.sleep(0.2)
    assert state["status"] == "succeeded", state
    assert Path(state["report_path"]).is_file()
    assert Path(state["recipe_output"]).is_file()

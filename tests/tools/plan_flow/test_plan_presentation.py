import json
from pathlib import Path

from data_juicer.tools.plan_flow.presentation import build_plan_view
from data_juicer.tools.plan_flow.service import PlanFlowService


def _plan() -> dict:
    return {
        "plan_id": "task_example/plan_v001",
        "plan_version": "plan_v001",
        "recipe": {
            "process": [
                {"clean_html_mapper": {}},
                {"text_length_filter": {"min_len": 2}},
                {"text_length_filter": {"min_len": 5}},
            ]
        },
    }


def test_build_plan_view_groups_real_steps_and_fills_ungrouped_steps():
    view, warnings = build_plan_view(
        _plan(),
        "hash-1",
        {"groups": [{"title": "初步清洗", "summary": "去除网页噪声", "process_indexes": [0, 1]}]},
    )

    assert warnings == []
    assert [group["title"] for group in view["groups"]] == ["初步清洗", "text_length_filter"]
    assert view["groups"][0]["step_refs"] == ["step-000", "step-001"]
    assert [step["execution_key"] for step in view["steps"]] == [
        "op_001_clean_html_mapper",
        "op_002_text_length_filter",
        "op_003_text_length_filter",
    ]


def test_invalid_group_falls_back_without_changing_execution_steps():
    view, warnings = build_plan_view(
        _plan(), "hash-1", {"groups": [{"title": "错误分组", "process_indexes": [0, 3]}]}
    )

    assert warnings
    assert [group["step_refs"] for group in view["groups"]] == [
        ["step-000"],
        ["step-001"],
        ["step-002"],
    ]
    assert [step["process_index"] for step in view["steps"]] == [0, 1, 2]


def test_prepare_plan_persists_view_sidecar_without_hashing_presentation(tmp_path: Path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    plan = {
        "user_intent": "Clean text",
        "modality": "text",
        "recipe": {
            "dataset_path": str(dataset),
            "export_path": "result.jsonl",
            "process": [
                {"clean_html_mapper": {}},
                {"text_length_filter": {"min_len": 2}},
            ],
        },
    }
    service = PlanFlowService()

    prepared = service.prepare_plan(
        str(tmp_path),
        plan,
        view_spec={"groups": [{"title": "初步清洗", "process_indexes": [0, 1]}]},
    )

    sidecar = Path(prepared["plan_path"]).with_name("plan-view.json")
    assert json.loads(sidecar.read_text(encoding="utf-8")) == prepared["view"]
    assert prepared["view"]["recipe_content_hash"] == prepared["content_hash"]
    original_hash = prepared["content_hash"]
    stored_view = json.loads(sidecar.read_text(encoding="utf-8"))
    stored_view["groups"][0]["title"] = "只改展示"
    sidecar.write_text(json.dumps(stored_view), encoding="utf-8")
    assert service.get_plan(str(tmp_path), prepared["task_id"])["content_hash"] == original_hash


def test_get_plan_rejects_a_stale_view_identity_and_returns_safe_fallback(tmp_path: Path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    plan = {
        "user_intent": "Clean text",
        "modality": "text",
        "recipe": {
            "dataset_path": str(dataset),
            "export_path": "result.jsonl",
            "process": [{"text_length_filter": {"min_len": 2}}],
        },
    }
    service = PlanFlowService()
    prepared = service.prepare_plan(str(tmp_path), plan)
    sidecar = Path(prepared["plan_path"]).with_name("plan-view.json")
    stale = json.loads(sidecar.read_text(encoding="utf-8"))
    stale["recipe_content_hash"] = "sha256:stale"
    sidecar.write_text(json.dumps(stale), encoding="utf-8")

    loaded = service.get_plan(str(tmp_path), prepared["task_id"])

    assert loaded["view"]["recipe_content_hash"] == prepared["content_hash"]
    assert loaded["presentation_warnings"]

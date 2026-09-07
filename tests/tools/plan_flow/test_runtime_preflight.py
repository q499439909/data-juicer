import json

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.runtime_preflight import RuntimePreflight
from data_juicer.tools.plan_flow.service import PlanFlowService


def _plan(dataset, *, operator="image_nsfw_filter", params=None, profile=None):
    plan = {
        "user_intent": "Inspect images",
        "modality": "image",
        "recipe": {
            "dataset_path": str(dataset),
            "export_path": "result.jsonl",
            "process": [{operator: params or {}}],
        },
    }
    if profile:
        plan["execution_profile"] = profile
    return plan


def _cpu_preflight():
    return RuntimePreflight(cuda_available=lambda: False, platform_name=lambda: "cpu-test-host")


def test_prepare_plan_reports_gpu_tagged_operator_cpu_fallback_without_blocking(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text(json.dumps({"text": "image", "images": ["one.jpg"]}) + "\n", encoding="utf-8")
    service = PlanFlowService.native(runtime_preflight=_cpu_preflight())

    prepared = service.prepare_plan(str(tmp_path), _plan(dataset))

    assessment = prepared["runtime_assessment"]
    assert assessment["ok"] is True
    assert assessment["execution_backend"] == "local-process"
    assert assessment["host"]["available_accelerators"] == ["cpu"]
    assert assessment["host"]["probe_source"] == "local_host"
    assert assessment["operators"] == [
        {
            "process_index": 0,
            "operator": "image_nsfw_filter",
            "preferred_accelerator": "cuda",
            "resolved_device": "cpu",
            "resolution": "cpu_fallback",
        }
    ]
    assert {item["code"] for item in assessment["warnings"]} >= {
        "CPU_FALLBACK",
        "MODEL_DOWNLOAD_MAY_BE_REQUIRED",
    }
    assert assessment["model_inputs"][0]["value"] == "Falconsai/nsfw_image_detection"


def test_prepare_plan_blocks_explicit_gpu_profile_on_cpu_host(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    service = PlanFlowService.native(runtime_preflight=_cpu_preflight())

    prepared = service.prepare_plan(
        str(tmp_path),
        _plan(dataset, operator="text_length_filter", params={"min_len": 1}, profile="local-gpu"),
    )

    assert prepared["runtime_assessment"]["ok"] is False
    assert prepared["runtime_assessment"]["blocking_issues"][0]["code"] == "GPU_REQUIRED"


def test_run_plan_rechecks_explicit_gpu_requirement(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    service = PlanFlowService.native(runtime_preflight=_cpu_preflight())
    prepared = service.prepare_plan(
        str(tmp_path),
        _plan(dataset, operator="text_length_filter", params={"min_len": 1}, profile="local-gpu"),
    )
    service.approve_plan(str(tmp_path), prepared["task_id"], prepared["plan_version"], prepared["content_hash"])

    with pytest.raises(PlanFlowError) as blocked:
        service.run_plan(str(tmp_path), prepared["task_id"], prepared["plan_version"])

    assert blocked.value.code == "RUNTIME_PREFLIGHT_FAILED"


def test_broker_preflight_reports_missing_runtime_seams(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    service = PlanFlowService(runtime_preflight=_cpu_preflight())
    plan = _plan(dataset, operator="custom_mapper")
    plan["capability_bindings"] = [
        {"capability_id": "custom-capability", "operators": ["custom_mapper"]}
    ]

    prepared = service.prepare_plan(str(tmp_path), plan)

    assert {item["code"] for item in prepared["runtime_assessment"]["blocking_issues"]} == {
        "BROKER_REQUIRED",
        "RUNTIME_RESOLVER_REQUIRED",
    }


def test_broker_preflight_uses_target_profile_instead_of_control_host_cuda(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text(json.dumps({"text": "image", "images": ["one.jpg"]}) + "\n", encoding="utf-8")
    service = PlanFlowService(
        runtime_resolver=object(),
        broker_client=object(),
        runtime_preflight=_cpu_preflight(),
    )

    prepared = service.prepare_plan(str(tmp_path), _plan(dataset, profile="gpu-small"))

    assessment = prepared["runtime_assessment"]
    assert assessment["ok"] is True
    assert assessment["host"] == {
        "platform": "broker-managed",
        "probe_source": "execution_profile",
        "available_accelerators": ["cpu", "cuda"],
        "cuda_available": True,
    }
    assert assessment["operators"][0]["resolved_device"] == "cuda"
    assert "CPU_FALLBACK" not in {item["code"] for item in assessment["warnings"]}


def test_broker_cpu_profile_blocks_explicit_operator_gpu_request(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding="utf-8")
    service = PlanFlowService(
        runtime_resolver=object(),
        broker_client=object(),
        runtime_preflight=_cpu_preflight(),
    )

    prepared = service.prepare_plan(
        str(tmp_path),
        _plan(dataset, operator="text_length_filter", params={"min_len": 1, "num_gpus": 1}),
    )

    assert {item["code"] for item in prepared["runtime_assessment"]["blocking_issues"]} == {"GPU_REQUIRED"}

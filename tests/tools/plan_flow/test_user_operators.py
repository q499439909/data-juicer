import json
import tempfile
import time
from pathlib import Path

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.user_operator_store import (
    UserOperatorStore,
    current_user,
    resolve_bindings,
)
from data_juicer.tools.plan_flow.user_operator_validation import UserOperatorValidation

SOURCE = '''from data_juicer.ops.base_op import Mapper
from data_juicer.ops import OPERATORS

@OPERATORS.register_module("personal_uppercase_mapper")
class PersonalUppercaseMapper(Mapper):
    """Uppercase text for an account-specific example."""
    def __init__(self, suffix: str = "", **kwargs):
        super().__init__(**kwargs)
        self.suffix = suffix

    def process_single(self, sample):
        sample[self.text_key] = sample[self.text_key].upper() + self.suffix
        return sample
'''


INCOMPATIBLE_FILTER_SOURCE = """from data_juicer.ops.base_filter import BaseFilter, OP_REGISTRY

@OP_REGISTRY.register()
class FaceClarityFilter(BaseFilter):
    def compute_stats(self, sample, context=False):
        return sample

    def is_keep(self, sample):
        return True
"""


@pytest.fixture
def account(tmp_path, monkeypatch):
    # Dataset backends generate deep cache paths; use a short Windows test root.
    parent = Path.cwd() / ".test-tmp"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="op-", dir=parent) as directory:
        monkeypatch.setenv("DSH_USER_DATA_ROOT", directory)
        token = current_user.set("usr_A")
        yield UserOperatorStore()
        current_user.reset(token)


def publish(store, source="source", status="validated"):
    return store.publish(
        source,
        {
            "name": "personal_uppercase_mapper",
            "type": "mapper",
            "tags": ["text"],
            "description": "uppercase",
            "parameters": {},
        },
        {},
        {},
        {"status": status},
    )


def test_immutable_and_account_scoped(account):
    first = publish(account)
    second = publish(account, "changed", "experimental")
    assert account.candidates()[0]["version"] == first["version"]
    assert first["version"] != second["version"]
    with pytest.raises(PlanFlowError, match="not available"):
        UserOperatorStore(user_id="usr_B").resolve(first["candidate_id"])
    from pathlib import Path

    Path(first["_path"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(PlanFlowError, match="changed after validation"):
        account.resolve(first["candidate_id"])


def test_path_and_binding_rejection(account):
    with pytest.raises(PlanFlowError):
        UserOperatorStore(user_id="../usr_B")
    with pytest.raises(PlanFlowError):
        resolve_bindings({"recipe": {"custom_operator_paths": ["D:/someone/secret.py"]}})
    with pytest.raises(PlanFlowError):
        account.resolve("user:../../secret:hash")


def test_low_relevance_user_candidate_does_not_displace_relevant_builtin(monkeypatch):
    from data_juicer.tools.plan_flow import operator_catalog_service

    monkeypatch.setattr(
        operator_catalog_service,
        "personal",
        lambda: [
            {
                "candidate_id": "user:image_watermark_remove_mapper:test-version",
                "name": "image_watermark_remove_mapper",
                "type": "mapper",
                "tags": ["cpu", "image"],
                "description": "Remove text and logo watermarks from images and repair the masked region.",
                "parameters": {},
                "provider": "user",
                "operator_id": "image_watermark_remove_mapper",
                "status": "experimental",
                "version": "test-version",
            }
        ],
    )

    result = operator_catalog_service.search(
        ["face detection filter keeping images with exactly one face and face area ratio in a given range"],
        modality="image",
        top_k=3,
    )

    names = result["results"][0]["operator_names"]
    assert "image_face_count_filter" in names
    assert "image_watermark_remove_mapper" not in names


def test_relevant_user_candidate_can_join_builtin_shortlist(monkeypatch):
    from data_juicer.tools.plan_flow import operator_catalog_service

    monkeypatch.setattr(
        operator_catalog_service,
        "personal",
        lambda: [
            {
                "candidate_id": "user:image_watermark_remove_mapper:test-version",
                "name": "image_watermark_remove_mapper",
                "type": "mapper",
                "tags": ["cpu", "image"],
                "description": "Remove text and logo watermarks from images and repair the masked region.",
                "parameters": {},
                "provider": "user",
                "operator_id": "image_watermark_remove_mapper",
                "status": "experimental",
                "version": "test-version",
            }
        ],
    )

    result = operator_catalog_service.search(
        ["remove text and logo watermarks from images and repair the masked region"],
        modality="image",
        top_k=3,
    )

    names = result["results"][0]["operator_names"]
    assert "image_watermark_remove_mapper" in names
    candidate = next(item for item in result["operators"] if item["name"] == "image_watermark_remove_mapper")
    assert candidate["match_score"] > 0


def test_incompatible_source_is_rejected_before_creating_validation_artifacts(account):
    jobs = UserOperatorValidation()
    proposal = {
        "name": "face_clarity_filter",
        "category": "filter",
        "source": INCOMPATIBLE_FILTER_SOURCE,
        "validation_contract": {"purpose": "filter clear faces"},
    }

    with pytest.raises(PlanFlowError) as failure:
        jobs.develop(proposal, [{"text": "fixture"}])

    assert failure.value.code == "OPERATOR_SOURCE_INCOMPATIBLE"
    assert failure.value.details["phase"] == "source_preflight"
    diagnostics = failure.value.details["diagnostics"]
    assert any(
        item.get("replacement") == "from data_juicer.ops.base_op import OPERATORS, Filter" for item in diagnostics
    )
    assert any(item.get("replacement") == "compute_stats_single" for item in diagnostics)
    assert not (account.home / "operator_jobs").exists()
    assert not (account.home / "operator_tmp").exists()
    assert not (account.catalog / "filter" / "face_clarity_filter").exists()


def wait_job(jobs, job_id, timeout=150):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = jobs.get(job_id)["job"]
        if job["status"] != "testing":
            return job
        time.sleep(0.2)
    pytest.fail("Validation did not finish")


@pytest.mark.integration
def test_real_executor_publish_cleanup_and_catalog(account, monkeypatch):
    from data_juicer.ops import OPERATORS
    from data_juicer.tools.plan_flow import operator_catalog_service

    before = set(OPERATORS.modules)
    jobs = UserOperatorValidation()
    proposal = {
        "name": "personal_uppercase_mapper",
        "category": "mapper",
        "source": SOURCE,
        "validation_contract": {
            "purpose": "uppercase text",
            "row_count": 1,
            "equals": [{"row": 0, "field": "text", "value": "HELLO"}],
        },
    }
    job = jobs.develop(proposal, [{"text": "hello"}])["job"]
    final = wait_job(jobs, job["job_id"])
    assert final["status"] == "validated", final.get("error", final)
    assert not final["cleanup_pending"]
    assert list((account.home / "operator_tmp").iterdir()) == []
    assert set(OPERATORS.modules) == before
    assert any(
        item["provider"] == "user"
        for item in operator_catalog_service.search(["personal_uppercase_mapper"])["operators"]
    )
    assert operator_catalog_service.schemas([final["candidate"]["candidate_id"]])["ok"]
    # Bind and execute a real mixed native pipeline, then reject a different account.
    from data_juicer.tools.plan_flow.service import PlanFlowService

    workspace = str(account.root).removeprefix("\\\\?\\")
    dataset = Path(workspace) / "business.jsonl"
    dataset.write_text('{"text":"hello"}\n{"text":"x"}\n', encoding="utf-8")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(Path(workspace) / "cache"))
    plan = {
        "user_intent": "Mixed operators",
        "modality": "text",
        "recipe": {
            "dataset_path": str(dataset),
            "np": 1,
            "export_path": "result.jsonl",
            "use_cache": False,
            "process": [
                {"text_length_filter": {"min_len": 2}},
                {"personal_uppercase_mapper": {}},
            ],
        },
        "operator_bindings": [
            {
                "step_index": 1,
                "provider": "user",
                "operator_id": "personal_uppercase_mapper",
                "version": final["candidate"]["version"],
            }
        ],
    }
    service = PlanFlowService.native()
    prepared = service.prepare_plan(workspace, plan)
    assert prepared["validation"]["ok"], prepared["validation"]
    args = (workspace, prepared["task_id"], prepared["plan_version"])
    service.approve_plan(*args, prepared["content_hash"])
    token = current_user.set("usr_B")
    try:
        with pytest.raises(PlanFlowError, match="another account"):
            service.run_plan(*args)
        with pytest.raises(PlanFlowError, match="another account"):
            service.get_plan(*args)
    finally:
        current_user.reset(token)
    started = service.run_plan(*args)["run"]
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        state = service.get_run(workspace, prepared["task_id"], started["run_id"])["run"]
        if state["status"] not in {"running", "starting"}:
            break
        time.sleep(0.2)
    assert state["status"] == "succeeded", state
    output = Path(state["output_dir"]) / "result.jsonl"
    assert json.loads(output.read_text(encoding="utf-8"))["text"] == "HELLO"
    assert dataset.read_text(encoding="utf-8") == '{"text":"hello"}\n{"text":"x"}\n'
    assert set(OPERATORS.modules) == before
    # A failed revision must not overwrite the last validated pointer.
    proposal["source"] = SOURCE.replace(".upper()", ".lower()")
    failed = wait_job(jobs, jobs.develop(proposal, [{"text": "hello"}])["job"]["job_id"])
    assert failed["status"] == "failed"
    assert not failed["cleanup_pending"]
    assert account.candidates()[0]["version"] == final["candidate"]["version"]
    proposal["validation_contract"] = {}
    with pytest.raises(PlanFlowError, match="frozen"):
        jobs.develop(proposal, [{"text": "hello"}])


def test_cancel_cleanup(account):
    jobs = UserOperatorValidation()
    proposal = {
        "name": "personal_uppercase_mapper",
        "category": "mapper",
        "source": "import time\ntime.sleep(120)\n" + SOURCE,
    }
    job = jobs.develop(proposal, [{"text": "hello"}])["job"]
    jobs.get(job["job_id"], cancel=True)
    final = wait_job(jobs, job["job_id"], 30)
    assert final["status"] == "cancelled"
    assert not final["cleanup_pending"]


def test_dependency_conflicts_and_model_fingerprints(account):
    from data_juicer.tools.plan_flow.user_operator_resources import model_refs
    from data_juicer.tools.plan_flow.user_operator_runtime import (
        dependency_lock,
    )

    with pytest.raises(PlanFlowError, match="Conflicting"):
        dependency_lock(["example==1.0", "example==2.0"])
    with pytest.raises(PlanFlowError):
        dependency_lock(["https://example.com/package.whl"])
    model = account.home / "model.bin"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"first")
    refs = model_refs([{"path": str(model)}], freeze=True)
    assert model_refs(refs) == refs
    model.write_bytes(b"changed")
    with pytest.raises(PlanFlowError, match="changed"):
        model_refs(refs)


def test_timeout_and_smoke_are_not_validated(account):
    jobs = UserOperatorValidation()
    proposal = {
        "name": "personal_uppercase_mapper",
        "category": "mapper",
        "source": SOURCE,
    }
    final = wait_job(jobs, jobs.develop(proposal, [{"text": "hello"}])["job"]["job_id"])
    assert final["status"] == "experimental", final.get("error")
    proposal["source"] = "import time\ntime.sleep(120)\n" + SOURCE
    final = wait_job(
        jobs,
        jobs.develop(proposal, [{"text": "hello"}], timeout_seconds=1)["job"]["job_id"],
    )
    assert final["status"] == "failed"
    assert "timeout" in final["error"]
    assert not final["cleanup_pending"]


@pytest.mark.integration
def test_async_failure_identifies_the_validation_phase(account):
    jobs = UserOperatorValidation()
    proposal = {
        "name": "missing_dependency_mapper",
        "category": "mapper",
        "source": SOURCE.replace(
            "from data_juicer.ops.base_op import Mapper",
            "import package_that_does_not_exist\nfrom data_juicer.ops.base_op import Mapper",
        ).replace("personal_uppercase_mapper", "missing_dependency_mapper"),
    }

    submitted = jobs.develop(proposal, [{"text": "hello"}])["job"]
    final = wait_job(jobs, submitted["job_id"])

    assert final["status"] == "failed"
    assert final["error_details"]["phase"] == "schema"
    assert not final["cleanup_pending"]


def test_http_gateway_injects_account_outside_tool_arguments(account, monkeypatch):
    from starlette.testclient import TestClient

    from data_juicer.tools.plan_flow import server

    item = publish(account)
    monkeypatch.setenv("DSH_DJ_INTERNAL_TOKEN", "unit-test-internal-token")
    mcp = server.create_mcp_server()
    token = current_user.set(None)
    try:
        with TestClient(mcp.streamable_http_app()) as client:
            assert client.get("/internal/operator-tools").status_code == 403
            headers = {
                "x-dsh-internal-token": "unit-test-internal-token",
                "x-dsh-user-id": "usr_A",
            }
            listing = client.get("/internal/operator-tools", headers=headers)
            assert listing.status_code == 200
            tool_names = {tool["name"] for tool in listing.json()["tools"]}
            assert "develop_custom_operator" in tool_names
            assert "get_custom_operator_authoring_spec" in tool_names
            authoring = client.post(
                "/internal/operator-tools",
                headers=headers,
                json={
                    "name": "get_custom_operator_authoring_spec",
                    "arguments": {"category": "filter", "modality": "image"},
                },
            ).json()
            assert authoring["ok"]
            assert authoring["spec"]["required_methods"] == [
                "compute_stats_single",
                "process_single",
            ]
            request = {
                "name": "get_capability_schemas",
                "arguments": {"operator_names": [item["candidate_id"]]},
            }
            assert client.post("/internal/operator-tools", headers=headers, json=request).json()["ok"]
            headers["x-dsh-user-id"] = "usr_B"
            assert not client.post("/internal/operator-tools", headers=headers, json=request).json()["ok"]
            request["arguments"]["user_id"] = "usr_A"
            assert client.post("/internal/operator-tools", headers=headers, json=request).status_code == 400
            public = client.get("/operator-catalog").json()
            assert all(item["provider"] == "dj" for item in public["operators"])
    finally:
        current_user.reset(token)

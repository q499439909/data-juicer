import ast

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.operator_authoring import operator_authoring


def test_filter_authoring_spec_matches_the_running_dj_interface():
    spec = operator_authoring.get_spec("filter", "image")

    assert spec["contract_version"] == "dj-operator-authoring/v2"
    assert spec["category"] == "filter"
    assert spec["modality"] == "image"
    assert spec["base_class"] == "Filter"
    assert spec["required_methods"] == ["compute_stats_single", "process_single"]
    assert spec["forbidden_overrides"] == ["compute_stats", "process"]
    assert "from data_juicer.ops.base_op import OPERATORS, Filter" in spec["source_scaffold"]
    assert "@OPERATORS.register_module(OP_NAME)" in spec["source_scaffold"]
    assert "def compute_stats_single(self, sample, context=False):" in spec["source_scaffold"]
    assert "def process_single(self, sample):" in spec["source_scaffold"]
    assert spec["optional_integrations"] == [
        "from data_juicer.ops.op_fusion import LOADED_IMAGES",
        "@LOADED_IMAGES.register_module(OP_NAME)",
    ]
    assert "self.image_key" in spec["modality_notes"]


def test_authoring_spec_defines_a_complete_model_decision_policy():
    policy = operator_authoring.get_spec("mapper", "image")["model_policy"]

    assert policy["decision_order"] == [
        "reuse_a_matching_dj_builtin",
        "none",
        "api_vlm",
        "remote_model",
        "local_model",
    ]
    assert set(policy["strategies"]) == {"none", "api_vlm", "remote_model", "local_model"}
    assert "model_id, revision" in policy["strategies"]["remote_model"]
    assert "absolute model file path" in policy["strategies"]["local_model"]
    assert "exact name==version" in policy["dependencies"]
    assert "backend model caches" in policy["cache"]
    assert "never model weights" in policy["assets"]
    assert "develop_custom_operator" in policy["next_step"]


def test_unknown_authoring_spec_category_is_rejected():
    with pytest.raises(PlanFlowError) as failure:
        operator_authoring.get_spec("processor", "image")

    assert failure.value.code == "INVALID_OPERATOR_TYPE"


def test_unknown_authoring_spec_modality_is_rejected():
    with pytest.raises(PlanFlowError) as failure:
        operator_authoring.get_spec("mapper", "spreadsheet")

    assert failure.value.code == "INVALID_MODALITY"


def test_authoring_lint_requires_the_real_base_class_import():
    source = """class Mapper:
    pass

class Registry:
    def register_module(self, name):
        return lambda cls: cls

OPERATORS = Registry()

@OPERATORS.register_module("fake_mapper")
class FakeMapper(Mapper):
    def process_single(self, sample):
        return sample
"""

    with pytest.raises(PlanFlowError) as failure:
        operator_authoring.lint("fake_mapper", "mapper", source)

    assert failure.value.code == "OPERATOR_SOURCE_INCOMPATIBLE"
    assert any(item["symbol"] == "data_juicer.ops.base_op.Mapper" for item in failure.value.details["diagnostics"])


def test_authoring_lint_rejects_a_builtin_registration_name():
    source = operator_authoring.get_spec("mapper")["source_scaffold"].replace(
        '"replace_with_operator_name"', '"text_length_filter"'
    )

    with pytest.raises(PlanFlowError) as failure:
        operator_authoring.lint("text_length_filter", "mapper", source)

    assert any(
        item["symbol"] == "operator_name" and "built-in" in item["message"]
        for item in failure.value.details["diagnostics"]
    )


def test_authoring_lint_rejects_a_second_registered_class():
    source = (
        operator_authoring.get_spec("mapper")["source_scaffold"].replace(
            '"replace_with_operator_name"', '"custom_mapper"'
        )
        + """

@OPERATORS.register_module("hidden_mapper")
class HiddenMapper:
    pass
"""
    )

    with pytest.raises(PlanFlowError) as failure:
        operator_authoring.lint("custom_mapper", "mapper", source)

    assert any(
        item["symbol"] == "OPERATORS.register_module" and "exactly one" in item["message"]
        for item in failure.value.details["diagnostics"]
    )


@pytest.mark.parametrize(
    ("category", "base_class", "required_methods", "forbidden_overrides"),
    [
        ("mapper", "Mapper", ["process_single"], ["process"]),
        ("filter", "Filter", ["compute_stats_single", "process_single"], ["compute_stats", "process"]),
        ("deduplicator", "Deduplicator", ["compute_hash", "process"], []),
        ("selector", "Selector", ["process"], []),
        ("grouper", "Grouper", ["process"], []),
        ("aggregator", "Aggregator", ["process_single"], []),
        ("pipeline", "Pipeline", ["run"], []),
    ],
)
def test_every_supported_category_has_a_parseable_versioned_spec(
    category, base_class, required_methods, forbidden_overrides
):
    spec = operator_authoring.get_spec(category, "generic")

    ast.parse(spec["source_scaffold"])
    assert spec["base_class"] == base_class
    assert spec["required_methods"] == required_methods
    assert spec["forbidden_overrides"] == forbidden_overrides

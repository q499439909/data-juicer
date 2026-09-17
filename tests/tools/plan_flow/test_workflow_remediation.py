import json
import importlib.metadata
import pytest
from data_juicer.tools.plan_flow.runtime_environment_lock import inspect_plan_runtime
from data_juicer.tools.plan_flow.service import PlanFlowService
from data_juicer.tools.plan_flow.common import PlanFlowError


def test_locked_closure_reports_drift_before_execution(monkeypatch):
    original=importlib.metadata.version
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'2.13.0' if name=='torch' else original(name))
    report=inspect_plan_runtime({'model_bindings':[{'runtime_packages':['torch']}]})
    assert not report['ok']
    assert any(i['package']=='torch' and i['expected']=='2.8.0' and i['installed']=='2.13.0' for i in report['blocking_issues'])
    assert 'nvidia-cublas-cu12' not in report['manifest']['packages']  # Windows marker
    assert report['recovery_actions'][0]['action']=='restore_locked_runtime'


def test_prepare_retry_uses_one_version_and_rejects_changed_content(tmp_path):
    source=tmp_path/'input.jsonl';source.write_text('{"text":"hello"}\n',encoding='utf-8')
    plan={'user_intent':'clean','recipe':{'dataset_path':str(source),'export_path':'out.jsonl','process':[{'clean_links_mapper':{}}]}}
    service=PlanFlowService.native()
    first=service.prepare_plan(str(tmp_path),plan,request_id='host-call-1')
    again=service.prepare_plan(str(tmp_path),plan,request_id='host-call-1')
    assert (first['task_id'],first['plan_version'],first['content_hash'])==(again['task_id'],again['plan_version'],again['content_hash'])
    assert again['replayed']
    with pytest.raises(PlanFlowError,match='cannot change Plan content'):
        service.prepare_plan(str(tmp_path),{**plan,'user_intent':'changed'},request_id='host-call-1')


def test_public_schema_exposes_audit_without_source_reading():
    from data_juicer.tools.plan_flow.server import create_mcp_server
    tools={t.name:t for t in create_mcp_server()._tool_manager.list_tools()}
    schema=tools['prepare_plan'].parameters
    text=json.dumps(schema)
    assert 'ImageAudit' in text and 'expected_outputs' in text and 'postprocess' in text
    recipe_schema = schema['$defs']['Recipe']
    assert recipe_schema['properties']['keep_stats_in_res_ds']['default'] is True
    from data_juicer.tools.plan_flow.operator_catalog_service import schemas
    result=schemas(['image_face_count_filter'])['operators'][0]
    assert result['output_contract']['stats_field']=='__dj__stats__.face_counts'
    assert 'clarity' in result['limitations'][-1]


def test_image_tagging_schema_exposes_machine_readable_response_contract():
    from data_juicer.tools.plan_flow.operator_catalog_service import schemas

    contract = schemas(['image_tagging_vlm_mapper'])['operators'][0]['output_contract']

    assert contract['model_response_schema']['required'] == ['tags']
    assert contract['model_response_schema']['properties']['tags'] == {
        'type': 'array',
        'items': {'type': 'string'},
        'minItems': 1,
        'maxItems': 10,
    }
    assert contract['canonical_example'] == {'tags': ['tag1', 'tag2']}
    assert contract['ordering_guaranteed'] is False
    assert contract['storage_path'] == '__dj__meta__.<tag_field_name>'


def test_plan_rejects_image_tagging_prompt_with_wrong_response_shape(tmp_path):
    from data_juicer.tools.plan_flow.validation import normalize_and_validate

    dataset = tmp_path / 'input.jsonl'
    dataset.write_text(json.dumps({'text': '', 'images': ['one.jpg']}) + '\n', encoding='utf-8')
    plan = {
        'user_intent': 'Classify cat and dog photos',
        'recipe': {
            'dataset_path': str(dataset),
            'export_path': 'result.jsonl',
            'process': [{
                'image_tagging_vlm_mapper': {
                    'system_prompt': 'Return {"animal":"cat","photo_type":"real"}.',
                },
            }],
        },
    }

    _, validation, _ = normalize_and_validate(str(tmp_path), plan)

    mismatch = [item for item in validation['errors'] if item['code'] == 'MODEL_RESPONSE_CONTRACT_MISMATCH']
    assert len(mismatch) == 1
    assert mismatch[0]['path'].endswith('.image_tagging_vlm_mapper.system_prompt')


@pytest.mark.parametrize(
    'system_prompt',
    [
        None,
        'Return only JSON in this exact format: {"tags":["animal-cat","photo-type-real"]}',
        "Return only this shape: {'tags': ['cat', 'real']}",
    ],
)
def test_plan_accepts_default_or_contract_compatible_image_tagging_prompt(tmp_path, system_prompt):
    from data_juicer.tools.plan_flow.validation import normalize_and_validate

    dataset = tmp_path / 'input.jsonl'
    dataset.write_text(json.dumps({'text': '', 'images': ['one.jpg']}) + '\n', encoding='utf-8')
    params = {} if system_prompt is None else {'system_prompt': system_prompt}
    plan = {
        'user_intent': 'Classify cat and dog photos',
        'recipe': {
            'dataset_path': str(dataset),
            'export_path': 'result.jsonl',
            'process': [{'image_tagging_vlm_mapper': params}],
        },
    }

    _, validation, _ = normalize_and_validate(str(tmp_path), plan)

    assert not [item for item in validation['errors'] if item['code'] == 'MODEL_RESPONSE_CONTRACT_MISMATCH']

import json
from pathlib import Path

import pytest

from data_juicer.tools.plan_flow.discovery import inspect_input
from data_juicer.tools.plan_flow.common import PlanFlowError


def test_profile_separates_exact_count_from_sample_statistics(tmp_path):
    source = tmp_path / 'rows.jsonl'
    source.write_text('\n'.join(json.dumps(row) for row in [{'body':'a'}, {'body':None}, {'other':1}]), encoding='utf-8')
    result = inspect_input(str(tmp_path), {'path':'rows.jsonl', 'bindings':{'text_keys':['body']}}, sample_size=2)
    assert result['record_count'] == 3
    assert result['profile']['status'] == 'complete'
    assert result['profile']['sample'] == {'method':'head','limit':2,'size':2,'representative':False,'values_truncated':False}
    assert result['fields'] == ['body']
    assert result['profile']['field_statistics']['fields']['body']['null'] == 1
    assert result['profile']['field_statistics']['scope'] == 'sample'
    assert result['profile']['quality'] == 'not_evaluated'
    assert result['modality'] == 'text'


def test_partial_empty_unsupported_and_malformed_are_distinct(tmp_path):
    (tmp_path / 'rows.jsonl').write_text('{}\n{}\n{}\n')
    result = inspect_input(str(tmp_path), {'path':'rows.jsonl', 'max_scan_records':2})
    assert result['record_count'] is None
    assert result['profile']['status'] == 'partial'
    assert result['profile']['record_count']['observed'] == 2
    (tmp_path / 'empty.jsonl').write_text('')
    result = inspect_input(str(tmp_path), {'path':'empty.jsonl'})
    assert result['profile']['status'] == 'empty'
    assert result['modality'] == 'unknown'
    (tmp_path / 'unknown.bin').write_bytes(b'abc')
    assert inspect_input(str(tmp_path), {'path':'unknown.bin'})['profile']['status'] == 'unsupported'
    (tmp_path / 'bad.jsonl').write_text('not json')
    with pytest.raises(PlanFlowError, match='Cannot profile'):
        inspect_input(str(tmp_path), {'path':'bad.jsonl'})


def test_directory_modality_uses_inventory_outside_head_sample(tmp_path):
    (tmp_path / 'a.jpg').write_bytes(b'fixture')
    (tmp_path / 'z.wav').write_bytes(b'fixture')
    result = inspect_input(str(tmp_path), {'path':'.'}, sample_size=1)
    assert result['modality'] == 'multimodal'
    assert result['record_count'] == 2
    assert result['bindings']['audio_key'] == 'audios'
    assert result['profile']['media_content'] == 'not_decoded'
    # Generated .dj manifests must not be inventoried on the next inspection.
    assert inspect_input(str(tmp_path), {'path':'.'})['record_count'] == 2


def test_parquet_uses_metadata_count_and_bounded_batch(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({'text':['a','b','c']}), tmp_path / 'rows.parquet')
    result = inspect_input(str(tmp_path), {'path':'rows.parquet'}, sample_size=1)
    assert result['record_count'] == 3
    assert result['samples'] == [{'text':'a'}]
    assert result['profile']['field_statistics']['records'] == 1


def test_rank_merge_never_compares_provider_score_scales(monkeypatch):
    from data_juicer.tools.plan_flow import operator_catalog_service as catalog
    def raw(requirements, *args):
        return {'ok':True,'top_k':3,'operators':[{'name':'native','type':'mapper','tags':['text'],'description':'text','match_score':999999,'matched_requirements':[]}],
                'results':[{'requirement':query,'operator_names':['native'],'retrieval':[{'name':'native','rank':1,'method':'bm25','raw_score':999999,'exact':False}]} for query in requirements]}
    monkeypatch.setattr(catalog.discovery,'search_capabilities',raw)
    monkeypatch.setattr(catalog, 'personal',lambda:[{'candidate_id':'user:custom:v1','name':'custom','type':'mapper','tags':['text'],
        'description':'keep text','parameters':{},'provider':'user','status':'validated','version':'v1'}])
    result = catalog.search(['keep text', 'custom'])
    assert result['results'][0]['candidate_ids'] == ['dj:native','user:custom:v1']
    assert result['results'][1]['candidate_ids'][0] == 'user:custom:v1'
    ranks = result['results'][0]['ranking']
    assert ranks[0]['fusion_score'] == ranks[1]['fusion_score']
    assert ranks[0]['raw_score'] != ranks[1]['raw_score']
    assert result['ranking_policy']['method'] == 'reciprocal_rank_merge'


def test_protocol_and_health_are_authenticated_and_instance_bound(tmp_path, monkeypatch):
    from starlette.testclient import TestClient
    from data_juicer.tools.plan_flow.server import create_mcp_server
    monkeypatch.setenv('DSH_DJ_INTERNAL_TOKEN','fixture')
    monkeypatch.setenv('DSH_USER_DATA_ROOT',str(tmp_path / 'users'))
    with TestClient(create_mcp_server().streamable_http_app()) as client:
        assert client.get('/internal/health').status_code == 403
        headers = {'x-dsh-internal-token':'fixture','x-dsh-user-id':'alice'}
        health = client.get('/internal/health',headers=headers).json()
        assert health['ok']
        assert health['layers']['execution']['status'] == 'per_plan'
        listing = client.get('/internal/operator-tools',headers=headers).json()
        request = {'name':'search_capabilities','arguments':{'requirements':[]}}
        assert client.post('/internal/operator-tools',headers=headers,json=request).json()['error']['code'] == 'PROTOCOL_MISMATCH'
        headers['x-dsh-dj-protocol'] = '1'
        headers['x-dsh-dj-instance'] = 'old'
        assert client.post('/internal/operator-tools',headers=headers,json=request).json()['error']['code'] == 'SERVICE_RESTARTED'
        headers['x-dsh-dj-instance'] = listing['instance_id']
        assert client.post('/internal/operator-tools',headers=headers,json=request).json()['ok']
        invalid = {'name':'search_capabilities','arguments':{'requirements':[], 'top_k':{}}}
        assert client.post('/internal/operator-tools',headers=headers,json=invalid).status_code == 400


def test_build_identity_detects_dirty_source_and_model_catalog(tmp_path):
    from data_juicer.tools.plan_flow.deployment import build_identity
    root = tmp_path / 'data_juicer/tools/plan_flow'; root.mkdir(parents=True)
    source = root / 'module.py'; source.write_text('a=1')
    before = build_identity(tmp_path)
    source.write_text('a=2')
    assert build_identity(tmp_path)['source_sha256'] != before['source_sha256']
    (root / 'builtin_model_catalog.json').write_text('{}')
    assert build_identity(tmp_path)['model_catalog_sha256'] is not None


def test_overlay_cache_changes_when_base_runtime_changes(tmp_path, monkeypatch):
    import os
    from data_juicer.tools.plan_flow.user_operator_runtime import runtime_python
    from data_juicer.tools.plan_flow.user_operator_store import UserOperatorStore, current_user
    from data_juicer.tools.plan_flow import deployment
    monkeypatch.setenv('DSH_USER_DATA_ROOT',str(tmp_path / 'users'))
    identity = {'source_sha256':'one', 'runtime_packages_sha256':'base-one'}
    monkeypatch.setattr(deployment, 'build_identity', lambda: dict(identity))
    token = current_user.set('alice')
    try:
        store = UserOperatorStore()
        def create_env(command):
            assert command[1:3] == ['-m','venv']
            target = Path(command[3])
            python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            python.parent.mkdir(parents=True)
            python.touch()
        first = runtime_python(store, [], create_env)
        assert runtime_python(store, []) == first
        identity['runtime_packages_sha256'] = 'base-two'
        with pytest.raises(PlanFlowError, match='Validate this dependency combination'):
            runtime_python(store, [])
        assert runtime_python(store, [], create_env) != first
    finally: current_user.reset(token)


@pytest.mark.parametrize('has_hash', [True, False])
def test_overlay_requires_artifact_identity_and_discards_index_urls(tmp_path, monkeypatch, has_hash):
    import os
    from data_juicer.tools.plan_flow import deployment, user_operator_runtime as runtime
    from data_juicer.tools.plan_flow.user_operator_store import UserOperatorStore, current_user
    monkeypatch.setenv('DSH_USER_DATA_ROOT',str(tmp_path / 'users'))
    monkeypatch.setattr(deployment, 'build_identity', lambda: {'source_sha256':'fixture'})
    real_version = runtime.importlib.metadata.version
    def version(name):
        if name == 'fixture-extra': raise runtime.importlib.metadata.PackageNotFoundError(name)
        return real_version(name)
    monkeypatch.setattr(runtime.importlib.metadata,'version',version)
    token = current_user.set('alice')
    reports = []
    try:
        store = UserOperatorStore()
        def command(args):
            if args[1:3] == ['-m','venv']:
                target=Path(args[3]); python=target / ('Scripts/python.exe' if os.name=='nt' else 'bin/python')
                python.parent.mkdir(parents=True);python.touch()
            elif '--report' in args:
                report=Path(args[args.index('--report')+1]);reports.append(report)
                report.write_text(json.dumps({'install':[{'metadata':{'name':'fixture-extra','version':'1.0'},'download_info':{'url':'https://private-index.example/package','archive_info':{'hashes':{'sha256':'a'*64} if has_hash else {}}}}]}))
        if has_hash:
            runtime.runtime_python(store,['fixture-extra==1.0'],command)
        else:
            with pytest.raises(PlanFlowError, match='each installed artifact'):
                runtime.runtime_python(store,['fixture-extra==1.0'],command)
        assert 'private-index' not in reports[0].read_text()
    finally: current_user.reset(token)

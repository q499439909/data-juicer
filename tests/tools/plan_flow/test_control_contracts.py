import json
from pathlib import Path
from unittest.mock import patch

import pytest

from data_juicer.tools.plan_flow.common import PlanFlowError, read_yaml
from data_juicer.tools.plan_flow.store import PlanStore
from data_juicer.tools.plan_flow.runner import PlanRunner
from data_juicer.tools.plan_flow.task_control import register_workspace, authorize_workspace
from data_juicer.tools.plan_flow.user_operator_store import current_user
from data_juicer.tools.plan_flow.execution import RunHandle, RunStatus


class FakeBackend:
    name = 'fake'
    def __init__(self): self.starts = []
    def start(self, spec):
        self.starts.append(spec)
        return RunHandle(self.name, spec.run_id, spec.created_at, spec.deadline, 'test')
    def inspect(self, handle):
        from datetime import datetime, timezone
        return RunStatus('running', datetime.now(timezone.utc))


def plan_fixture(root):
    source = root / 'input.jsonl'
    media = root / 'a.jpg'
    media.write_bytes(b'original media')
    source.write_text(json.dumps({'text': 'hello', 'images': [str(media)]}) + '\n', encoding='utf-8')
    store = PlanStore(root)
    task, _ = store.create_task('contract')
    saved = store.save_plan(task_id=task, plan={'user_intent':'contract','recipe':{'dataset_path':str(source),'export_path':'${RUN_OUTPUT}/result.jsonl','process':[{'text_length_filter':{'min_len':1}}]},'postprocess':[]}, validation={'ok':True}, artifact_paths=[], base_plan_version=None)
    store.approve(task, saved['plan_version'], saved['content_hash'], 'local test')
    return store, task, saved


def test_native_input_is_frozen_and_tamper_detected(tmp_path):
    store, task, saved = plan_fixture(tmp_path)
    (tmp_path / 'a.jpg').write_bytes(b'changed')
    (tmp_path / 'input.jsonl').write_text('{}\n')
    assert store.verify_bundle(task, saved['plan_version']) == saved['content_hash']
    plan = store.get_plan(task, saved['plan_version'])['plan']
    frozen = Path(plan['recipe']['dataset_path'])
    row = json.loads(frozen.read_text())
    assert Path(row['images'][0]).read_bytes() == b'original media'
    frozen.write_text('{}\n')
    with pytest.raises(PlanFlowError, match='Frozen input'):
        store.verify_bundle(task, saved['plan_version'])


def test_workspace_grant_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setenv('DSH_USER_DATA_ROOT', str(tmp_path / 'users'))
    monkeypatch.setenv('DSH_DJ_ALLOWED_WORKSPACES', str(tmp_path))
    register_workspace(tmp_path, 'alice')
    token = current_user.set('bob')
    try:
        with pytest.raises(PlanFlowError):
            authorize_workspace(tmp_path)
        with pytest.raises(PlanFlowError):
            register_workspace(tmp_path, 'bob')
    finally:
        current_user.reset(token)


def test_owner_cannot_self_approve_without_user_event(tmp_path, monkeypatch):
    store, task, saved = plan_fixture(tmp_path)
    token = current_user.set('alice')
    try:
        with pytest.raises(PlanFlowError, match='user decision'):
            store.approve(task, saved['plan_version'], saved['content_hash'], 'confirmed=true')
    finally:
        current_user.reset(token)


def test_retry_reuses_run_but_new_request_can_rerun(tmp_path):
    store, task, saved = plan_fixture(tmp_path)
    backend = FakeBackend()
    runner = PlanRunner(tmp_path, backend)
    first = runner.start(task, saved['plan_version'], request_id='same')
    replay = runner.start(task, saved['plan_version'], request_id='same')
    assert first['run_id'] == replay['run_id']
    assert len(backend.starts) == 1
    with pytest.raises(PlanFlowError, match='different arguments'):
        runner.start(task, saved['plan_version'], request_id='same', timeout_seconds=2)
    assert runner.start(task, saved['plan_version'], request_id='new')['run_id'] != first['run_id']


def test_deadline_stops_real_process_without_get_run(tmp_path):
    import os, subprocess, sys
    from datetime import datetime, timedelta, timezone
    import psutil
    from data_juicer.tools.plan_flow.execution.supervisor import watch
    from data_juicer.tools.plan_flow.common import write_json_atomic, read_json
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    state = tmp_path / 'run.json'
    record = tmp_path / 'backend.json'
    write_json_atomic(state, {'status':'running'})
    write_json_atomic(record, {'pid':process.pid, 'pid_create_time':psutil.Process(process.pid).create_time(), 'run_state_path':str(state), 'runtime_spec':{'deadline':(datetime.now(timezone.utc)+timedelta(seconds=.3)).isoformat()}})
    try:
        watch(record)
        process.wait(timeout=5)
        result = read_json(state)
        assert result['error_code'] == 'RUN_TIMEOUT'
        assert result['cleanup_pending'] is False
    finally:
        if process.poll() is None: process.kill()


def test_output_contract_rejects_unimplemented_producer_and_reports_missing(tmp_path):
    from data_juicer.tools.plan_flow.delivery import validate_contract, verify_delivery
    plan = {'recipe':{'export_path':'${RUN_OUTPUT}/out.jsonl'}, 'postprocess':[], 'expected_outputs':[{'id':'report','path':'report.json','format':'json','producer':{'kind':'postprocess','index':0}}]}
    errors = []
    validate_contract(plan, errors, required=True)
    assert errors[0]['code'] == 'OUTPUT_PRODUCER_INVALID'
    result = verify_delivery(plan, tmp_path)
    assert result['delivery_status'] == 'failed'
    assert result['task_status'] != 'completed'


def test_quality_without_evidence_remains_unverified(tmp_path):
    from data_juicer.tools.plan_flow.delivery import verify_delivery
    (tmp_path / 'out.jsonl').write_text('{"id":1}\n')
    plan = {'expected_outputs':[{'id':'corpus','path':'out.jsonl','format':'jsonl','required_fields':['id']}], 'acceptance_criteria':['images look good']}
    result = verify_delivery(plan, tmp_path)
    assert result['delivery_status'] == 'passed'
    assert result['acceptance_status'] == 'unverified'


def test_http_approval_requires_trusted_route_and_resource_owner(tmp_path, monkeypatch):
    import uuid
    from starlette.testclient import TestClient
    from data_juicer.tools.plan_flow import server
    from data_juicer.tools.plan_flow.service import PlanFlowService
    monkeypatch.setenv('DSH_USER_DATA_ROOT', str(tmp_path / 'users'))
    monkeypatch.setenv('DSH_DJ_ALLOWED_WORKSPACES', str(tmp_path))
    monkeypatch.setenv('DSH_DJ_INTERNAL_TOKEN', 'test-control-token')
    register_workspace(tmp_path, 'alice')
    token = current_user.set('alice')
    try:
        source = tmp_path / 'input.jsonl'
        source.write_text('{"text":"hello"}\n')
        service = PlanFlowService.native()
        monkeypatch.setattr(server, 'service', service)
        prepared = service.prepare_plan(str(tmp_path), {'user_intent':'test','recipe':{'dataset_path':str(source),'process':[{'text_length_filter':{}}],'export_path':'out.jsonl'},'expected_outputs':[{'id':'corpus','path':'out.jsonl','format':'jsonl','producer':{'kind':'recipe'}}]})
        assert prepared['validation']['ok'], prepared
    finally:
        current_user.reset(token)
    args = {k:prepared[k] for k in ('workspace_root','task_id','plan_version','content_hash')}
    headers = {'x-dsh-internal-token':'test-control-token','x-dsh-user-id':'alice'}
    with TestClient(server.create_mcp_server().streamable_http_app()) as client:
        assert client.post('/internal/operator-tools',headers=headers,json={'name':'approve_plan','arguments':args}).status_code == 403
        decision = {**args,'decision_id':str(uuid.uuid4())}
        assert client.post('/internal/plan-decision',json=decision).status_code == 403
        bad = {**headers,'x-dsh-user-id':'bob'}
        assert client.post('/internal/plan-decision',headers=bad,json=decision).status_code == 403
        assert client.get('/plan-view',headers=bad,params=args).status_code != 200
        assert client.post('/internal/plan-decision',headers=headers,json=decision).json()['approval']['actor'] == 'alice'
        decision['content_hash'] = 'sha256:wrong'
        assert client.post('/internal/plan-decision',headers=headers,json=decision).status_code == 403


def test_cancel_waits_for_real_process_tree(tmp_path):
    import subprocess, sys, time, psutil
    from data_juicer.tools.plan_flow.execution.supervisor import stop_record
    from data_juicer.tools.plan_flow.common import write_json_atomic, read_json
    child_file = tmp_path / 'child.pid'
    code = "import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    process = subprocess.Popen([sys.executable, '-c', code, str(child_file)], creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        until = time.monotonic() + 5
        while not child_file.exists() and time.monotonic() < until: time.sleep(.05)
        child = psutil.Process(int(child_file.read_text()))
        state, record = tmp_path / 'run.json', tmp_path / 'backend.json'
        write_json_atomic(state, {'status':'running'})
        write_json_atomic(record, {'pid':process.pid,'pid_create_time':psutil.Process(process.pid).create_time(),'run_state_path':str(state)})
        stop_record(record)
        process.wait(timeout=5)
        assert not child.is_running()
        assert read_json(state)['status'] == 'cancelled'
        assert read_json(state)['cleanup_pending'] is False
    finally:
        if process.poll() is None: process.kill()


def test_develop_retry_and_budget_are_persistent(tmp_path, monkeypatch):
    from data_juicer.tools.plan_flow.user_operator_validation import UserOperatorValidation
    monkeypatch.setenv('DSH_USER_DATA_ROOT', str(tmp_path / 'users'))
    token = current_user.set('alice')
    validator = UserOperatorValidation()
    calls = []
    def develop(*args):
        calls.append(args)
        return {'ok':True,'job':{'job_id':'job_test'}}
    monkeypatch.setattr(validator, '_develop', develop)
    monkeypatch.setattr(validator, 'get', lambda job_id: {'ok':True,'job':{'job_id':job_id}})
    try:
        for _ in range(2): validator.develop({'name':'a'}, [], request_id='one', task_id='task')
        assert len(calls) == 1
        with pytest.raises(PlanFlowError, match='different arguments'):
            validator.develop({'name':'changed'}, [], request_id='one', task_id='task')
        validator.develop({'name':'a'}, [], request_id='two', task_id='task')
        validator.develop({'name':'b'}, [], request_id='three', task_id='task')
        with pytest.raises(PlanFlowError, match='Three attempts'):
            validator.develop({'name':'c'}, [], request_id='four', task_id='task')
        assert len(calls) == 3
    finally: current_user.reset(token)


@pytest.mark.parametrize('cancelled', [False, True])
def test_model_prepare_obeys_shared_deadline_and_persisted_cancel(tmp_path, monkeypatch, cancelled):
    import subprocess, sys, threading
    from datetime import datetime, timedelta, timezone
    from data_juicer.tools.plan_flow.common import write_json_atomic, read_json
    from data_juicer.tools.plan_flow.user_operator_store import UserOperatorStore
    from data_juicer.tools.plan_flow.user_operator_validation import UserOperatorValidation
    monkeypatch.setenv('DSH_USER_DATA_ROOT', str(tmp_path / 'users'))
    token = current_user.set('alice')
    spawned = []
    real_popen = subprocess.Popen
    def slow_model(command, **kwargs):
        assert 'data_juicer.tools.plan_flow.user_operator_prepare' in command
        process = real_popen([sys.executable,'-c','import time;time.sleep(60)'], **kwargs)
        spawned.append(process)
        return process
    monkeypatch.setattr(subprocess, 'Popen', slow_model)
    try:
        store = UserOperatorStore()
        temp = store.path(store.home / 'operator_tmp' / 'job_test'); temp.mkdir(parents=True)
        path = store.path(store.home / 'operator_jobs' / 'job_test.json')
        job = {'job_id':'job_test','status':'testing','cleanup_pending':True,'deadline':(datetime.now(timezone.utc)+timedelta(seconds=.4)).isoformat()}
        write_json_atomic(path, job)
        if cancelled: path.with_suffix('.cancel').write_text('cancel')
        UserOperatorValidation()._run(store,temp,path,job,'',{'model_refs':[{'model_id':'test'}]}, {},180,threading.Event())
        final = read_json(path)
        assert final['status'] == ('cancelled' if cancelled else 'failed')
        assert final['error_details']['phase'] == 'model_prepare'
        assert final['cleanup_pending'] is False
        assert not temp.exists()
        assert all(process.poll() is not None for process in spawned)
        assert len(spawned) == (0 if cancelled else 1)
    finally: current_user.reset(token)


def test_workspace_grants_cannot_overlap_across_accounts(tmp_path, monkeypatch):
    import os
    child = tmp_path / 'nested'; child.mkdir()
    monkeypatch.setenv('DSH_USER_DATA_ROOT', str(tmp_path / 'users'))
    monkeypatch.setenv('DSH_DJ_ALLOWED_WORKSPACES', os.pathsep.join([str(tmp_path),str(child)]))
    register_workspace(tmp_path, 'alice')
    with pytest.raises(PlanFlowError, match='Overlapping'):
        register_workspace(child, 'bob')


def test_late_worker_cannot_overwrite_cancelled_state(tmp_path):
    from data_juicer.tools.plan_flow.execution.local_worker import _write_worker_state
    from data_juicer.tools.plan_flow.common import read_json, write_json_atomic
    path = tmp_path / 'run.json'
    write_json_atomic(path, {'status':'cancelled','cleanup_pending':False})
    _write_worker_state(path, {'status':'succeeded'})
    assert read_json(path)['status'] == 'cancelled'


def test_operator_watchdog_cleans_after_supervisor_loss(tmp_path, monkeypatch):
    from data_juicer.tools.plan_flow.user_operator_store import UserOperatorStore
    from data_juicer.tools.plan_flow.user_operator_validation import UserOperatorValidation
    from data_juicer.tools.plan_flow.operator_watchdog import watch
    from data_juicer.tools.plan_flow.common import read_json, write_json_atomic
    monkeypatch.setenv('DSH_USER_DATA_ROOT', str(tmp_path / 'users'))
    token = current_user.set('alice')
    try:
        store = UserOperatorStore()
        temp = store.path(store.home / 'operator_tmp' / 'job_lost'); temp.mkdir(parents=True)
        (temp / 'partial.txt').write_text('partial')
        path = store.path(store.home / 'operator_jobs' / 'job_lost.json')
        write_json_atomic(path, {'status':'testing','cleanup_pending':True,'supervisor_pid':99999999})
        monkeypatch.setattr(UserOperatorValidation, '_same_process', staticmethod(lambda *args: False))
        watch(store.root, store.user_id, 'job_lost')
        state = read_json(path)
        assert state['status'] == 'failed'
        assert not state['cleanup_pending']
        assert not temp.exists()
    finally: current_user.reset(token)

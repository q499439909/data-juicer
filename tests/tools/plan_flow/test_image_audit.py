import json
import pytest
from data_juicer.tools.plan_flow.common import PlanFlowError
from data_juicer.tools.plan_flow.image_audit import run_image_audit
from data_juicer.tools.plan_flow.presentation import build_plan_view
from data_juicer.tools.plan_flow.plan_contract import PlanDraft


def fixture(tmp_path, rows):
    source=tmp_path/'scores.jsonl'; manifest=tmp_path/'input.jsonl'; out=tmp_path/'output';out.mkdir()
    for row in rows:
        for image in row['images']: (tmp_path/image).write_bytes(b'image-fixture')
        row['images']=[str(tmp_path/p) for p in row['images']]
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    manifest.write_text(''.join(json.dumps({'images':r['images']})+'\n' for r in rows),encoding='utf-8')
    return source,out,manifest


def test_per_image_scores_all_reasons_and_stage_conservation(tmp_path):
    source,out,manifest=fixture(tmp_path,[{'images':['a.jpg','b.jpg','c.jpg'],'__dj__stats__':{'faces':[0,1,1],'ratio':[0,.2,.9]}}])
    step={'kind':'image_audit','copy_kept':True,'rules':[{'id':'face','field':'faces','min':1,'max':1},{'id':'area','field':'ratio','min':.15,'max':.6}]}
    summary=run_image_audit(step,source,out,manifest)['summary']
    assert summary['input']==summary['kept']+summary['removed']==3
    assert summary['kept']==1
    assert [s['entered'] for s in summary['stages']]==[3,2]
    assert all(s['entered']==s['removed']+s['remaining'] for s in summary['stages'])
    rows=[json.loads(x) for x in (out/'audit/decisions.jsonl').read_text().splitlines()]
    assert rows[0]['all_reasons']==['face','area']
    assert len(list((out/'audit/images').iterdir()))==1


@pytest.mark.parametrize('rows',[[],[{'images':['a.jpg'],'__dj__stats__':{'faces':[0]}}]])
def test_empty_and_zero_retained_are_valid(tmp_path,rows):
    args=fixture(tmp_path,rows)
    result=run_image_audit({'kind':'image_audit','rules':[{'id':'face','field':'faces','min':1}]},*args)
    assert result['summary']['kept']==0


def test_missing_score_is_never_zero_or_implicit_pass(tmp_path):
    args=fixture(tmp_path,[{'images':['a.jpg'],'__dj__stats__':{'faces':[0]}}])
    rules=[{'id':'face','field':'faces','min':1},{'id':'clear','field':'clarity','min':.9}]
    with pytest.raises(PlanFlowError,match='Missing numeric clarity'):
        run_image_audit({'kind':'image_audit','rules':rules},*args)
    out=tmp_path/'cascade';out.mkdir()
    run_image_audit({'kind':'image_audit','rules':rules,'audit_mode':'cascade'},args[0],out,args[2])
    row=json.loads((out/'audit/decisions.jsonl').read_text())
    assert row['scores']['clarity'] is None
    assert row['checks']['clear']=='not_evaluated'
    assert row['all_reasons_complete'] is False


def test_plan_contract_and_view_include_decision_report_copy(tmp_path):
    plan=PlanDraft.model_validate({'user_intent':'audit','recipe':{'process':[{'image_face_count_filter':{}}]},'postprocess':[{'kind':'image_audit','rules':[{'id':'faces','field':'face_counts','min':1}]}]}).model_dump()
    view,_=build_plan_view(plan,'sha256:x')
    assert view['steps'][0]['mode']=='score_only'
    assert view['steps'][1]['configuration']['kind']=='image_audit'


def test_exclusive_boundary_has_one_decision_and_report(tmp_path):
    from data_juicer.tools.plan_flow.audit_report import verify_audit
    source,out,manifest=fixture(tmp_path,[{'images':['watermark.jpg'],'__dj__stats__':{'watermark':[.8]}}])
    step={'kind':'image_audit','copy_kept':True,'rules':[{'id':'wm','field':'watermark','max':.8,'max_inclusive':False}]}
    result=run_image_audit(step,source,out,manifest,coverage=[{'requirement':'clear face','status':'gap','evidence':'not measured'}])
    assert result['summary']['kept']==0
    assert (out/'audit/kept.jsonl').read_text()==''
    report=(out/'audit/report.md').read_text(encoding='utf-8')
    assert '<0.8' in report and 'clear face：gap' in report
    assert verify_audit(out/'audit/summary.json',out)['status']=='passed'
    summary=json.loads((out/'audit/summary.json').read_text())
    summary['stages'][0]['removed']=0
    (out/'audit/summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='stage counts'):verify_audit(out/'audit/summary.json',out)


def test_native_scoring_preserves_rejected_records(tmp_path,monkeypatch):
    from data_juicer.tools.plan_flow.image_audit import score_audit_input
    from data_juicer.config import init_configs
    from data_juicer.tools.plan_flow.common import write_yaml_atomic
    source,out,manifest=fixture(tmp_path,[{'text':'a','images':['a.jpg'],'__dj__stats__':{'face_counts':[99]}},
                                          {'text':'b','images':['b.jpg'],'__dj__stats__':{'face_counts':[99]}}])
    config=tmp_path/'recipe.yaml'
    write_yaml_atomic(config,{'dataset':{'configs':[{'type':'local','path':str(source)}]},'export_path':str(out/'scored.jsonl'),
        'work_dir':str(tmp_path/'work'),'np':1,'keep_stats_in_res_ds':True,'use_cache':False,
        'process':[{'image_face_count_filter':{'min_face_count':1,'max_face_count':1}}]})
    cfg=init_configs(['--config',str(config)],load_configs_only=False)
    from data_juicer.ops.base_op import Filter
    import data_juicer.ops
    class FixtureScore(Filter):
        _name='fixture_score_filter'
        def compute_stats_single(self,sample,context=False):
            if 'face_counts' in sample['__dj__stats__']: return sample
            sample['__dj__stats__']['face_counts']=[0 if sample['text']=='a' else 1]
            return sample
        def process_single(self,sample):
            return sample['__dj__stats__']['face_counts'][0]==1
    monkeypatch.setattr(data_juicer.ops,'load_ops',lambda process:[FixtureScore()])
    transitions=[]
    score_audit_input(cfg,lambda steps:transitions.append(json.loads(json.dumps(steps))),source)
    rows=[json.loads(line) for line in (out/'scored.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==2, 'The zero-face record must survive scoring so it gets a decision and deletion reason'
    assert [r['__dj__stats__']['face_counts'][0] for r in rows]==[0,1], 'Untrusted input scores must not bypass computation'
    assert transitions[-1][0]['status']=='succeeded'

def test_consistency_rejects_changed_scores_and_accepts_cascade_counts(tmp_path):
    from data_juicer.tools.plan_flow.audit_report import verify_audit
    source,out,manifest=fixture(tmp_path,[{'images':['a.jpg'],'__dj__stats__':{'a':[2],'b':[2]}}])
    run_image_audit({'kind':'image_audit','audit_mode':'cascade','rules':[{'id':'a','field':'a','max':1},{'id':'b','field':'b','max':1}]},source,out,manifest)
    assert verify_audit(out/'audit/summary.json',out)['status']=='passed'
    path=out/'audit/decisions.jsonl';row=json.loads(path.read_text());row['scores']['a']=0
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='recorded score'):verify_audit(out/'audit/summary.json',out)

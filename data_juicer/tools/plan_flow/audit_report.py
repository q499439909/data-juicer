"""Presentation and independent structural checks of the one audit decision stream."""
import json
from pathlib import Path
from .plan_contract import AuditDecision, AuditSummary


def _cell(value):
    return str(value).replace('|','\\|').replace('\n',' ')


def render_report(target, coverage):
    target=Path(target)
    summary=json.loads((target/'summary.json').read_text(encoding='utf-8'))
    lines=['# 图片处理报告','',f"输入 {summary['input']}；保留 {summary['kept']}；排除 {summary['removed']}。",'',
           '本报告仅展示同目录 decisions.jsonl 和 summary.json 中的实际结果，不重新判定。','',
           '## 阶段计数','','| 规则 | 进入 | 排除 | 剩余 | 独立命中 |','|---|---:|---:|---:|---:|']
    for stage in summary['stages']:lines.append('| '+' | '.join(_cell(stage[k]) for k in ('id','entered','removed','remaining','independent_hits'))+' |')
    lines+=['','## 规则','']
    for rule in summary['rules']:
        bounds=[]
        if rule.get('min') is not None:bounds.append(('≥' if rule.get('min_inclusive',True) else '>')+str(rule['min']))
        if rule.get('max') is not None:bounds.append(('≤' if rule.get('max_inclusive',True) else '<')+str(rule['max']))
        lines.append(f"- {_cell(rule['id'])}：{_cell(rule['field'])} {' 且 '.join(bounds)}")
    lines+=['','## 需求覆盖','']
    if not coverage:lines.append('未提供语义验收证据。检测到人脸不等于已验证清晰度。')
    for c in coverage:lines.append(f"- {_cell(c['requirement'])}：{_cell(c['status'])}；{_cell(c.get('evidence',''))}")
    lines+=['','## 逐图结果','','| 图片 | 保留 | 实际分数 | 排除原因 |','|---|---|---|---|']
    with (target/'decisions.jsonl').open(encoding='utf-8') as handle:
        for i,line in enumerate(handle):
            if i>=1000:lines+=['','表格仅展示前 1000 项；完整结果见 decisions.jsonl。'];break
            d=json.loads(line)
            lines.append('| '+' | '.join(_cell(v) for v in (Path(d['source']).name,'是' if d['kept'] else '否',json.dumps(d['scores'],ensure_ascii=False),', '.join(d['all_reasons']) or '无'))+' |')
    (target/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def verify_audit(summary_path, output_root):
    from .image_audit import rule_violation
    path=Path(summary_path);root=Path(output_root).resolve()
    summary=AuditSummary.model_validate_json(path.read_text(encoding='utf-8'))
    def read(name):
        file=path.parent/name
        if file.is_symlink() or not file.resolve().is_relative_to(root):raise ValueError('Unsafe audit output')
        return [AuditDecision.model_validate_json(line) for line in file.read_text(encoding='utf-8').splitlines() if line.strip()]
    decisions=read('decisions.jsonl');kept=read('kept.jsonl')
    ids=[r.id for r in summary.rules]
    if [s.id for s in summary.stages]!=ids:raise ValueError('Audit stages differ from declared rules')
    if summary.input!=len(decisions) or summary.kept!=len(kept) or summary.input!=summary.kept+summary.removed:raise ValueError('Audit counts do not conserve input')
    if len({d.image_id for d in decisions})!=len(decisions):raise ValueError('Duplicate audit image identity')
    if [d.model_dump() for d in decisions if d.kept]!=[d.model_dump() for d in kept]:raise ValueError('Retained manifest differs from decisions')
    for d in decisions:
        if set(d.checks)!=set(ids):raise ValueError('Audit decision checks differ from declared rules')
        if set(d.scores)!={r.field for r in summary.rules}:raise ValueError('Audit scores differ from declared rules')
        rejected=False
        for rule in summary.rules:
            expected='not_evaluated' if summary.audit_mode=='cascade' and rejected else 'failed' if rule_violation(rule,d.scores[rule.field]) else 'passed'
            if d.checks[rule.id]!=expected:raise ValueError('Audit decision differs from its recorded score and rule')
            rejected=rejected or expected=='failed'
        failed=[k for k,v in d.checks.items() if v=='failed']
        if failed!=d.all_reasons or d.kept==bool(failed) or d.first_reason!=(failed[0] if failed else None):raise ValueError('Inconsistent decision reasons')
        if summary.audit_mode=='full' and (any(v is None for v in d.scores.values()) or any(v=='not_evaluated' for v in d.checks.values())):raise ValueError('Incomplete full audit scores')
        if d.output_image:
            copy=root/d.output_image
            if not copy.resolve().is_relative_to(root) or copy.is_symlink() or not copy.is_file():raise ValueError('Retained image copy missing or unsafe')
    remaining=decisions
    for stage in summary.stages:
        rule=next(r for r in summary.rules if r.id==stage.id)
        if stage.independent_hits!=sum(rule_violation(rule,d.scores[rule.field]) for d in decisions):raise ValueError('Independent hit count differs from decisions')
        removed=[d for d in remaining if d.checks.get(stage.id)=='failed']
        if (stage.entered,stage.removed,stage.remaining)!=(len(remaining),len(removed),len(remaining)-len(removed)):raise ValueError('Audit stage counts disagree with decisions')
        remaining=[d for d in remaining if d.checks.get(stage.id)!='failed']
    return {'status':'passed','input':summary.input,'kept':summary.kept,'removed':summary.removed}

"""Deterministic per-image decisions and reports inside an approved Run."""
import hashlib
import json
import math
import shutil
from pathlib import Path
from .common import PlanFlowError, is_within, write_json_atomic
from .plan_contract import ImageAudit


def rule_violation(rule, value):
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
        return False
    return ((rule.min is not None and (value < rule.min or (not rule.min_inclusive and value == rule.min))) or
            (rule.max is not None and (value > rule.max or (not rule.max_inclusive and value == rule.max))))


def score_audit_input(cfg, progress, input_manifest):
    from data_juicer.core.executor.default_executor import DefaultExecutor
    from data_juicer.ops import load_ops
    from data_juicer.ops.base_op import Filter
    executor=DefaultExecutor(cfg)
    if not Path(input_manifest).read_text(encoding='utf-8').strip():
        Path(cfg.export_path).write_text('',encoding='utf-8')
        progress([{'process_index':i,'operator_name':next(iter(op)),'status':'skipped','reason':'empty_input'} for i,op in enumerate(cfg.process)])
        return
    dataset=executor.dataset_builder.load_dataset(num_proc=cfg.np or 1)
    # Input JSONL is data, not previously validated model evidence. Filters skip
    # computation when a stats field is already present, so remove supplied stats.
    if '__dj__stats__' in dataset.column_names:
        dataset=dataset.remove_columns(['__dj__stats__'])
    stages=[]
    for index,op in enumerate(load_ops(cfg.process)):
        if not isinstance(op,Filter):
            raise PlanFlowError('INVALID_IMAGE_AUDIT','Only Filter scoring is supported')
        stage={'process_index':index,'operator_name':op._name,'status':'running','mode':'score_only'}
        stages.append(stage);progress(stages)
        dataset=op.run(dataset,exporter=executor.exporter,reduce=False)
        stage['status']='succeeded';progress(stages)
    executor.exporter.export(dataset)


def run_image_audit(step, source, output_root, input_manifest, *, coverage=None):
    config = ImageAudit.model_validate(step)
    root = Path(output_root).resolve()
    target = (root / config.output_prefix).resolve()
    if not is_within(target, root):
        raise PlanFlowError('PATH_NOT_ALLOWED', 'Audit output escaped Run')
    target.mkdir(parents=True, exist_ok=False)
    if config.copy_kept:
        (target / 'images').mkdir()
    # Only media paths from this frozen input can be copied, never arbitrary
    # model-supplied paths or another task's internals.
    allowed = {}
    with Path(input_manifest).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip(): continue
            row = json.loads(line)
            if not isinstance(row,dict) or not isinstance(row.get(config.image_key),list):
                raise PlanFlowError('AUDIT_INPUT_INVALID','Frozen input must explicitly contain the configured image list')
            for raw in row.get(config.image_key, []):
                path = Path(raw).resolve()
                allowed[str(path)] = allowed.get(str(path), 0) + 1
    expected = sum(allowed.values())
    stages = [{'id':r.id, 'entered':0, 'removed':0, 'remaining':0, 'independent_hits':0} for r in config.rules]
    seen, kept, count = {}, 0, 0
    with Path(source).open(encoding='utf-8') as inp, (target/'decisions.jsonl').open('w',encoding='utf-8') as decisions, (target/'kept.jsonl').open('w',encoding='utf-8') as retained:
        for line in inp:
            if not line.strip(): continue
            row = json.loads(line)
            images = row.get(config.image_key, [])
            stats = row.get('__dj__stats__', {})
            if not isinstance(images, list) or not isinstance(stats, dict):
                raise PlanFlowError('AUDIT_INPUT_INVALID', 'Expected image list and stats object')
            for index, raw in enumerate(images):
                path = Path(raw).resolve()
                occurrence = seen.get(str(path), 0)
                seen[str(path)] = occurrence + 1
                if seen[str(path)] > allowed.get(str(path), 0):
                    raise PlanFlowError('AUDIT_INPUT_INVALID', 'Recipe contains media not present in frozen input')
                # Frozen media path contains its content digest; remove version
                # directory from identity so the same input survives replanning.
                identity='/'.join(path.parts[-2:])
                identifier = hashlib.sha256(f'{identity}\0{occurrence}'.encode()).hexdigest()
                count += 1
                scores, checks, reasons = {}, {}, []
                for rule, stage in zip(config.rules, stages):
                    values = stats.get(rule.field)
                    value = values[index] if isinstance(values, list) and len(values)==len(images) else None
                    scores[rule.field] = value
                    numeric = isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value)
                    if not numeric and not (config.audit_mode=='cascade' and reasons):
                        raise PlanFlowError('AUDIT_SCORE_MISSING', f'Missing numeric {rule.field} for image {identifier}')
                    violated = numeric and rule_violation(rule, value)
                    if violated: stage['independent_hits'] += 1
                    if config.audit_mode=='cascade' and reasons:
                        checks[rule.id] = 'not_evaluated'
                        continue
                    checks[rule.id] = 'failed' if violated else 'passed'
                    if not reasons:
                        stage['entered'] += 1
                        stage['removed' if violated else 'remaining'] += 1
                    if violated: reasons.append(rule.id)
                record = {'image_id':identifier, 'source':str(path), 'image_index':index, 'scores':scores,
                          'checks':checks, 'kept':not reasons, 'first_reason':reasons[0] if reasons else None,
                          'all_reasons':reasons, 'all_reasons_complete':config.audit_mode=='full'}
                if not reasons:
                    kept += 1
                    if config.copy_kept:
                        destination=target/'images'/f'{identifier}{path.suffix}'
                        shutil.copyfile(path,destination)
                        record['output_image']=str(destination.relative_to(root))
                    retained.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                decisions.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    if count != expected:
        raise PlanFlowError('AUDIT_INPUT_INCOMPLETE', f'Expected {expected} images, observed {count}; audit must retain the original population')
    summary={'schema_version':1,'audit_mode':config.audit_mode,'input':count,'kept':kept,'removed':count-kept,
             'stages':stages,'rules':[r.model_dump() for r in config.rules], 'zero_retained_valid':True,
             'independent_hits_complete':config.audit_mode=='full'}
    write_json_atomic(target/'summary.json',summary)
    from .audit_report import render_report
    render_report(target,coverage or [])
    return {'kind':'image_audit','status':'succeeded','summary':summary,'output':str(target)}

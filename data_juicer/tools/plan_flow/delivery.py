"""Structural output contracts and independently reported acceptance evidence."""
import json
from pathlib import Path, PurePosixPath

from .common import is_within


def validate_contract(plan, errors, *, required=False):
    outputs = plan.get('expected_outputs', [])
    if not isinstance(outputs, list) or (required and not outputs):
        errors.append({'code':'OUTPUT_CONTRACT_REQUIRED','path':'expected_outputs','message':'Declare every required output and its recipe/postprocess producer'})
        return
    ids = set()
    for i, item in enumerate(outputs):
        valid = isinstance(item, dict)
        if valid:
            name = item.get('id')
            raw = str(item.get('path', ''))
            path = PurePosixPath(raw)
            producer = item.get('producer', {})
            valid = bool(isinstance(name, str) and name and name not in ids and raw and raw != '.' and not path.is_absolute() and '..' not in path.parts and ':' not in raw and '\\' not in raw)
            if isinstance(name, str): ids.add(name)
            if not isinstance(producer, dict):
                valid = False
            elif producer.get('kind') == 'recipe':
                valid = valid and raw == str(plan.get('recipe', {}).get('export_path', '')).removeprefix('${RUN_OUTPUT}/')
            elif producer.get('kind') == 'postprocess':
                index = producer.get('index')
                valid = valid and type(index) is int and 0 <= index < len(plan.get('postprocess', []))
            else:
                valid = False
            valid = valid and item.get('format', 'file') in ('file','json','jsonl','directory')
            for number_key in ('min_records', 'min_files'):
                if number_key in item and (type(item[number_key]) is not int or item[number_key] < 0): valid = False
            if not isinstance(item.get('required_fields', []), list) or any(not isinstance(f, str) for f in item.get('required_fields', [])): valid = False
        if not valid:
            errors.append({'code':'OUTPUT_PRODUCER_INVALID','path':f'expected_outputs[{i}]','message':'Output needs a safe relative path and a real matching producer'})
    checks = plan.get('acceptance_checks', [])
    if not isinstance(checks, list):
        errors.append({'code':'INVALID_ACCEPTANCE_CHECK','message':'acceptance_checks must be a list'})
        return
    check_ids = set()
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get('id'), str) or not check.get('id') or check['id'] in check_ids or check.get('kind') not in ('row_count','field_equals','audit_consistency','manual') or (check.get('kind') != 'manual' and (not isinstance(check.get('output_id'), str) or check.get('output_id') not in ids)):
            errors.append({'code':'INVALID_ACCEPTANCE_CHECK','message':'Each check needs a unique id, supported kind and declared output'})
        else:
            check_ids.add(check['id'])
            if check.get('criterion') and plan.get('acceptance_criteria') and check['criterion'] not in plan['acceptance_criteria']:
                errors.append({'code':'ACCEPTANCE_CRITERION_UNBOUND','path':f"acceptance_checks.{check['id']}",'message':'criterion must match an existing acceptance_criteria entry exactly'})
            if check['kind']=='audit_consistency':
                output=next(o for o in outputs if o['id']==check['output_id'])
                producer=output.get('producer',{})
                index=producer.get('index')
                step=plan.get('postprocess',[])[index] if type(index) is int and 0<=index<len(plan.get('postprocess',[])) else {}
                if producer.get('kind')!='postprocess' or step.get('kind')!='image_audit' or output['path']!=step.get('output_prefix','audit')+'/summary.json':
                    errors.append({'code':'INVALID_AUDIT_CHECK','message':'audit_consistency must reference the native audit summary output'})
            if check['kind'] == 'row_count' and (any(type(check[k]) is not int or check[k] < 0 for k in ('min','max') if k in check) or check.get('min',0) > check.get('max',float('inf'))):
                errors.append({'code':'INVALID_ACCEPTANCE_CHECK','message':'Row count bounds must be nonnegative integers'})
            if check['kind'] == 'field_equals' and (not isinstance(check.get('field'), str) or not check.get('field') or 'value' not in check):
                errors.append({'code':'INVALID_ACCEPTANCE_CHECK','message':'field_equals requires field and value'})


def verify_delivery(plan, output_root):
    root = Path(output_root).resolve()
    output_results, data = [], {}
    for item in plan.get('expected_outputs', []):
        result = {'id':item['id'], 'path':item['path'], 'status':'passed'}
        path = (root / item['path']).resolve()
        try:
            if not is_within(path, root) or not path.exists() or path.is_symlink():
                raise ValueError('Required output is missing or unsafe')
            fmt = item.get('format', 'file')
            if fmt == 'directory':
                if not path.is_dir():
                    raise ValueError('Expected a directory')
                files = [p for p in path.rglob('*') if p.is_file()]
                if any(p.is_symlink() or not is_within(p, root) for p in files) or len(files) < item.get('min_files', 0):
                    raise ValueError('Directory does not satisfy file inventory')
            elif not path.is_file():
                raise ValueError('Expected a file')
            elif fmt in {'json','jsonl'}:
                with path.open(encoding='utf-8') as handle:
                    records = [json.loads(line) for line in handle if line.strip()] if fmt == 'jsonl' else json.load(handle)
                if not isinstance(records, list):
                    records = [records]
                if len(records) < item.get('min_records', 0):
                    raise ValueError('Too few output records')
                for row in records:
                    if any(not isinstance(row, dict) or field not in row for field in item.get('required_fields', [])):
                        raise ValueError('Required output field is missing')
                data[item['id']] = records
                result['record_count'] = len(records)
        except (ValueError, OSError, TypeError) as exc:
            result.update(status='failed', reason=str(exc))
        output_results.append(result)
    acceptance = []
    for check in plan.get('acceptance_checks', []):
        result = {'id':check['id'], 'status':'unverified', 'criterion':check.get('criterion')}
        records = data.get(check.get('output_id'))
        if check['kind'] != 'manual' and records is not None:
            if check['kind'] == 'row_count':
                value = len(records)
                passed = check.get('min', 0) <= value <= check.get('max', float('inf'))
                result['value'] = value
            elif check['kind']=='audit_consistency':
                from .audit_report import verify_audit
                output=next(o for o in plan['expected_outputs'] if o['id']==check['output_id'])
                try:
                    result['evidence']=verify_audit(root/output['path'],root);passed=True
                except (ValueError,OSError,TypeError,KeyError) as exc:
                    result['reason']=str(exc);passed=False
            else:
                passed = bool(records) and all(isinstance(row, dict) and row.get(check.get('field')) == check.get('value') for row in records)
            result['status'] = 'passed' if passed else 'failed'
        acceptance.append(result)
    covered = {item.get('criterion') for item in acceptance if isinstance(item.get('criterion'), str)}
    for index, criterion in enumerate(plan.get('acceptance_criteria', [])):
        if str(criterion) not in covered:
            acceptance.append({'id':f'criterion-{index+1}', 'criterion':str(criterion), 'status':'unverified'})
    delivery = 'passed' if output_results and all(item['status'] == 'passed' for item in output_results) else 'failed' if any(item['status'] == 'failed' for item in output_results) else 'unverified'
    status = 'failed' if any(item['status'] == 'failed' for item in acceptance) else 'unverified' if not acceptance or any(item['status'] == 'unverified' for item in acceptance) else 'passed'
    return {'delivery_status':delivery, 'acceptance_status':status, 'task_status':'completed' if delivery == status == 'passed' else 'needs_review', 'outputs':output_results, 'checks':acceptance}

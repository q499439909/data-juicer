"""Bounded input profiling. A head sample is never whole-dataset quality evidence."""
from __future__ import annotations

import csv
import io
import json
import os
import uuid
from pathlib import Path

from .common import PlanFlowError, is_within, require_workspace, resolve_workspace_path, sha256_file, write_json_atomic

MEDIA = {
    'image': {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff'},
    'video': {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'},
    'audio': {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'},
}
MAX_BYTES = 8 * 1024 * 1024
MAX_RECORDS = 100000


def _profile(samples, *, scanned, total, complete, supported, limit, bindings=None, modalities=None, warnings=None):
    fields = sorted({key for row in samples if isinstance(row, dict) for key in row})
    inferred = {}
    for binding, choices in [('image_key', ['images', 'image', 'image_path']),
                             ('video_key', ['videos', 'video', 'video_path']),
                             ('audio_key', ['audios', 'audio', 'audio_path'])]:
        found = next((field for field in choices if field in fields), None)
        if found: inferred[binding] = found
    if 'text' in fields: inferred['text_keys'] = ['text']
    selected = bindings if bindings is not None else inferred
    if not isinstance(selected, dict) or any(key not in {'text_keys', 'image_key', 'video_key', 'audio_key'} for key in selected):
        raise PlanFlowError('INVALID_INPUT_BINDINGS', 'Use text_keys, image_key, video_key and audio_key')
    for key, value in selected.items():
        if (key == 'text_keys' and (not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value))) or (key != 'text_keys' and (not isinstance(value, str) or not value)):
            raise PlanFlowError('INVALID_INPUT_BINDINGS', 'Bindings must name fields, with text_keys as a list')
    bound_fields = [v for k, value in selected.items() for v in (value if k == 'text_keys' else [value])]
    missing = sorted(set(bound_fields) - set(fields))
    warnings = list(warnings or [])
    if missing: warnings.append({'code': 'BINDING_NOT_OBSERVED', 'fields': missing})
    kinds = modalities if modalities is not None else [kind for key, kind in [('text_keys', 'text'), ('image_key', 'image'), ('video_key', 'video'), ('audio_key', 'audio')] if selected.get(key)]
    stats = {}
    for field in fields:
        values = [row[field] for row in samples if isinstance(row, dict) and field in row]
        stats[field] = {'present': len(values), 'missing': len(samples) - len(values),
                        'null': sum(value is None for value in values), 'types': sorted({type(value).__name__ for value in values})}
    status = 'unsupported' if not supported else 'empty' if complete and total == 0 else 'complete' if complete else 'partial'
    return {'fields': fields, 'bindings': selected, 'modality': 'multimodal' if len(kinds) > 1 else kinds[0] if kinds else 'unknown',
            'record_count': total, 'profile': {'schema_version': 1, 'status': status,
                'record_count': {'value': total, 'exact': total is not None, 'observed': scanned},
                'sample': {'method': 'head', 'limit': limit, 'size': len(samples), 'representative': False},
                'field_statistics': {'scope': 'sample', 'records': len(samples), 'fields': stats},
                'bindings': {'source': 'explicit' if bindings is not None else 'inferred', 'verified_scope': 'sample', 'unobserved_fields': missing},
                'media_content': 'not_decoded', 'quality': 'not_evaluated', 'warnings': warnings}}


def _public_samples(samples):
    """Keep binary/date values serializable and tool responses bounded."""
    truncated = False
    def compact(value, depth=0):
        nonlocal truncated
        if depth > 8:
            truncated = True
            return '<depth limit>'
        if isinstance(value, bytes): return {'binary_bytes': len(value)}
        if isinstance(value, str) and len(value) > 2048:
            truncated = True
            return value[:2048] + '…'
        if isinstance(value, dict):
            if len(value) > 100: truncated = True
            return {str(k):compact(v, depth+1) for k,v in list(value.items())[:100]}
        if isinstance(value, (list, tuple)):
            if len(value) > 100: truncated = True
            return [compact(v, depth+1) for v in value[:100]]
        if isinstance(value, float):
            import math
            return value if math.isfinite(value) else str(value)
        if value is None or isinstance(value, (str,int,bool)): return value
        return str(value)
    result = [compact(row) for row in samples]
    return result, truncated


def inspect_input(workspace_root, input, sample_size=20):
    workspace = require_workspace(workspace_root)
    if not isinstance(input, dict) or not input.get('path'):
        raise PlanFlowError('INPUT_REQUIRED', 'input.path is required')
    path = resolve_workspace_path(input['path'], workspace)
    if not is_within(path, workspace): raise PlanFlowError('PATH_NOT_ALLOWED', 'Input must be inside workspace')
    if not path.exists(): raise PlanFlowError('INPUT_NOT_FOUND', 'Input does not exist')
    try:
        limit = max(1, min(int(sample_size), 100))
        max_records = max(1, min(int(input.get('max_scan_records', MAX_RECORDS)), MAX_RECORDS))
    except (TypeError, ValueError) as exc:
        raise PlanFlowError('INVALID_PROFILE_LIMIT', 'Profile limits must be integers') from exc
    base = {'ok': True, 'workspace_root': str(workspace), 'source_path': str(path), 'dataset_path': str(path)}
    if path.is_dir():
        media = []
        visited = 0
        for directory, dirs, files in os.walk(path, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != '.dj' and not (Path(directory) / d).is_symlink() and not getattr(Path(directory) / d, 'is_junction', lambda: False)())
            for name in sorted(files):
                visited += 1
                if visited > MAX_RECORDS: raise PlanFlowError('INPUT_TOO_MANY_FILES', 'Split directories exceeding 100000 inspected files')
                file = Path(directory) / name
                kind = next((kind for kind, suffixes in MEDIA.items() if file.suffix.lower() in suffixes), None)
                if not kind: continue
                if not is_within(file.resolve(), workspace) or file.is_symlink():
                    raise PlanFlowError('PATH_NOT_ALLOWED', 'Media references must stay inside workspace')
                media.append((file, kind))
        samples = []
        kinds = sorted({kind for _, kind in media})
        if media:
            input_id = 'input_' + uuid.uuid4().hex[:12]
            input_dir = workspace / '.dj' / 'inputs' / input_id
            input_dir.mkdir(parents=True)
            manifest = input_dir / 'manifest.jsonl'
            with manifest.open('w', encoding='utf-8') as handle:
                for file, kind in media:
                    row = {'text': f'<__dj__{kind}>', kind + 's': [str(file)]}
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                    if len(samples) < limit: samples.append(row)
            base.update(input_id=input_id, dataset_path=str(manifest), manifest_sha256=sha256_file(manifest))
        result = {**base, 'kind': 'directory', 'samples': samples,
                  **_profile(samples, scanned=len(media), total=len(media), complete=True, supported=True,
                             limit=limit, modalities=kinds, bindings=input.get('bindings'))}
        result['profile']['directory'] = {'inspected_files': visited, 'media_files': len(media), 'excluded_directories': ['.dj', 'symlinks', 'junctions']}
        # Generated columns are known from the full inventory, even outside the head sample.
        if media and 'bindings' not in input:
            result['bindings'] = {'text_keys': ['text'], **{kind + '_key': kind + 's' for kind in kinds}}
            result['profile']['bindings']['source'] = 'generated_manifest'
            result['profile']['bindings']['verified_scope'] = 'inventory'
        if media: write_json_atomic(input_dir / 'input.json', {k: v for k, v in result.items() if k != 'samples'})
        return result

    samples, scanned, total, complete, supported, warnings = [], 0, None, False, True, []
    suffix = path.suffix.lower()
    before = path.stat()
    try:
        if suffix == '.parquet':
            import pyarrow.parquet as pq
            parquet = pq.ParquetFile(path)
            total = parquet.metadata.num_rows
            for batch in parquet.iter_batches(batch_size=min(limit, max_records)):
                samples.extend(batch.to_pylist()[:limit - len(samples)])
                break
            scanned, complete = len(samples), True  # Exact metadata count; field statistics still sampled.
        elif suffix in {'.json', '.jsonl', '.csv', '.tsv', '.txt', '.text'}:
            with path.open('rb') as handle: raw = handle.read(MAX_BYTES + 1)
            bounded = len(raw) > MAX_BYTES
            if bounded:
                raw = raw[:MAX_BYTES]
                raw = raw[:raw.rfind(b'\n') + 1]  # Do not parse a cut record or UTF-8 character.
                warnings.append({'code': 'BYTE_SCAN_LIMIT', 'max_bytes': MAX_BYTES})
            if suffix == '.json' and bounded:
                warnings.append({'code': 'JSON_TOO_LARGE', 'message': 'Use JSONL or Parquet for bounded profiling'})
            else:
                content = raw.decode('utf-8-sig')
                if suffix == '.json':
                    value = json.loads(content)
                    rows = value if isinstance(value, list) else [value]
                elif suffix == '.jsonl': rows = (json.loads(line) for line in content.splitlines() if line.strip())
                elif suffix in {'.csv', '.tsv'}: rows = csv.DictReader(io.StringIO(content, newline=''), delimiter='\t' if suffix == '.tsv' else ',', strict=True)
                else: rows = ({'text': line} for line in content.splitlines())
                exhausted = True
                for row in rows:
                    if scanned >= max_records:
                        exhausted = False
                        warnings.append({'code': 'RECORD_SCAN_LIMIT', 'max_records': max_records})
                        break
                    scanned += 1
                    if len(samples) < limit: samples.append(row)
                complete = exhausted and not bounded
                total = scanned if complete else None
        else:
            supported = False
            warnings.append({'code': 'UNSUPPORTED_FORMAT', 'suffix': suffix})
    except (ValueError, UnicodeError, csv.Error, OSError) as exc:
        raise PlanFlowError('INPUT_PARSE_FAILED', f'Cannot profile {path.name}: {exc}') from exc
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PlanFlowError('INPUT_CHANGED', 'Input changed while profiling; inspect again')
    public_samples, truncated = _public_samples(samples)
    result = {**base, 'kind': 'file', 'samples': public_samples, 'sha256': sha256_file(path),
              **_profile(samples, scanned=scanned, total=total, complete=complete, supported=supported,
                         limit=limit, bindings=input.get('bindings'), warnings=warnings)}
    result['profile']['sample']['values_truncated'] = truncated
    result['profile']['fingerprint_scope'] = 'whole_file_bytes'
    result['profile']['count_method'] = 'parquet_metadata' if suffix == '.parquet' else 'bounded_scan'
    return result

"""Freeze local inputs and referenced media for native execution."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from .common import PlanFlowError, is_within, sha256_file

MEDIA = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.gif', '.mp4', '.mov', '.avi', '.mkv', '.wav', '.mp3', '.flac'}


def freeze_inputs(plan: dict, version_path: Path, workspace: Path) -> None:
    recipe = plan['recipe']
    sources = []
    if recipe.get('dataset_path'):
        sources.append((recipe, 'dataset_path'))
    elif isinstance(recipe.get('dataset'), dict):
        for config in recipe['dataset'].get('configs', []):
            if config.get('type') != 'local' or not config.get('path'):
                raise PlanFlowError('INPUT_SNAPSHOT_UNSUPPORTED', 'Controlled plans require versioned local input')
            sources.append((config, 'path'))
    if not sources:
        raise PlanFlowError('INPUT_SNAPSHOT_UNSUPPORTED', 'Use a local dataset manifest for controlled execution')
    root = version_path / 'input'
    root.mkdir()
    inventory = []
    copied = {}
    source_identities = []
    total = 0
    media_keys = {'images', 'image', 'image_path', 'videos', 'video', 'video_path', 'audios', 'audio', 'audio_path'}
    media_keys.update(str(recipe[key]) for key in ('image_key', 'video_key', 'audio_key') if recipe.get(key))

    def copy_media(value, base, media=False):
        nonlocal total
        if isinstance(value, dict):
            return {k: copy_media(v, base, media or k in media_keys) for k, v in value.items()}
        if isinstance(value, list):
            return [copy_media(v, base, media) for v in value]
        if not media or not isinstance(value, str):
            return value
        if '://' in value:
            raise PlanFlowError('INPUT_REMOTE_UNVERSIONED', 'Materialize remote media before preparing a plan')
        source = (base / value).resolve()
        if not is_within(source, workspace) or not source.is_file():
            raise PlanFlowError('INPUT_REFERENCE_MISSING', f'Unavailable local media: {source}')
        if source not in copied:
            if len(copied) >= 100000:
                raise PlanFlowError('INPUT_TOO_MANY_FILES', 'Input exceeds 100000 referenced files')
            total += source.stat().st_size
            if total > 10 * 1024**3:
                raise PlanFlowError('INPUT_TOO_LARGE', 'Input exceeds the 10 GiB snapshot limit')
            digest = sha256_file(source)
            target = root / 'media' / digest.removeprefix('sha256:') / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
            if sha256_file(target) != digest or sha256_file(source) != digest:
                raise PlanFlowError('INPUT_CHANGED', 'Input changed while snapshotting; prepare again')
            inventory.append({'path': target.relative_to(version_path).as_posix(), 'sha256': digest})
            copied[source] = str(target)
        return copied[source]

    for index, (container, key) in enumerate(sources):
        source = Path(container[key]).resolve()
        if not is_within(source, workspace) or not source.is_file():
            raise PlanFlowError('INPUT_SNAPSHOT_UNSUPPORTED', 'Inspect raw directories to create a manifest first')
        if source.stat().st_size + total > 10 * 1024**3:
            raise PlanFlowError('INPUT_TOO_LARGE', 'Input exceeds the 10 GiB snapshot limit')
        before = sha256_file(source)
        source_identities.append(before)
        target = root / f'dataset-{index}{source.suffix.lower()}'
        suffix = source.suffix.lower()
        if suffix in {'.json', '.jsonl'}:
            with source.open(encoding='utf-8-sig') as reader, target.open('w', encoding='utf-8') as writer:
                if suffix == '.json':
                    json.dump(copy_media(json.load(reader), source.parent), writer, ensure_ascii=False)
                else:
                    for line in reader:
                        if line.strip():
                            writer.write(json.dumps(copy_media(json.loads(line), source.parent), ensure_ascii=False) + '\n')
        elif suffix in {'.csv', '.tsv'}:
            with source.open(encoding='utf-8-sig', newline='') as reader, target.open('w', encoding='utf-8', newline='') as writer:
                rows = csv.DictReader(reader, delimiter='\t' if suffix == '.tsv' else ',')
                output = csv.DictWriter(writer, fieldnames=rows.fieldnames or [], delimiter='\t' if suffix == '.tsv' else ',')
                output.writeheader()
                for row in rows:
                    output.writerow(copy_media(row, source.parent))
        elif suffix == '.parquet':
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pq.read_table(source)
            pq.write_table(pa.Table.from_pylist(copy_media(table.to_pylist(), source.parent), schema=table.schema), target)
        elif suffix in {'.txt', '.text'}:
            shutil.copyfile(source, target)
        else:
            raise PlanFlowError('INPUT_SNAPSHOT_UNSUPPORTED', f'Unsupported input snapshot format: {suffix}')
        if sha256_file(source) != before:
            raise PlanFlowError('INPUT_CHANGED', 'Dataset changed while snapshotting; prepare again')
        total += target.stat().st_size
        inventory.append({'path': target.relative_to(version_path).as_posix(), 'sha256': sha256_file(target)})
        container[key] = str(target)
    from .common import canonical_json, sha256_bytes
    content_id=sha256_bytes(canonical_json({'datasets':source_identities,'media':sorted(item['sha256'] for item in inventory if item['path'].startswith('input/media/'))}))
    plan['input_snapshot'] = {'schema_version': 1, 'content_id':content_id, 'files': inventory, 'size_bytes': total}


def verify_inputs(plan: dict, version_path: Path):
    snapshot = plan.get('input_snapshot')
    if not snapshot or not snapshot.get('files'):
        raise PlanFlowError('INPUT_SNAPSHOT_REQUIRED', 'Create a new plan version with frozen inputs')
    for item in snapshot['files']:
        path = (version_path / item['path']).resolve()
        if not is_within(path, version_path) or not path.is_file() or sha256_file(path) != item['sha256']:
            raise PlanFlowError('INPUT_SNAPSHOT_CHANGED', 'Frozen input changed; create a new plan version')

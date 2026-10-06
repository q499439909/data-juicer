"""Freeze local manifests while keeping referenced media in their source tree."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from .common import PlanFlowError, is_within, sha256_file

MEDIA = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".wav",
    ".mp3",
    ".flac",
}


def freeze_inputs(plan: dict, version_path: Path, workspace: Path) -> None:
    recipe = plan["recipe"]
    sources = []
    if recipe.get("dataset_path"):
        sources.append((recipe, "dataset_path"))
    elif isinstance(recipe.get("dataset"), dict):
        for config in recipe["dataset"].get("configs", []):
            if config.get("type") != "local" or not config.get("path"):
                raise PlanFlowError("INPUT_SNAPSHOT_UNSUPPORTED", "Controlled plans require versioned local input")
            sources.append((config, "path"))
    if not sources:
        raise PlanFlowError("INPUT_SNAPSHOT_UNSUPPORTED", "Use a local dataset manifest for controlled execution")
    root = version_path / "input"
    root.mkdir()
    inventory = []
    referenced = {}
    source_identities = []
    referenced_bytes = 0
    media_keys = {"images", "image", "image_path", "videos", "video", "video_path", "audios", "audio", "audio_path"}
    media_keys.update(str(recipe[key]) for key in ("image_key", "video_key", "audio_key") if recipe.get(key))

    def reference_media(value, base, media=False):
        nonlocal referenced_bytes
        if isinstance(value, dict):
            return {k: reference_media(v, base, media or k in media_keys) for k, v in value.items()}
        if isinstance(value, list):
            return [reference_media(v, base, media) for v in value]
        if not media or not isinstance(value, str):
            return value
        if "://" in value:
            raise PlanFlowError("INPUT_REMOTE_UNVERSIONED", "Materialize remote media before preparing a plan")
        source = (base / value).resolve()
        if not is_within(source, workspace) or not source.is_file():
            raise PlanFlowError("INPUT_REFERENCE_MISSING", f"Unavailable local media: {source}")
        if source not in referenced:
            before = source.stat()
            digest = sha256_file(source)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise PlanFlowError("INPUT_CHANGED", "Input changed while indexing; prepare again")
            referenced_bytes += after.st_size
            referenced[source] = {"path": str(source), "size_bytes": after.st_size, "sha256": digest}
        return str(source)

    for index, (container, key) in enumerate(sources):
        source = Path(container[key]).resolve()
        if not is_within(source, workspace) or not source.is_file():
            raise PlanFlowError("INPUT_SNAPSHOT_UNSUPPORTED", "Inspect raw directories to create a manifest first")
        before = sha256_file(source)
        source_identities.append(before)
        target = root / f"dataset-{index}{source.suffix.lower()}"
        suffix = source.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            with source.open(encoding="utf-8-sig") as reader, target.open("w", encoding="utf-8") as writer:
                if suffix == ".json":
                    json.dump(reference_media(json.load(reader), source.parent), writer, ensure_ascii=False)
                else:
                    for line in reader:
                        if line.strip():
                            writer.write(
                                json.dumps(reference_media(json.loads(line), source.parent), ensure_ascii=False) + "\n"
                            )
        elif suffix in {".csv", ".tsv"}:
            with (
                source.open(encoding="utf-8-sig", newline="") as reader,
                target.open("w", encoding="utf-8", newline="") as writer,
            ):
                rows = csv.DictReader(reader, delimiter="\t" if suffix == ".tsv" else ",")
                output = csv.DictWriter(
                    writer, fieldnames=rows.fieldnames or [], delimiter="\t" if suffix == ".tsv" else ","
                )
                output.writeheader()
                for row in rows:
                    output.writerow(reference_media(row, source.parent))
        elif suffix == ".parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pq.read_table(source)
            pq.write_table(
                pa.Table.from_pylist(reference_media(table.to_pylist(), source.parent), schema=table.schema), target
            )
        elif suffix in {".txt", ".text"}:
            shutil.copyfile(source, target)
        else:
            raise PlanFlowError("INPUT_SNAPSHOT_UNSUPPORTED", f"Unsupported input snapshot format: {suffix}")
        if sha256_file(source) != before:
            raise PlanFlowError("INPUT_CHANGED", "Dataset changed while snapshotting; prepare again")
        inventory.append({"path": target.relative_to(version_path).as_posix(), "sha256": sha256_file(target)})
        container[key] = str(target)
    media_inventory = root / "media-inventory.jsonl"
    with media_inventory.open("w", encoding="utf-8") as writer:
        for item in sorted(referenced.values(), key=lambda value: value["path"].casefold()):
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")
    inventory.append(
        {"path": media_inventory.relative_to(version_path).as_posix(), "sha256": sha256_file(media_inventory)}
    )
    from .common import canonical_json, sha256_bytes

    media_inventory_hash = sha256_file(media_inventory)
    content_id = sha256_bytes(
        canonical_json(
            {
                "datasets": source_identities,
                "media_inventory": media_inventory_hash,
            }
        )
    )
    stored_bytes = sum((version_path / item["path"]).stat().st_size for item in inventory)
    plan["input_snapshot"] = {
        "schema_version": 2,
        "content_id": content_id,
        "files": inventory,
        "size_bytes": stored_bytes,
        "source_media": {
            "inventory_path": media_inventory.relative_to(version_path).as_posix(),
            "count": len(referenced),
            "size_bytes": referenced_bytes,
        },
    }


def verify_inputs(plan: dict, version_path: Path, workspace: Path | None = None):
    snapshot = plan.get("input_snapshot")
    if not snapshot or not snapshot.get("files"):
        raise PlanFlowError("INPUT_SNAPSHOT_REQUIRED", "Create a new plan version with frozen inputs")
    for item in snapshot["files"]:
        path = (version_path / item["path"]).resolve()
        if not is_within(path, version_path) or not path.is_file() or sha256_file(path) != item["sha256"]:
            raise PlanFlowError("INPUT_SNAPSHOT_CHANGED", "Frozen input changed; create a new plan version")
    source_media = snapshot.get("source_media")
    if not source_media:
        return
    workspace = workspace.resolve() if workspace is not None else None
    inventory_path = (version_path / str(source_media.get("inventory_path", ""))).resolve()
    if not is_within(inventory_path, version_path) or not inventory_path.is_file():
        raise PlanFlowError("INPUT_SNAPSHOT_CHANGED", "Source media inventory is missing")
    observed_count = 0
    observed_bytes = 0
    with inventory_path.open(encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            item = json.loads(line)
            source = Path(str(item.get("path", ""))).resolve()
            if (
                workspace is None
                or not is_within(source, workspace)
                or not source.is_file()
                or source.stat().st_size != item.get("size_bytes")
                or sha256_file(source) != item.get("sha256")
            ):
                raise PlanFlowError("INPUT_SNAPSHOT_CHANGED", f"Referenced input changed: {source}")
            observed_count += 1
            observed_bytes += source.stat().st_size
    if observed_count != source_media.get("count") or observed_bytes != source_media.get("size_bytes"):
        raise PlanFlowError("INPUT_SNAPSHOT_CHANGED", "Referenced input inventory changed")

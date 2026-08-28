"""Input inspection and Data-Juicer operator discovery."""

from __future__ import annotations

import csv
import inspect
import json
import os
import uuid
from functools import lru_cache
from typing import Any

from .common import (
    PlanFlowError,
    is_within,
    require_workspace,
    resolve_workspace_path,
    sha256_file,
    write_json_atomic,
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
_SEARCH_MODALITIES = {"text", "image", "audio", "video", "multimodal"}
_MEDIA_MODALITIES = {"image", "audio", "video"}
_MAX_SEARCH_TOP_K = 5


@lru_cache(maxsize=1)
def _searcher():
    from data_juicer.tools.op_search import OPSearcher

    return OPSearcher()


def operator_record(name: str):
    for record in _searcher().op_records:
        if record.name == name:
            return record
    return None


def operator_schema(name: str) -> dict[str, Any] | None:
    record = operator_record(name)
    if record is None:
        return None
    parameters = {}
    for parameter in record.sig.parameters.values():
        if parameter.name in {"self", "args", "kwargs"}:
            continue
        default = None if parameter.default is inspect.Parameter.empty else parameter.default
        annotation = "Any" if parameter.annotation is inspect.Parameter.empty else str(parameter.annotation)
        parameters[parameter.name] = {
            "type": annotation,
            "required": parameter.default is inspect.Parameter.empty,
            "default": default,
            "description": record.param_desc_map.get(parameter.name, ""),
        }
    return {
        "name": record.name,
        "type": record.type,
        "description": record.desc.strip(),
        "tags": list(record.tags),
        "parameters": parameters,
    }


def capability_schemas(operator_names: list[str]) -> dict[str, Any]:
    """Return full schemas for already-discovered operators by exact name."""
    operators = []
    missing = []
    seen = set()
    for value in operator_names:
        name = str(value or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        schema = operator_schema(name)
        if schema is None:
            missing.append(name)
        else:
            operators.append(schema)
    return {"ok": not missing, "operators": operators, "missing": missing}


def runtime_capabilities() -> dict[str, Any]:
    """Report safe runtime configuration while keeping credentials secret."""
    vlm_model = str(os.environ.get("DJ_VLM_MODEL") or "").strip()
    return {
        "api_credentials_configured": bool(
            os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("SK")
        ),
        "api_base_url_configured": bool(
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_URL")
            or os.environ.get("DASHSCOPE_BASE_URL")
        ),
        "vlm_model_configured": bool(vlm_model),
        "default_models": {
            "vlm": {
                "configured": bool(vlm_model),
                "model": vlm_model or None,
                "role": "vision-language",
                "source": "server_environment",
            }
        },
    }


def _search_tags(modality: str | None) -> list[str] | None:
    if modality not in _SEARCH_MODALITIES:
        return None
    if modality in _MEDIA_MODALITIES:
        return [modality, "multimodal"]
    return [modality]


def _compact_candidate(record, modality: str | None, executor_type: str) -> dict[str, Any]:
    """Build the compact definition returned during capability discovery."""
    tags = set(record.tags)
    return {
        "name": record.name,
        "type": record.type,
        "description": record.desc.strip(),
        "tags": list(record.tags),
        "signature": str(record.sig),
        "parameter_descriptions": record.param_desc.strip(),
        "modality_compatible": modality not in _SEARCH_MODALITIES or modality in tags or (
            modality in _MEDIA_MODALITIES and "multimodal" in tags
        ),
        "executor_compatible": not (
            executor_type.startswith("ray")
            and record.name.startswith("document_")
            and "deduplicator" in record.name
        ),
    }


def search_capabilities(
    requirements: list[str],
    modality: str | None = None,
    executor_type: str = "default",
    top_k: int = 3,
) -> dict[str, Any]:
    """Search all requirements with BM25 and return deduplicated compact definitions."""
    searcher = _searcher()
    rows = []
    unique_operators = {}
    tags = _search_tags(modality)
    limit = max(1, min(int(top_k), _MAX_SEARCH_TOP_K))
    for requirement in requirements:
        query = str(requirement or "").strip()
        if not query:
            continue
        candidate_names = []
        seen = set()
        exact_record = operator_record(query)
        if exact_record is not None:
            candidate_names.append(query)
            seen.add(query)
            if query in unique_operators:
                unique_operators[query]["matched_requirements"].append(query)
            else:
                compact = _compact_candidate(exact_record, modality, executor_type)
                compact["matched_requirements"] = [query]
                unique_operators[query] = compact
        matches = searcher.search_by_bm25(
            query=query,
            fields=["name", "desc", "param_desc", "sig"],
            top_k=limit,
            tags=tags,
            match_all=False,
        )
        for match in matches:
            if match["name"] in seen:
                continue
            record = operator_record(match["name"])
            if record is not None:
                candidate_names.append(record.name)
                seen.add(record.name)
                if record.name in unique_operators:
                    unique_operators[record.name]["matched_requirements"].append(query)
                else:
                    compact = _compact_candidate(record, modality, executor_type)
                    compact["matched_requirements"] = [query]
                    unique_operators[record.name] = compact
            if len(candidate_names) >= limit:
                break
        rows.append(
            {
                "requirement": query,
                "coverage": "candidates" if candidate_names else "gap",
                "operator_names": candidate_names,
                "fallbacks": [] if candidate_names else ["postprocess_script", "custom_operator"],
            }
        )
    return {
        "ok": True,
        "executor_type": executor_type,
        "top_k": limit,
        "runtime": runtime_capabilities(),
        "results": rows,
        "operators": list(unique_operators.values()),
    }


def inspect_input(workspace_root: str, input: dict[str, Any], sample_size: int = 20) -> dict[str, Any]:
    """Inspect a local dataset or turn a raw media directory into a DJ JSONL manifest."""
    workspace = require_workspace(workspace_root)
    if not isinstance(input, dict) or not input.get("path"):
        raise PlanFlowError("INPUT_REQUIRED", "input.path is required")
    path = resolve_workspace_path(input["path"], workspace)
    if not is_within(path, workspace):
        raise PlanFlowError("PATH_NOT_ALLOWED", f"Input must be inside workspace: {path}")
    if not path.exists():
        raise PlanFlowError("INPUT_NOT_FOUND", f"Input does not exist: {path}")
    limit = max(1, min(int(sample_size), 100))
    if path.is_dir():
        media_files = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES | _VIDEO_SUFFIXES
        )
        if media_files:
            input_id = f"input_{uuid.uuid4().hex[:12]}"
            input_dir = workspace / ".dj" / "inputs" / input_id
            input_dir.mkdir(parents=True, exist_ok=False)
            manifest = input_dir / "manifest.jsonl"
            records = []
            with manifest.open("w", encoding="utf-8") as handle:
                for media in media_files:
                    if media.suffix.lower() in _IMAGE_SUFFIXES:
                        record = {"text": "<__dj__image>", "images": [str(media)]}
                    else:
                        record = {"text": "<__dj__video>", "videos": [str(media)]}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if len(records) < limit:
                        records.append(record)
            descriptor = {
                "input_id": input_id,
                "source_path": str(path),
                "dataset_path": str(manifest),
                "record_count": len(media_files),
                "modality": (
                    "multimodal"
                    if any("images" in r for r in records) and any("videos" in r for r in records)
                    else ("image" if any("images" in r for r in records) else "video")
                ),
                "bindings": {"text_keys": ["text"], "image_key": "images", "video_key": "videos"},
                "manifest_sha256": sha256_file(manifest),
            }
            write_json_atomic(input_dir / "input.json", descriptor)
            return {"ok": True, "workspace_root": str(workspace), **descriptor, "samples": records}
        return {
            "ok": True,
            "workspace_root": str(workspace),
            "dataset_path": str(path),
            "kind": "directory",
            "modality": "unknown",
            "samples": [],
        }

    suffix = path.suffix.lower()
    samples: list[Any] = []
    if suffix in {".jsonl", ".json"}:
        text = path.read_text(encoding="utf-8")
        if suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    samples.append(json.loads(line))
                    if len(samples) >= limit:
                        break
        else:
            value = json.loads(text)
            samples = (value if isinstance(value, list) else [value])[:limit]
    elif suffix in {".csv", ".tsv"}:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")
            samples = [row for _, row in zip(range(limit), reader)]
    elif suffix == ".parquet":
        import pandas as pd

        samples = pd.read_parquet(path).head(limit).to_dict(orient="records")
    keys = sorted({key for sample in samples if isinstance(sample, dict) for key in sample})
    modality = (
        "multimodal"
        if sum(bool(set(keys) & group) for group in ({"text"}, {"images", "image"}, {"audios"}, {"videos"})) > 1
        else (
            "image"
            if set(keys) & {"images", "image", "image_bytes"}
            else "audio" if "audios" in keys else "video" if "videos" in keys else "text"
        )
    )
    return {
        "ok": True,
        "workspace_root": str(workspace),
        "dataset_path": str(path),
        "kind": "file",
        "modality": modality,
        "fields": keys,
        "samples": samples,
        "sha256": sha256_file(path),
    }

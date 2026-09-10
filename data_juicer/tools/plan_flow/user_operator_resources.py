"""Resource references reuse backend caches; test cleanup never owns model caches."""

import importlib.util
import os
import re
from pathlib import Path

from .common import PlanFlowError, sha256_file


def model_refs(refs, *, freeze=False):
    if not isinstance(refs, list):
        raise PlanFlowError("INVALID_MODEL_REFS", "model_refs must be a list")
    result = []
    for ref in refs:
        if not isinstance(ref, dict):
            raise PlanFlowError("INVALID_MODEL_REFS", "Every model reference must be an object")
        item = dict(ref)
        backend = "local-file" if item.get("path") else str(item.get("backend") or "huggingface")
        if backend not in {"local-file", "huggingface", "modelscope", "http-file", "torch-hub"}:
            raise PlanFlowError("INVALID_MODEL_REFS", f"Unsupported model backend: {backend}")
        item["backend"] = backend
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(item.get("parameter", ""))):
            raise PlanFlowError(
                "MODEL_REQUIREMENT_UNDECLARED",
                "Every custom model reference must name its constructor parameter",
            )
        if backend == "local-file":
            path = Path(item["path"]).expanduser()
            if not path.is_absolute() or not path.is_file():
                raise PlanFlowError(
                    "MODEL_NOT_FOUND",
                    "Local model references must point to an existing absolute file",
                )
            actual = sha256_file(path)
            if item.get("sha256") and item["sha256"] != actual:
                raise PlanFlowError("MODEL_HASH_MISMATCH", "Referenced local model changed")
            if not freeze and not item.get("sha256"):
                raise PlanFlowError(
                    "MODEL_HASH_REQUIRED",
                    "Local models must be fingerprinted by validation",
                )
            item["sha256"] = actual
        elif backend in {"huggingface", "modelscope"} and (
            not item.get("model_id") or not re.fullmatch(r"[a-fA-F0-9]{40,64}", str(item.get("revision", "")))
        ):
            raise PlanFlowError(
                "MODEL_REVISION_REQUIRED",
                "Remote model references need model_id and an immutable commit revision",
            )
        elif backend == "http-file":
            digest = str(item.get("sha256", "")).removeprefix("sha256:")
            if (
                not str(item.get("url", "")).startswith("https://")
                or not item.get("filename")
                or type(item.get("size")) is not int
                or not re.fullmatch(r"[a-f0-9]{64}", digest)
            ):
                raise PlanFlowError(
                    "MODEL_HASH_REQUIRED", "HTTP model refs require HTTPS URL, filename, size and SHA256"
                )
            item["sha256"] = digest
        elif backend == "torch-hub" and (
            not str(item.get("repository_url", "")).startswith("https://")
            or not re.fullmatch(r"[a-fA-F0-9]{40,64}", str(item.get("revision", "")))
            or not item.get("files")
        ):
            raise PlanFlowError(
                "MODEL_REVISION_REQUIRED", "Torch Hub refs require repository_url, immutable commit and files"
            )
        if item.get("files") is not None:
            if not isinstance(item["files"], list):
                raise PlanFlowError("INVALID_MODEL_REFS", "Model files must be an array")
            for file in item["files"]:
                relative = Path(str(file.get("path", ""))) if isinstance(file, dict) else Path("..")
                if (
                    not isinstance(file, dict)
                    or set(file) != {"path", "size", "sha256"}
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or type(file["size"]) is not int
                    or file["size"] < 0
                    or not re.fullmatch(r"[a-f0-9]{64}", str(file["sha256"]))
                ):
                    raise PlanFlowError("INVALID_MODEL_REFS", "Model file identity is invalid")
        result.append(item)
    return result


def cache_snapshot():
    from data_juicer.utils import cache_utils

    snapshot = {
        key: str(getattr(cache_utils, key, ""))
        for key in (
            "DATA_JUICER_CACHE_HOME",
            "DATA_JUICER_ASSETS_CACHE",
            "DATA_JUICER_MODELS_CACHE",
            "DATA_JUICER_EXTERNAL_MODELS_HOME",
        )
    }
    if importlib.util.find_spec("huggingface_hub"):
        from huggingface_hub import constants

        snapshot.update(HF_HOME=constants.HF_HOME, HF_HUB_CACHE=constants.HF_HUB_CACHE)
    if importlib.util.find_spec("torch"):
        import torch

        snapshot["TORCH_HUB"] = torch.hub.get_dir()
    snapshot["MODELSCOPE_CACHE"] = os.environ.get("MODELSCOPE_CACHE", "backend default (not probed)")
    return snapshot


def validate_assets(assets):
    if not isinstance(assets, dict) or sum(len(str(value).encode()) for value in assets.values()) > 256_000:
        raise PlanFlowError(
            "INVALID_OPERATOR_ASSETS",
            "assets must be at most 256 KB of small text resources, not models",
        )
    for name, value in assets.items():
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)?", name):
            raise PlanFlowError(
                "INVALID_OPERATOR_ASSETS",
                "Asset names must be plain filenames without paths",
            )
    return {name: value.replace("\r\n", "\n") for name, value in assets.items()}

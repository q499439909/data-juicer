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
        item = dict(ref)
        if item.get("path"):
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
        elif not item.get("model_id") or not re.fullmatch(r"[a-fA-F0-9]{40,64}", str(item.get("revision", ""))):
            raise PlanFlowError(
                "MODEL_REVISION_REQUIRED",
                "Remote model references need model_id and an immutable commit revision",
            )
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

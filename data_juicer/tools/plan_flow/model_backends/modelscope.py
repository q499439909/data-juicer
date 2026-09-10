"""Resolve a ModelScope snapshot at an immutable revision."""

from __future__ import annotations

import os
from pathlib import Path

from ..common import PlanFlowError
from .huggingface import HuggingFaceModelBackend


class ModelScopeModelBackend(HuggingFaceModelBackend):
    name = "modelscope"

    def _find(self, binding: dict) -> Path | None:
        roots = []
        if self.preloaded_root:
            roots.append(self.preloaded_root / "modelscope" / binding["model_id"] / binding["revision"])
        configured = self.cache_root or Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
        roots.append(Path(configured) / "hub" / binding["model_id"])
        for candidate in roots:
            if candidate.is_dir():
                return candidate.resolve()
        return None

    def prepare(self, binding: dict, *, offline: bool) -> Path:
        existing = self._find(binding)
        if existing is not None:
            self.verify(binding, existing)
            return existing
        if offline:
            raise PlanFlowError("MODEL_NOT_PREPARED", f"Locked ModelScope model is absent: {binding['lock_id']}")
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise PlanFlowError(
                "MODEL_DOWNLOAD_BLOCKED", "modelscope must already be installed from uv.lock"
            ) from exc
        try:
            path = Path(snapshot_download(binding["model_id"], revision=binding["revision"])).resolve()
        except Exception as exc:
            raise PlanFlowError("MODEL_DOWNLOAD_BLOCKED", f"Could not download locked ModelScope model: {exc}") from exc
        self.verify(binding, path)
        return path

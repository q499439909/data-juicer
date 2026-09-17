"""Resolve a Torch Hub repository by commit without running its code."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..common import FileLock, PlanFlowError
from .huggingface import HuggingFaceModelBackend


class TorchHubModelBackend(HuggingFaceModelBackend):
    name = "torch-hub"

    def __init__(self, *, cache_root: str | Path | None = None, preloaded_root: str | Path | None = None):
        super().__init__(cache_root=cache_root, preloaded_root=preloaded_root)
        configured = cache_root or os.environ.get("DSH_MODEL_ARTIFACT_CACHE")
        self.repo_root = Path(configured or Path.home() / ".cache" / "data_juicer" / "model-artifacts").resolve()

    def _find(self, binding: dict) -> Path | None:
        candidates = []
        if self.preloaded_root:
            candidates.append(self.preloaded_root / "torch-hub" / binding["lock_id"] / binding["revision"])
        candidates.append(self.repo_root / "torch-hub" / binding["lock_id"] / binding["revision"])
        return next((item.resolve() for item in candidates if item.is_dir()), None)

    def prepare(self, binding: dict, *, offline: bool) -> Path:
        existing = self._find(binding)
        if existing is not None:
            self.verify(binding, existing)
            return existing
        if offline:
            raise PlanFlowError("MODEL_NOT_PREPARED", f"Locked Torch Hub repository is absent: {binding['lock_id']}")
        target = self.repo_root / "torch-hub" / binding["lock_id"] / binding["revision"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(target.parent / ".prepare.lock"):
            existing = self._find(binding)
            if existing is not None:
                return existing
            temporary = Path(tempfile.mkdtemp(prefix=".git-", dir=target.parent))
            try:
                subprocess.run(["git", "init", "-q", str(temporary)], check=True)
                subprocess.run(
                    ["git", "-C", str(temporary), "remote", "add", "origin", binding["repository_url"]], check=True
                )
                subprocess.run(
                    ["git", "-C", str(temporary), "fetch", "-q", "--depth", "1", "origin", binding["revision"]],
                    check=True,
                )
                subprocess.run(["git", "-C", str(temporary), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
                shutil.rmtree(temporary / ".git", ignore_errors=True)
                self.verify(binding, temporary)
                os.replace(temporary, target)
            except Exception as exc:
                if isinstance(exc, PlanFlowError):
                    raise
                raise PlanFlowError(
                    "MODEL_DOWNLOAD_BLOCKED", f"Could not fetch locked Torch Hub repository: {exc}"
                ) from exc
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        return target.resolve()

"""Resolve immutable Hugging Face revisions without persisting host paths."""

from __future__ import annotations

import os
from pathlib import Path

from ..common import FileLock, PlanFlowError, sha256_file


class HuggingFaceModelBackend:
    name = "huggingface"

    def __init__(self, *, cache_root: str | Path | None = None, preloaded_root: str | Path | None = None):
        self.cache_root = Path(cache_root).expanduser().resolve() if cache_root else None
        configured = preloaded_root or os.environ.get("DSH_PRELOADED_MODEL_ROOT")
        self.preloaded_root = Path(configured).expanduser().resolve() if configured else None

    def prepare(self, binding: dict, *, offline: bool) -> Path:
        existing = self._find(binding)
        if existing is not None:
            try:
                self.verify(binding, existing)
                return existing
            except PlanFlowError:
                if offline:
                    raise
        if offline:
            raise PlanFlowError(
                "MODEL_NOT_PREPARED",
                f"Locked model is not available locally: {binding['model_id']}@{binding['revision']}",
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise PlanFlowError("MODEL_DOWNLOAD_BLOCKED", "huggingface_hub is required to prepare this model") from exc
        kwargs = {
            "repo_id": binding["model_id"],
            "revision": binding["revision"],
            "local_files_only": False,
        }
        if self.cache_root:
            kwargs["cache_dir"] = str(self.cache_root)
        files = [item["path"] for item in binding.get("files", [])]
        if files:
            kwargs["allow_patterns"] = files
        if existing is not None:
            kwargs["force_download"] = True
        lock_root = self.cache_root or self._default_cache_root()
        lock_name = binding["lock_id"].replace("/", "--") + ".lock"
        try:
            with FileLock(lock_root / ".dsh-locks" / lock_name):
                existing = self._find(binding)
                path = existing or Path(snapshot_download(**kwargs)).resolve()
        except PlanFlowError:
            raise
        except Exception as exc:
            raise PlanFlowError(
                "MODEL_DOWNLOAD_BLOCKED",
                f"Could not download locked model {binding['model_id']}@{binding['revision']}: {exc}",
            ) from exc
        self.verify(binding, path)
        return path

    def verify(self, binding: dict, root: Path) -> None:
        if not root.is_dir():
            raise PlanFlowError("MODEL_FILE_MISSING", f"Model snapshot directory is missing: {root}")
        for item in binding.get("files", []):
            path = (root / item["path"]).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError as exc:
                raise PlanFlowError("MODEL_PATH_FORBIDDEN", f"Model file escaped snapshot: {item['path']}") from exc
            if not path.is_file() or path.stat().st_size != int(item["size"]):
                raise PlanFlowError("MODEL_FILE_MISSING", f"Locked model file is missing or truncated: {item['path']}")
            if sha256_file(path).removeprefix("sha256:") != item["sha256"]:
                raise PlanFlowError("MODEL_HASH_MISMATCH", f"Locked model file changed: {item['path']}")

    def _find(self, binding: dict) -> Path | None:
        relative = Path(*("models--" + binding["model_id"].replace("/", "--"), "snapshots", binding["revision"]))
        roots = []
        if self.preloaded_root:
            roots.extend((self.preloaded_root, self.preloaded_root / "hub"))
        roots.append(self.cache_root or self._default_cache_root())
        for root in roots:
            candidate = (root / relative).resolve()
            if candidate.is_dir():
                return candidate
        return None

    @staticmethod
    def _default_cache_root() -> Path:
        try:
            from huggingface_hub import constants

            return Path(constants.HF_HUB_CACHE).expanduser().resolve()
        except ImportError:
            return Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub")).resolve()

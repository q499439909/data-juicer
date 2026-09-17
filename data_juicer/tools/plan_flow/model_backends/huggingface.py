"""Verify and prepare exact model files; a directory is not a ready model."""

from __future__ import annotations

import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from ..common import FileLock, PlanFlowError, sha256_file


@lru_cache(maxsize=1024)
def _stat_hash(path, size, mtime, ctime):
    return sha256_file(Path(path)).removeprefix("sha256:")


class HuggingFaceModelBackend:
    name = "huggingface"

    def __init__(self, *, cache_root=None, preloaded_root=None):
        self.cache_root = Path(cache_root).expanduser().resolve() if cache_root else None
        configured = preloaded_root or os.environ.get("DSH_PRELOADED_MODEL_ROOT")
        self.preloaded_root = Path(configured).expanduser().resolve() if configured else None

    @staticmethod
    def _offline():
        return any(
            os.environ.get(k, "").upper() in {"1", "TRUE", "YES", "ON"} for k in ("HF_HUB_OFFLINE", "DSH_MODEL_OFFLINE")
        )

    @staticmethod
    def _relative(binding):
        return Path("models--" + binding["model_id"].replace("/", "--"), "snapshots", binding["revision"])

    @staticmethod
    def _file(root, item):
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or ":" in item["path"]:
            raise PlanFlowError("MODEL_PATH_FORBIDDEN", "Invalid model file path")
        path = root / relative
        resolved = path.resolve()
        if not any(resolved.is_relative_to(base.resolve()) for base in (root, root.parent.parent / "blobs")):
            raise PlanFlowError("MODEL_PATH_FORBIDDEN", "Model file escaped snapshot and repository blob store")
        return path

    def _issues(self, binding, root, *, fresh=False):
        missing, invalid = [], []
        for item in binding.get("files", []):
            path = self._file(root, item)
            if not path.is_file() or path.stat().st_size != int(item["size"]):
                missing.append(item["path"])
                continue
            stat = path.stat()
            digest = (
                sha256_file(path).removeprefix("sha256:")
                if fresh
                else _stat_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            )
            if digest != item["sha256"]:
                invalid.append(item["path"])
        return missing, invalid

    def inspect(self, binding, *, offline=None):
        offline = self._offline() if offline is None else offline
        cache = self.cache_root or self._default_cache_root()
        root = self._find(binding) or cache / self._relative(binding)
        missing, invalid = self._issues(binding, root)
        writable = cache
        while not writable.exists() and writable != writable.parent:
            writable = writable.parent
        ready = not missing and not invalid
        return {
            "model_id": binding["model_id"],
            "revision": binding["revision"],
            "lock_id": binding["lock_id"],
            "locked": True,
            "cached": root.is_dir(),
            "verified": ready,
            "loadable": "not_checked",
            "status": "ready" if ready else "preparation_required",
            "cache_root": str(cache),
            "resolved_path": str(root),
            "missing_files": missing,
            "invalid_files": invalid,
            "offline": offline,
            "cache_writable": os.access(writable, os.W_OK),
            "can_prepare": not offline and os.access(writable, os.W_OK),
            "download_bytes": sum(i["size"] for i in binding.get("files", []) if i["path"] in missing + invalid),
        }

    def verify(self, binding, root):
        missing, invalid = self._issues(binding, Path(root), fresh=True)
        if missing or invalid:
            code = "MODEL_FILE_MISSING" if missing else "MODEL_HASH_MISMATCH"
            details = {
                "model_id": binding["model_id"],
                "revision": binding["revision"],
                "missing_files": missing,
                "invalid_files": invalid,
                "resolved_path": str(root),
            }
            raise PlanFlowError(
                code,
                f"Model {binding['model_id']}@{binding['revision']}: missing or invalid files: {', '.join(missing+invalid)}",
                details=details,
            )

    def prepare(self, binding, *, offline):
        existing = self._find(binding)
        if existing:
            try:
                self.verify(binding, existing)
                return existing
            except PlanFlowError:
                if offline:
                    raise
        if offline:
            raise PlanFlowError(
                "MODEL_NOT_PREPARED",
                f"Model {binding['model_id']}@{binding['revision']} is unavailable offline",
                details=self.inspect(binding, offline=True),
            )
        cache = self.cache_root or self._default_cache_root()
        target = cache / self._relative(binding)
        lock_name = binding["lock_id"].replace("/", "--") + ".lock"
        try:
            with FileLock(cache / ".dsh-locks" / lock_name):
                existing = self._find(binding)
                if existing:
                    missing, invalid = self._issues(binding, existing, fresh=True)
                    if not missing and not invalid:
                        return existing
                target.mkdir(parents=True, exist_ok=True)
                missing, invalid = self._issues(binding, target, fresh=True)
                needed = set(missing + invalid)
                source_bad = set(sum(self._issues(binding, existing), [])) if existing else set()
                stage_root = cache / ".dsh-staging"
                stage_root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="model-", dir=stage_root) as temporary:
                    stage = Path(temporary)
                    for item in binding.get("files", []):
                        if item["path"] not in needed:
                            continue
                        candidate = self._file(existing, item) if existing and item["path"] not in source_bad else None
                        blobs = target.parent.parent / "blobs"
                        if candidate is None and blobs.is_dir():
                            for blob in blobs.iterdir():
                                if (
                                    blob.is_file()
                                    and not blob.is_symlink()
                                    and blob.stat().st_size == item["size"]
                                    and sha256_file(blob).removeprefix("sha256:") == item["sha256"]
                                ):
                                    candidate = blob
                                    break
                        staged = stage / item["path"]
                        staged.parent.mkdir(parents=True, exist_ok=True)
                        if candidate is not None:
                            shutil.copyfile(candidate, staged)
                        else:
                            from huggingface_hub import hf_hub_download

                            hf_hub_download(
                                repo_id=binding["model_id"],
                                revision=binding["revision"],
                                filename=item["path"],
                                local_dir=str(stage),
                                local_files_only=False,
                            )
                        self.verify({**binding, "files": [item]}, stage)
                        destination = target / item["path"]
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(staged, destination)
                self.verify(binding, target)
                return target
        except PlanFlowError:
            raise
        except Exception as exc:
            raise PlanFlowError(
                "MODEL_PREPARATION_FAILED",
                f"Could not prepare {binding['model_id']}@{binding['revision']}: {type(exc).__name__}",
                details={
                    **self.inspect(binding),
                    "cause_type": type(exc).__name__,
                    "errno": getattr(exc, "errno", None),
                },
            ) from exc

    def _find(self, binding):
        roots = []
        if self.preloaded_root:
            roots.extend((self.preloaded_root, self.preloaded_root / "hub"))
        roots.append(self.cache_root or self._default_cache_root())
        for root in roots:
            candidate = (root / self._relative(binding)).resolve()
            if candidate.is_dir():
                return candidate
        return None

    @staticmethod
    def _default_cache_root():
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE).expanduser().resolve()
